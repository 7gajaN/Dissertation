"""
Force Consistency Score (FCS) - A Physics-Based Evaluation Metric

This metric evaluates dance motion quality by checking if the ground reaction forces
required to produce the observed Center of Mass (CoM) accelerations are physically
feasible given the feet contact positions and biomechanical constraints.

Key improvements over PFC:
1. Uses actual physics equations (F = ma)
2. Considers Center of Mass dynamics
3. Validates force generation capacity based on contact points
4. Accounts for biomechanical limits
"""

import argparse
import glob
import os
import pickle
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


# SMPL joint hierarchy and anthropometric data
# Segment masses as percentage of total body mass (Winter, 2009)
SMPL_SEGMENT_MASSES = {
    'head': 0.081,
    'trunk': 0.497,  # thorax + abdomen + pelvis
    'upper_arm_r': 0.028,
    'upper_arm_l': 0.028,
    'forearm_r': 0.016,
    'forearm_l': 0.016,
    'hand_r': 0.006,
    'hand_l': 0.006,
    'thigh_r': 0.100,
    'thigh_l': 0.100,
    'shank_r': 0.0465,
    'shank_l': 0.0465,
    'foot_r': 0.0145,
    'foot_l': 0.0145,
}

# SMPL joint indices (AIST++ format)
JOINT_NAMES = [
    'root',           # 0
    'lhip',           # 1
    'rhip',           # 2
    'belly',          # 3
    'lknee',          # 4
    'rknee',          # 5
    'spine',          # 6
    'lankle',         # 7
    'rankle',         # 8
    'chest',          # 9
    'ltoes',          # 10
    'rtoes',          # 11
    'neck',           # 12
    'linshoulder',    # 13
    'rinshoulder',    # 14
    'head',           # 15
    'lshoulder',      # 16
    'rshoulder',      # 17
    'lelbow',         # 18
    'relbow',         # 19
    'lwrist',         # 20
    'rwrist',         # 21
    'lhand',          # 22
    'rhand',          # 23
]

# Mapping SMPL joints to body segments for CoM calculation
# Each segment is represented by its representative joints and mass fraction
SEGMENT_JOINT_MAPPING = {
    'head': ([15], 0.081),
    'trunk': ([0, 3, 6, 9, 12], 0.497),  # root to chest
    'upper_arm_r': ([17, 19], 0.028),
    'upper_arm_l': ([16, 18], 0.028),
    'forearm_r': ([19, 21], 0.016),
    'forearm_l': ([18, 20], 0.016),
    'hand_r': ([23], 0.006),
    'hand_l': ([22], 0.006),
    'thigh_r': ([2, 5], 0.100),
    'thigh_l': ([1, 4], 0.100),
    'shank_r': ([5, 8], 0.0465),
    'shank_l': ([4, 7], 0.0465),
    'foot_r': ([8, 11], 0.0145),
    'foot_l': ([7, 10], 0.0145),
}


class ForceConsistencyEvaluator:
    """Evaluates motion physics using biomechanical force analysis."""
    
    def __init__(self, fps=30, gravity=9.81, body_mass=70.0):
        """
        Args:
            fps: Frames per second of motion data
            gravity: Gravitational acceleration (m/s^2)
            body_mass: Average human body mass in kg
        """
        self.dt = 1.0 / fps
        self.gravity = gravity
        self.body_mass = body_mass
        self.up_axis = 2  # z is up
        self.horizontal_axes = [0, 1]  # x, y
        
        # Biomechanical constraints
        self.max_ground_force_multiplier = 3.0  # Max GRF = 3x body weight (running/jumping)
        self.min_contact_velocity = 0.02  # m/s - threshold for foot contact
        self.foot_force_radius = 0.15  # m - effective radius for force application
        
    def calculate_segment_com(self, joint_positions):
        """
        Calculate position of each body segment's center of mass.
        
        Args:
            joint_positions: (S, J, 3) array of joint positions over time
            
        Returns:
            (S, num_segments, 3) array of segment CoM positions
        """
        S = joint_positions.shape[0]
        segment_coms = []
        segment_masses = []
        
        for segment_name, (joint_indices, mass_fraction) in SEGMENT_JOINT_MAPPING.items():
            # Get joints for this segment
            segment_joints = joint_positions[:, joint_indices, :]  # (S, n_joints, 3)
            # CoM is average position of segment joints
            segment_com = segment_joints.mean(axis=1)  # (S, 3)
            segment_coms.append(segment_com)
            segment_masses.append(mass_fraction)
            
        return np.stack(segment_coms, axis=1), np.array(segment_masses)
    
    def calculate_body_com(self, joint_positions):
        """
        Calculate whole-body Center of Mass position over time.
        
        Args:
            joint_positions: (S, J, 3) array of joint positions
            
        Returns:
            com_positions: (S, 3) array of CoM positions
        """
        segment_coms, segment_masses = self.calculate_segment_com(joint_positions)
        
        # Weighted average of segment CoMs
        segment_masses = segment_masses.reshape(1, -1, 1)  # (1, num_segments, 1)
        com_positions = (segment_coms * segment_masses).sum(axis=1)  # (S, 3)
        
        return com_positions
    
    def calculate_com_acceleration(self, com_positions):
        """
        Calculate Center of Mass acceleration using finite differences.
        
        Args:
            com_positions: (S, 3) array of CoM positions
            
        Returns:
            com_acceleration: (S-2, 3) array of CoM accelerations
        """
        # Velocity: first derivative
        com_velocity = (com_positions[1:] - com_positions[:-1]) / self.dt  # (S-1, 3)
        
        # Acceleration: second derivative
        com_acceleration = (com_velocity[1:] - com_velocity[:-1]) / self.dt  # (S-2, 3)
        
        return com_acceleration
    
    def detect_foot_contacts(self, joint_positions, foot_indices=[7, 8, 10, 11]):
        """
        Detect when feet are in contact with the ground.
        
        Args:
            joint_positions: (S, J, 3) array of joint positions
            foot_indices: List of foot joint indices [left_ankle, right_ankle, left_toe, right_toe]
            
        Returns:
            contacts: (S-2, 4) boolean array indicating contact state
            foot_positions: (S-2, 4, 3) array of foot positions during analysis window
        """
        feet = joint_positions[:, foot_indices, :]  # (S, 4, 3)
        
        # Calculate horizontal foot velocity
        foot_velocity = np.linalg.norm(
            feet[2:, :, self.horizontal_axes] - feet[1:-1, :, self.horizontal_axes],
            axis=-1
        )  # (S-2, 4)
        
        # Check foot height (feet near ground)  
        foot_height = feet[1:-1, :, self.up_axis]  # (S-2, 4) - height of feet in analysis window
        ground_threshold = 0.10  # 10cm above ground
        near_ground = foot_height < ground_threshold  # (S-2, 4)
        
        # Contact when velocity is low AND foot is near ground
        contacts = (foot_velocity < self.min_contact_velocity) & near_ground  # (S-2, 4)
        
        # Get foot positions in analysis window
        foot_positions = feet[1:-1, :, :]  # (S-2, 4, 3) - middle frames
        
        return contacts, foot_positions
    
    def calculate_required_force(self, com_acceleration):
        """
        Calculate the ground reaction force required to produce CoM acceleration.
        
        Args:
            com_acceleration: (S-2, 3) array of CoM accelerations
            
        Returns:
            required_force: (S-2, 3) array of required forces in Newtons
        """
        # F = ma (Newton's second law)
        required_force = self.body_mass * com_acceleration  # (S-2, 3)
        
        # Add gravitational force (body must support its weight + accelerate)
        required_force[:, self.up_axis] += self.body_mass * self.gravity
        
        return required_force
    
    def calculate_maximum_available_force(self, contacts, foot_positions, com_position):
        """
        Calculate maximum force that can be generated by feet in contact.
        
        Args:
            contacts: (S-2, 4) boolean array of foot contacts
            foot_positions: (S-2, 4, 3) array of foot positions
            com_position: (S-2, 3) array of CoM positions
            
        Returns:
            max_available_force: (S-2, 3) array of maximum forces available
            force_margin: (S-2,) array of force capacity utilization
        """
        S = contacts.shape[0]
        max_available_force = np.zeros((S, 3))
        
        # Maximum force per contact point (biomechanical limit)
        max_force_per_foot = self.body_mass * self.gravity * self.max_ground_force_multiplier
        
        for i in range(S):
            # Get contact feet for this frame
            contact_mask = contacts[i]  # (4,)
            
            if not np.any(contact_mask):
                # No contacts - zero available force (problematic!)
                continue
            
            # Number of feet in contact
            n_contacts = contact_mask.sum()
            
            # Simple model: distribute force equally among contact points
            # In reality, force distribution depends on CoM position relative to base of support
            contact_feet = foot_positions[i, contact_mask, :]  # (n_contacts, 3)
            
            # Calculate base of support center
            support_center = contact_feet.mean(axis=0)  # (3,)
            
            # Vertical force capacity (can always generate upward force if in contact)
            max_available_force[i, self.up_axis] = max_force_per_foot * n_contacts
            
            # Horizontal force capacity (limited by friction and moment arms)
            # Simplified model: proportional to number of contacts
            horizontal_capacity = max_force_per_foot * n_contacts * 0.5  # friction coefficient ~0.5
            max_available_force[i, self.horizontal_axes] = horizontal_capacity
            
        return max_available_force
    
    def calculate_force_inconsistency(self, required_force, max_available_force):
        """
        Calculate how much the required force exceeds what's physically possible.
        
        Args:
            required_force: (S-2, 3) array of required forces
            max_available_force: (S-2, 3) array of maximum available forces
            
        Returns:
            inconsistency_score: Scalar value (higher = worse physics)
            per_frame_violations: (S-2,) array of violations per frame
        """
        # Calculate force deficit (positive = impossible)
        force_deficit = np.maximum(0, np.abs(required_force) - max_available_force)  # (S-2, 3)
        
        # Normalize by body weight
        normalized_deficit = force_deficit / (self.body_mass * self.gravity)
        
        # L2 norm per frame
        per_frame_violations = np.linalg.norm(normalized_deficit, axis=-1)  # (S-2,)
        
        # Overall score: mean violation
        inconsistency_score = per_frame_violations.mean()
        
        return inconsistency_score, per_frame_violations
    
    def evaluate_motion(self, joint_positions):
        """
        Full pipeline: evaluate force consistency for a motion sequence.
        
        Args:
            joint_positions: (S, J, 3) array of joint positions
            
        Returns:
            Dictionary with detailed metrics
        """
        # Step 1: Calculate Center of Mass
        com_positions = self.calculate_body_com(joint_positions)
        
        # Step 2: Calculate CoM acceleration
        com_acceleration = self.calculate_com_acceleration(com_positions)
        
        # Step 3: Detect foot contacts
        contacts, foot_positions = self.detect_foot_contacts(joint_positions)
        
        # Step 4: Calculate required ground reaction force
        required_force = self.calculate_required_force(com_acceleration)
        
        # Step 5: Calculate maximum available force from contacts
        max_available_force = self.calculate_maximum_available_force(
            contacts, foot_positions, com_positions[1:-1]
        )
        
        # Step 6: Calculate inconsistency
        inconsistency_score, per_frame_violations = self.calculate_force_inconsistency(
            required_force, max_available_force
        )
        
        # Additional metrics
        contact_ratio = contacts.any(axis=1).mean()  # Fraction of time with at least one foot contact
        avg_contacts = contacts.sum(axis=1).mean()  # Average number of feet in contact
        
        return {
            'fcs_score': inconsistency_score,
            'per_frame_violations': per_frame_violations,
            'com_positions': com_positions,
            'com_acceleration': com_acceleration,
            'required_force': required_force,
            'max_available_force': max_available_force,
            'contacts': contacts,
            'contact_ratio': contact_ratio,
            'avg_contacts': avg_contacts,
        }


def calculate_pfc_score(joint_positions, fps=30):
    """
    Original PFC metric for comparison.
    
    Args:
        joint_positions: (S, J, 3) array of joint positions
        fps: Frames per second
        
    Returns:
        pfc_score: Scalar PFC score
    """
    up_dir = 2  # z is up
    flat_dirs = [0, 1]  # x, y
    DT = 1 / fps
    
    # Root acceleration
    root_v = (joint_positions[1:, 0, :] - joint_positions[:-1, 0, :]) / DT
    root_a = (root_v[1:] - root_v[:-1]) / DT
    root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)
    root_a = np.linalg.norm(root_a, axis=-1)
    
    # Normalize
    scaling = root_a.max() if root_a.max() > 0 else 1.0
    root_a /= scaling
    
    # Foot velocities
    foot_idx = [7, 10, 8, 11]  # ankle L, toe L, ankle R, toe R
    feet = joint_positions[:, foot_idx, :]
    foot_v = np.linalg.norm(
        feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
    )
    
    # Minimum velocities per leg
    foot_mins = np.zeros((len(foot_v), 2))
    foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])  # left leg
    foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])  # right leg
    
    # PFC score
    foot_loss = foot_mins[:, 0] * foot_mins[:, 1] * root_a
    pfc_score = foot_loss.mean() * 10000
    
    return pfc_score


def evaluate_directory(motion_dir, output_file=None, max_samples=None):
    """
    Evaluate all motions in a directory.
    
    Args:
        motion_dir: Path to directory containing .pkl motion files
        output_file: Path to save detailed results (JSON)
        max_samples: Maximum number of samples to process (None = all)
        
    Returns:
        Dictionary with aggregate statistics
    """
    evaluator = ForceConsistencyEvaluator()
    
    pkl_files = glob.glob(os.path.join(motion_dir, "*.pkl"))
    if max_samples and len(pkl_files) > max_samples:
        np.random.seed(42)
        pkl_files = np.random.choice(pkl_files, max_samples, replace=False)
    
    results = {
        'fcs_scores': [],
        'pfc_scores': [],
        'contact_ratios': [],
        'avg_contacts': [],
        'filenames': [],
    }
    
    print(f"Evaluating {len(pkl_files)} motions from {motion_dir}...")
    
    for pkl_path in tqdm(pkl_files):
        try:
            # Load motion data
            with open(pkl_path, 'rb') as f:
                info = pickle.load(f)
            
            joint_positions = info['full_pose']  # (S, J, 3)
            
            # Evaluate with FCS
            fcs_result = evaluator.evaluate_motion(joint_positions)
            
            # Evaluate with PFC for comparison
            pfc_score = calculate_pfc_score(joint_positions)
            
            # Store results
            results['fcs_scores'].append(fcs_result['fcs_score'])
            results['pfc_scores'].append(pfc_score)
            results['contact_ratios'].append(fcs_result['contact_ratio'])
            results['avg_contacts'].append(fcs_result['avg_contacts'])
            results['filenames'].append(os.path.basename(pkl_path))
            
        except Exception as e:
            print(f"Error processing {pkl_path}: {e}")
            continue
    
    # Calculate statistics
    fcs_scores = np.array(results['fcs_scores'])
    pfc_scores = np.array(results['pfc_scores'])
    
    stats = {
        'num_samples': len(fcs_scores),
        'fcs_mean': float(fcs_scores.mean()),
        'fcs_std': float(fcs_scores.std()),
        'fcs_median': float(np.median(fcs_scores)),
        'fcs_min': float(fcs_scores.min()),
        'fcs_max': float(fcs_scores.max()),
        'pfc_mean': float(pfc_scores.mean()),
        'pfc_std': float(pfc_scores.std()),
        'pfc_median': float(np.median(pfc_scores)),
        'contact_ratio_mean': float(np.mean(results['contact_ratios'])),
        'avg_contacts_mean': float(np.mean(results['avg_contacts'])),
    }
    
    print(f"\n{'='*60}")
    print(f"Results for: {motion_dir}")
    print(f"{'='*60}")
    print(f"Samples evaluated: {stats['num_samples']}")
    print(f"\nForce Consistency Score (FCS):")
    print(f"  Mean:   {stats['fcs_mean']:.4f} ± {stats['fcs_std']:.4f}")
    print(f"  Median: {stats['fcs_median']:.4f}")
    print(f"  Range:  [{stats['fcs_min']:.4f}, {stats['fcs_max']:.4f}]")
    print(f"\nPhysical Foot Contact (PFC) - for comparison:")
    print(f"  Mean:   {stats['pfc_mean']:.4f} ± {stats['pfc_std']:.4f}")
    print(f"  Median: {stats['pfc_median']:.4f}")
    print(f"\nContact Statistics:")
    print(f"  Average contact ratio: {stats['contact_ratio_mean']:.2%}")
    print(f"  Average feet in contact: {stats['avg_contacts_mean']:.2f}")
    print(f"{'='*60}\n")
    
    # Save detailed results if requested
    if output_file:
        output_data = {
            'statistics': stats,
            'per_sample_results': results,
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Detailed results saved to: {output_file}")
    
    return stats


def compare_datasets(real_data_dir, generated_data_dir, output_dir=None):
    """
    Compare FCS scores between real and generated data.
    
    Args:
        real_data_dir: Path to real mocap data
        generated_data_dir: Path to generated motion data
        output_dir: Directory to save comparison results
    """
    print("\n" + "="*60)
    print("GROUND TRUTH VALIDATION TEST")
    print("="*60 + "\n")
    
    print("Evaluating REAL human data (AIST++)...")
    real_stats = evaluate_directory(real_data_dir)
    
    print("\nEvaluating GENERATED data (EDGE)...")
    gen_stats = evaluate_directory(generated_data_dir)
    
    # Statistical comparison
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    fcs_diff = gen_stats['fcs_mean'] - real_stats['fcs_mean']
    fcs_diff_pct = (fcs_diff / real_stats['fcs_mean']) * 100 if real_stats['fcs_mean'] > 0 else 0
    
    print(f"\nFCS Score Comparison:")
    print(f"  Real data:      {real_stats['fcs_mean']:.4f} ± {real_stats['fcs_std']:.4f}")
    print(f"  Generated data: {gen_stats['fcs_mean']:.4f} ± {gen_stats['fcs_std']:.4f}")
    print(f"  Difference:     {fcs_diff:+.4f} ({fcs_diff_pct:+.1f}%)")
    
    pfc_diff = gen_stats['pfc_mean'] - real_stats['pfc_mean']
    pfc_diff_pct = (pfc_diff / real_stats['pfc_mean']) * 100 if real_stats['pfc_mean'] > 0 else 0
    
    print(f"\nPFC Score Comparison:")
    print(f"  Real data:      {real_stats['pfc_mean']:.4f} ± {real_stats['pfc_std']:.4f}")
    print(f"  Generated data: {gen_stats['pfc_mean']:.4f} ± {gen_stats['pfc_std']:.4f}")
    print(f"  Difference:     {pfc_diff:+.4f} ({pfc_diff_pct:+.1f}%)")
    
    print(f"\nSeparability:")
    # Calculate effect size (Cohen's d)
    pooled_std_fcs = np.sqrt((real_stats['fcs_std']**2 + gen_stats['fcs_std']**2) / 2)
    cohens_d_fcs = abs(fcs_diff) / pooled_std_fcs if pooled_std_fcs > 0 else 0
    
    pooled_std_pfc = np.sqrt((real_stats['pfc_std']**2 + gen_stats['pfc_std']**2) / 2)
    cohens_d_pfc = abs(pfc_diff) / pooled_std_pfc if pooled_std_pfc > 0 else 0
    
    print(f"  FCS Cohen's d: {cohens_d_fcs:.3f} {get_effect_size_label(cohens_d_fcs)}")
    print(f"  PFC Cohen's d: {cohens_d_pfc:.3f} {get_effect_size_label(cohens_d_pfc)}")
    
    print(f"\n{'='*60}\n")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        comparison_file = os.path.join(output_dir, 'fcs_comparison.json')
        comparison_data = {
            'real_data': real_stats,
            'generated_data': gen_stats,
            'comparison': {
                'fcs_difference': float(fcs_diff),
                'fcs_difference_percent': float(fcs_diff_pct),
                'pfc_difference': float(pfc_diff),
                'pfc_difference_percent': float(pfc_diff_pct),
                'fcs_cohens_d': float(cohens_d_fcs),
                'pfc_cohens_d': float(cohens_d_pfc),
            }
        }
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"Comparison results saved to: {comparison_file}")
    
    return real_stats, gen_stats


def get_effect_size_label(cohens_d):
    """Return label for effect size."""
    if cohens_d < 0.2:
        return "(negligible)"
    elif cohens_d < 0.5:
        return "(small)"
    elif cohens_d < 0.8:
        return "(medium)"
    else:
        return "(large)"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Force Consistency Score (FCS) - Physics-based motion evaluation"
    )
    parser.add_argument(
        "--motion_path",
        type=str,
        required=True,
        help="Path to motion directory (pkl files)"
    )
    parser.add_argument(
        "--compare_path",
        type=str,
        default=None,
        help="Path to second dataset for comparison (e.g., real vs generated)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results (JSON)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.compare_path:
        # Comparison mode: real vs generated
        compare_datasets(
            real_data_dir=args.motion_path,
            generated_data_dir=args.compare_path,
            output_dir=args.output
        )
    else:
        # Single dataset evaluation
        evaluate_directory(
            motion_dir=args.motion_path,
            output_file=args.output,
            max_samples=args.max_samples
        )
