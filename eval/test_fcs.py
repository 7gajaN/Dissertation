"""
Quick test script for Force Consistency Score (FCS)

This script demonstrates FCS usage and validates the implementation
with synthetic test cases.
"""

import numpy as np
from eval_fcs import ForceConsistencyEvaluator, calculate_pfc_score


def create_synthetic_motion(motion_type='static', duration=60, fps=30):
    """
    Create synthetic motion for testing.
    
    Args:
        motion_type: 'static', 'walking', 'jumping', 'floating'
        duration: Number of frames
        fps: Frames per second
    
    Returns:
        joint_positions: (S, 24, 3) array
    """
    S = duration
    J = 24  # SMPL joints
    
    # Initialize with T-pose
    joint_positions = np.zeros((S, J, 3))
    
    # Basic skeleton (approximate)
    # Root at origin
    joint_positions[:, 0, :] = [0, 0, 1.0]  # root at 1m height
    
    # Legs
    joint_positions[:, 1, :] = [-0.1, 0, 0.95]  # left hip
    joint_positions[:, 2, :] = [0.1, 0, 0.95]   # right hip
    joint_positions[:, 4, :] = [-0.1, 0, 0.5]   # left knee
    joint_positions[:, 5, :] = [0.1, 0, 0.5]    # right knee
    joint_positions[:, 7, :] = [-0.1, 0, 0.05]  # left ankle
    joint_positions[:, 8, :] = [0.1, 0, 0.05]   # right ankle
    joint_positions[:, 10, :] = [-0.1, 0.15, 0.0]  # left toe
    joint_positions[:, 11, :] = [0.1, 0.15, 0.0]   # right toe
    
    # Torso
    joint_positions[:, 3, :] = [0, 0, 1.05]   # belly
    joint_positions[:, 6, :] = [0, 0, 1.15]   # spine
    joint_positions[:, 9, :] = [0, 0, 1.3]    # chest
    joint_positions[:, 12, :] = [0, 0, 1.45]  # neck
    joint_positions[:, 15, :] = [0, 0, 1.6]   # head
    
    # Arms (simplified)
    joint_positions[:, 16, :] = [-0.2, 0, 1.3]   # left shoulder
    joint_positions[:, 17, :] = [0.2, 0, 1.3]    # right shoulder
    joint_positions[:, 18, :] = [-0.3, 0, 1.1]   # left elbow
    joint_positions[:, 19, :] = [0.3, 0, 1.1]    # right elbow
    joint_positions[:, 20, :] = [-0.35, 0, 0.9]  # left wrist
    joint_positions[:, 21, :] = [0.35, 0, 0.9]   # right wrist
    joint_positions[:, 22, :] = [-0.4, 0, 0.85]  # left hand
    joint_positions[:, 23, :] = [0.4, 0, 0.85]   # right hand
    
    if motion_type == 'static':
        # No motion - should have perfect physics (no acceleration)
        pass
    
    elif motion_type == 'walking':
        # Simple walking: shift weight, move forward
        for i in range(S):
            t = i / fps
            # Forward motion
            joint_positions[i, :, 1] += 0.5 * t  # move forward
            # Weight shift (subtle)
            shift = 0.02 * np.sin(2 * np.pi * t)
            joint_positions[i, :, 0] += shift
            # Slight bounce
            bounce = 0.01 * abs(np.sin(2 * np.pi * t))
            joint_positions[i, :, 2] += bounce
    
    elif motion_type == 'jumping':
        # Jump: rapid upward acceleration then fall
        for i in range(S):
            t = i / fps
            if t < 0.3:
                # Crouch
                joint_positions[i, :, 2] *= 0.9
            elif t < 0.5:
                # Jump up (high acceleration)
                jump_height = 0.5
                joint_positions[i, :, 2] += jump_height
            elif t < 1.0:
                # Fall down
                fall_t = t - 0.5
                g = 9.81
                fall_distance = 0.5 * g * fall_t**2
                joint_positions[i, :, 2] = joint_positions[int(0.5*fps), :, 2] - fall_distance
                joint_positions[i, :, 2] = np.maximum(joint_positions[i, :, 2], joint_positions[0, :, 2])
            else:
                # Landed
                pass
    
    elif motion_type == 'floating':
        # Violates physics: body rises without ground contact
        for i in range(S):
            # Gradually float upward with no foot contact
            joint_positions[i, :, 2] += 0.02 * i
            # Move feet off ground
            joint_positions[i, 7, 2] += 0.5  # left ankle
            joint_positions[i, 8, 2] += 0.5  # right ankle
            joint_positions[i, 10, 2] += 0.5  # left toe
            joint_positions[i, 11, 2] += 0.5  # right toe
    
    return joint_positions


def test_synthetic_motions():
    """Test FCS on synthetic motions with known physics."""
    print("="*60)
    print("SYNTHETIC MOTION TEST")
    print("="*60 + "\n")
    
    evaluator = ForceConsistencyEvaluator(fps=30)
    
    motion_types = ['static', 'walking', 'jumping', 'floating']
    
    for motion_type in motion_types:
        print(f"\nTesting: {motion_type.upper()}")
        print("-" * 40)
        
        # Create synthetic motion
        joint_positions = create_synthetic_motion(motion_type=motion_type, duration=60)
        
        # Evaluate with FCS
        result = evaluator.evaluate_motion(joint_positions)
        
        # Evaluate with PFC for comparison
        pfc_score = calculate_pfc_score(joint_positions)
        
        print(f"FCS Score:      {result['fcs_score']:.4f}")
        print(f"PFC Score:      {pfc_score:.4f}")
        print(f"Contact ratio:  {result['contact_ratio']:.2%}")
        print(f"Avg contacts:   {result['avg_contacts']:.2f}")
        print(f"Max violation:  {result['per_frame_violations'].max():.4f}")
        
        # Interpret
        if motion_type == 'static':
            expectation = "Very low FCS (no acceleration, good physics)"
        elif motion_type == 'walking':
            expectation = "Low FCS (realistic motion)"
        elif motion_type == 'jumping':
            expectation = "Medium FCS (high forces during jump)"
        elif motion_type == 'floating':
            expectation = "High FCS (violates physics, no ground contact)"
        
        print(f"Expected:       {expectation}")
    
    print("\n" + "="*60 + "\n")


def test_single_file(pkl_path):
    """Test FCS on a single motion file."""
    import pickle
    
    print(f"Testing file: {pkl_path}")
    
    # Load motion
    with open(pkl_path, 'rb') as f:
        info = pickle.load(f)
    
    joint_positions = info['full_pose']
    
    # Evaluate
    evaluator = ForceConsistencyEvaluator()
    result = evaluator.evaluate_motion(joint_positions)
    pfc_score = calculate_pfc_score(joint_positions)
    
    print("\nResults:")
    print(f"  FCS Score:     {result['fcs_score']:.4f}")
    print(f"  PFC Score:     {pfc_score:.4f}")
    print(f"  Duration:      {len(joint_positions)} frames")
    print(f"  Contact ratio: {result['contact_ratio']:.2%}")
    print(f"  Avg contacts:  {result['avg_contacts']:.2f}")
    
    # Show worst violations
    violations = result['per_frame_violations']
    worst_frames = np.argsort(violations)[-5:][::-1]
    
    print(f"\nTop 5 violation frames:")
    for rank, frame_idx in enumerate(worst_frames, 1):
        print(f"  {rank}. Frame {frame_idx}: {violations[frame_idx]:.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific file
        test_single_file(sys.argv[1])
    else:
        # Test synthetic motions
        test_synthetic_motions()
        
        print("\nTo test a specific motion file:")
        print("  python eval/test_fcs.py path/to/motion.pkl")
