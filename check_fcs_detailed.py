"""
Diagnostic script to analyze FCS metric in detail
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from eval.eval_fcs import ForceConsistencyEvaluator
from vis import SMPLSkeleton


def analyze_single_sequence():
    """Analyze one sequence in detail to understand FCS calculation"""
    
    print("Loading test dataset...")
    cached_path = 'data/dataset_backups/test_tensor_dataset.pkl'
    dataset = pickle.load(open(cached_path, 'rb'))
    
    # Initialize SMPL skeleton
    smpl = SMPLSkeleton()
    
    # Get first sequence
    motion_data, _, _, _ = dataset[0]
    motion_data = motion_data.unsqueeze(0)
    motion_data = dataset.normalizer.unnormalize(motion_data)
    motion_data = motion_data.squeeze(0).cpu()
    
    # Parse motion data: [contacts(4), root_pos(3), local_q(24*6)]
    seq_len, features = motion_data.shape
    contact = motion_data[:, :4]
    root_pos = motion_data[:, 4:7]
    local_q_6d = motion_data[:, 7:]
    
    # Convert 6D rotation to axis-angle
    from dataset.quaternion import ax_from_6v
    local_q_6d = local_q_6d.reshape(seq_len, 24, 6)
    local_q = ax_from_6v(local_q_6d)  # (seq_len, 24, 3)
    
    # Forward kinematics to get joint positions
    joint_positions = smpl.forward(local_q.unsqueeze(0), root_pos.unsqueeze(0))  # (1, seq_len, 24, 3)
    joint_positions = joint_positions.squeeze(0).cpu().numpy()  # (seq_len, 24, 3)
    
    print(f"Motion: {seq_len} frames, 24 joints")
    
    # Evaluate with FCS
    fcs_evaluator = ForceConsistencyEvaluator(fps=30)
    result = fcs_evaluator.evaluate_motion(joint_positions)
    
    print(f"\n{'='*70}")
    print(f"FCS DIAGNOSTIC ANALYSIS")
    print(f"{'='*70}")
    print(f"FCS Score: {result['fcs_score']:.6f}")
    print(f"Contact ratio: {result['contact_ratio']:.2%} (frames with ≥1 foot contact)")
    print(f"Average contacts per frame: {result['avg_contacts']:.2f} feet")
    print(f"\nContact detection breakdown:")
    contacts = result['contacts']  # (S-2, 4)
    for foot_idx, foot_name in enumerate(['L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe']):
        contact_pct = contacts[:, foot_idx].mean() * 100
        print(f"  {foot_name}: {contact_pct:.1f}% of frames")
    
    # Analyze forces
    required = result['required_force']  # (S-2, 3)
    available = result['max_available_force']  # (S-2, 3)
    
    print(f"\nForce Analysis:")
    print(f"  Required force (norm): mean={np.linalg.norm(required, axis=1).mean():.2f} N")
    print(f"  Available force (norm): mean={np.linalg.norm(available, axis=1).mean():.2f} N")
    
    # Find frames with violations
    violations = result['per_frame_violations']
    violation_frames = np.where(violations > 0)[0]
    if len(violation_frames) > 0:
        print(f"\nViolations found in {len(violation_frames)}/{len(violations)} frames ({len(violation_frames)/len(violations)*100:.1f}%)")
        print(f"  Max violation: {violations.max():.6f}")
        print(f"  Mean violation (non-zero): {violations[violations > 0].mean():.6f}")
        
        # Show a few violation frames
        print(f"\nExample violation frames:")
        for i in violation_frames[:5]:
            n_contacts = contacts[i].sum()
            req_norm = np.linalg.norm(required[i])
            avail_norm = np.linalg.norm(available[i])
            print(f"  Frame {i}: {n_contacts} contacts, req={req_norm:.1f}N, avail={avail_norm:.1f}N, violation={violations[i]:.6f}")
    else:
        print(f"\nNo violations found! (This is suspicious for real mocap)")
        print("Checking frames with no contacts:")
        no_contact_frames = np.where(~contacts.any(axis=1))[0]
        print(f"  Frames with 0 contacts: {len(no_contact_frames)}/{len(contacts)} ({len(no_contact_frames)/len(contacts)*100:.1f}%)")
        if len(no_contact_frames) > 0:
            print(f"\nExample no-contact frames:")
            for i in no_contact_frames[:5]:
                req_norm = np.linalg.norm(required[i])
                avail_norm = np.linalg.norm(available[i])
                print(f"  Frame {i}: req={req_norm:.1f}N, avail={avail_norm:.1f}N")


if __name__ == '__main__':
    analyze_single_sequence()
