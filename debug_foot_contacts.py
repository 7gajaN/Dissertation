"""
Debug script to analyze foot heights and velocities in AIST++ data
"""

import numpy as np
import torch
import pickle
from pathlib import Path

from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton

# Load test dataset
print("Loading test dataset...")
dataset = pickle.load(open('data/dataset_backups/test_tensor_dataset.pkl', 'rb'))
print(f"Dataset loaded: {len(dataset)} sequences")

# Get a few samples
smpl = SMPLSkeleton()
foot_indices = [7, 8, 10, 11]  # left_ankle, right_ankle, left_toe, right_toe

all_foot_heights = []
all_foot_velocities = []

print("\nAnalyzing 10 random samples...")

for idx in range(10):
    # Get sample
    motion_data, _, _, _ = dataset[idx]
    
    # Unnormalize
    motion_data = motion_data.unsqueeze(0)
    motion_data = dataset.normalizer.unnormalize(motion_data)
    motion_data = motion_data.squeeze(0)
    
    # Parse
    contact, motion = torch.split(motion_data, (4, motion_data.shape[-1] - 4), dim=1)
    root_pos = motion[:, :3]
    local_q_6d = motion[:, 3:]
    local_q = ax_from_6v(local_q_6d.reshape(motion.shape[0], 24, 6))
    
    # FK
    joint_positions = smpl.forward(local_q.unsqueeze(0), root_pos.unsqueeze(0))
    joint_positions = joint_positions.squeeze(0).cpu().numpy()
    
    # Get foot data
    feet = joint_positions[:, foot_indices, :]  # (S, 4, 3)
    
    # Heights (z-axis)
    foot_heights = feet[:, :, 2]  # (S, 4)
    all_foot_heights.extend(foot_heights.flatten())
    
    # Velocities
    foot_velocity = np.linalg.norm(
        feet[1:, :, [0,1]] - feet[:-1, :, [0,1]], axis=-1
    ) * 30  # multiply by fps for m/s
    all_foot_velocities.extend(foot_velocity.flatten())
    
    print(f"\nSample {idx}:")
    print(f"  Min foot height: {foot_heights.min():.4f} m")
    print(f"  Max foot height: {foot_heights.max():.4f} m")
    print(f"  Mean foot height: {foot_heights.mean():.4f} m")
    print(f"  % below 0.10m: {(foot_heights < 0.10).sum() / foot_heights.size * 100:.1f}%")
    print(f"  % below 0.20m: {(foot_heights < 0.20).sum() / foot_heights.size * 100:.1f}%")
    print(f"  % below 0.30m: {(foot_heights < 0.30).sum() / foot_heights.size * 100:.1f}%")
    print(f"  Min foot velocity: {foot_velocity.min():.4f} m/s")
    print(f"  Mean foot velocity: {foot_velocity.mean():.4f} m/s")
    print(f"  % below 0.02 m/s: {(foot_velocity < 0.02).sum() / foot_velocity.size * 100:.1f}%")
    print(f"  % below 0.05 m/s: {(foot_velocity < 0.05).sum() / foot_velocity.size * 100:.1f}%")
    print(f"  % below 0.10 m/s: {(foot_velocity < 0.10).sum() / foot_velocity.size * 100:.1f}%")

# Overall statistics
all_foot_heights = np.array(all_foot_heights)
all_foot_velocities = np.array(all_foot_velocities)

print("\n" + "="*70)
print("OVERALL STATISTICS (10 samples)")
print("="*70)
print(f"\nFoot Heights:")
print(f"  Min:    {all_foot_heights.min():.4f} m")
print(f"  Max:    {all_foot_heights.max():.4f} m")
print(f"  Mean:   {all_foot_heights.mean():.4f} m")
print(f"  Median: {np.median(all_foot_heights):.4f} m")
print(f"  10th percentile: {np.percentile(all_foot_heights, 10):.4f} m")
print(f"  25th percentile: {np.percentile(all_foot_heights, 25):.4f} m")

print(f"\nFoot Velocities:")
print(f"  Min:    {all_foot_velocities.min():.4f} m/s")
print(f"  Max:    {all_foot_velocities.max():.4f} m/s")
print(f"  Mean:   {all_foot_velocities.mean():.4f} m/s")
print(f"  Median: {np.median(all_foot_velocities):.4f} m/s")
print(f"  10th percentile: {np.percentile(all_foot_velocities, 10):.4f} m/s")

print(f"\nContact Detection Analysis:")
print(f"  Current thresholds: height < 0.10m AND velocity < 0.02 m/s")

# Try different threshold combinations
for height_thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
    for vel_thresh in [0.02, 0.05, 0.10, 0.20, 0.50]:
        contact_pct = ((all_foot_heights < height_thresh) & (all_foot_velocities < vel_thresh)).sum() / len(all_foot_heights) * 100
        print(f"  height < {height_thresh:.2f}m, vel < {vel_thresh:.2f} m/s → {contact_pct:.1f}% contacts")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("Based on above, adjust thresholds in eval_fcs.py for realistic contact %")
print("="*70)
