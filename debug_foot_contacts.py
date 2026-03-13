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
    
    # Heights (z-axis) - NOTE: may need to subtract root/floor offset
    foot_heights = feet[:, :, 2]  # (S, 4)
    
    # Find the minimum height across all joints to establish ground level
    min_height_in_sequence = joint_positions[:, :, 2].min()
    
    # Adjust heights relative to minimum (ground level)
    foot_heights_adjusted = foot_heights - min_height_in_sequence
    
    # Use ADJUSTED heights for analysis
    all_foot_heights.extend(foot_heights_adjusted[:-1].flatten())  # Match velocity shape
    
    # Velocities
    foot_velocity = np.linalg.norm(
        feet[1:, :, [0,1]] - feet[:-1, :, [0,1]], axis=-1
    ) * 30  # multiply by fps for m/s
    all_foot_velocities.extend(foot_velocity.flatten())
    
    print(f"\nSample {idx}:")
    print(f"  Raw min height: {min_height_in_sequence:.4f} m (ground level)")
    print(f"  Adjusted min foot height: {foot_heights_adjusted.min():.4f} m")
    print(f"  Adjusted max foot height: {foot_heights_adjusted.max():.4f} m")
    print(f"  Adjusted mean foot height: {foot_heights_adjusted.mean():.4f} m")
    print(f"  % below 0.10m: {(foot_heights_adjusted < 0.10).sum() / foot_heights_adjusted.size * 100:.1f}%")
    print(f"  % below 0.20m: {(foot_heights_adjusted < 0.20).sum() / foot_heights_adjusted.size * 100:.1f}%")
    print(f"  % below 0.30m: {(foot_heights_adjusted < 0.30).sum() / foot_heights_adjusted.size * 100:.1f}%")
    print(f"  Min foot velocity: {foot_velocity.min():.4f} m/s")
    print(f"  Mean foot velocity: {foot_velocity.mean():.4f} m/s")
    print(f"  % below 0.02 m/s: {(foot_velocity < 0.02).sum() / foot_velocity.size * 100:.1f}%")
    print(f"  % below 0.05 m/s: {(foot_velocity < 0.05).sum() / foot_velocity.size * 100:.1f}%")
    print(f"  % below 0.10 m/s: {(foot_velocity < 0.10).sum() / foot_velocity.size * 100:.1f}%")

# Overall statistics
all_foot_heights = np.array(all_foot_heights)
all_foot_velocities = np.array(all_foot_velocities)

print("\n" + "="*70)
print("OVERALL STATISTICS (10 samples) - Heights Adjusted to Ground Level")
print("="*70)
print(f"\nFoot Heights (adjusted):")
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
print(f"  Arrays: heights={len(all_foot_heights)}, velocities={len(all_foot_velocities)}")
print(f"  Current thresholds: height < 0.10m AND velocity < 0.02 m/s")

# Ensure arrays are same length
min_len = min(len(all_foot_heights), len(all_foot_velocities))
heights_subset = all_foot_heights[:min_len]
velocities_subset = all_foot_velocities[:min_len]

# Try different threshold combinations
for height_thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
    for vel_thresh in [0.02, 0.05, 0.10, 0.20, 0.50]:
        contact_pct = ((heights_subset < height_thresh) & (velocities_subset < vel_thresh)).sum() / min_len * 100
        print(f"  height < {height_thresh:.2f}m, vel < {vel_thresh:.2f} m/s → {contact_pct:.1f}% contacts")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print("1. Heights are adjusted relative to minimum joint height per sequence")
print("2. Original data has feet ~0.88m above origin (coordinate system offset)")
print("3. FCS evaluation needs similar ground-level adjustment")
print("4. Based on above stats, suggested thresholds for 30-60% contact:")
print("   - Option A: height < 0.15m, velocity < 0.10 m/s")
print("   - Option B: height < 0.20m, velocity < 0.15 m/s")
print("5. Update detect_foot_contacts() in eval/eval_fcs.py accordingly")
print("="*70)
