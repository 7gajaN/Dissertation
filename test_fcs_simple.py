"""Quick test to verify FCS evaluator works correctly."""

import numpy as np
import sys
sys.path.append('d:/Master/EDGE')

from eval.eval_fcs import ForceConsistencyEvaluator

# Create FCS evaluator
evaluator = ForceConsistencyEvaluator(fps=30, gravity=9.81, body_mass=70.0)

# Test 1: Static standing pose (should have low FCS - physically consistent)
print("=" * 60)
print("TEST 1: Static standing pose")
joint_positions = np.random.randn(150, 24, 3) * 0.01  # minimal movement
joint_positions[:, :, 2] += 1.0  # body at height 1m
# Put feet on the ground (indices 7, 8, 10, 11 are foot joints)
joint_positions[:, [7, 8, 10, 11], 2] = 0.02  # feet at 2cm (on ground)
result = evaluator.evaluate_motion(joint_positions)
print(f"FCS Score: {result['fcs_score']:.6f}")
print(f"Contact ratio: {result['contact_ratio']:.3f}")
print(f"Avg contacts: {result['avg_contacts']:.3f}")
print(f"COM acceleration shape: {result['com_acceleration'].shape}")
print(f"COM acceleration magnitude: {np.linalg.norm(result['com_acceleration'], axis=-1).mean():.6f} m/s^2")
print(f"Required force magnitude: {np.linalg.norm(result['required_force'], axis=-1).mean():.2f} N")
print(f"Max available force (vertical): {result['max_available_force'][:, 2].mean():.2f} N")
print()

# Test 2: Impossible motion - teleportation (should have high FCS)
print("=" * 60)
print("TEST 2: Impossible motion - sudden teleportation")
joint_positions = np.zeros((150, 24, 3))
joint_positions[:, :, 2] = 1.0  # body at height 1m
joint_positions[:, [7, 8, 10, 11], 2] = 0.02  # feet on ground
joint_positions[75, :, :] += np.array([5.0, 0, 0])  # sudden jump sideways 5m
result = evaluator.evaluate_motion(joint_positions)
print(f"FCS Score: {result['fcs_score']:.6f}")
print(f"Contact ratio: {result['contact_ratio']:.3f}")
print(f"Avg contacts: {result['avg_contacts']:.3f}")
print(f"COM acceleration shape: {result['com_acceleration'].shape}")
print(f"COM acceleration magnitude: {np.linalg.norm(result['com_acceleration'], axis=-1).mean():.6f} m/s^2")
print(f"COM acceleration MAX: {np.linalg.norm(result['com_acceleration'], axis=-1).max():.2f} m/s^2") 
print(f"Required force magnitude: {np.linalg.norm(result['required_force'], axis=-1).mean():.2f} N")
print(f"Max available force (vertical): {result['max_available_force'][:, 2].mean():.2f} N")
print()

# Test 3: Smoothly varying motion (walking-like)
print("=" * 60)
print("TEST 3: Smooth walking-like motion")
t = np.linspace(0, 5, 150)
joint_positions = np.zeros((150, 24, 3))
# Body sways gently
for j in range(24):
    joint_positions[:, j, 0] = 0.05 * np.sin(2 * np.pi * t)  # gentle sway
    joint_positions[:, j, 1] = 0.03 * np.cos(2 * np.pi * t)
    joint_positions[:, j, 2] = 1.0 + 0.05 * np.sin(4 * np.pi * t)
# Feet alternate: left foot (7,10) and right foot (8,11) 
# Left foot stationary first half, then lifts
left_foot_height = np.where(t < 2.5, 0.02, 0.15)
# Right foot lifts first half, then stationary
right_foot_height = np.where(t < 2.5, 0.15, 0.02)
joint_positions[:, [7, 10], 2] = left_foot_height[:, None]  # left ankle, left toe
joint_positions[:, [8, 11], 2] = right_foot_height[:, None]  # right ankle, right toe
# Feet stay put horizontally when on ground
joint_positions[:, [7, 10], :2] = 0.0  # left foot stays at origin
joint_positions[:, [8, 11], :2] = 0.0  # right foot stays at origin
result = evaluator.evaluate_motion(joint_positions)
print(f"FCS Score: {result['fcs_score']:.6f}")
print(f"Contact ratio: {result['contact_ratio']:.3f}")
print(f"Avg contacts: {result['avg_contacts']:.3f}")
print(f"COM acceleration magnitude: {np.linalg.norm(result['com_acceleration'], axis=-1).mean():.6f} m/s^2")
print(f"Required force magnitude: {np.linalg.norm(result['required_force'], axis=-1).mean():.2f} N")
print(f"Max available force (vertical): {result['max_available_force'][:, 2].mean():.2f} N")
print()

print("=" * 60)
print("If all three tests show FCS = 0.000000, there's a bug in the metric!")
print("Expected: Test 1 < Test 3 < Test 2")
