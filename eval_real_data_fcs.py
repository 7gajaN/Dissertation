"""
Evaluate Force Consistency Score (FCS) on real AIST++ dataset
Compares real motion data vs generated samples
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from eval.eval_fcs import ForceConsistencyEvaluator
from vis import SMPLSkeleton


def evaluate_dataset_fcs(data_path, split='test', max_samples=100):
    """
    Evaluate FCS on real AIST++ dataset.
    
    Args:
        data_path: Path to AIST++ data
        split: 'train' or 'test'
        max_samples: Maximum number of sequences to evaluate
    
    Returns:
        Dictionary with FCS statistics
    """
    print(f"Loading AIST++ {split} dataset from {data_path}...")
    
    # Load dataset
    dataset = AISTPPDataset(
        data_path=data_path,
        backup_path="data/dataset_backups",
        train=split == 'train',
        feature_type='jukebox'
    )
    
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    # Initialize FCS evaluator and SMPL skeleton
    fcs_evaluator = ForceConsistencyEvaluator(fps=30)
    smpl = SMPLSkeleton()
    
    # Evaluate FCS on samples
    fcs_scores = []
    num_samples = min(max_samples, len(dataset))
    
    print(f"Evaluating FCS on {num_samples} sequences...")
    
    for idx in tqdm(range(num_samples)):
        try:
            # Get a sample from dataset
            # Dataset returns: (pose, feature, filename, wav)
            motion_data, _, _, _ = dataset[idx]
            
            # Unnormalize the data
            motion_data = motion_data.unsqueeze(0)  # (1, seq_len, features)
            motion_data = dataset.normalizer.unnormalize(motion_data)
            motion_data = motion_data.squeeze(0)  # (seq_len, features)
            
            # Parse motion data
            seq_len, features = motion_data.shape
            
            # Split: [contacts(4), root_pos(3), local_q(24*6)]
            contact = motion_data[:, :4]
            root_pos = motion_data[:, 4:7]
            local_q_6d = motion_data[:, 7:]
            
            # Convert 6D rotation to axis-angle
            from dataset.quaternion import ax_from_6v
            local_q_6d = local_q_6d.reshape(seq_len, 24, 6)
            local_q = ax_from_6v(local_q_6d)  # (seq_len, 24, 3)
            
            # Forward kinematics to get joint positions
            joint_positions = smpl.forward(local_q.unsqueeze(0), root_pos.unsqueeze(0))  # (1, seq_len, 24, 3)
            joint_positions = joint_positions.squeeze(0)  # (seq_len, 24, 3)
            
            # Convert to numpy
            joint_positions_np = joint_positions.cpu().numpy()
            
            # Evaluate FCS
            result = fcs_evaluator.evaluate_motion(joint_positions_np)
            fcs_scores.append(result['fcs_score'])
            
        except Exception as e:
            print(f"\nError processing sequence {idx}: {e}")
            continue
    
    # Compute statistics
    fcs_scores = np.array(fcs_scores)
    
    if len(fcs_scores) == 0:
        print("\nERROR: All samples failed to evaluate! Check the errors above.")
        return None
    
    results = {
        'mean': np.mean(fcs_scores),
        'std': np.std(fcs_scores),
        'median': np.median(fcs_scores),
        'min': np.min(fcs_scores),
        'max': np.max(fcs_scores),
        'percentile_25': np.percentile(fcs_scores, 25),
        'percentile_75': np.percentile(fcs_scores, 75),
        'num_evaluated': len(fcs_scores),
        'all_scores': fcs_scores
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate FCS on AIST++ dataset')
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to AIST++ dataset (parent folder containing train/test)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of sequences to evaluate')
    parser.add_argument('--output', type=str, default='fcs_real_data_results.txt',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EVALUATING FORCE CONSISTENCY SCORE ON REAL AIST++ DATA")
    print("=" * 70)
    
    # Evaluate
    results = evaluate_dataset_fcs(args.data_path, args.split, args.max_samples)
    
    if results is None:
        print("\nFailed to evaluate dataset. Exiting.")
        return
    
    # Print results
    print("\n" + "=" * 70)
    print(f"FCS EVALUATION RESULTS - {args.split.upper()} SET")
    print("=" * 70)
    print(f"Number of sequences evaluated: {results['num_evaluated']}")
    print(f"\nFCS Score Statistics:")
    print(f"  Mean:       {results['mean']:.6f}")
    print(f"  Std Dev:    {results['std']:.6f}")
    print(f"  Median:     {results['median']:.6f}")
    print(f"  Min:        {results['min']:.6f}")
    print(f"  Max:        {results['max']:.6f}")
    print(f"  25th %ile:  {results['percentile_25']:.6f}")
    print(f"  75th %ile:  {results['percentile_75']:.6f}")
    print("=" * 70)
    
    # Save results
    with open(args.output, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"FCS EVALUATION - AIST++ {args.split.upper()} SET\n")
        f.write("=" * 70 + "\n")
        f.write(f"Sequences evaluated: {results['num_evaluated']}\n\n")
        f.write("Statistics:\n")
        f.write(f"  Mean:       {results['mean']:.6f}\n")
        f.write(f"  Std Dev:    {results['std']:.6f}\n")
        f.write(f"  Median:     {results['median']:.6f}\n")
        f.write(f"  Min:        {results['min']:.6f}\n")
        f.write(f"  Max:        {results['max']:.6f}\n")
        f.write(f"  25th %ile:  {results['percentile_25']:.6f}\n")
        f.write(f"  75th %ile:  {results['percentile_75']:.6f}\n")
        f.write("\nAll scores:\n")
        for i, score in enumerate(results['all_scores']):
            f.write(f"  Sequence {i:3d}: {score:.6f}\n")
    
    print(f"\nResults saved to: {args.output}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("FCS measures physics violations (lower = more realistic)")
    print("  - FCS ≈ 0:     Perfect physics (rare in real data)")
    print("  - FCS < 0.5:   Very good physics")
    print("  - FCS < 2.0:   Acceptable physics")
    print("  - FCS > 5.0:   Significant violations")
    print("  - FCS > 10:    Severe violations (impossible motions)")
    print("\nCompare this with your generated samples (FCS ≈ 1.578)")
    print("=" * 70)


if __name__ == '__main__':
    main()
