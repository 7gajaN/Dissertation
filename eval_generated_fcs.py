"""
Generate dances from trained EDGE model and evaluate with FCS
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from EDGE import EDGE
from dataset.dance_dataset import AISTPPDataset
from eval.eval_fcs import ForceConsistencyEvaluator


def generate_and_evaluate(checkpoint_path, data_path, num_samples=50, output_dir='generated_samples'):
    """
    Generate dance samples from trained model and evaluate FCS.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to AIST++ dataset (for music conditioning)
        num_samples: Number of samples to generate
        output_dir: Directory to save results
    
    Returns:
        Dictionary with FCS statistics
    """
    print("=" * 70)
    print("GENERATING SAMPLES FROM TRAINED MODEL")
    print("=" * 70)
    
    # Load dataset (for music features)
    print(f"\nLoading dataset from {data_path}...")
    dataset = AISTPPDataset(
        data_path=data_path,
        backup_path="",
        train=False,  # Use test set
        feature_type='jukebox'
    )
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    # Load trained model
    print(f"\nLoading model from {checkpoint_path}...")
    model = EDGE(
        feature_type='jukebox',
        checkpoint_path=checkpoint_path,
        normalizer=dataset.normalizer
    )
    model.eval()
    print("Model loaded successfully!")
    
    # Initialize FCS evaluator
    fcs_evaluator = ForceConsistencyEvaluator(fps=30)
    
    # Generate and evaluate samples
    fcs_scores = []
    contact_ratios = []
    avg_contacts_list = []
    
    print(f"\nGenerating and evaluating {num_samples} samples...")
    
    for idx in tqdm(range(min(num_samples, len(dataset)))):
        try:
            # Get music conditioning from dataset
            sample = dataset[idx]
            music_features = sample['feature'].unsqueeze(0)  # (1, seq_len, feature_dim)
            
            # Move to device
            device = next(model.parameters()).device
            music_features = music_features.to(device)
            
            # Generate dance motion
            with torch.no_grad():
                # Generate sample using DDIM
                shape = (1, music_features.shape[1], model.repr_dim)
                generated_motion = model.diffusion.ddim_sample(shape, music_features)
                
                # Unnormalize
                generated_motion = model.normalizer.unnormalize(generated_motion)
                
                # Parse motion data
                b, s, c = generated_motion.shape
                
                # Split: [contacts(4), root_pos(3), local_q(24*6)]
                contact, motion = torch.split(generated_motion, (4, c - 4), dim=2)
                root_pos = motion[:, :, :3]
                local_q_6d = motion[:, :, 3:]
                
                # Convert 6D to axis-angle
                from dataset.quaternion import ax_from_6v
                local_q = ax_from_6v(local_q_6d.reshape(b, s, 24, 6))
                
                # Forward kinematics
                joint_positions = model.diffusion.smpl.forward(local_q, root_pos)
                joint_positions = joint_positions.squeeze(0).cpu().numpy()  # (seq_len, 24, 3)
                
                # Evaluate FCS
                result = fcs_evaluator.evaluate_motion(joint_positions)
                fcs_scores.append(result['fcs_score'])
                contact_ratios.append(result['contact_ratio'])
                avg_contacts_list.append(result['avg_contacts'])
                
        except Exception as e:
            print(f"\nError generating/evaluating sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute statistics
    fcs_scores = np.array(fcs_scores)
    contact_ratios = np.array(contact_ratios)
    avg_contacts_list = np.array(avg_contacts_list)
    
    results = {
        'fcs_mean': np.mean(fcs_scores),
        'fcs_std': np.std(fcs_scores),
        'fcs_median': np.median(fcs_scores),
        'fcs_min': np.min(fcs_scores),
        'fcs_max': np.max(fcs_scores),
        'fcs_percentile_25': np.percentile(fcs_scores, 25),
        'fcs_percentile_75': np.percentile(fcs_scores, 75),
        'contact_ratio_mean': np.mean(contact_ratios),
        'avg_contacts_mean': np.mean(avg_contacts_list),
        'num_evaluated': len(fcs_scores),
        'all_scores': fcs_scores,
        'all_contact_ratios': contact_ratios,
        'all_avg_contacts': avg_contacts_list
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate and evaluate samples from trained EDGE model')
    parser.add_argument('--checkpoint', type=str, default='weights/train-2000.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/aistpp',
                        help='Path to AIST++ dataset')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='fcs_generated_results.txt',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        weights_dir = Path('weights')
        if weights_dir.exists():
            for ckpt in sorted(weights_dir.glob('*.pt')):
                print(f"  - {ckpt}")
        return
    
    # Generate and evaluate
    results = generate_and_evaluate(args.checkpoint, args.data_path, args.num_samples)
    
    # Print results
    print("\n" + "=" * 70)
    print("FCS EVALUATION RESULTS - GENERATED SAMPLES")
    print("=" * 70)
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Number of samples: {results['num_evaluated']}")
    print(f"\nFCS Score Statistics:")
    print(f"  Mean:       {results['fcs_mean']:.6f}")
    print(f"  Std Dev:    {results['fcs_std']:.6f}")
    print(f"  Median:     {results['fcs_median']:.6f}")
    print(f"  Min:        {results['fcs_min']:.6f}")
    print(f"  Max:        {results['fcs_max']:.6f}")
    print(f"  25th %ile:  {results['fcs_percentile_25']:.6f}")
    print(f"  75th %ile:  {results['fcs_percentile_75']:.6f}")
    print(f"\nContact Statistics:")
    print(f"  Avg contact ratio: {results['contact_ratio_mean']:.3f}")
    print(f"  Avg num contacts:  {results['avg_contacts_mean']:.3f}")
    print("=" * 70)
    
    # Save results
    with open(args.output, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FCS EVALUATION - GENERATED SAMPLES\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Samples evaluated: {results['num_evaluated']}\n\n")
        f.write("FCS Statistics:\n")
        f.write(f"  Mean:       {results['fcs_mean']:.6f}\n")
        f.write(f"  Std Dev:    {results['fcs_std']:.6f}\n")
        f.write(f"  Median:     {results['fcs_median']:.6f}\n")
        f.write(f"  Min:        {results['fcs_min']:.6f}\n")
        f.write(f"  Max:        {results['fcs_max']:.6f}\n")
        f.write(f"  25th %ile:  {results['fcs_percentile_25']:.6f}\n")
        f.write(f"  75th %ile:  {results['fcs_percentile_75']:.6f}\n\n")
        f.write("Contact Statistics:\n")
        f.write(f"  Contact ratio: {results['contact_ratio_mean']:.3f}\n")
        f.write(f"  Avg contacts:  {results['avg_contacts_mean']:.3f}\n\n")
        f.write("All FCS scores:\n")
        for i, score in enumerate(results['all_scores']):
            f.write(f"  Sample {i:3d}: {score:.6f}\n")
    
    print(f"\nResults saved to: {args.output}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("FCS measures physics violations (lower = better)")
    print(f"\nYour model's FCS: {results['fcs_mean']:.3f}")
    print("\nRecommended next steps:")
    print("1. Compare with real AIST++ data (run eval_real_data_fcs.py)")
    print("2. If FCS is high (>2), consider:")
    print("   - Training with FCS as an additional loss term")
    print("   - Post-processing to enforce physics constraints")
    print("   - Analyzing which motions have highest violations")
    print("=" * 70)


if __name__ == '__main__':
    main()
