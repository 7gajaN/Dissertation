"""
Utility script to visualize training metrics saved during EDGE training.

Usage:
    python plot_metrics.py --experiment runs/exp1/training_metrics.json
    python plot_metrics.py --csv runs/exp1/training_metrics.csv
"""

import argparse
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics_json(json_file):
    """Load metrics from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def load_metrics_csv(csv_file):
    """Load metrics from CSV file."""
    metrics = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append({
                'epoch': int(row['Epoch']),
                'total_loss': float(row['Total_Loss']),
                'train_loss': float(row['Train_Loss']),
                'v_loss': float(row['V_Loss']),
                'fk_loss': float(row['FK_Loss']),
                'foot_loss': float(row['Foot_Loss']),
                'fcs_score': None if row['FCS_Score'] == 'N/A' else float(row['FCS_Score']),
                'type': row['Type']
            })
    return {'metrics': metrics}


def plot_metrics(data, output_dir=None):
    """Create comprehensive plots of training metrics."""
    metrics = data['metrics']
    
    # Separate progress and checkpoint metrics
    progress_metrics = [m for m in metrics if m['type'] == 'progress']
    checkpoint_metrics = [m for m in metrics if m['type'] == 'checkpoint']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Metrics - {data.get('experiment', 'EDGE')}", fontsize=16)
    
    # Plot 1: Total Loss over epochs
    ax = axes[0, 0]
    if progress_metrics:
        epochs_prog = [m['epoch'] for m in progress_metrics]
        total_loss_prog = [m['total_loss'] for m in progress_metrics]
        ax.plot(epochs_prog, total_loss_prog, 'o-', label='Progress (50 epoch)', alpha=0.7)
    
    if checkpoint_metrics:
        epochs_ckpt = [m['epoch'] for m in checkpoint_metrics]
        total_loss_ckpt = [m['total_loss'] for m in checkpoint_metrics]
        ax.plot(epochs_ckpt, total_loss_ckpt, 's-', label='Checkpoint', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Individual Loss Components (from checkpoints)
    ax = axes[0, 1]
    if checkpoint_metrics:
        epochs = [m['epoch'] for m in checkpoint_metrics]
        ax.plot(epochs, [m['train_loss'] for m in checkpoint_metrics], 'o-', label='Train Loss')
        ax.plot(epochs, [m['v_loss'] for m in checkpoint_metrics], 's-', label='V Loss')
        ax.plot(epochs, [m['fk_loss'] for m in checkpoint_metrics], '^-', label='FK Loss')
        ax.plot(epochs, [m['foot_loss'] for m in checkpoint_metrics], 'd-', label='Foot Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title('Loss Components (Checkpoints)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: FCS Score over time
    ax = axes[1, 0]
    if checkpoint_metrics:
        epochs = [m['epoch'] for m in checkpoint_metrics]
        fcs_scores = [m['fcs_score'] for m in checkpoint_metrics if m['fcs_score'] is not None]
        fcs_epochs = [m['epoch'] for m in checkpoint_metrics if m['fcs_score'] is not None]
        
        if fcs_scores:
            ax.plot(fcs_epochs, fcs_scores, 'o-', color='red', linewidth=2)
            ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Target (< 0.5)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('FCS Score')
            ax.set_title('Force Consistency Score (Physics Quality)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No FCS data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Loss Component Percentages
    ax = axes[1, 1]
    if checkpoint_metrics and len(checkpoint_metrics) > 0:
        # Use last checkpoint
        last = checkpoint_metrics[-1]
        components = ['Train', 'V', 'FK', 'Foot']
        values = [last['train_loss'], last['v_loss'], last['fk_loss'], last['foot_loss']]
        total = sum(values)
        percentages = [100 * v / total for v in values]
        
        colors = plt.cm.Set3(range(len(components)))
        ax.pie(percentages, labels=components, autopct='%1.1f%%', colors=colors)
        ax.set_title(f"Loss Distribution (Epoch {last['epoch']})")
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_file = Path(output_dir) / 'training_metrics_plot.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def print_summary(data):
    """Print summary statistics."""
    metrics = data['metrics']
    checkpoint_metrics = [m for m in metrics if m['type'] == 'checkpoint']
    
    if not checkpoint_metrics:
        print("No checkpoint metrics found.")
        return
    
    print(f"\n{'='*70}")
    print(f"TRAINING METRICS SUMMARY - {data.get('experiment', 'EDGE')}")
    print(f"{'='*70}")
    print(f"  Total Epochs:    {data.get('total_epochs', 'N/A')}")
    print(f"  Batch Size:      {data.get('batch_size', 'N/A')}")
    print(f"  Feature Type:    {data.get('feature_type', 'N/A')}")
    print(f"  Checkpoints:     {len(checkpoint_metrics)}")
    print(f"  Progress points: {len([m for m in metrics if m['type'] == 'progress'])}")
    
    # First and last checkpoint comparison
    first = checkpoint_metrics[0]
    last = checkpoint_metrics[-1]
    
    print(f"\n  FIRST CHECKPOINT (Epoch {first['epoch']}):")
    print(f"    Total Loss:  {first['total_loss']:.6f}")
    print(f"    FCS Score:   {first.get('fcs_score', 'N/A')}")
    
    print(f"\n  LAST CHECKPOINT (Epoch {last['epoch']}):")
    print(f"    Total Loss:  {last['total_loss']:.6f}")
    print(f"    FCS Score:   {last.get('fcs_score', 'N/A')}")
    
    # Calculate improvements
    loss_improvement = ((first['total_loss'] - last['total_loss']) / first['total_loss']) * 100
    print(f"\n  IMPROVEMENT:")
    print(f"    Total Loss:  {loss_improvement:+.2f}%")
    
    if first.get('fcs_score') and last.get('fcs_score'):
        if first['fcs_score'] > 0:
            fcs_improvement = ((first['fcs_score'] - last['fcs_score']) / first['fcs_score']) * 100
            print(f"    FCS Score:   {fcs_improvement:+.2f}%")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize EDGE training metrics")
    parser.add_argument('--json', type=str, help='Path to training_metrics.json file')
    parser.add_argument('--csv', type=str, help='Path to training_metrics.csv file')
    parser.add_argument('--output', type=str, help='Output directory for plots (default: show plot)')
    parser.add_argument('--summary', action='store_true', help='Print summary only (no plot)')
    
    args = parser.parse_args()
    
    # Load data
    if args.json:
        data = load_metrics_json(args.json)
    elif args.csv:
        data = load_metrics_csv(args.csv)
    else:
        print("Error: Provide either --json or --csv argument")
        return
    
    # Print summary
    print_summary(data)
    
    # Plot (unless --summary only)
    if not args.summary:
        try:
            plot_metrics(data, args.output)
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")
            print("Showing data summary only.")


if __name__ == "__main__":
    main()
