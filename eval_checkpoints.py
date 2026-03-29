"""
Evaluate FCS and PFC metrics on saved training checkpoints.

Usage:
    # Evaluate all checkpoints in a run
    python eval_checkpoints.py --run_dir runs/baseline/no_fcs --num_samples 50

    # Evaluate a single checkpoint
    python eval_checkpoints.py --checkpoint runs/baseline/no_fcs/weights/train-100.pt --num_samples 50

    # Evaluate with more samples for final results
    python eval_checkpoints.py --run_dir runs/baseline/no_fcs --num_samples 200
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.quaternion import ax_from_6v
from eval.eval_fcs import ForceConsistencyEvaluator, calculate_pfc_score
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton


def load_model(checkpoint_path, feature_type="jukebox", device="cuda"):
    """Load a model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    feature_dim = 4800 if feature_type == "jukebox" else 35
    horizon = 150  # 5 seconds at 30 fps
    repr_dim = 151  # 4 contact + 3 root + 24*6 rotation

    model = DanceDecoder(
        nfeats=repr_dim,
        seq_len=horizon,
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=feature_dim,
        activation=None,
    )

    smpl = SMPLSkeleton(device=device)

    diffusion = GaussianDiffusion(
        model,
        horizon,
        repr_dim,
        smpl,
        schedule="cosine",
        n_timestep=1000,
        predict_epsilon=False,
        loss_type="l2",
        clip_denoised=True,
        use_p2=True,
        cond_drop_prob=0.25,
        guidance_weight=2,
    )

    # Load EMA weights
    state_dict = checkpoint["ema_state_dict"]
    diffusion.master_model.load_state_dict(state_dict)
    diffusion.model.load_state_dict(state_dict)
    diffusion = diffusion.to(device)
    diffusion.eval()

    normalizer = checkpoint["normalizer"]
    return diffusion, normalizer, smpl


def load_test_data(data_path="data/", processed_dir="data/dataset_backups/", feature_type="jukebox"):
    """Load the test dataset."""
    train_cache = os.path.join(processed_dir, "train_tensor_dataset.pkl")
    test_cache = os.path.join(processed_dir, "test_tensor_dataset.pkl")

    if os.path.isfile(train_cache) and os.path.isfile(test_cache):
        train_dataset = pickle.load(open(train_cache, "rb"))
        test_dataset = pickle.load(open(test_cache, "rb"))
    else:
        train_dataset = AISTPPDataset(data_path=data_path, backup_path=processed_dir, train=True)
        test_dataset = AISTPPDataset(
            data_path=data_path, backup_path=processed_dir, train=False,
            normalizer=train_dataset.normalizer,
        )
    return test_dataset


def generate_and_evaluate(diffusion, normalizer, smpl, test_dataset, num_samples, device="cuda"):
    """Generate samples and compute FCS/PFC scores."""
    evaluator = ForceConsistencyEvaluator(fps=30)
    horizon = 150
    repr_dim = 151

    # Collect conditioning from test set
    all_cond = []
    for i in range(len(test_dataset)):
        _, cond, _, _ = test_dataset[i]
        all_cond.append(cond)

    fcs_scores = []
    pfc_scores = []
    batch_size = min(32, num_samples)
    generated = 0

    pbar = tqdm(total=num_samples, desc="Evaluating")
    while generated < num_samples:
        current_batch = min(batch_size, num_samples - generated)
        shape = (current_batch, horizon, repr_dim)

        # Pick random conditioning from test set
        indices = np.random.randint(0, len(all_cond), size=current_batch)
        cond = torch.stack([all_cond[i] for i in indices]).to(device)

        with torch.no_grad():
            samples = diffusion.ddim_sample(shape, cond)
            samples = normalizer.unnormalize(samples)

            b, s, c = samples.shape
            _, samples_motion = torch.split(samples, (4, c - 4), dim=2)
            pos = samples_motion[:, :, :3]
            q = ax_from_6v(samples_motion[:, :, 3:].reshape(b, s, 24, 6))
            joint_positions = smpl.forward(q.to(device), pos.to(device))

            for i in range(current_batch):
                joints_np = joint_positions[i].cpu().numpy()
                try:
                    fcs_result = evaluator.evaluate_motion(joints_np)
                    fcs_scores.append(fcs_result['fcs_score'])
                    pfc_score = calculate_pfc_score(joints_np)
                    pfc_scores.append(pfc_score)
                except Exception as e:
                    print(f"\nWarning: Failed to evaluate sample: {e}")

        generated += current_batch
        pbar.update(current_batch)

    pbar.close()
    return np.array(fcs_scores), np.array(pfc_scores)


def evaluate_checkpoint(checkpoint_path, test_dataset, num_samples, feature_type="jukebox", device="cuda"):
    """Evaluate a single checkpoint."""
    diffusion, normalizer, smpl = load_model(checkpoint_path, feature_type, device)
    fcs_scores, pfc_scores = generate_and_evaluate(
        diffusion, normalizer, smpl, test_dataset, num_samples, device
    )

    result = {
        "fcs_mean": float(np.mean(fcs_scores)),
        "fcs_std": float(np.std(fcs_scores)),
        "fcs_median": float(np.median(fcs_scores)),
        "pfc_mean": float(np.mean(pfc_scores)),
        "pfc_std": float(np.std(pfc_scores)),
        "pfc_median": float(np.median(pfc_scores)),
        "num_samples": len(fcs_scores),
    }

    # Free GPU memory
    del diffusion, normalizer, smpl
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints for FCS and PFC metrics")
    parser.add_argument("--run_dir", type=str, default="", help="Run directory containing weights/")
    parser.add_argument("--checkpoint", type=str, default="", help="Single checkpoint path")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per checkpoint")
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--processed_data_dir", type=str, default="data/dataset_backups/")
    parser.add_argument("--output", type=str, default="", help="Output file path (default: run_dir/eval_results.json)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.checkpoint and not args.run_dir:
        print("Error: provide --run_dir or --checkpoint")
        sys.exit(1)

    # Find checkpoints
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        pattern = os.path.join(args.run_dir, "weights", "train-*.pt")
        checkpoints = sorted(glob.glob(pattern), key=lambda x: int(x.split("train-")[-1].split(".pt")[0]))

    if not checkpoints:
        print(f"No checkpoints found")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoint(s)")
    print(f"Evaluating with {args.num_samples} samples each\n")

    # Load test data once
    print("Loading test dataset...")
    test_dataset = load_test_data(args.data_path, args.processed_data_dir, args.feature_type)
    print(f"Test dataset: {len(test_dataset)} samples\n")

    # Evaluate each checkpoint
    results = {}
    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.split("train-")[-1].split(".pt")[0])
        print(f"{'='*60}")
        print(f"Epoch {epoch}: {ckpt_path}")
        print(f"{'='*60}")

        result = evaluate_checkpoint(ckpt_path, test_dataset, args.num_samples, args.feature_type, args.device)
        results[epoch] = result

        print(f"  FCS: {result['fcs_mean']:.4f} +/- {result['fcs_std']:.4f} (median: {result['fcs_median']:.4f})")
        print(f"  PFC: {result['pfc_mean']:.4f} +/- {result['pfc_std']:.4f} (median: {result['pfc_median']:.4f})")
        print(f"  Samples evaluated: {result['num_samples']}\n")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Epoch':>6} | {'FCS Mean':>10} {'FCS Std':>10} {'FCS Med':>10} | {'PFC Mean':>10} {'PFC Std':>10} {'PFC Med':>10} | {'N':>4}")
    print(f"{'-'*80}")
    for epoch in sorted(results.keys()):
        r = results[epoch]
        print(f"{epoch:>6} | {r['fcs_mean']:>10.4f} {r['fcs_std']:>10.4f} {r['fcs_median']:>10.4f} | "
              f"{r['pfc_mean']:>10.4f} {r['pfc_std']:>10.4f} {r['pfc_median']:>10.4f} | {r['num_samples']:>4}")
    print(f"{'='*80}")

    # Save results
    output_path = args.output
    if not output_path:
        output_dir = args.run_dir if args.run_dir else os.path.dirname(args.checkpoint)
        output_path = os.path.join(output_dir, "eval_results.json")

    with open(output_path, "w") as f:
        json.dump({"num_samples": args.num_samples, "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save CSV
    csv_path = output_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "FCS_Mean", "FCS_Std", "FCS_Median", "PFC_Mean", "PFC_Std", "PFC_Median", "N"])
        for epoch in sorted(results.keys()):
            r = results[epoch]
            writer.writerow([epoch, f"{r['fcs_mean']:.6f}", f"{r['fcs_std']:.6f}", f"{r['fcs_median']:.6f}",
                             f"{r['pfc_mean']:.6f}", f"{r['pfc_std']:.6f}", f"{r['pfc_median']:.6f}", r['num_samples']])
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
