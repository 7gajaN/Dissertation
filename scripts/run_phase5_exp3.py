"""
Phase 5 Experiment 3: guidance start-step ablation on the baseline.

Sweeps guidance_start_step ∈ {0, 10, 25, 35, 45} at the sweet-spot λ=1.0 found
in Experiment 1. Tests whether late-step guidance is the right default or if
earlier guidance helps. Only runs on the baseline because the trained models
are saturated and would show no effect regardless of start step.

Usage:
    /venv/edge/bin/python scripts/run_phase5_exp3.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_checkpoints import load_model, load_test_data, generate_and_evaluate


CHECKPOINT = "runs/baseline/no_fcs/weights/train-2000.pt"
LAMBDA = 1.0  # sweet spot from Experiment 1
START_STEPS = [0, 10, 25, 35, 45]
NUM_SAMPLES = 50
FCS_PREDICTOR_PATH = "models/fcs_predictor.pt"
DEVICE = "cuda"
SEED = 42

OUT_PATH = Path("runs/phase5/exp3_start_step.json")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading test dataset + model...")
    test_dataset = load_test_data()
    diffusion, normalizer, smpl = load_model(
        CHECKPOINT,
        feature_type="jukebox",
        device=DEVICE,
        fcs_predictor_path=FCS_PREDICTOR_PATH,
    )
    if diffusion.fcs_predictor is None:
        print("ERROR: predictor failed to load")
        sys.exit(1)

    results = {}
    for start in START_STEPS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        print(f"\nstart_step = {start}")
        fcs, pfc = generate_and_evaluate(
            diffusion, normalizer, smpl, test_dataset, NUM_SAMPLES, DEVICE,
            guidance_scale=LAMBDA,
            guidance_start_step=start,
        )
        row = {
            "fcs_mean": float(np.mean(fcs)),
            "fcs_std": float(np.std(fcs)),
            "fcs_median": float(np.median(fcs)),
            "pfc_mean": float(np.mean(pfc)),
            "pfc_std": float(np.std(pfc)),
            "pfc_median": float(np.median(pfc)),
            "n": len(fcs),
        }
        results[str(start)] = row
        print(f"  FCS = {row['fcs_mean']:.4f} (med {row['fcs_median']:.4f})   "
              f"PFC = {row['pfc_mean']:.4f} (med {row['pfc_median']:.4f})")

    # ─────── Summary ───────
    print()
    print("=" * 78)
    print("Phase 5 Experiment 3 — Start-step ablation (baseline, λ=1.0)")
    print("=" * 78)
    header = f"{'start_step':>10} {'FCS mean':>10} {'FCS med':>10} {'PFC mean':>10} {'PFC med':>10}"
    print(header)
    print("-" * 78)
    for start in START_STEPS:
        r = results[str(start)]
        print(f"{start:>10} {r['fcs_mean']:>10.4f} {r['fcs_median']:>10.4f} "
              f"{r['pfc_mean']:>10.4f} {r['pfc_median']:>10.4f}")
    print("=" * 78)

    with open(OUT_PATH, "w") as f:
        json.dump({
            "checkpoint": CHECKPOINT,
            "lambda": LAMBDA,
            "start_steps": START_STEPS,
            "num_samples": NUM_SAMPLES,
            "seed": SEED,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
