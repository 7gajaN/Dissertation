"""
Phase 5 Experiment 1: inference-time physics guidance sweep.

Sweeps guidance_scale over a per-model lambda list, with the model and test
dataset loaded only once per checkpoint. Writes results to
runs/phase5/exp1_results.json and prints a summary table.

The baseline gets a finer lambda sweep around its sweet spot (1.0). The other
models use a coarser sweep since they are expected to saturate.

Usage:
    /venv/edge/bin/python scripts/run_phase5_exp1.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_checkpoints import load_model, load_test_data, generate_and_evaluate


SWEEP = {
    "baseline_no_fcs": {
        "ckpt": "runs/baseline/no_fcs/weights/train-2000.pt",
        "lambdas": [0.0, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
    },
    "phase3_fcs_w12": {
        "ckpt": "runs/fcs/physics_w12/weights/train-2000.pt",
        "lambdas": [0.0, 0.01, 0.1, 1.0, 10.0],
    },
    "phase4_fcs_com_bilat": {
        "ckpt": "runs/phase4/fcs_com_bilateral/weights/train-2000.pt",
        "lambdas": [0.0, 0.01, 0.1, 1.0, 10.0],
    },
}

NUM_SAMPLES = 50
GUIDANCE_START_STEP = 25
FCS_PREDICTOR_PATH = "models/fcs_predictor.pt"
DEVICE = "cuda"
SEED = 42

OUT_PATH = Path("runs/phase5/exp1_results.json")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading test dataset...")
    test_dataset = load_test_data()
    print(f"Test dataset: {len(test_dataset)} samples\n")

    all_results = {}

    for ckpt_name, cfg in SWEEP.items():
        ckpt_path = cfg["ckpt"]
        lambdas = cfg["lambdas"]
        if not os.path.exists(ckpt_path):
            print(f"WARN: {ckpt_path} not found, skipping {ckpt_name}")
            continue

        print("=" * 70)
        print(f"Checkpoint: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"  Lambdas: {lambdas}")
        print("=" * 70)

        # Load once, attach predictor + normalizer
        diffusion, normalizer, smpl = load_model(
            ckpt_path,
            feature_type="jukebox",
            device=DEVICE,
            fcs_predictor_path=FCS_PREDICTOR_PATH,
        )
        if diffusion.fcs_predictor is None:
            print("ERROR: predictor failed to load, aborting this checkpoint")
            continue

        ckpt_results = {}
        for lam in lambdas:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            torch.cuda.manual_seed_all(SEED)

            print(f"\n  λ = {lam}")
            fcs_scores, pfc_scores = generate_and_evaluate(
                diffusion, normalizer, smpl, test_dataset, NUM_SAMPLES, DEVICE,
                guidance_scale=lam,
                guidance_start_step=GUIDANCE_START_STEP,
            )
            row = {
                "fcs_mean": float(np.mean(fcs_scores)),
                "fcs_std": float(np.std(fcs_scores)),
                "fcs_median": float(np.median(fcs_scores)),
                "pfc_mean": float(np.mean(pfc_scores)),
                "pfc_std": float(np.std(pfc_scores)),
                "pfc_median": float(np.median(pfc_scores)),
                "n": len(fcs_scores),
            }
            ckpt_results[str(lam)] = row
            print(f"    FCS = {row['fcs_mean']:.4f} ± {row['fcs_std']:.4f}   "
                  f"PFC = {row['pfc_mean']:.4f} ± {row['pfc_std']:.4f}")

        all_results[ckpt_name] = {
            "checkpoint": ckpt_path,
            "lambdas": ckpt_results,
        }

        # Free GPU
        del diffusion, normalizer, smpl
        torch.cuda.empty_cache()

    # ─────────────── Summary table ───────────────
    print()
    print("=" * 80)
    print("Phase 5 Experiment 1 — Summary")
    print("=" * 80)
    header = f"{'Model':<28} {'λ':>6} {'FCS':>9} {'FCS std':>9} {'PFC':>9} {'PFC std':>9}"
    print(header)
    print("-" * 80)
    for ckpt_name, info in all_results.items():
        for lam_str, row in info["lambdas"].items():
            print(f"{ckpt_name:<28} {lam_str:>6} {row['fcs_mean']:>9.4f} {row['fcs_std']:>9.4f} "
                  f"{row['pfc_mean']:>9.4f} {row['pfc_std']:>9.4f}")
    print("=" * 80)

    # Save
    with open(OUT_PATH, "w") as f:
        json.dump({
            "num_samples": NUM_SAMPLES,
            "guidance_start_step": GUIDANCE_START_STEP,
            "seed": SEED,
            "sweep": {k: {"ckpt": v["ckpt"], "lambdas": v["lambdas"]} for k, v in SWEEP.items()},
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
