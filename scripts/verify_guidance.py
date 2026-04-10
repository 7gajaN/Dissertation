"""
Verification harness for inference-time physics guidance (Phase 5).

Runs three checks:
  1. Regression: with guidance_scale=0, output is bit-identical to vanilla DDIM
     (uses the same RNG seed for both runs).
  2. Gradient sanity: with guidance_scale>0, prints x_start / joints / FCS / grad
     magnitudes per guidance step to confirm they look sane.
  3. Memory: tracks GPU peak memory across both runs.

Usage:
    /venv/edge/bin/python scripts/verify_guidance.py \
        --checkpoint runs/phase4/fcs_com_bilateral/weights/train-2000.pt \
        --fcs_predictor_path models/fcs_predictor.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

# Make project imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.quaternion import ax_from_6v
from eval_checkpoints import load_model, load_test_data


def hook_predictor_for_logging(diffusion):
    """Register a forward hook on fcs_predictor to log magnitudes."""
    if diffusion.fcs_predictor is None:
        return None
    log_state = {"calls": 0}

    def hook(module, inputs, output):
        if log_state["calls"] < 5:
            joints = inputs[0]
            print(
                f"      [call {log_state['calls']}] "
                f"joints |.|mean={joints.abs().mean().item():.4f}, "
                f"joints range=[{joints.min().item():.3f}, {joints.max().item():.3f}], "
                f"fcs={output.mean().item():.6e}"
            )
        log_state["calls"] += 1

    return diffusion.fcs_predictor.register_forward_hook(hook)


def standalone_predictor_check(diffusion, normalizer, smpl, test_dataset, device):
    """Sanity-check the predictor by feeding it a real (unnormalized) motion."""
    print()
    print("=" * 60)
    print("PRE-TEST: Predictor sanity on real (unnormalized) motion")
    print("=" * 60)
    # Pull a real sample from the test set
    pose, cond, _, _ = test_dataset[0]
    pose = pose.unsqueeze(0).to(device)  # (1, 150, 151)
    # The dataset already returns normalized motion. Test both normalized and unnormalized.
    with torch.no_grad():
        # ── Normalized (matches training-time predictor input) ──
        x_norm = pose
        b_, s_, _ = x_norm.shape
        pos_n = x_norm[:, :, 4:7]
        q_n = ax_from_6v(x_norm[:, :, 7:].reshape(b_, s_, -1, 6))
        joints_n = smpl.forward(q_n, pos_n)
        fcs_n = diffusion.fcs_predictor(joints_n).item()
        print(f"  normalized:    joints |.|mean={joints_n.abs().mean().item():.4f}, fcs={fcs_n:.6e}")

        # ── Unnormalized ──
        x_unnorm = normalizer.unnormalize(pose)
        pos_u = x_unnorm[:, :, 4:7]
        q_u = ax_from_6v(x_unnorm[:, :, 7:].reshape(b_, s_, -1, 6))
        joints_u = smpl.forward(q_u, pos_u)
        fcs_u = diffusion.fcs_predictor(joints_u).item()
        print(f"  unnormalized:  joints |.|mean={joints_u.abs().mean().item():.4f}, fcs={fcs_u:.6e}")
    return fcs_n, fcs_u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fcs_predictor_path", type=str, default="models/fcs_predictor.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device

    print("=" * 60)
    print("Loading model + FCS predictor...")
    print("=" * 60)
    diffusion, normalizer, smpl = load_model(
        args.checkpoint,
        feature_type="jukebox",
        device=device,
        fcs_predictor_path=args.fcs_predictor_path,
    )
    if diffusion.fcs_predictor is None:
        print("ERROR: FCS predictor failed to load. Aborting.")
        sys.exit(1)

    test_dataset = load_test_data()
    _, cond, _, _ = test_dataset[0]
    cond = cond.unsqueeze(0).to(device)
    shape = (1, 150, 151)

    standalone_predictor_check(diffusion, normalizer, smpl, test_dataset, device)

    # ── Test 1: Regression — guidance_scale=0 must equal vanilla output ──
    print()
    print("=" * 60)
    print("TEST 1: Regression check (guidance_scale=0 vs vanilla)")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        out_vanilla = diffusion.ddim_sample(shape, cond)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        out_guided_zero = diffusion.ddim_sample(shape, cond, guidance_scale=0.0)

    diff = (out_vanilla - out_guided_zero).abs().max().item()
    print(f"  Max abs difference: {diff:.2e}")
    if diff < 1e-6:
        print("  PASS: outputs are identical")
    else:
        print("  FAIL: outputs differ — guidance_scale=0 path is not equivalent")
        sys.exit(1)

    # ── Test 2: Gradient sanity ──
    print()
    print("=" * 60)
    print("TEST 2: Gradient sanity (guidance_scale=0.1)")
    print("=" * 60)

    hook_predictor_for_logging(diffusion)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out_guided = diffusion.ddim_sample(
        shape, cond, guidance_scale=0.1, guidance_start_step=25,
    )
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU memory during guided sampling: {peak_mb:.1f} MB")

    diff_guided_max = (out_vanilla - out_guided).abs().max().item()
    diff_guided_mean = (out_vanilla - out_guided).abs().mean().item()
    print(f"  Max abs diff vs vanilla:  {diff_guided_max:.6e} (expect > 0)")
    print(f"  Mean abs diff vs vanilla: {diff_guided_mean:.6e}")
    if diff_guided_max <= 1e-6:
        print("  WARN: guided output is identical to vanilla — gradient may not be applied")
    else:
        print("  PASS: guidance is changing the output")

    # ── Test 3: NaN check ──
    print()
    print("=" * 60)
    print("TEST 3: NaN/Inf check at extreme guidance scale")
    print("=" * 60)
    for scale in [1.0, 10.0]:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            out = diffusion.ddim_sample(shape, cond, guidance_scale=scale, guidance_start_step=25)
        n_nan = torch.isnan(out).sum().item()
        n_inf = torch.isinf(out).sum().item()
        print(f"  scale={scale}: NaN={n_nan}, Inf={n_inf}, range=[{out.min().item():.3f}, {out.max().item():.3f}]")

    print()
    print("=" * 60)
    print("All checks done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
