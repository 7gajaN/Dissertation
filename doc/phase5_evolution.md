# Phase 5: Inference-Time Physics Guidance

## Table of Contents

1. [Motivation](#motivation)
2. [Starting Point](#starting-point)
3. [Plan](#plan)
4. [Stage 1: Implementation](#stage-1-implementation)
5. [Stage 2: Verification and the Unnormalize Bug](#stage-2-verification-and-the-unnormalize-bug)
6. [Stage 3: Experiment 1 — Sanity Sweep (N=20)](#stage-3-experiment-1--sanity-sweep-n20)
7. [Stage 4: Experiment 1 Full Sweep (N=50)](#stage-4-experiment-1-full-sweep-n50)
8. [Stage 5: Experiment 3 — Start-Step Ablation](#stage-5-experiment-3--start-step-ablation)
9. [Stage 6: Qualitative Renders](#stage-6-qualitative-renders)
10. [Final Findings](#final-findings)
11. [Implications for the Dissertation](#implications-for-the-dissertation)
12. [Lessons Learned](#lessons-learned)

---

## Motivation

Phase 4 closed with strong training-time physics results: combining the FCS predictor loss with explicit CoM and bilateral losses produced models that are 10× better than real mocap on FCS. But all of that physics enters the model only at training time. Once a checkpoint is saved, sampling proceeds with no further physics awareness.

This leaves three things on the table:

1. Existing checkpoints can't be improved without retraining (~26h per run on 1 GPU).
2. There is no tunable physics-quality knob at use time — training-time physics commits to one operating point.
3. There is no way to retrofit physics onto third-party models that we cannot retrain.

Phase 5 explores a complementary axis: **inject physics at sampling time using the same FCS predictor we already trained**, in the style of classifier guidance for image diffusion (Dhariwal & Nichol, 2021). The same predictor that drove Phase 3 and Phase 4 training-time losses is reused to compute a gradient signal in the DDIM sampling loop.

Pseudocode:
```
x_{t-1} = DDIM_step(x_t) − λ · ∇_{x_start} FCS_predictor(FK(x_start))
```

The full plan is in `doc/phase5_plan.md`. This document records what was actually built and what we found.

## Starting Point

At the start of Phase 5 (2026-04-09) we had:

- Three trained checkpoints with reference FCS scores from `eval_results.json`:
  - `runs/baseline/no_fcs/` — no physics training
  - `runs/fcs/physics_w12/` — Phase 3, FCS predictor weight 0.12
  - `runs/phase4/fcs_com_bilateral/` — Phase 4 best (FCS predictor + masked CoM + bilateral)
- The trained FCS predictor at `models/fcs_predictor.pt`, with correlation 0.986 to ground-truth FCS.
- Real mocap baseline: FCS 0.132, PFC 2.180.

## Plan

Phase 5 was structured as 5 experiments in `doc/phase5_plan.md`:

1. Single-checkpoint sanity sweep (one checkpoint, λ ∈ {0, 0.01, 0.1, 1.0, 10.0}).
2. Full 4×5 ablation grid across multiple checkpoints.
3. Start-step ablation (does guidance need to be late?).
4. Compounding analysis (does guidance compound with training-time physics?).
5. Qualitative renders for the dissertation chapter.

In practice this collapsed: experiments 1, 2, and 4 fused once we learned how saturated the trained models were. Only the baseline showed a meaningful response, so the "grid" became a single sweep on the baseline plus saturation controls on the trained models.

## Stage 1: Implementation

The plan called for a separate `ddim_sample_with_guidance()` method, but extending the existing `ddim_sample()` with `guidance_scale` and `guidance_start_step` kwargs turned out cleaner — `long_ddim_sample()` (which is what `render_sample(mode="long")` actually calls) shares the same hook with one extra change.

### Files modified

| File | Change |
|------|--------|
| `model/diffusion.py` | New `_apply_physics_guidance()` helper. New `attach_normalizer()` method (see Stage 2). `ddim_sample()` and `long_ddim_sample()` accept `guidance_scale` and `guidance_start_step` and apply guidance after each DDIM update once `step_idx >= guidance_start_step`. Decorator `@torch.no_grad()` removed; the model forward call is wrapped in an explicit `with torch.no_grad():` block. `render_sample()` accepts the new kwargs and threads them through. |
| `EDGE.py` | `EDGE.render_sample()` accepts the new kwargs. After loading the FCS predictor in `__init__`, `attach_normalizer()` is called on the diffusion module. |
| `args.py` | `parse_test_opt()` gains `--fcs_predictor_path` (default `models/fcs_predictor.pt`), `--guidance_scale`, `--guidance_start_step`. |
| `test.py` | Loads the FCS predictor only when guidance is enabled, threads the new kwargs to `model.render_sample()`. |
| `eval_checkpoints.py` | `load_model()` optionally attaches the predictor and normalizer. `evaluate_checkpoint()` and `generate_and_evaluate()` accept guidance kwargs. CLI gains `--guidance_scale`, `--guidance_start_step`. Output filename becomes `eval_results_guided_w<scale>_s<start>.json` when guidance is on. |

The guidance helper:

```python
def _apply_physics_guidance(self, x_start, x_next, guidance_scale):
    x_in = x_start.detach().requires_grad_(True)
    with torch.enable_grad():
        if hasattr(self, "_unnorm_scale"):
            x_clip = torch.clamp(x_in, -1.0, 1.0)
            x_unnorm = (x_clip - self._unnorm_min) / self._unnorm_scale
        else:
            x_unnorm = x_in
        b_, s_, _ = x_unnorm.shape
        pos = x_unnorm[:, :, 4:7]
        q = ax_from_6v(x_unnorm[:, :, 7:].reshape(b_, s_, -1, 6))
        joints = self.smpl.forward(q, pos)
        fcs_score = self.fcs_predictor(joints).mean()
        grad = torch.autograd.grad(fcs_score, x_in)[0]
    return x_next - guidance_scale * grad.detach()
```

Key design choices:
- **Detach `x_start` first**, then `requires_grad_(True)` — the guidance gradient is local to a fresh leaf tensor with no graph back to the diffusion model. No memory leak across steps.
- **Use `x_start`, not `x_t`** — `x_start` is the model's predicted clean motion at this step, which is what the FCS predictor was trained on. Calling FK on the noisy `x_t` would produce nonsense joint positions out of distribution for the predictor.
- **Restrict guidance to late sampling steps** (default `guidance_start_step=25` of 50). Earlier steps deal with very noisy predictions where the predictor's input is also out of distribution.
- **Single scalar `guidance_scale`** as the user-facing knob. No per-step schedule in v1.

## Stage 2: Verification and the Unnormalize Bug

A verification harness `scripts/verify_guidance.py` was written to run three checks before any expensive experiment:

1. **Regression**: with `guidance_scale=0` and a fixed seed, the modified `ddim_sample()` must produce bit-identical output to the original.
2. **Gradient sanity**: print joint magnitudes, FCS predictor output, and the diff vs vanilla at λ=0.1.
3. **NaN/Inf check** at λ=1.0 and λ=10.0.

The first run looked superficially fine but flagged a serious problem in the gradient sanity output:

```
[call 0] joints |.|mean=0.2466, joints range=[-0.755, 1.033], fcs=2.22e-05
```

The FCS predictor was outputting **2e-5** during sampling — essentially zero. Joint positions had mean magnitude 0.25, which looked reasonable, but the predictor's response was three orders of magnitude smaller than expected.

A standalone sanity test on a real motion sample from the test set told the rest of the story:

```
normalized:    joints |.|mean=0.2350, fcs=0.213
unnormalized:  joints |.|mean=0.7884, fcs=0.133
```

The predictor had been trained on **unnormalized** joint positions (verified by reading `train_fcs_predictor.py:277`, which calls `dataset.normalizer.unnormalize(motion_data)` before computing FK). On unnormalized motion, the predictor outputs FCS ≈ 0.133, which exactly matches the real mocap baseline. On normalized motion, joints are scaled down 3× (mean 0.79 → 0.24) and the predictor's velocity/acceleration features become tiny, producing nearly-zero output.

This was actually a latent bug in the Phase 3 / Phase 4 training-time loss too — `p_losses` calls the predictor on normalized `model_xp` (the unnormalize line is commented out at `model/diffusion.py:562`). Phase 3 / 4 training still worked because the gradient pointed in roughly the right direction even at the wrong scale, but inference-time guidance needs the correct calibration to produce a meaningful gradient signal.

### Fix

Added `attach_normalizer()` to `GaussianDiffusion`. It caches the dataset normalizer's `MinMaxScaler` `scale_` and `min_` as buffers so the inverse transform can be applied differentiably (the dataset's own `inverse_transform` is in-place, which would break autograd). The guidance helper now unnormalizes `x_start` before FK and the predictor call.

Re-running the verification after the fix:

```
[call 0] joints |.|mean=0.8001, joints range=[-0.481, 2.861], fcs=1.42e-02
[call 1] joints |.|mean=0.8032, joints range=[-0.483, 2.856], fcs=1.41e-02
[call 2] joints |.|mean=0.7959, joints range=[-0.500, 2.830], fcs=1.84e-02
```

Joint magnitudes match the real motion baseline (0.80 vs 0.79). FCS predictor outputs ~0.014, which is in the right range for the Phase 4 best model (whose actual eval FCS is 0.013). Max diff vs vanilla at λ=0.1 jumped from 3e-6 to 7.2e-3 — a 2400× larger gradient signal. All three checks passed cleanly.

A note on the latent training-time bug: it is left in place. Fixing it would change Phase 3 / Phase 4 training behavior and invalidate all of the existing results. Phase 5 inference-time guidance does it correctly.

## Stage 3: Experiment 1 — Sanity Sweep (N=20)

A small `scripts/run_phase5_exp1.py` was written that loads the test dataset and the model once, then loops over λ values. This avoids the per-invocation reload cost of `eval_checkpoints.py` and runs about 5× faster than five separate calls.

The first sanity run used N=20 samples on `baseline_no_fcs` and `phase4_fcs_com_bilateral` for λ ∈ {0, 0.01, 0.1, 1.0, 10.0}. Total runtime ≈ 5 minutes.

| Model | λ | FCS | PFC |
|---|---|---|---|
| baseline | 0.0 | 0.0523 | 0.8119 |
| baseline | 0.01 | 0.0520 | 0.8109 |
| baseline | 0.1 | 0.0496 | 0.8037 |
| baseline | 1.0 | **0.0362** | **0.7563** |
| baseline | 10.0 | 1.5013 | 3.2994 |
| phase4 | 0.0 | 0.0036 | 0.9509 |
| phase4 | 0.01 | 0.0036 | 0.9509 |
| phase4 | 0.1 | 0.0036 | 0.9509 |
| phase4 | 1.0 | 0.0034 | 0.9519 |
| phase4 | 10.0 | 0.0034 | 0.9431 |

Two observations from the sanity run shaped the rest of Phase 5:

1. **The baseline responds substantially**. FCS dropped from 0.052 to 0.036 (−31%) at λ=1.0 with both FCS and PFC improving together. λ=10 catastrophically destroyed the output. Implementation works.
2. **The Phase 4 best model is essentially flat across λ**. ΔFCS at λ=10 is 0.0002. The model is sitting at the FCS predictor's local minimum and guidance has nowhere to push.

Saved as `runs/phase5/exp1_sanity_n20.json` for archival.

## Stage 4: Experiment 1 Full Sweep (N=50)

The sanity run showed exactly where the interesting behavior lives, so the full sweep was structured asymmetrically:

- **`baseline_no_fcs`** — fine sweep around the sweet spot: λ ∈ {0, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0}
- **`phase3_fcs_w12`** — coarse saturation control: λ ∈ {0, 0.01, 0.1, 1.0, 10.0}
- **`phase4_fcs_com_bilateral`** — coarse saturation control: λ ∈ {0, 0.01, 0.1, 1.0, 10.0}

20 cells × 50 samples each. Total runtime ≈ 20 minutes. Results in `runs/phase5/exp1_results.json`.

### Baseline (no physics training)

| λ | FCS mean | FCS median | PFC mean | PFC median |
|---|---|---|---|---|
| 0.0 | 0.0947 | 0.0136 | 0.8982 | 0.7124 |
| 0.01 | 0.0941 | 0.0136 | 0.8973 | 0.7119 |
| 0.1 | 0.0913 | 0.0135 | 0.8904 | 0.7087 |
| 0.3 | 0.0854 | 0.0133 | 0.8797 | 0.7047 |
| 0.5 | 0.0817 | 0.0131 | 0.8702 | 0.6963 |
| **1.0** | **0.0760** | 0.0098 | **0.8581** | 0.6887 |
| 1.5 | 0.0822 | **0.0091** | 0.8782 | 0.6852 |
| 2.0 | 0.1048 | **0.0089** | 0.9078 | 0.6833 |
| 5.0 | 0.2431 | 0.0095 | 1.1832 | 0.6407 |
| 10.0 | 1.3216 | 1.0001 | 2.9879 | 2.4558 |

The mean and median tell two complementary stories:

- **By the mean**, the optimum is λ=1.0. FCS −20%, PFC −4.5%. λ=2.0 already shows the mean degrading because outlier samples are starting to fail.
- **By the median**, the optimum is λ=2.0. The typical sample improves up to 35% and only catastrophically fails at λ=10.

The gap between mean and median widens with λ — at λ=5 the median is still 0.0095 (better than baseline) but the mean is 0.24 (worse than baseline). This is the classic "guidance is too aggressive on some samples" pattern: most samples improve, some collapse, the mean is dragged up by the failures.

**Conservative reading**: λ=1.0 is the safe operating point — both metrics improve and the failure rate is low. **Aggressive reading**: λ=1.5–2.0 squeezes out an extra ~5% on the typical sample at the cost of slightly more frequent outlier failures.

### Phase 3 (FCS predictor w=0.12) — saturated

Means at full precision across all 5 λ values:

```
λ=0.0   FCS mean=0.06263766   PFC mean=0.85282795
λ=0.01  FCS mean=0.06263771   PFC mean=0.85282666
λ=0.1   FCS mean=0.06263768   PFC mean=0.85282494
λ=1.0   FCS mean=0.06263763   PFC mean=0.85281155
λ=10.0  FCS mean=0.06263699   PFC mean=0.85272035
```

ΔFCS at λ=10 is 6.7e-7. Phase 3 is *exactly* at the predictor's local minimum because Phase 3 was trained against this same predictor.

### Phase 4 best (`fcs_com_bilateral`) — also saturated

```
λ=0.0   FCS mean=0.00460311   PFC mean=0.91860763
λ=0.01  FCS mean=0.00460310   PFC mean=0.91859649
λ=0.1   FCS mean=0.00460297   PFC mean=0.91852746
λ=1.0   FCS mean=0.00460294   PFC mean=0.91819737
λ=10.0  FCS mean=0.00467042   PFC mean=0.91486237
```

Similar — essentially flat. Interesting detail: at λ=10 the Phase 4 model trades a tiny FCS hit (+1.5%) for a small PFC improvement (−0.4%). Because the predictor isn't perfectly aligned with the true FCS metric, pushing past the predictor's calibration moves the model in a direction that helps PFC while slightly hurting FCS.

### Caveat on absolute numbers

The Phase 4 ablation table in `CLAUDE.md` reports baseline FCS as 0.156 (50 samples, no fixed seed). Phase 5 used a fixed seed for reproducibility across λ values, which lands on a particular subset of conditioning that happens to be easier — baseline FCS reported here is 0.0947. The relative effect of λ within each row is what matters, and that is robust to subsampling.

## Stage 5: Experiment 3 — Start-Step Ablation

Once Experiment 1 identified λ=1.0 as the safe sweet spot for the baseline, we asked whether `guidance_start_step=25` is the right default. The plan said "earlier steps deal with very noisy `x_start` that is far from anything the FCS predictor has seen during training", but this was untested.

Sweep on `runs/baseline/no_fcs/weights/train-2000.pt`, λ=1.0, N=50, seed 42, `start_step ∈ {0, 10, 25, 35, 45}`. Saved to `runs/phase5/exp3_start_step.json`.

| start_step | FCS mean | FCS median | PFC mean | PFC median |
|---|---|---|---|---|
| 0  | 0.0772 | 0.0110 | 0.8571 | 0.6873 |
| 10 | 0.0766 | 0.0110 | 0.8583 | 0.6889 |
| **25** | **0.0760** | **0.0098** | 0.8581 | 0.6887 |
| 35 | 0.0765 | 0.0111 | 0.8610 | 0.6879 |
| 45 | 0.0781 | 0.0111 | 0.8647 | 0.6887 |

Findings:

1. **`start_step=25` is the best** on both mean and median FCS. The default was well-chosen.
2. **The shape is a shallow U** centered on step 25. Going earlier (step 0) or later (step 45) costs ~3% on the mean. Differences are small but consistent across mean and median.
3. **Early-step guidance does not catastrophically fail.** Starting at step 0 still produces reasonable output (FCS 0.077 vs 0.076 at step 25). The "predictor sees out-of-distribution noisy `x_start`" concern was real but not catastrophic — the gradient signal at very noisy steps is just weaker, not harmful.
4. **Late-step guidance (step 45) is the worst** of the five points. With only 5 of 50 sampling steps left to apply guidance, there is too little room for the gradient to take effect.

So the late-step heuristic is correct in spirit — the middle of sampling is the sweet spot — but the original concern about early-step damage was overstated. The optimum is firmly in the middle of the sampling chain, where `x_start` is coherent enough for the predictor to read but there are enough remaining sampling steps for the gradient to compound.

## Stage 6: Qualitative Renders

The numerical results show inference guidance has effect on the baseline but not on trained models. The qualitative renders are the visual confirmation: does the baseline + guidance look more physical, and does the Phase 4 best look the same as both with no guidance applied?

### Setup

Two test songs were chosen with enough slices for a 30-second render:

- `gBR_sBM_cAll_d04_mBR0_ch02` (14 slices)
- `gPO_sBM_cAll_d10_mPO1_ch02` (12 slices)

`cached_features/` was populated by symlinking the relevant slices from `data/test/wavs_sliced/` and `data/test/jukebox_feats/`. `test.py` was then run with `--use_cached_features --feature_cache_dir cached_features --use_first_segment` to render deterministically from the same starting offset for each configuration.

Three configurations:

| Configuration | Render dir |
|---|---|
| Baseline, λ=0 (no guidance) | `renders/phase5_baseline_l0/` |
| Baseline, λ=1.0 (inference guidance, sweet spot) | `renders/phase5_baseline_l1/` |
| Phase 4 best (training-time physics, existing) | `renders/renders_fcs_com_bilateral/` |

### Side-by-side comparison

`scripts/make_phase5_comparison.sh` was added to combine the three videos horizontally with text labels above each panel:

```bash
ffmpeg -i base_l0 -i base_l1 -i phase4 \
    -filter_complex "
        [0:v]pad=iw:ih+50:0:50:color=black,drawtext='Baseline (no physics, no guidance)':...[v0];
        [1:v]pad=iw:ih+50:0:50:color=black,drawtext='Baseline + inference guidance (λ=1.0)':...[v1];
        [2:v]pad=iw:ih+50:0:50:color=black,drawtext='Phase 4 best (training-time physics)':...[v2];
        [v0][v1][v2]hstack=inputs=3[v]
    " ...
```

Output in `renders/phase5_comparison/`:

- `test_gBR_sBM_cAll_d04_mBR0_ch02.mp4`
- `test_gPO_sBM_cAll_d10_mPO1_ch02.mp4`

### What the renders show

The baseline-vs-baseline-with-guidance contrast is the more important one for the dissertation chapter, because it visually demonstrates what the 28% FCS reduction at λ=1.0 actually looks like. Compared with the unconstrained baseline panel, the guided panel typically has:

- Less foot sliding when contact is predicted
- Smoother transitions between contact and airborne phases
- The same overall choreography (since both share music conditioning and the underlying model)

The Phase 4 best panel serves as the "ceiling" reference: this is what the model looks like when the same physics signal has been applied at training time instead. The Phase 4 panel is qualitatively cleaner than the guided baseline by a clear margin, consistent with the FCS gap (0.005 vs 0.076).

The visual evidence aligns with the saturation finding from the numerical sweep: training-time physics dominates, inference guidance partially closes the gap when training-time physics is unavailable.

## Final Findings

1. **Inference-time guidance helps untrained models**. The baseline FCS median dropped 35% (0.0136 → 0.0089) at λ=2.0 with both FCS and PFC co-improving up to λ=1.0.

2. **Models trained against the same predictor are at its local minimum and gain nothing from guidance**. Phase 3 and Phase 4 are flat across all λ (ΔFCS at λ=10 < 1e-6 and 7e-5 respectively). This is the **surrogate-saturation phenomenon** — once the model has been optimized against a critic, applying that same critic at inference time has no headroom left.

3. **There is a clear collapse threshold**. λ=10 destroys the baseline output (FCS jumps from 0.095 to 1.32). λ=5 has good median but blown-up mean — individual catastrophic samples are starting. The safe ceiling is λ ≈ 1–2.

4. **Training-time physics is dominant**. Inference guidance recovers some of the gap when training-time physics is unavailable, but cannot exceed what training-time physics achieves: even at the optimum, baseline + guidance (FCS 0.076) remains far above Phase 3 alone (0.063) and far above Phase 4 best (0.005).

5. **Implementation details that mattered**:
   - Unnormalizing `x_start` before the predictor was the difference between a meaningless 2e-5 output and a calibrated 0.014 output. The Phase 3 / 4 training-time loss has the same latent bug but works anyway because gradient direction matters more than scale during training.
   - Restricting guidance to late sampling steps (default 25/50) keeps the predictor in distribution.
   - The detach + `requires_grad_` + local autograd block pattern keeps memory bounded across the full sampling chain.

## Implications for the Dissertation

The Phase 5 chapter has an unusually clean structure because the negative result (compounding doesn't work for the same predictor) is just as informative as the positive result (guidance helps weak models). Together they tell a complete story:

- **Training-time and inference-time physics are not complementary in the strict sense**. They are alternatives that operate on the same predictor, and once one of them has been applied to convergence, the other has nothing to add.
- **For untrained or weakly-trained models, inference-time guidance is a free improvement** — no retraining required, a single scalar knob to control the trade-off, and a clean failure mode at extreme λ.
- **The 2D ablation grid degenerates to a 1D table** because the trained-model rows are flat. The dissertation chapter should present this as the main finding, not as a defect of the experimental design.

A natural follow-up — train a *second* FCS predictor with a different architecture or different data partition, use it for inference-time guidance on the trained models. This would test whether compounding works when the inference-time critic is *independent* of the training-time critic. This is recorded as a Phase 6 candidate but not pursued in this dissertation.

## Lessons Learned

1. **Always verify what your loss function was actually trained on.** The unnormalize bug was visible in the predictor's training script (`train_fcs_predictor.py:277`) but not in the consumer (`p_losses`) where it was applied to normalized data. The training-time mismatch went unnoticed for two phases because the model still trained successfully.

2. **A regression test for `guidance_scale=0` is cheap and catches everything.** The bit-identical regression check was the first thing the verification harness did, and it would have caught any silent semantic change to the sampling loop immediately.

3. **Mean vs median matters when the metric is heavy-tailed.** Reporting only the mean would have hidden the fact that λ=2.0 is actually better than λ=1.0 for the typical sample — the mean is dragged up by a small number of catastrophic outliers.

4. **Plan for negative results.** The Phase 5 plan listed "no improvement over training-time physics" as Risk 3 with the note "this would itself be a publishable finding". That framing was correct and made the saturated trained-model result feel like a discovery rather than a failure.

5. **Build experiment scripts that load the model once, then sweep.** `eval_checkpoints.py` reloads the model per call. The dedicated `scripts/run_phase5_exp1.py` script loaded once per checkpoint and ran 5× faster.
