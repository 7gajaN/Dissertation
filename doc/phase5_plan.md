# Phase 5 Plan: Inference-Time Physics Guidance

This document is the detailed plan for Phase 5 of the dissertation. Phase 4 closed with strong training-time results: combining the FCS predictor loss with explicit physics losses (CoM with acceleration masking, bilateral with contact gating) gave a 10× improvement over real mocap on FCS. Phase 5 adds an orthogonal axis: **physics applied at inference time, on top of any trained model**.

## Table of Contents

1. [Motivation](#motivation)
2. [Background: Classifier Guidance in Diffusion Models](#background-classifier-guidance-in-diffusion-models)
3. [Algorithm](#algorithm)
4. [Why This Is the Right Phase 5 Direction](#why-this-is-the-right-phase-5-direction)
5. [Implementation Plan](#implementation-plan)
6. [Experiment Plan](#experiment-plan)
7. [Risks and Mitigations](#risks-and-mitigations)
8. [Expected Outcomes](#expected-outcomes)
9. [Timeline](#timeline)
10. [Open Questions](#open-questions)

---

## Motivation

In Phase 3 and Phase 4, physics enters the model only at training time — the FCS predictor loss and explicit physics losses shape what the model learns to generate. Once a checkpoint is saved, sampling proceeds with no further physics awareness: the DDIM sampling loop runs 50 denoising steps, each step nudges the noisy sample toward a clean motion, and the only physics that survives is whatever was baked into the weights.

This leaves three things on the table:

1. **Existing checkpoints can't be improved.** The Phase 3 and Phase 4 best models are fixed. Any improvement requires retraining from scratch, which costs ~26h per model on 1 GPU.
2. **No tunable physics-quality knob at use time.** A user generating dance for a stiff demonstration scene wants different physics than a user generating dance for an expressive performance. Training-time physics commits to one trade-off.
3. **No way to retrofit physics onto third-party models.** When comparing against Bailando, FACT, or the original EDGE checkpoint, we can only evaluate them — we can't show how much physics could be added to their outputs without retraining.

Inference-time physics guidance addresses all three by injecting physics at sampling time using the same FCS predictor we already trained.

## Background: Classifier Guidance in Diffusion Models

Classifier guidance, introduced by Dhariwal & Nichol (2021) for image diffusion, modifies the DDIM sampling step to include the gradient of an auxiliary classifier with respect to the current sample:

```
x_{t-1} = DDIM_step(x_t) + λ · ∇_{x_t} log p(y | x_t)
```

For class-conditional image generation, the auxiliary network is a classifier that outputs `log p(class | image)`. The gradient says "how should I move the image to make the classifier more confident in this class?", and adding it to the denoising step nudges the sample toward higher class confidence.

For physics guidance in motion generation, the auxiliary network is the FCS predictor, which outputs a scalar physics score (lower = more physical). The analogous formulation:

```
x_{t-1} = DDIM_step(x_t) − λ · ∇_{x_t} FCS_predictor(FK(x_{t-1}))
```

The negative sign because we want to decrease physics violations rather than increase a class probability. The structure is identical otherwise.

This is mathematically equivalent to sampling from a tilted distribution `p(x) · exp(−λ · FCS(x))`, which sharpens the prior toward physically plausible regions.

## Algorithm

Pseudocode for the modified `ddim_sample()` method:

```python
def ddim_sample_with_guidance(self, shape, cond, guidance_scale=0.0, guidance_start_step=25):
    """
    DDIM sampling with optional physics guidance from the FCS predictor.

    Args:
        shape: shape of the sample to generate
        cond: music conditioning
        guidance_scale: λ. 0 = no guidance (vanilla sampling), >0 = physics-guided
        guidance_start_step: step index at which to start applying guidance.
                             Earlier steps are too noisy for the predictor (out of distribution).
    """
    x = torch.randn(shape, device=device)

    for step_idx, (time, time_next) in enumerate(time_pairs):
        time_cond = torch.full((batch,), time, ...)

        # Standard DDIM step
        pred_noise, x_start, _ = self.model_predictions(x, cond, time_cond, ...)
        if time_next < 0:
            x = x_start
            continue
        x_next = compute_ddim_update(x_start, pred_noise, alpha, alpha_next, sigma)

        # Physics guidance
        if guidance_scale > 0 and step_idx >= guidance_start_step:
            x_next = x_next.detach().requires_grad_(True)
            with torch.enable_grad():
                # Run FK on the predicted clean motion
                model_xp = self.smpl.forward(
                    ax_from_6v(x_next[:, :, 7:].reshape(b, s, -1, 6)),
                    x_next[:, :, 4:7]
                )
                fcs_score = self.fcs_predictor(model_xp).mean()
                grad = torch.autograd.grad(fcs_score, x_next)[0]
            x_next = x_next.detach() - guidance_scale * grad

        x = x_next

    return x
```

Key design choices:

- **Apply guidance only in late sampling steps** (e.g., from step 25 of 50 onward). Early steps deal with very noisy `x` that is far from anything the FCS predictor has seen during training. Computing `FK(x_t)` on a noisy sample produces nonsensical joint positions, and the gradient of a network on nonsense input is itself nonsense. Restricting guidance to late steps keeps the predictor in distribution.
- **Use `x_start` (the predicted clean motion at step t)**, not the noisy `x_t` directly. The DDIM step already computes `x_start = (x_t - sqrt(1-alpha)*pred_noise) / sqrt(alpha)`. The FCS predictor was trained on clean motions, so `x_start` is the right input.
- **Detach and re-enable grad** around the predictor call to keep the computation graph local — we don't want gradients flowing back through the entire sampling chain.
- **Single scalar `guidance_scale`** as the user-facing knob. No per-step schedule in v1.

## Why This Is the Right Phase 5 Direction

This was selected from the full Phase 5 backlog (in `doc/ideas.md`) for several reasons:

### Scientific contribution
Inference-time physics guidance is **novel for motion generation**. Classifier guidance is standard in image diffusion (Dhariwal & Nichol 2021, Ho & Salimans 2021) but has not been applied to motion generation with a learned physics critic. This gives the dissertation a clean theoretical framing — "we adapt classifier guidance from image diffusion to motion generation, using a learned physics surrogate as the classifier" — that connects to a well-established literature.

### Compounds with existing work
The same FCS predictor that drives Phase 3 and Phase 4 training-time losses is reused at inference. No new networks, no new losses, no new data. The infrastructure is already in place — we're just calling the predictor in a new place.

### Works on any checkpoint
Crucially, this approach **does not require retraining**. We can apply guidance to:
- The Phase 0 baseline (no physics training)
- The Phase 3 FCS predictor model
- The Phase 4 best models (`fcs_com_bilateral`, `fcs1_com_bilateral`)
- Third-party models if we obtain their checkpoints (Bailando, FACT, original EDGE)

This gives a **2D ablation grid**: training-time physics × inference-time physics. Filling this grid is a much richer story than any single new training run could produce.

### Tunable trade-off
Training-time physics commits to one operating point. Inference guidance lets the user choose where to operate at use time. A single model can produce more or less physically constrained dance just by changing `λ`. This is a real practical advantage and a discussion point for the dissertation.

### Cheap to run
Each guided sampling run takes only modestly longer than vanilla sampling (~2× per sample). No GPU-weeks of training. We can sweep many `λ` values across many checkpoints in days, not weeks. The full experiment grid is feasible within Phase 5.

## Implementation Plan

### Files to Modify

| File | Change |
|------|--------|
| `model/diffusion.py` | Add `ddim_sample_with_guidance()` method (or extend existing `ddim_sample()` with `guidance_scale` and `guidance_start_step` parameters). Pass through to `sample()` and `render_sample()`. |
| `EDGE.py` | Pipe guidance args from CLI through to the sampling call. Add `guidance_scale` parameter to inference helpers. |
| `args.py` | Add `--guidance_scale` (float, default 0.0) and `--guidance_start_step` (int, default 25) to the test argument parser. |
| `test.py` | Pass guidance args to model.render_sample() / sampling call. |
| `eval_checkpoints.py` | Add `--guidance_scale` argument so checkpoint evaluation can be re-run with guidance enabled. Outputs go to a parallel `eval_results_guided_w<scale>.json` file alongside the existing `eval_results.json`. |

### New file (optional)

`scripts/run_guidance_sweep.sh` — convenience script to run a guidance scale sweep across all relevant checkpoints. Roughly:

```bash
for ckpt in baseline/no_fcs fcs/physics_w12 phase4/fcs_com_bilateral phase4/fcs1_com_bilateral; do
    for lambda in 0.0 0.01 0.1 1.0 10.0; do
        python eval_checkpoints.py \
            --run_dir runs/$ckpt \
            --num_samples 50 \
            --guidance_scale $lambda \
            --epochs_to_eval 2000
    done
done
```

### Code review checklist

Before running long experiments, verify:

1. **Gradient flows correctly**: confirm the gradient is non-zero, has reasonable magnitude (compare to typical `x` values during sampling), and points in the expected direction.
2. **Predictor receives clean joint positions**: print `model_xp` statistics inside the guidance call to confirm it's in the same range as training data, not noise-dominated garbage.
3. **No memory leak**: each sampling call should release the autograd graph after `torch.autograd.grad`. Watch GPU memory across many samples.
4. **Vanilla sampling still works**: with `guidance_scale=0`, the modified function must produce identical samples to the original `ddim_sample()`. This is a regression test.

## Experiment Plan

### Experiment 1: Sanity check on a single checkpoint
- One checkpoint: `runs/phase4/fcs_com_bilateral/weights/train-2000.pt` (current best)
- One song: a single test wav
- Five `λ` values: 0.0, 0.01, 0.1, 1.0, 10.0
- Generate 4 samples per setting (20 samples total)
- Compute FCS and PFC, render videos

**Goal**: confirm guidance produces a monotonic decrease in FCS as `λ` increases, and that high `λ` doesn't blow up the sample.

### Experiment 2: Guidance sweep across checkpoints
- Four checkpoints: `baseline/no_fcs`, `fcs/physics_w12`, `phase4/fcs_com_bilateral`, `phase4/fcs1_com_bilateral`
- Five `λ` values: 0.0, 0.01, 0.1, 1.0, 10.0
- 50 samples per (checkpoint, λ) cell
- Output: 4 × 5 = 20 cells in the ablation grid
- Compute FCS, PFC, and qualitative comparison renders for the most interesting cells

**Goal**: build the central ablation table for the dissertation chapter.

### Experiment 3: Guidance start-step ablation
- Best checkpoint from Experiment 2
- Best `λ` from Experiment 2
- Five start-step values: 0, 10, 25, 35, 45 (out of 50 sampling steps)
- 50 samples each

**Goal**: confirm or refute the assumption that guidance should be restricted to late sampling steps. May find that earlier guidance helps for some operating points.

### Experiment 4: Guidance compounding analysis
- Show the relationship between training-time physics strength and benefit from inference-time guidance
- Plot: x-axis is training-time physics strength (none → FCS predictor → FCS + CoM + Bilateral), y-axis is FCS, two curves (no guidance vs optimal guidance)
- 50 samples per point

**Goal**: answer "do training-time and inference-time physics compound, or does training-time physics already saturate the gain that guidance could provide?"

### Experiment 5: Qualitative comparison videos
- Pick the most interesting cells from Experiment 2
- Render 30s clips from the presentation music set
- Side-by-side videos: same model, different `λ` values, same song
- Same model with `λ=0` vs another model with strong training-time physics, same song

**Goal**: visual evidence that guidance produces meaningfully different output and that the trade-off between physics and expressiveness is observable, not just numerical.

## Risks and Mitigations

### Risk 1: Sampling collapses at high `λ`
Large guidance scales can push samples outside the training distribution and produce garbage. This is documented in image diffusion (Dhariwal & Nichol).

**Mitigation**: anneal `λ` over the sampling schedule (smaller `λ` early, larger `λ` late), or clip the gradient norm. Start with a small fixed `λ` (0.01–1.0) before testing extremes.

### Risk 2: The predictor is out of distribution on partially-denoised samples
The FCS predictor was trained on clean motions. Calling it on `x_t` for large `t` (very noisy) will produce nonsense outputs and nonsense gradients.

**Mitigation**: only apply guidance in late sampling steps (default `guidance_start_step=25`), where `x_start` is close to a clean motion. Validate by printing FCS predictor outputs at different `t` values and confirming they make sense in the late steps.

### Risk 3: No improvement over training-time physics
If the Phase 4 training-time physics has already pushed FCS to saturation, guidance may not help. The model might already be at the operating point guidance would push it toward.

**Mitigation**: this would itself be a publishable finding ("training-time physics dominates inference-time guidance for this task; the two approaches are not complementary"). The negative result is still valuable. Also: even if guidance doesn't help on the best model, it likely helps on the baseline and Phase 3 model, which is still worth showing.

### Risk 4: Implementation gradients are wrong
Easy to get the autograd graph wrong, especially with the FK operation in the middle. Could end up with zero gradient or NaN gradient.

**Mitigation**: write a unit test that takes a known motion, perturbs it slightly, and confirms the guidance gradient points in the direction of decreasing FCS. Also confirm that with `guidance_scale=0` the modified sampler produces bit-identical output to the original.

### Risk 5: Predictor evaluation is too slow at inference
Calling the predictor + computing gradients at every sampling step doubles or triples sampling time. For 50-sample evaluation runs, this scales to noticeable cost.

**Mitigation**: this is acceptable for evaluation but could be a problem for real-time use. Discussion point only — Phase 5 is about scientific contribution, not deployment efficiency.

## Expected Outcomes

### Likely findings (high confidence)
1. **Vanilla sampling reproduces with `λ=0`**. (Sanity check.)
2. **Small `λ` (0.01–0.1) reduces FCS without visible artifacts**, on at least the baseline and Phase 3 models.
3. **Large `λ` (≥10) produces visible deformation or collapse**. The interesting question is where the failure threshold is.
4. **Guidance helps more on weak baselines than strong ones**. Baseline + guidance might catch up to Phase 3 model without guidance. Phase 4 best model + guidance might gain little.

### Possible findings (medium confidence)
5. **Compounding works**: Phase 4 best + guidance produces the new best result.
6. **OR compounding fails**: Phase 4 best + guidance is no better than Phase 4 best alone, suggesting training-time physics has already saturated.
7. **An optimal `λ` exists per model**: bigger weight is needed for weaker models, smaller for stronger ones.

### Speculative findings (low confidence)
8. **Guidance is strong enough to make the baseline competitive with Phase 3**, suggesting some of the Phase 3 training-time work could be replaced by inference-time guidance. Would be surprising but not impossible.

Any combination of these is publishable.

## Timeline

Assuming 1 GPU and a few hours per evaluation run:

| Week | Activity |
|------|----------|
| 1 | Implementation + unit tests + sanity check (Experiment 1) |
| 2 | Run Experiment 2 (full sweep across checkpoints) — slowest part, ~3-5 days of GPU time |
| 3 | Run Experiments 3 and 4 (start-step ablation, compounding analysis) |
| 4 | Run Experiment 5 (qualitative renders), update documentation, integrate findings into the dissertation chapter |

Total: ~4 weeks. Leaves the remaining 4 weeks of the budget for writing, polishing, and any follow-up experiments suggested by the findings.

## Open Questions

These should be resolved early in implementation, not by guessing now:

1. **Does the FCS predictor's gradient have the right magnitude relative to the DDIM update?** The denoising step moves `x` by some amount per step; the guidance term should be comparable, not 1000× bigger or smaller. May need to normalize.
2. **Should we use the EMA model (`master_model`) for predictions during guidance, or the live model?** The existing sampling uses EMA. Guidance should be consistent.
3. **Should guidance be applied to the rotation channels (slots 7-150) or all 151 dimensions?** The contact channels (0-3) and root position (4-6) are not used by the FCS predictor; gradients on those are zero anyway, but might want to mask them explicitly.
4. **For batch sampling, should each sample get its own gradient, or share one across the batch?** Per-sample is correct but uses more compute. Default to per-sample.
5. **Does classifier-free guidance (Ho & Salimans 2021) apply here, given the model already uses CFG for music conditioning?** If yes, the two could be combined (one CFG term for music, one for physics). Worth exploring as a stretch goal.

These are not blockers — they will be answered by the sanity-check experiment.

---

## Summary

Phase 5 adds **inference-time physics guidance** as an orthogonal complement to the training-time physics losses developed in Phases 3 and 4. It reuses the existing FCS predictor with no retraining required, gives a tunable physics knob at inference time, applies to any checkpoint, and produces a 2D ablation grid (training-time × inference-time physics) that is the centerpiece of the Phase 5 dissertation chapter. Implementation is ~1 week, full experimental sweep is ~4 weeks, leaving ample time for writing.
