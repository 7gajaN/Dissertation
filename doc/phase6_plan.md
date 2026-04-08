# Phase 6 Plan: Rescuing the Foot Height Loss

This document is the detailed plan for Phase 6. It addresses the one Phase 4 loss that failed to converge: the foot height during contact loss. The Phase 4 narrative currently has an honest negative result — "two of three explicit losses worked, the third did not." Phase 6 attempts to rescue the third loss using the same family of techniques (masking) that worked for the CoM loss.

## Table of Contents

1. [Motivation](#motivation)
2. [What the Loss Does](#what-the-loss-does)
3. [Why It Failed in Phase 4](#why-it-failed-in-phase-4)
4. [Diagnosis: Three Hypotheses](#diagnosis-three-hypotheses)
5. [Proposed Fixes](#proposed-fixes)
6. [Implementation Plan](#implementation-plan)
7. [Experiment Plan](#experiment-plan)
8. [Risks and Mitigations](#risks-and-mitigations)
9. [Expected Outcomes](#expected-outcomes)
10. [Timeline](#timeline)
11. [Why This Is Phase 6, Not Phase 5](#why-this-is-phase-6-not-phase-5)

---

## Motivation

The Phase 4 ablation table includes one negative result: the foot height during contact loss could not be made to work at any tested weight. At weight 5.0 (suggested by the phase2 branch), it destabilized training and produced FCS scores 19× worse than baseline. At weight 0.5, results were *worse* (FCS 5.288 vs 2.558), which is unusual — the loss didn't behave like a simple "too strong" weight problem.

The dissertation can publish this as an honest negative result, but the story is much cleaner if **all three explicit losses** can be made to work. The Phase 4 lesson — physics losses derived from quasi-static assumptions need motion-aware masking — is a stronger claim if it generalizes beyond CoM. Phase 6 tests whether the same fix family works for foot height.

If Phase 6 succeeds, the dissertation can say:
> "All three explicit physics losses are viable when properly gated by motion dynamics. Naive application of any of them is harmful, but a single family of fixes — acceleration or velocity masking — rescues each one."

This is a much cleaner academic statement than what we have now.

## What the Loss Does

Current implementation (`model/diffusion.py:614-620`):

```python
if self.foot_height_loss_weight > 0:
    min_h = model_feet[:, :, :, 2].min(dim=1, keepdim=True)[0].detach()  # (B,1,4)
    adj_h = model_feet[:, :, :, 2] - min_h                                # (B,S,4)
    contact_w2 = (model_contact > 0.95).float()
    height_loss = (adj_h * contact_w2).mean()
```

Step by step:
1. **Per-sequence ground reference**: take the minimum foot height over all timesteps for each sequence and each foot (4 foot indices: L_Ankle, R_Ankle, L_Toe, R_Toe). Detach this so the model can't satisfy the loss by pushing all feet underground.
2. **Adjusted height**: subtract the ground reference. Now `adj_h[b, t, f] = 0` means foot `f` is at the lowest point it ever reaches in this sequence; `adj_h > 0` means it is lifted above that minimum.
3. **Contact mask**: if the model predicts contact at this frame (sigmoid output > 0.95), the corresponding entry of `contact_w2` is 1; otherwise 0.
4. **Loss**: the mean of `adj_h * contact_w2` — average lifted-height during predicted contact.

Intent: enforce **geometric consistency between contact prediction and foot position**. If the model says "this foot is touching the ground", the foot's height must actually be at ground level. Otherwise the model is producing physically impossible motion: a planted foot that hovers in mid-air.

## Why It Failed in Phase 4

Two 500-epoch runs were trained with this loss in isolation:

| Run | Weight | FCS@500 | PFC@500 | Notes |
|-----|--------|---------|---------|-------|
| `height_test` | 5.0 | 2.558 | 1.299 | Total loss oscillating 1.4–2.3, FK loss 100× higher than baseline runs, height loss 37% of total and not decreasing |
| `height_w05` | 0.5 | 5.288 | 1.333 | Worse FCS than at higher weight |

For comparison, the baseline at the same epoch was FCS 0.134 / PFC 1.183. Both height runs were dramatically worse.

The training dynamics showed three pathological signs:

1. **FK loss exploded** (~1.0 vs typical ~0.01). The model could not satisfy both the foot height loss and the FK reconstruction loss simultaneously. Pulling feet to the ground deformed the rest of the joint chain.
2. **Total loss oscillated** instead of decreasing. The two losses fought each other in alternating directions step by step.
3. **Lower weight was worse than higher**. This is the unusual signal — typically a too-strong loss is fixed by reducing the weight, but here halving the weight (5.0 → 0.5) made FCS worse, not better. This suggests something more structural than weight tuning could fix.

## Diagnosis: Three Hypotheses

### Hypothesis 1: Loss assumes static contact, but contact is often transient

The contact mask `model_contact > 0.95` triggers whenever the model predicts contact, regardless of whether the foot is actually in a stable planted phase or in a transition (about to lift off, just touched down). During transitions, the foot's instantaneous position is *supposed* to be slightly above the ground — the contact label is correct in spirit but the height target is too aggressive.

**Evidence**: same failure mode as the unmasked CoM loss, which forced balance during dynamic frames where balance is not appropriate. The fix that worked for CoM was acceleration masking. The same fix family probably applies here.

### Hypothesis 2: The "ground reference" is too noisy

Using the per-sequence minimum foot height as the ground reference is sensitive to a single frame. A foot that briefly dips low during a stomp or jump landing becomes the floor for the entire sequence — and now every other contact frame is penalized for being "above the floor", even if "above the floor" is the correct geometric position.

**Evidence**: this failure mode doesn't depend on dynamic vs static frames; it would also affect long stationary phases. We'd need a more robust ground estimate (5th percentile, fixed assumption that data is centered at z=0, or learned per-foot ground level).

### Hypothesis 3: Linear penalty has unbounded contribution from outliers

The current loss is a mean of `adj_h`, which is linear. A single foot lifted by 1 meter contributes the same as 100 feet lifted by 1 cm each. Combined with Hypothesis 2, a single mis-detected contact during a leg swing produces a huge contribution that dominates the gradient.

**Evidence**: this would explain the oscillation — the model fixes one outlier, another appears, the gradient swings.

## Proposed Fixes

The fix combines all three diagnoses. Each fix is independent and we can ablate to find which matter most.

### Fix A: Velocity masking (mirror of CoM acceleration masking)

Only apply the loss to frames where the **foot velocity is low** (the foot is actually planted, not transitioning). Compute foot velocity as the temporal difference of foot positions, take the magnitude per foot, and gate the loss on that:

```python
foot_v = torch.norm(
    model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :], dim=-1
)  # (B, S-1, 4)
foot_v = F.pad(foot_v, (0, 0, 0, 1), value=0.0)  # pad to (B, S, 4)
is_planted = (foot_v < velocity_threshold).float()
contact_w2 = (model_contact > 0.95).float() * is_planted
```

Threshold suggestion: 0.01 m/s (initial guess, tunable). At 30 fps, this corresponds to motion of less than 0.33 mm per frame — essentially stationary.

**Why velocity, not acceleration like CoM?** CoM acceleration distinguishes static poses from dynamic ones at the body level. For feet specifically, the natural quantity is velocity — a planted foot has near-zero velocity by definition, regardless of what the rest of the body is doing.

### Fix B: Robust ground reference

Replace per-sequence minimum with a more robust estimate. Three options to test:

1. **Per-sequence 5th percentile**: still per-sequence but less sensitive to outliers
2. **Fixed at z=0**: assume the data normalization centers ground at zero (requires checking the dataset)
3. **Per-foot min, smoothed over a window**: take the minimum over a sliding window rather than the entire sequence

Option 1 is the safest first try.

### Fix C: Quadratic penalty with threshold

Replace the linear `adj_h * mask` with a thresholded squared penalty:

```python
clearance = 0.02  # 2 cm tolerance for natural foot height variation
height_above = F.relu(adj_h - clearance)
height_loss = ((height_above ** 2) * contact_w2).mean()
```

This:
- Allows up to 2 cm of natural variation without penalty (handles toe bend, heel lift)
- Penalizes larger violations quadratically (proportional to square of how far above ground)
- Gradient is bounded by `2 * height_above`, well-behaved at zero

### Combined fix

The most likely successful version applies all three:

```python
if self.foot_height_loss_weight > 0:
    # Robust ground reference (5th percentile per-sequence per-foot)
    ground_ref = torch.quantile(
        model_feet[:, :, :, 2], 0.05, dim=1, keepdim=True
    ).detach()                                                   # (B,1,4)

    # Adjusted height with tolerance
    adj_h = model_feet[:, :, :, 2] - ground_ref                  # (B,S,4)
    clearance = 0.02
    height_above = F.relu(adj_h - clearance)                     # (B,S,4)

    # Foot velocity for planted-foot masking
    foot_v = torch.norm(
        model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :], dim=-1
    )                                                             # (B,S-1,4)
    foot_v = F.pad(foot_v, (0, 0, 0, 1), value=0.0)              # (B,S,4)
    is_planted = (foot_v < 0.01).float()                          # (B,S,4)

    # Predicted contact gated by planted state
    contact_w2 = (model_contact > 0.95).float() * is_planted     # (B,S,4)

    height_loss = ((height_above ** 2) * contact_w2).mean()
```

Each component is independently testable; we'll ablate them in the experiment plan.

## Implementation Plan

### Files to Modify

| File | Change |
|------|--------|
| `model/diffusion.py` | Replace the existing foot height loss block with the combined fix above. Add 2 hyperparameters as constructor args: `foot_height_clearance` (default 0.02), `foot_height_velocity_threshold` (default 0.01). Keep the existing `foot_height_loss_weight` arg. |
| `EDGE.py` | Pass the 2 new hyperparameters through to `GaussianDiffusion`. |
| `args.py` | Add `--foot_height_clearance` and `--foot_height_velocity_threshold` (both float, with the same defaults). |
| `train.py` | Pass through. |

The change is localized to the loss computation block (~25 lines). No changes to the training loop, logging, or evaluation infrastructure.

### Verification

Before running long experiments:

1. **Sanity check on a real motion**: take a real AIST++ test sequence, compute the new loss with `weight=1.0`. Expect a small, finite, non-NaN value. Print the breakdown of `is_planted`, `contact_w2`, `height_above` to confirm each component is doing what it should.
2. **Sanity check on a synthetic broken motion**: take the same sequence and lift one foot by 0.5m at a frame where it should be planted. The loss should jump significantly.
3. **Backward compatibility**: with `foot_height_loss_weight=0.0`, training behavior must be identical to the current Phase 4 best. Quick smoke test: 5 epochs with the new code, all three new args at default, compare loss values to a known-good run.

## Experiment Plan

### Experiment 1: Component ablation (4 × 500-epoch runs)

Each run isolates one component of the combined fix to identify which parts are necessary.

| Run | Velocity mask | Robust ground ref | Quadratic + clearance |
|-----|--------------|-------------------|----------------------|
| `height_a_velmask` | yes | no (per-seq min) | no (linear) |
| `height_b_robustref` | no | yes (5th percentile) | no (linear) |
| `height_c_quadratic` | no | no (per-seq min) | yes (quadratic, 0.02 clearance) |
| `height_d_combined` | yes | yes | yes |

Weight: start with 1.0 for all four. 500 epochs each. Compare against the baseline (FCS 0.134) and the Phase 4 unmasked failure (FCS 5.288 at w=0.5, FCS 2.558 at w=5.0).

**Goal**: identify which components are necessary and which are sufficient. If `height_a_velmask` alone reaches baseline-or-better, the velocity mask is the key fix and the others are nice-to-have.

### Experiment 2: Weight tuning on the winning combination

Once a working configuration is found, sweep weights at 500 epochs:
- Weights to try: 0.1, 0.5, 1.0, 5.0
- Pick the best one.

### Experiment 3: Full combined run at 2000 epochs

If a working foot height variant exists, train it at 2000 epochs combined with the rest of the Phase 4 stack:

```bash
/venv/edge/bin/accelerate launch train.py \
    --fcs_loss_weight 0.12 --fcs_predictor_path models/fcs_predictor.pt \
    --com_loss_weight 0.05 --bilateral_loss_weight 5.0 \
    --foot_height_loss_weight <best_weight> \
    --foot_height_velocity_threshold 0.01 --foot_height_clearance 0.02 \
    --epochs 2000 --exp_name fcs_com_bilateral_height --project runs/phase6
```

This is the experiment that decides whether foot height adds anything on top of the existing Phase 4 best, or whether it just makes things worse.

### Experiment 4 (optional): Threshold tuning

If Experiment 3 succeeds, run a small sweep over `clearance` (0.01, 0.02, 0.05, 0.10) and `velocity_threshold` (0.005, 0.01, 0.02, 0.05). Each is 500 epochs since we're refining a working configuration, not finding one from scratch.

## Risks and Mitigations

### Risk 1: All four ablation runs fail
The loss is fundamentally incompatible with the task and no masking saves it.

**Mitigation**: this is also a publishable result. Foot height is the only quantity in the loss family that requires *exact* geometric matching (CoM and bilateral are softer constraints). It may be that the contact prediction is too noisy for any height target to work. We'd document this and use it as evidence that the contact prediction itself is the bottleneck — which motivates Phase 7 work on better contact heads (E13 in `ideas.md`).

### Risk 2: Velocity threshold is too sensitive
0.01 m/s is a guess. Too low and almost no frames are masked in (the loss has no effect). Too high and the loss is back to its original failure mode.

**Mitigation**: print the masking ratio in early training (what fraction of contact frames pass the velocity gate). Aim for ~50-80%. Adjust threshold based on this diagnostic before committing to full runs.

### Risk 3: Robust ground reference fights with the contact-detection signal
If the 5th percentile is consistently above the actual ground, the loss never triggers. If it's consistently below, the loss is back to its outlier-sensitive original.

**Mitigation**: print the ground reference for a few sequences during a sanity check. Compare against the dataset's actual minimum foot height distribution to see whether 5th percentile is in a reasonable range.

### Risk 4: Quadratic loss creates a new gradient issue
The CoM fix moved from `torch.norm` to squared distance specifically to avoid gradient singularities. The proposed quadratic loss here is `(F.relu(adj_h - clearance) ** 2)`, which is well-defined everywhere but has a kink at `adj_h = clearance`. This shouldn't cause numerical issues but is worth noting.

**Mitigation**: confirm gradients are well-behaved during the sanity check. If issues appear, switch to a smooth approximation like `softplus(adj_h - clearance) ** 2`.

### Risk 5: Even if the loss converges, it doesn't add anything to the Phase 4 best
The Phase 4 combined model already has FCS 0.013, well below real mocap. There may be no remaining "physics gain" available from a third loss.

**Mitigation**: even a no-improvement result is valuable for the dissertation — it would say "all three explicit losses are individually viable, but they don't all stack." The story changes from "two losses worked, one didn't" to "two losses stack productively, the third is viable in isolation but adds no marginal gain when combined", which is a richer finding.

## Expected Outcomes

### Likely findings
1. **Velocity masking alone is enough** to make the loss not destabilize training. (High confidence, since it directly addresses the failure mode observed in Phase 4.)
2. **Component ablation will show velocity masking is the most important component.** Robust ground reference and quadratic penalty are nice-to-have but not load-bearing. (Medium confidence.)
3. **The fixed loss works in isolation but adds little when combined with Phase 4 best.** The marginal gain from a third explicit loss on top of FCS predictor + CoM + bilateral is likely small. (Medium confidence.)

### Possible findings
4. **None of the four ablation runs work**, indicating the loss is structurally incompatible with this task regardless of masking.
5. **The combined run improves the FCS or PFC marginally** (e.g., FCS 0.012 vs 0.013), giving a small new best.
6. **The combined run is *worse* than Phase 4 best**, indicating that even a working foot height loss conflicts with the other losses in the combined setting.

Any of these is publishable.

## Timeline

| Week | Activity |
|------|----------|
| 1 | Implementation + sanity checks (~2-3 days) + Experiment 1 first runs launched (4 × 500 epochs = ~24h on 1 GPU, sequential) |
| 2 | Experiment 1 finishes, eval, decide winning configuration. Experiment 2 weight sweep (~3 × 500ep = ~18h sequential). |
| 3 | Experiment 3 full 2000-epoch run (~26h). Optional Experiment 4 threshold tuning. |
| 4 | Eval, qualitative renders, document results. Update `phase4_evolution.md` with the rescued result, or update `phase6_results.md` as a new chapter depending on outcome. |

Total: ~3-4 weeks. Should fit comfortably in the remaining timeline after Phase 5 if both phases run.

## Why This Is Phase 6, Not Phase 5

Phase 5 (inference-time guidance) and Phase 6 (foot height rescue) are independent and could run in either order. They are sequenced this way because:

1. **Phase 5 has higher novelty.** Inference-time physics guidance is a clean theoretical contribution that connects to a well-established literature (classifier guidance). Foot height rescue is an engineering refinement of an existing approach.
2. **Phase 5 doesn't require additional training.** It only needs new inference runs on existing checkpoints, which is much faster than the multiple training runs Phase 6 requires.
3. **Phase 6 builds on the same insight as Phase 4** (motion-aware masking). Doing it after Phase 5 lets the dissertation present masking as a well-validated principle (it worked for CoM, then Phase 5 adds inference guidance, then Phase 6 generalizes the masking principle to a third loss). The narrative arc is cleaner this way.
4. **If time runs short, Phase 6 can be dropped without losing the main contribution.** Phase 5 cannot be — it is the main novel contribution beyond Phase 4. Sequencing the more novel work first protects the dissertation against schedule slip.

If Phase 5 finishes faster than expected, Phase 6 can start immediately after. If Phase 5 runs over, Phase 6 is dropped or downgraded to a brief "future work" section in the dissertation.
