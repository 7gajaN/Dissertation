# Phase 4 (Done) and Phase 5 Ideas

> **Phase 4 status (2026-04-07): COMPLETE.** The reintegration plan below was executed. Best result: FCS 0.013 (10× better than real mocap), PFC 0.858. Full narrative in `doc/phase4_evolution.md`.
>
> Items in this document are kept as historical context and as a backlog for Phase 5. Sections marked **[DONE]** were finished in Phase 4. Sections marked **[FUTURE]** remain open.

## ~~PRIORITY 1~~ [DONE] Reintegrate Phase 2 Physics Losses (CoM, Bilateral, Foot Height)

**Status**: completed in Phase 4. CoM and bilateral were reintegrated and found to stack with the FCS predictor. Foot height did not converge at any tested weight and was abandoned. CoM required acceleration masking to avoid suppressing dynamic dance moves — see `doc/phase4_evolution.md`.

The `phase2` branch (`origin/phase2`) added three explicit physics losses directly to `p_losses` in `model/diffusion.py`. They were explored in isolation on that branch but never merged into the main line alongside the FCS predictor. The goal is to combine both approaches — the learned FCS predictor loss from Phase 3 AND these direct physics penalties — to see if they stack.

### The three losses

**1. CoM / Base-of-Support Balance Loss**

Penalizes the center of mass (horizontal XY) being far from the mean position of active foot contacts. Uses SMPL joint mass fractions (Winter 2009) to compute a biomechanically accurate CoM. Only active on frames where at least one foot is in contact.

- Addresses the "weightless slide" artifact: the model must shift weight over the standing foot before lifting the other.
- **Known bug (fixed)**: The original implementation had a circular gradient loop — moving feet changed the support center target, which moved the feet again. Fix: `.detach()` the foot positions used to compute the support center, so the target is fixed each step.
- Caveat: May be too strict for dynamic dance (jumps, spins) where CoM routinely leaves the base of support. Needs a reasonable weight or masking for high-acceleration frames.

**2. Bilateral Foot Exclusivity Loss**

Penalizes both feet moving fast simultaneously. Computed as the product of left-foot velocity and right-foot velocity. A high product means both feet are sliding at the same time with no support — the signature of "airborne sliding."

- Simple and targeted: directly penalizes the most common physics artifact.
- Low risk of side effects since the product is near zero whenever one foot is stationary.

**3. Foot Height During Contact Loss**

Penalizes feet hovering above the ground while the model predicts they are in contact. Forces geometric consistency: if the model says "foot is planted," the foot must actually be at ground level.

- Uses minimum foot height per sequence as the ground reference (same approach as FCS evaluator).
- The ground reference is `.detach()`ed to prevent the model from "solving" this loss by pushing all feet underground.

### Implementation plan

Changes to 4 files — all code already exists on `origin/phase2` (commits `6fd5e84` + `7ec7577`):

| File | Change |
|------|--------|
| `model/diffusion.py` | Add `_SMPL_JOINT_MASSES` constant, 3 new loss weight attributes, 3 loss computations after foot skate loss in `p_losses`, extend losses tuple from 5 to 8 elements |
| `EDGE.py` | Accept 3 new weight params in `__init__`, pass to diffusion, unpack 8-element loss tuple, accumulate/log the 3 new losses in training loop |
| `args.py` | Add `--com_loss_weight`, `--bilateral_loss_weight`, `--foot_height_loss_weight` (all default 0.0) |
| `train.py` | Pass the 3 new args to `EDGE()` constructor |

### Experiments to run

1. **Phase 2 losses only** (no FCS predictor) — isolate their effect
2. **Phase 2 losses + FCS predictor** — do they stack?
3. **Weight sweep** — try different combinations for the 3 weights

### Suggested starting weights (to be tuned)

Based on the phase2 branch exploration and the magnitude of existing losses:
- `--com_loss_weight 1.0`
- `--bilateral_loss_weight 1.0`
- `--foot_height_loss_weight 5.0`

---

## Key Signals from Phase 3 Results

What worked:
- FCS predictor loss gives 3.5× better FCS scores (0.048 avg vs 0.170 baseline)
- The surrogate-network approach is validated — differentiable physics loss via learned proxy

What's leaving performance on the table:
- FCS train loss is ~0.0003 — tiny compared to other losses (~0.006–0.01). Weight of 0.12 is likely too conservative.
- 30% higher reconstruction/FK losses — model pays a tax but we haven't explored the Pareto frontier.
- No curriculum: physics loss applied from epoch 1, before the model learns basic motion.
- Evaluation only uses FCS/PFC — no perceptual or motion quality metrics.
- FCS predictor was trained once and frozen — never updated as the diffusion model improves.

---

## A. Low-Hanging Fruit — Weight & Schedule Tuning

### [DONE] A1. Higher FCS Weight

Try weights of 0.5, 1.0, 2.0, 5.0. The current FCS loss magnitude (~0.0003) is orders of magnitude below other losses even after the 0.12 multiplier. The effective contribution to total loss is ~0.00004, compared to reconstruction at ~0.006. There is likely significant room to push physics harder without destabilizing training.

**Effort**: Low — just change `--fcs_loss_weight` and retrain.
**Risk**: Too high a weight could degrade motion quality. Need motion quality metrics (see D9) to find the sweet spot.

**[Phase 4 result]**: w=1.0 alone improved FCS from 0.047 (w=0.12) to 0.019 — 2.5× better. When combined with CoM + bilateral, w=0.12 gave the best FCS (0.013) while w=1.0 gave the best PFC (0.858). Higher weight does not stack additively with explicit losses.

### A2. Curriculum / Warm-Up

Train without FCS loss for the first N epochs (e.g., 200–500), then enable it. Rationale: in early training, the model hasn't learned basic motion structure yet. Applying physics constraints at this stage forces the model to satisfy two conflicting objectives simultaneously — learn what dance looks like AND obey physics. Letting it learn motion first, then refine for physics, could reduce the reconstruction trade-off.

**Implementation**: Add `--fcs_warmup_epochs` argument. In `p_losses`, set `fcs_loss = 0` when `epoch < warmup`.
**Effort**: Low — a few lines of code.

### A3. Loss Annealing

Linearly or exponentially increase FCS weight over training rather than a hard switch. For example, ramp from 0 to target weight over 500 epochs. Smoother than curriculum, avoids the sudden loss landscape change.

**Implementation**: `effective_weight = min(1.0, epoch / ramp_epochs) * fcs_loss_weight`
**Effort**: Low.

---

## B. Richer Physics Losses

### B4. Ground Penetration Loss

Directly penalize feet below the ground plane in `p_losses`. This doesn't need the FCS predictor — it's a simple differentiable penalty: `ReLU(-foot_z_adjusted)`. The FCS predictor captures this implicitly, but an explicit loss would provide a stronger, more direct gradient signal for this specific violation.

**Implementation**: After FK in `p_losses`, compute foot heights relative to ground, apply `ReLU(-height).mean()`.
**Effort**: Low — ~10 lines in `diffusion.py`.

### B5. Foot Skating Loss v2

The existing foot contact loss (weight 10.942) uses the model's **own contact predictions** to decide when feet should be stationary. This creates a loophole: the model can predict "not in contact" and slide feet freely. A physics-grounded skating loss would use actual height and velocity (like the FCS evaluator) to detect contact independently of the model's predictions, then penalize sliding.

**Implementation**: In `p_losses`, detect contacts from `model_xp` foot heights/velocities (soft sigmoid like the FCS predictor), penalize foot velocity during detected contact.
**Effort**: Medium — need to add contact detection logic to `diffusion.py`.

### [DONE] B6. CoM Balance Loss

Revisit the phase2 branch idea: penalize center-of-mass positions that fall outside the base of support (convex hull of contact feet). This enforces static balance — a standing or slow-moving dancer should have CoM above their feet.

**Caveat**: The phase2 implementation had a circular gradient bug. The fix is straightforward (detach the base-of-support computation from the gradient graph), but the loss itself may be too strict for dynamic dance where the CoM routinely travels outside the base of support during jumps, spins, and weight transfers.

**Implementation**: Compute CoM from `model_xp`, compute support polygon from contact feet, penalize CoM-to-polygon distance. Detach polygon computation.
**Effort**: Medium. **Risk**: May not suit dynamic dance — needs careful testing.

**[Phase 4 result]**: Initial implementation was strictly worse than baseline at every weight tried (12-126× worse FCS). Required acceleration masking — only apply the loss on frames where CoM acceleration < 0.01 (static poses) — to become useful. With masking at w=0.05 it gave +23% over baseline, and combined with bilateral + FCS predictor it produced the best Phase 4 model. The "may not suit dynamic dance" caveat above turned out to be the central design challenge.

---

## C. Inference-Time Physics (No Retraining Needed)

### C7. Physics-Guided Sampling

During DDIM inference, after each denoising step, compute the FCS predictor's gradient with respect to the current sample and nudge it toward lower FCS. This is analogous to classifier guidance in image diffusion models but uses the physics predictor as the "classifier."

```
x_{t-1} = DDIM_step(x_t) - λ * ∇_x FCS_predictor(FK(x_t))
```

**Advantages**:
- No retraining required — works on any existing checkpoint
- Adjustable at inference time via guidance scale λ
- Can be combined with a retrained model for compounding improvements

**Implementation**: Modify `ddim_sample()` to optionally apply predictor gradient after each step.
**Effort**: Medium — need to handle gradient computation during sampling without breaking the denoising.

### C8. Post-Hoc Optimization

Take a fully generated sample and directly optimize it to minimize FCS while staying close to the original via an L2 regularizer:

```
x* = argmin_x  FCS_predictor(FK(x)) + α * ||x - x_generated||²
```

Run gradient descent for N steps on the motion representation. This is a test-time refinement that can improve any generated sample.

**Advantages**: Dead simple, works on any checkpoint, tunable trade-off.
**Disadvantages**: Slow (optimization per sample), may introduce artifacts if pushed too hard.
**Effort**: Low — standalone script, ~50 lines.

---

## D. Evaluation & Analysis (For the Dissertation)

### D9. Motion Quality Metrics

**Priority: MUST-DO.** Reviewers will ask whether physics improvement comes at the cost of motion quality. Need to measure:

- **FID (Fréchet Inception Distance)** on motion features — standard metric for generative motion quality. Compares distribution of generated motions to real motions in a learned feature space.
- **Diversity** — variance across generated samples for the same music. Physics constraints might reduce diversity.
- **Beat alignment score** — does the generated dance still match the music beats? Physics loss shouldn't affect this, but need to verify.
- **Foot skating ratio** — percentage of frames with foot sliding during detected contact. A direct, interpretable physics metric.

**Implementation**: Use the motion evaluation framework from the original EDGE paper or from MDM/MoFusion codebases. Many of these metrics are standard in the dance generation literature.
**Effort**: Medium — need to set up feature extractors and compute statistics.

### D10. Ablation Study

Systematic comparison to isolate the contribution of each design decision:

| Experiment | Description |
|------------|-------------|
| Baseline | No FCS loss |
| + FCS predictor v1 | Generic convolutions (expected: minimal improvement) |
| + FCS predictor v2 | Physics-informed (current best) |
| + Higher weight | Weight = 1.0 instead of 0.12 |
| + Curriculum | Warm-up for 500 epochs, then FCS |
| + Ground penetration | Explicit floor penalty |
| + Guided sampling | Inference-time physics guidance |

Each row needs FCS, PFC, and motion quality metrics (D9). This table is the core contribution evidence for the dissertation.

**Effort**: High — multiple training runs (each ~26 hours based on Phase 3 timing).

### [DONE] D11. Qualitative Comparison

Render side-by-side videos of baseline vs physics-trained model on the same music clips. Select examples that highlight:
- Foot skating reduction
- Physically impossible accelerations removed
- Ground penetration fixed
- Cases where physics loss hurts (if any)

**Implementation**: Use existing `test.py` + render pipeline. Pick 5–10 representative test clips.
**Effort**: Low — infrastructure already exists.

**[Phase 4 result]**: Three models rendered (baseline, best FCS, best balance) on the AIST++ test wavs. Side-by-side comparison videos in `renders/comparison/` made via `scripts/make_comparison_videos.sh` (ffmpeg hstack with model labels). Qualitative finding: physics-trained models are visibly more stable (planted feet, no gross balance violations) but at the cost of dampening some larger expressive moves — see Key Finding #7 in `phase4_evolution.md`.

### [DONE] D12. Real Data Baseline

Compute FCS on the AIST++ test set (real mocap) to establish the "ceiling" — what real motion scores look like. This contextualizes the generated results:

- If real mocap FCS ≈ 0.05 and physics model FCS ≈ 0.048, we're at human level.
- If real mocap FCS ≈ 0.01 and physics model FCS ≈ 0.048, there's still a gap.

**Implementation**: `eval_real_data_fcs.py` already exists. Just run it and record results.
**Effort**: Very low — run existing script.

**[Phase 4 result]**: Real mocap (100 AIST++ test sequences) scores FCS 0.132 mean / 0.050 median, PFC 2.180 mean / 1.788 median. Best Phase 4 model is **10× better than real mocap on FCS** and 2.5× better on PFC. Surprising finding: even the untrained baseline beats real mocap on PFC, indicating PFC is unreliable as an absolute target on this dataset (it heavily penalizes natural mocap micro-jitter). Script was extended to compute PFC alongside FCS in the same pass.

---

## E. Architecture Improvements (Bigger Scope)

### E13. Contact Prediction Head

Add an auxiliary output head to the DanceDecoder transformer that explicitly predicts binary foot contact labels, supervised by ground-truth contacts from AIST++. Currently the model predicts contacts as part of the 151D output but they're only supervised by the reconstruction loss (weak signal). A dedicated head with binary cross-entropy loss would give sharper contact predictions.

Better contact predictions → better foot contact loss → less skating.

**Implementation**: Add a small MLP head after the transformer, branching from the same features. Add BCE loss.
**Effort**: Medium — architecture change + training.

### E14. Iterative Predictor Refinement

The FCS predictor was trained on real mocap + augmented data. As the diffusion model improves, its output distribution shifts — the predictor may become less accurate on the new distribution. Solution: periodically retrain the predictor on samples from the current diffusion model.

Pipeline:
1. Train diffusion model for N epochs with current predictor
2. Generate samples, compute real FCS
3. Retrain predictor on these samples + real data
4. Resume diffusion training with updated predictor
5. Repeat

**Effort**: High — complex pipeline, multiple training phases.
**Risk**: Predictor could overfit to current model's failure modes.

---

## Priority Ranking

### Phase 4 — completed
- ~~**Phase 2 losses reintegration**~~ — done. CoM (with masking) and bilateral worked; foot height did not.
- ~~**A1**~~ — Higher FCS weight (w=1.0) tested, improved solo result but did not stack with explicit losses.
- ~~**D12**~~ — Real data FCS + PFC baseline computed.
- ~~**D11**~~ — Side-by-side renders produced.

### Phase 5 — open backlog (priority order)

**Most likely to yield gains:**
- **Foot height with masking** — the only Phase 4 loss that failed. An acceleration- or contact-onset-masked variant analogous to what worked for CoM may rescue it.
- **A2 / A3** — Curriculum / warm-up / annealing for FCS weight. Not tried.
- **CoM acceleration threshold tuning** — currently 0.01, picked once and never tuned.

**Worth trying for completeness:**
- **Missing combo**: FCS w=1.0 + CoM (no bilateral), to fill the ablation grid.
- **C8** — Post-hoc test-time optimization.
- **C7** — Physics-guided sampling at inference (novel contribution).

**Bigger scope:**
- **B5** — Foot skating loss v2 (model-independent contact detection)
- **E13** — Contact prediction head
- **E14** — Iterative predictor refinement (retrain predictor on current model output)

**Skipped — out of scope:**
- **D9** — Motion quality metrics (FID, diversity, beat alignment). The EDGE paper itself argues FID is unreliable for AIST++. User decided not to pursue these.
