# Phase 4: Combining Explicit and Learned Physics Losses

This document traces the full development of Phase 4, from the initial plan through the final results. It records every experiment run, every bug discovered, every design decision, and the final numbers that will go into the dissertation. It is intended as a complete reference for the write-up.

## Table of Contents

1. [Motivation](#motivation)
2. [Starting Point](#starting-point)
3. [Plan](#plan)
4. [Stage 1: Code Reintegration](#stage-1-code-reintegration)
5. [Stage 2: Code Review Fixes](#stage-2-code-review-fixes)
6. [Stage 3: Individual Loss Validation — First Pass](#stage-3-individual-loss-validation--first-pass)
7. [Stage 4: Weight Tuning — Second Pass](#stage-4-weight-tuning--second-pass)
8. [Stage 5: The CoM Loss Failure and Acceleration Masking Fix](#stage-5-the-com-loss-failure-and-acceleration-masking-fix)
9. [Stage 6: Combined Runs](#stage-6-combined-runs)
10. [Stage 7: Real Data Baseline (D12)](#stage-7-real-data-baseline-d12)
11. [Stage 8: Qualitative Renders](#stage-8-qualitative-renders)
12. [Final Results Table](#final-results-table)
13. [Key Findings](#key-findings)
14. [Lessons Learned](#lessons-learned)

---

## Motivation

Phase 3 established the FCS predictor — a learned surrogate network that approximates the Force Consistency Score metric differentiably, allowing physics to be optimized via backpropagation without needing to differentiate through the (non-differentiable) ground-truth FCS computation. The Phase 3 result was a 3.5× improvement in FCS scores over the unconstrained baseline (FCS 0.047 vs 0.156 at epoch 2000).

Separately, the abandoned `origin/phase2` branch had developed three explicit, hand-crafted physics losses — Center of Mass balance, bilateral foot exclusivity, and foot height during contact — that were never combined with the FCS predictor. Phase 4's central question: **does combining the learned FCS predictor loss with explicit physics losses yield improvements beyond either approach alone?**

## Starting Point

At the start of Phase 4 (2026-04-01) we had:

- **Baseline model** (`runs/baseline/no_fcs/`) — standard EDGE, no physics, 2000 epochs. FCS@2000 = 0.156, PFC@2000 = 1.204.
- **FCS predictor model** (`runs/fcs/physics_w12/`) — FCS predictor loss at weight 0.12, 2000 epochs. FCS@2000 = 0.047, PFC@2000 = 0.907.
- **FCS predictor network** (`models/fcs_predictor.pt`) — trained on real and augmented data, correlation 0.986 with ground-truth FCS.
- **Three explicit physics losses** fully implemented on `origin/phase2` (commits `6fd5e84` + `7ec7577`) but never merged with the FCS predictor. The phase2 branch had in fact *removed* the FCS predictor integration.

Constraints: 1 GPU (sequential experiments, ~26h per 2000-epoch run), ~1-2 month timeline, budget of ~4-6 full training runs.

## Plan

The Phase 4 plan was:

1. **Code reintegration**: port the 3 phase2 losses into the current codebase alongside the FCS predictor, extending the losses tuple from 5 → 8 elements. Files: `model/diffusion.py`, `EDGE.py`, `args.py`, `train.py`.
2. **Individual loss validation**: train 500-epoch models with each loss enabled in isolation, verify gradient flow and training stability.
3. **Weight tuning**: find working weights for any loss that didn't converge cleanly at the initial suggested values.
4. **Combined runs**: 2000-epoch training with the working losses combined, including a higher FCS weight variant.
5. **Real data baseline**: compute FCS on real AIST++ test mocap to establish the "ceiling".
6. **Qualitative comparison**: render side-by-side videos of baseline vs best physics models.

## Stage 1: Code Reintegration

**Goal**: add 3 new physics losses alongside the existing FCS predictor loss.

The phase2 branch had removed the FCS predictor and replaced its slot in the losses tuple with the three new losses. The current main line had the FCS predictor in slot 4. The port needed to **keep** the FCS predictor in its existing slot and **append** the three new losses as slots 5, 6, 7 — growing the tuple from 5 to 8 elements.

Critically, the current architecture computes the FCS predictor loss inside `p_losses()` on `model_xp` — the FK joint positions of the model's *predicted* clean motion, which have a direct gradient path to the model weights. The phase2 branch had tried computing the FCS loss in the `EDGE.py` training loop on the unnormalized ground-truth input (`x`), which has no gradient to the model. Phase 2's approach was preserved in a separate wrapper and would have produced zero gradient on the FCS term. We kept the current `p_losses` approach.

### Files Modified

**`model/diffusion.py`**:
- Added `_SMPL_JOINT_MASSES` constant (24 values, Winter 2009 anthropometric model).
- Added 3 weight parameters to `GaussianDiffusion.__init__`: `com_loss_weight`, `bilateral_loss_weight`, `foot_height_loss_weight`, all defaulting to 0.0.
- Registered `_joint_masses` as a buffer so it moves to GPU with the model.
- In `p_losses()`, after the existing foot skate loss block (~line 516), added three new loss computations, each gated by a weight check so they are no-ops when disabled.
- Extended losses tuple from 5 to 8 elements: `(recon, velocity, fk, foot, fcs, com, bilateral, height)`.

**`EDGE.py`**:
- Added 3 weight parameters to `EDGE.__init__`, passed through to `GaussianDiffusion`.
- Unpacked 8-element loss tuple in the training loop.
- Added accumulators `avg_comloss`, `avg_bilateralloss`, `avg_heightloss`.
- Updated progress printing (every 50 epochs), checkpoint summary printing, CSV header + rows, JSON metrics, and wandb log_dict to track all 8 losses.
- Each new loss only printed when its weight > 0 (keeps logs tidy for FCS-only runs).

**`args.py`**:
- Added `--com_loss_weight`, `--bilateral_loss_weight`, `--foot_height_loss_weight`, all defaulting to 0.0 (backward compatible).

**`train.py`**:
- Passed the 3 new args to the `EDGE()` constructor.

The three losses as implemented:

```python
# CoM / Base-of-Support balance loss
if self.com_loss_weight > 0:
    masses = self._joint_masses.view(1, 1, 24, 1)
    com = (model_xp * masses).sum(dim=2)            # (B,S,3)
    com_h = com[:, :, :2]                            # XY plane
    foot_h = model_xp[:, :, foot_idx, :2].detach()   # detach to break gradient loop
    contact_w = (model_contact > 0.95).float()
    denom = contact_w.sum(dim=-1, keepdim=True).clamp(min=1.)
    support_center = (foot_h * contact_w.unsqueeze(-1)).sum(dim=2) / denom
    has_contact = (contact_w.sum(dim=-1) > 0).float()
    com_dist = torch.norm(com_h - support_center, dim=-1)
    com_loss = (com_dist * has_contact).mean()

# Bilateral foot exclusivity loss
if self.bilateral_loss_weight > 0:
    left_v = torch.norm(feet[:,1:,[0,2],:] - feet[:,:-1,[0,2],:], dim=-1).mean(dim=-1)
    right_v = torch.norm(feet[:,1:,[1,3],:] - feet[:,:-1,[1,3],:], dim=-1).mean(dim=-1)
    bilateral_loss = (left_v * right_v).mean()

# Foot height during contact loss
if self.foot_height_loss_weight > 0:
    min_h = model_feet[:,:,:,2].min(dim=1, keepdim=True)[0].detach()  # per-seq ground
    adj_h = model_feet[:,:,:,2] - min_h
    contact_w2 = (model_contact > 0.95).float()
    height_loss = (adj_h * contact_w2).mean()
```

Note: the CoM loss uses `.detach()` on the foot positions for the support center target. This is the same gradient-loop fix that commit `7ec7577` on the phase2 branch applied — without it, moving the feet changes the target which moves the feet again, creating a circular gradient loop.

## Stage 2: Code Review Fixes

Before training, the implementation was reviewed and three bugs were found and fixed.

### Bug 1: `torch.norm` gradient singularity in CoM loss

The initial implementation used `torch.norm(com_h - support_center, dim=-1)` to compute the distance. The gradient of `||x||` is `x/||x||`, which is undefined at zero and numerically huge near zero. This is specifically dangerous for a loss that is trying to drive the distance to zero — the loss becomes ill-conditioned exactly when it succeeds. Near-zero inputs (1e-7 to 1e-5) produce gradient magnitudes of ~1e5 to 1e7, which can cause spikes and NaN propagation.

**Fix**: switched to squared distance, removing the `sqrt` entirely:

```python
com_dist_sq = ((com_h - support_center) ** 2).sum(dim=-1)
com_loss = (com_dist_sq * has_contact).mean()
```

Squared L2 is equally valid as a loss and its gradient is just `2 * (com_h - support_center)` — well-behaved everywhere.

### Bug 2: Bilateral loss suppresses legitimate airborne dance

The bilateral loss as written penalized the product of left-foot and right-foot velocity unconditionally. This did not distinguish between:
- Both feet sliding simultaneously while at least one should be planted (the intended target)
- Both feet moving during a jump, hop, or aerial move (legitimate dance)

Dance has extensive jumping and hopping, and AIST++ includes genres (krump, breaking, locking) that are jump-heavy. The loss would actively suppress those dynamics.

**Fix**: mask by contact state — only penalize when at least one foot is predicted to be in contact:

```python
any_contact = (model_contact[:, 1:] > 0.95).any(dim=-1).float()  # (B, S-1)
bilateral_loss = (left_v * right_v * any_contact).mean()
```

This preserves the original intent (penalize simultaneous sliding when there's supposed to be a planted foot) while allowing legitimate airborne phases.

### Bug 3: SMPL mass fractions inconsistent with evaluation code

Three different CoM calculations existed in the codebase:

| File | Trunk joints | Collar joints (13,14) | Method |
|------|--------------|----------------------|--------|
| `diffusion.py` (training loss, as written) | 7 joints: 0,3,6,9,12,13,14 | 0.071 each | Per-joint weight |
| `model/fcs_predictor.py` (learned proxy) | 5 joints: 0,3,6,9,12 | 0.000 | Segment-mean |
| `eval/eval_fcs.py` (evaluation) | 5 joints: 0,3,6,9,12 | 0.000 | Segment-mean |

The training loss was optimizing toward a CoM that differed from what evaluation measured — `diffusion.py` allocated 14.2% of body mass to collarbones while the other two files allocated 0%. Hip joints also differed (the segment mapping splits thigh mass between hip and knee, but `diffusion.py` assigned the full thigh mass to the hip).

**Fix**: recomputed the per-joint weights from `SEGMENT_JOINT_MAPPING` so all three files use identical values. Key differences vs the original:

| Joint | Old | Canonical |
|-------|-----|-----------|
| Pelvis (0) | 0.071 | 0.0994 |
| L/R Hip (1,2) | 0.100 | 0.050 |
| L/R Knee (4,5) | 0.0465 | 0.07325 |
| L/R Ankle (7,8) | 0.00725 | 0.0305 |
| L/R Collar (13,14) | 0.071 | 0.000 |
| L/R Elbow (18,19) | 0.016 | 0.022 |

After the fix, `diffusion.py`, `fcs_predictor.py`, and `eval_fcs.py` all compute CoM identically — the training loss and evaluation metric agree.

### Minor: display total consistency

The per-epoch loss percentage display in `EDGE.py` included the new losses but excluded FCS, so percentages didn't sum to 100%. Fixed by including FCS in the total and percentage display. Also cleaned up a duplicate `temp_avg_fcs` computation.

## Stage 3: Individual Loss Validation — First Pass

Per the gradual validation plan, we trained each loss in isolation at the weights suggested in `doc/ideas.md`. Each run was 500 epochs on 1 GPU (~6 hours each).

| Run | Command flags | Purpose |
|-----|---------------|---------|
| `com_test3` | `--com_loss_weight 1.0` | CoM alone at phase2 suggested weight |
| `bilateral_test2` | `--bilateral_loss_weight 1.0` | Bilateral alone at phase2 suggested weight |
| `height_test` | `--foot_height_loss_weight 5.0` | Foot height alone at phase2 suggested weight |

### First-pass training behavior (from loss curves)

**CoM w=1.0**: total loss decreased steadily (0.035 → 0.029), CoM term stayed around 0.004 (~13% of total). Looked stable in training logs.

**Bilateral w=1.0**: total loss decreased cleanly (0.027 → 0.017). Bilateral term was tiny (0.000174 → 0.000197), ~1% of total. Loss magnitude clearly too small to steer training meaningfully at this weight.

**Foot height w=5.0**: total loss **did not converge** — oscillated between 1.4 and 2.3. FK loss was ~100× higher than the other runs (1.0 vs 0.01). Height loss was 37% of total and not decreasing. Clearly fighting the FK loss.

### First-pass evaluation (50 samples per checkpoint)

| Model | FCS@500 | PFC@500 | vs Baseline |
|-------|---------|---------|-------------|
| Baseline (no physics) | 0.134 | 1.183 | — |
| FCS predictor w=0.12 | 0.024 | 0.464 | 5.7× better FCS |
| **CoM w=1.0** | **16.92** | **123.27** | **126× worse** |
| **Bilateral w=1.0** | **0.131** | **0.786** | roughly neutral |
| **Foot height w=5.0** | **2.558** | **1.299** | 19× worse |

All three losses were either severely harmful or neutral at these weights. The training loss curves looked fine for CoM but the generated motions were catastrophically bad — a reminder that **training loss and output quality are not the same thing**.

## Stage 4: Weight Tuning — Second Pass

Based on the first-pass results, three new runs at more conservative weights:

| Run | Command flags | Rationale |
|-----|---------------|-----------|
| `com_w005` | `--com_loss_weight 0.05` | CoM was 100× too strong at 1.0 |
| `height_w05` | `--foot_height_loss_weight 0.5` | Height was destabilizing at 5.0 |
| `bilateral_w5` | `--bilateral_loss_weight 5.0` | Bilateral was too weak at 1.0 |

### Second-pass evaluation

| Model | FCS@500 | PFC@500 | vs Baseline |
|-------|---------|---------|-------------|
| Baseline | 0.134 | 1.183 | — |
| **CoM w=0.05** | **1.615** | **3.841** | 12× worse |
| **Bilateral w=5.0** | **0.159** | **0.679** | neutral FCS, slight PFC gain |
| **Foot height w=0.5** | **5.288** | **1.333** | 39× worse |

**Interpretations:**

1. **CoM** was no longer catastrophic but still 12× worse than baseline. The weight wasn't the only problem — the loss itself was structurally incompatible with dance.
2. **Bilateral w=5.0** was neutral on FCS and slightly helped PFC. Still a weak signal but not harmful. Safe to include in combined runs.
3. **Foot height** was actually *worse* at 0.5 than at 5.0 on FCS. Reducing the weight did not help. The loss seemed fundamentally broken for this task — no weight we tried worked. Abandoned.

## Stage 5: The CoM Loss Failure and Acceleration Masking Fix

The CoM loss was hurting at both tested weights, and the pattern hinted at a structural problem rather than a weight-tuning problem.

**Diagnosis**: the loss penalized the CoM being far from the support base **on every frame equally**. But in dance, the CoM is *supposed* to be far from the support base during dynamic moves — jumps, spins, lunges, weight transfers. AIST++ is dominated by these genres. The loss was pulling the model toward over-constrained motion, suppressing expressive choreography.

This was flagged as a caveat in `doc/ideas.md` from the beginning: *"May be too strict for dynamic dance (jumps, spins) where CoM routinely leaves the base of support. Needs a reasonable weight or masking for high-acceleration frames."*

### Acceleration masking

Computed the CoM acceleration at each frame (second-order finite difference) and only applied the loss on frames where the body was in a relatively static pose:

```python
com_acc = torch.norm(
    com[:, 2:, :] - 2 * com[:, 1:-1, :] + com[:, :-2, :], dim=-1
)
com_acc = F.pad(com_acc, (1, 1), value=0.0)   # pad to (B, S)
is_static = (com_acc < 0.01).float()
com_loss = (com_dist_sq * has_contact * is_static).mean()
```

During static poses, the model is pulled toward balance over the feet. During jumps and explosive moves, the mask is 0 and the loss has no effect. This is also physically correct: **static balance only matters when you're actually trying to maintain a pose**, not during ballistic motion.

The threshold of 0.01 was a first guess and was not further tuned.

### Fixing an off-by-one error

The initial implementation used `com[:, :-1, :]` in the finite difference, which produces a length-(S-1) tensor while `com[:, 2:, :]` and `com[:, 1:-1, :]` produce length-(S-2). Training crashed immediately with a shape mismatch. Fix: use `com[:, :-2, :]` — now all three slices produce (S-2) and the finite difference is correct.

### Masked CoM result (`com_masked2`)

| Model | FCS@500 | PFC@500 | vs Baseline |
|-------|---------|---------|-------------|
| Baseline | 0.134 | 1.183 | — |
| CoM w=0.05 unmasked | 1.615 | 3.841 | 12× worse |
| **CoM w=0.05 masked** | **0.103** | **1.022** | **23% better FCS, 14% better PFC** |

The masking turned the CoM loss from a 12× regression into a 23% improvement over baseline. Same weight, same computation, just masked on high-acceleration frames. This is the key design insight of Phase 4 — **physics losses derived from quasi-static assumptions must be gated by motion dynamics before they can be applied to dance**.

The foot height loss might benefit from similar treatment (e.g. masking during contact onset/offset frames) but we did not have the budget to explore that — it was set aside as future work.

## Stage 6: Combined Runs

With two working explicit losses (masked CoM at 0.05, bilateral at 5.0), the central question of Phase 4 could be tested: does combining them with the FCS predictor stack?

Four 2000-epoch combined runs were executed sequentially:

| Run | FCS predictor | CoM (masked) | Bilateral | Purpose |
|-----|---------------|--------------|-----------|---------|
| `fcs_com_combined` | 0.12 | 0.05 | 0.0 | FCS + CoM only |
| `fcs_w1` | 1.0 | 0.0 | 0.0 | Higher FCS weight alone |
| `fcs_com_bilateral` | 0.12 | 0.05 | 5.0 | Full combo at old FCS weight |
| `fcs1_com_bilateral` | 1.0 | 0.05 | 5.0 | Full combo at new FCS weight |

Each run was 2000 epochs, ~26h. Evaluated with 50 samples per checkpoint.

### Combined run evaluation

| Model | FCS@2000 | PFC@2000 | Best FCS (epoch) |
|-------|----------|----------|------------------|
| Baseline (no physics) | 0.156 | 1.204 | 0.068 (ep 1000) |
| FCS predictor w=0.12 (Phase 3 result) | 0.047 | 0.907 | 0.0045 (ep 100) |
| FCS w=0.12 + CoM | 0.060 | 1.179 | 0.0050 (ep 400) |
| FCS w=1.0 | 0.019 | 1.152 | 0.0018 (ep 200) |
| **FCS w=0.12 + CoM + Bilateral** | **0.013** | 1.040 | **0.0000 (ep 100)** |
| **FCS w=1.0 + CoM + Bilateral** | 0.018 | **0.858** | 0.0100 (ep 800) |

### Observations

1. **The combined approach stacks.** `fcs_com_bilateral` achieved FCS 0.013 at epoch 2000 — 3.6× better than the Phase 3 FCS predictor alone (0.047) and 12× better than the baseline (0.156). This answers Phase 4's central question affirmatively.
2. **Higher FCS weight works when solo but not when combined.** `fcs_w1` (FCS alone at 1.0) beat the old `physics_w12` (FCS alone at 0.12) 0.019 vs 0.047 — a 2.5× improvement from just raising the weight. However, when combined with the explicit losses, pushing FCS from 0.12 to 1.0 actually slightly *worsened* FCS (0.018 vs 0.013) while improving PFC (0.858 vs 1.040).
3. **`fcs_com` alone does not stack.** FCS + CoM without bilateral was slightly worse than FCS alone (0.060 vs 0.047). The bilateral loss appears to be the key complement — CoM only helps when bilateral is also present.
4. **Two "best" models, different strengths**. There is no single winner:
   - `fcs_com_bilateral` is the best on FCS (0.013) and has a strong PFC (1.040).
   - `fcs1_com_bilateral` has the best PFC (0.858) and still beats FCS predictor alone on FCS (0.018 vs 0.047).

## Stage 7: Real Data Baseline (D12)

Up to this point, results were interpreted relative to each other and to the baseline. We needed a real-world reference — what FCS and PFC does **actual human mocap** score? This is essential for the dissertation narrative: is a score of 0.013 "close to human" or "still far from human"?

The existing `eval_real_data_fcs.py` script only computed FCS. We extended it to also compute PFC:

```python
from eval.eval_fcs import ForceConsistencyEvaluator, calculate_pfc_score
# ... in the loop:
pfc = calculate_pfc_score(joint_positions_np)
if not (np.isnan(pfc) or np.isinf(pfc)):
    pfc_scores.append(pfc)
```

Ran on 100 sequences from the AIST++ test set (real mocap).

### Real mocap scores

| Metric | Mean | Median | Std |
|--------|------|--------|-----|
| FCS | **0.132** | 0.050 | 0.210 |
| PFC | **2.180** | 1.788 | 2.107 |

### The surprising finding

All of our models — **including the baseline** — score better than real mocap on PFC. PFC is clearly a noisy metric that heavily penalizes the natural micro-jitter present in real motion capture data. It cannot be used as an absolute target.

On FCS, the real data mean is 0.132 — comparable to our baseline (0.156) and **worse** than any of our physics-trained models. Our best model (`fcs_com_bilateral`) scores 10× better than real mocap on FCS.

This is either a genuine result (the model has learned physics that exceed the noise floor of real mocap) or the metric has an upper bound the model has saturated. The qualitative renders were the next step to distinguish these.

## Stage 8: Qualitative Renders

Three models rendered on the same music clips for side-by-side comparison:
- `baseline` (no physics)
- `fcs_com_bilateral` (best FCS)
- `fcs1_com_bilateral` (best balance)

Using the existing `test.py` + render pipeline. A few issues were worked through:

1. **Audio length too short**: default `--out_length 30` requires 11 slices per song, but the test wavs are 2-3 slices long. Fixed by reducing to `--out_length 5` (1 slice).
2. **Jukebox checkpoint download failure**: the ~10GB `prior_level_2.pth.tar` download was cut off at ~6GB due to network. Resolved by resuming with `wget -c` against the partial file.
3. **Renders overwriting each other**: the default `--render_dir` is `renders/`, so each model's videos overwrote the previous. Fixed by using `--render_dir renders/<model_name>` for each run. The motion `.pkl` files were saved to separate `--motion_save_dir` directories so nothing was lost.

A helper script `scripts/make_comparison_videos.sh` was written to combine the three renders side-by-side using `ffmpeg` with `hstack`, adding a 50-pixel black bar above each video with the model name centered. Output goes to `renders/comparison/`.

## Final Results Table

Evaluated with 50 samples per checkpoint. All physics-trained models are at epoch 2000.

| Model | FCS | PFC | vs Real FCS | vs Real PFC |
|-------|-----|-----|-------------|-------------|
| Real mocap (ceiling, 100 sequences) | 0.132 | 2.180 | 1.00× | 1.00× |
| Baseline (no physics, Phase 0) | 0.156 | 1.204 | 1.18× worse | 1.81× better |
| FCS predictor w=0.12 (Phase 3 best) | 0.047 | 0.907 | 2.81× better | 2.40× better |
| FCS w=0.12 + CoM (Phase 4) | 0.060 | 1.179 | 2.20× better | 1.85× better |
| FCS w=1.0 (Phase 4) | 0.019 | 1.152 | 6.95× better | 1.89× better |
| **FCS w=0.12 + CoM + Bilateral (Phase 4 best FCS)** | **0.013** | 1.040 | **10.2× better** | 2.10× better |
| **FCS w=1.0 + CoM + Bilateral (Phase 4 best balance)** | 0.018 | **0.858** | 7.33× better | **2.54× better** |

## Key Findings

1. **Phase 4 goal was met.** Combining the learned FCS predictor loss with explicit physics losses stacks — the best combined model achieves FCS 0.013, 3.6× better than the Phase 3 FCS predictor alone, and 12× better than the untrained baseline.
2. **Both best models surpass real mocap on both metrics.** FCS 0.013 vs real 0.132 (10.2× better), PFC 0.858 vs real 2.18 (2.54× better). This is a striking finding — either the model has learned physics beyond the noise floor of real mocap, or both metrics saturate at some level the model has reached. Qualitative inspection supports the former, though some visual over-smoothing is present in the physics-trained outputs.
3. **Higher FCS weight helps solo but not combined.** FCS w=1.0 alone beats FCS w=0.12 alone (0.019 vs 0.047), but when combined with CoM + bilateral, the lower FCS weight (0.12) gives the best FCS result while the higher weight gives the best PFC. There is a trade-off between the metrics.
4. **CoM loss required acceleration masking to be usable at all.** At 500 epochs without masking, any weight ≥0.05 was strictly worse than baseline. With masking at 0.05, it became +23% better FCS. The lesson is broader: **physics losses built on quasi-static assumptions (balance, planted-foot) cannot be applied unconditionally to dance — they must be gated by motion dynamics**.
5. **Foot height loss could not be made to work.** Both tested weights (0.5 and 5.0) produced models much worse than baseline, and the training dynamics showed the loss fighting FK reconstruction. An acceleration-masked or phase-specific variant may work but was not pursued in Phase 4. Honest negative result.
6. **PFC is unreliable as an absolute metric on AIST++.** Real mocap scores 2.18 while the untrained baseline scores 1.20 — the metric penalizes natural human motion more than it penalizes an unconstrained diffusion model. FCS is the trustworthy metric for this dataset.
7. **Stability vs expressiveness trade-off (qualitative).** Side-by-side comparison of the rendered videos shows the physics-trained models produce visibly more stable motion than the baseline — feet stay planted, weight shifts look grounded, gross balance violations are gone. The trade-off is that some of the more **voluptuous, expressive moves are dampened** — the larger swings, exaggerated reaches, and high-energy explosive motions that can be a deliberate part of the choreography are sometimes flattened into something more conservative. The acceleration-masked CoM loss helps preserve genuinely ballistic moves (jumps, spins) but does not fully recover the expressive amplitude of the baseline. This is the cost of physics regularization on this dataset and should be discussed openly in the dissertation as a known limitation rather than hidden — the metric improvements are real, but they are not free.

## Lessons Learned

1. **Training loss ≠ output quality.** The first CoM and bilateral runs had clean, decreasing loss curves but catastrophic eval scores (CoM) or neutral (bilateral). Always evaluate actual generated samples, not just training loss.
2. **Validate each loss in isolation before combining.** The initial plan was to jump straight to combined runs. The user insisted on solo 500-epoch validation runs first. This surfaced the CoM masking requirement and the foot height failure *before* committing to 26h combined runs — would have been wasted GPU otherwise.
3. **Gradient pathologies hide in simple losses.** `torch.norm` has an undefined gradient at zero. `sqrt(x)` gradients blow up near zero. Squared losses avoid both and are equally valid. Always check what the gradient of your loss looks like at the optimum.
4. **Consistency between training loss and evaluation matters.** The original CoM implementation used a different SMPL mass mapping than the evaluator, so the model was optimizing toward a different CoM than what was measured. Both files should import the mapping from a single source of truth — this is a codebase quality improvement for a future phase.
5. **Dance is hard for naive physics constraints.** Static balance, planted-foot assumptions, and foot-contact priors all assume quasi-static motion. Dance routinely violates all three. Every explicit physics loss needs an escape hatch for dynamic frames.
6. **The surrogate network approach remains the most reliable.** The FCS predictor loss "just works" in the sense that it steers the model toward lower FCS with no hand-tuned masking or weight juggling. The explicit losses only helped on top of it. For future physics metrics, investing in a learned surrogate is probably a better use of time than hand-crafting differentiable losses.
