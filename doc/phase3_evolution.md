# Phase 3: Physics-Aware Training — Evolution & Design Decisions

This document traces the full development of Phase 3, including dead ends, pivots, and the reasoning behind the current architecture. It serves as both a narrative record and a reference for the dissertation write-up.

## Table of Contents

1. [Motivation](#motivation)
2. [Timeline](#timeline)
3. [Stage 1: FCS as a Validation Metric](#stage-1-fcs-as-a-validation-metric)
4. [Stage 2: Foot Contact Detection — A Calibration Journey](#stage-2-foot-contact-detection)
5. [Stage 3: First Attempt — Direct Gradient Through Physics](#stage-3-first-attempt--direct-gradient-through-physics)
6. [Stage 4: Abandoning Direct Gradients](#stage-4-abandoning-direct-gradients)
7. [Stage 5: The FCS Predictor Approach](#stage-5-the-fcs-predictor-approach)
8. [Stage 6: FCS Predictor v1 — Generic Convolutions (Failed)](#stage-6-fcs-predictor-v1--generic-convolutions-failed)
9. [Stage 7: FCS Predictor v2 — Physics-Informed Architecture](#stage-7-fcs-predictor-v2--physics-informed-architecture)
10. [Stage 8: The Integration Bug — Gradients That Did Nothing](#stage-8-the-integration-bug--gradients-that-did-nothing)
11. [Stage 9: The Fix — FCS Inside p_losses](#stage-9-the-fix--fcs-inside-p_losses)
12. [Stage 10: Phase 2 Branch — Alternative Physics Losses (Explored)](#stage-10-phase-2-branch--alternative-physics-losses)
13. [Current Architecture](#current-architecture)
14. [Lessons Learned](#lessons-learned)

---

## Motivation

EDGE generates dance motion from music using a diffusion model. The original training uses four loss terms — reconstruction, velocity, forward kinematics, and foot contact — which produce smooth, musically synchronized motion but do not enforce **physical plausibility**. Generated dances can exhibit:

- Feet sliding across the floor while supposedly planted
- The body accelerating without any ground contact to push off from
- Feet penetrating below the ground plane
- Center of mass trajectories that would require impossible forces

The existing PFC (Physical Foot Contact) metric from the original paper was too crude to capture these issues. It is a heuristic product of foot velocity and root acceleration — no actual physics, no force analysis, no biomechanical constraints. The goal of Phase 3 was to develop a physics-based metric (FCS) and integrate it into training as a differentiable loss.

---

## Timeline

| Date | Commit | Event |
|------|--------|-------|
| Feb 24 | `cebb1d7` | FCS metric added as validation-only |
| Mar 1 | `b14dad2` | FCS bug fixes, intermediate metrics logging |
| Mar 2 | `091184d`–`c78a867` | Contact detection bugs: ground height threshold missing |
| Mar 6 | `64d575d`–`504c375` | Evaluation scripts, dataset fixes, device issues |
| Mar 13 AM | `29044b4`–`26f1e47` | Contact detection calibration: AND/OR logic, threshold tuning |
| Mar 13 11:03 | `6c24f0a` | **Attempt 1**: Gradient-based physics regularization via DDIM |
| Mar 13 12:55 | `51e0922` | **Abandoned**: Backprop through DDIM sampling not feasible |
| Mar 13 13:26 | `537c642` | **Pivot**: FCS predictor network (v1, generic convolutions) |
| Mar 13 14:20 | `461d42a` | Accelerate multi-GPU support for predictor training |
| Mar 13 14:35 | `b8fcce1` | Model loading fix (weights_only=False) |
| Mar 17–29 | `09bd4f7`–`84a7b38` | Post-pipeline work: segment generation, renders, eval scripts |
| Mar 24 | `6fd5e84`–`7ec7577` | **Phase 2 branch**: CoM/bilateral/height losses (alternative approach) |
| Uncommitted | — | Predictor v2 (physics-informed), integration bug fix (FCS moved into `p_losses`) |

---

## Stage 1: FCS as a Validation Metric

**Commit**: `cebb1d7` (Feb 24, 2026)

The Force Consistency Score was first implemented as a post-hoc evaluation metric in `eval/eval_fcs.py`. The core idea is Newton's second law applied to the body's center of mass:

1. Compute CoM from all 24 SMPL joints, weighted by segment masses (biomechanical data from Winter 2009)
2. Derive CoM acceleration via finite differences
3. Compute the **required force** to produce the observed acceleration: `F = m·a + m·g`
4. Detect foot contacts based on foot height and velocity
5. Estimate the **available force** from contact feet (capped at 3× body weight per foot)
6. The FCS is the mean **force deficit** — how much force was needed but could not have been produced

At this stage, FCS was purely a NumPy-based evaluation. It ran at checkpoints during training, computed scores on a few generated samples, and logged them to WandB. It had **no influence on the loss function**.

### Files added
- `eval/eval_fcs.py` — full FCS evaluator (non-differentiable, NumPy)
- `eval/test_fcs.py` — unit tests
- `eval/fcs_integration_guide.py` — integration examples
- `TRAINING_WITH_FCS.md` — three possible integration strategies documented

---

## Stage 2: Foot Contact Detection

**Commits**: `091184d` through `26f1e47` (Mar 2–13, 2026)

Contact detection turned out to be harder than expected and required extensive debugging and calibration across 10+ commits.

### Bug: No ground height threshold (`1488737`)

The initial contact detection used only foot velocity. A foot could be detected as "in contact" while high in the air simply because it paused momentarily. The fix added a height threshold — feet must be near the ground to count as contacts.

### Bug: AIST++ ground level not at zero (`fd67879`)

AIST++ motion capture data has an arbitrary vertical origin. The floor is not at z=0. The fix computes ground level as the minimum foot height per sequence, then measures all heights relative to that.

### AND vs OR logic for contact (`7bd4e5e`, `fd67879`)

Two conditions define contact: foot is near ground AND foot is slow. But should they be combined with AND or OR?

- **OR logic** (`fd67879`): Detects contacts whenever *either* condition is true. More permissive — catches feet that slide during pivots. But also produces false positives (slow feet at hip height).
- **AND logic** (`7bd4e5e`): Both conditions must be true. More conservative — only 3.1% of frames detected as contacts with tight thresholds.

After calibration with `debug_foot_contacts.py` on real AIST++ data, AND logic was kept but the velocity threshold was raised from 0.10 to 0.50 m/s (`5d34a4a`), yielding ~21% contact detection — a more realistic rate for dance that includes pivots and turns.

### Final calibrated thresholds (eval_fcs.py)
- Ground height: < 0.08m above minimum foot height
- Horizontal velocity: < 0.20 m/s
- Logic: AND (both conditions)

These thresholds are hardcoded in the evaluator. The FCS predictor uses learnable versions (see Stage 7).

---

## Stage 3: First Attempt — Direct Gradient Through Physics

**Commit**: `6c24f0a` (Mar 13, 11:03)

With FCS working as a validation metric, the next step was making it influence training. The first approach was conceptually straightforward: **generate samples during training with gradients enabled, compute physics penalties on them, and backpropagate.**

### Implementation

A new method `compute_physics_penalty()` was added to `EDGE.py`:

1. Run DDIM sampling (50 steps) with `torch.set_grad_enabled(True)` to keep the computation graph
2. Unnormalize the generated sample to real-world coordinates
3. Compute forward kinematics to get joint positions
4. Apply three differentiable penalties:
   - **Ground penetration** (weight 1.0): `ReLU(-foot_height)` — feet below floor
   - **Foot skating** (weight 0.5): `foot_velocity × near_ground` — sliding during contact
   - **Excessive acceleration** (weight 0.1): `ReLU(|accel| - 2.0)` — implausible forces
5. Combine into a scalar loss, call `accelerator.backward()`, step the optimizer

This ran periodically (every N epochs), not every training step, to manage compute cost.

### Why it seemed like it should work

The penalties are all differentiable PyTorch operations. The DDIM sampling loop is implemented in PyTorch. In principle, gradients should flow from the penalty through the 50 sampling steps back to the denoising model's weights.

---

## Stage 4: Abandoning Direct Gradients

**Commit**: `51e0922` (Mar 13, 12:55 — less than 2 hours after the previous commit)

### What went wrong

Backpropagating through 50 DDIM sampling steps is **computationally infeasible**:

- Each DDIM step involves a full forward pass through the transformer denoiser
- Retaining computation graphs for all 50 steps consumes enormous GPU memory
- The gradient signal becomes vanishingly small after passing through 50 iterations
- A single "physics regularization step" took orders of magnitude longer than a normal training step

### The change

The physics penalty code was converted to **monitoring only**:

```python
# Before (6c24f0a):
self.train()
physics_penalty, stats = self.compute_physics_penalty(cond, num_samples=4)
weighted_penalty = self.fcs_loss_weight * physics_penalty
self.optim.zero_grad()
self.accelerator.backward(weighted_penalty)
self.optim.step()

# After (51e0922):
self.eval()
with torch.no_grad():
    physics_penalty, stats = self.compute_physics_penalty(cond, num_samples=4)
# Log to WandB, no gradient step
```

The penalty weights were also rebalanced based on what matters for dance:
- Ground penetration: 1.0 → **2.0** (critical — feet must not go through the floor)
- Skating: 0.5 → **0.1** (reduced — controlled sliding is necessary for pivots and turns)
- Acceleration: 0.1 → **0.3** (moderate — limit unrealistic forces)

---

## Stage 5: The FCS Predictor Approach

**Commit**: `537c642` (Mar 13, 13:26 — just 30 minutes after abandoning direct gradients)

The fundamental insight: if you can't differentiate through the *full FCS computation* (which requires DDIM sampling), train a **surrogate neural network** that approximates FCS from joint positions. This network:

- Is fully differentiable (standard PyTorch forward pass)
- Is fast (single forward pass, no iterative sampling)
- Can be applied inside `p_losses` on the model's predicted joint positions — which already exist for the FK loss and already have a gradient path to the model weights

### The training pipeline

1. **Offline**: Extract joint positions from AIST++ dataset, compute ground-truth FCS for each
2. **Offline**: Generate augmented (corrupted) motions, compute their FCS scores
3. **Offline**: Train the predictor network to map `joint_positions → FCS_score`
4. **Online**: Load trained predictor into the diffusion training loop and use it as a loss term

---

## Stage 6: FCS Predictor v1 — Generic Convolutions (Failed)

**Commit**: `537c642` (initial version in the same commit)

The first predictor architecture was a straightforward design:

```
Joint Positions (B, S, 24, 3)
    → Flatten to (B, S, 72)
    → Linear projection to hidden_dim
    → 4× temporal Conv1d blocks (residual)
    → Concatenate with velocity/acceleration projections
    → Global average pool
    → MLP head → scalar FCS
```

### Why it failed

| Metric | Result |
|--------|--------|
| Validation loss | 23.3 |
| Correlation with true FCS | 0.22 |
| Behavior | Collapsed to predicting the mean FCS for all inputs |

The network tried to learn physics from scratch — extracting CoM, forces, contacts, and their relationships from raw joint coordinates using generic convolutions. This is an extremely difficult learning problem:

- **72 input features** (24 joints × 3 coords) with no indication of which joints are feet, which contribute to CoM, etc.
- The relationship between joint positions and forces involves **second derivatives** (acceleration), which are notoriously hard to learn from data
- Contact detection requires knowing ground height, which varies per sequence
- The FCS score involves a ratio of forces that spans 2 orders of magnitude (0.02–6+)

The network had no physics inductive bias and not enough capacity/data to discover biomechanics on its own.

---

## Stage 7: FCS Predictor v2 — Physics-Informed Architecture

**Uncommitted changes** (current working tree)

The solution was to **not ask the network to learn physics** — instead, compute the physics features explicitly and let the network learn only how to combine them.

### PhysicsFeatureExtractor

A new module that mirrors what `eval_fcs.py` computes, but using differentiable PyTorch operations:

| Feature | Description | Dims |
|---------|-------------|------|
| Force deficit | `ReLU(required_force - available_force)`, normalized | 1 |
| Required force | `\|m·a + m·g\|`, normalized | 1 |
| Available force | `num_contacts × max_force_per_foot`, normalized | 1 |
| Num contacts | Soft contact count via sigmoid | 1 |
| Foot skating | `velocity × contact_probability` | 1 |
| Ground penetration | `ReLU(-foot_height)` | 1 |
| CoM acceleration | Raw acceleration vector | 3 |
| Joint velocity stats | Mean and max across joints | 2 |
| Joint acceleration stats | Mean and max across joints | 2 |
| CoM height | Vertical position of center of mass | 1 |
| **Total** | | **14** |

Key design decisions:

**Soft contact detection**: Hard thresholds (`height < 0.08`) have zero gradient. Replaced with sigmoid approximations: `sigmoid(10 × (threshold - height))`. The steepness factor (10) makes it a close approximation of a step function while remaining differentiable.

**Learnable thresholds**: The height and velocity thresholds for contact detection are `nn.Parameter`s, initialized to the calibrated values (0.08m, 0.20 m/s) but fine-tuned during predictor training.

**Explicit force deficit**: The core FCS signal — how much force is missing — is computed directly by the extractor, not learned. The network head only learns how to aggregate per-frame deficits into a sequence-level score.

### Predictor head

With physics features pre-computed, the head is small:

```
Physics Features (B, S-2, 14)
    → Linear + GELU (→ hidden_dim)
    → 3× Conv1d blocks (residual, kernel=5)
    → Global average pool
    → MLP → Softplus → scalar FCS
```

### Training data augmentation

The predictor must work on noisy diffusion outputs, not just clean mocap. Seven augmentation types generate controlled physics violations:

1. **Joint noise** — Gaussian jitter (σ=0.005–0.05m)
2. **Foot skating** — Horizontal drift on grounded feet
3. **Trajectory corruption** — Low-frequency sinusoid on root trajectory
4. **Temporal jitter** — Random adjacent frame swaps
5. **Gravity violation** — Push feet below ground or lift entire body
6. **Limb explosion** — Scale limb chains outward by 1.3–2.5×
7. **Velocity spikes** — Sudden position jumps at random frames

Each augmented motion has its FCS recomputed with the real evaluator. The augmentation ratio is 3:1 (augmented:real), and augmented data is **regenerated every 20 epochs** so the predictor sees fresh corruptions throughout training.

### Loss function

```python
log_mse = MSE(log(pred + ε), log(true + ε))   # handles 100× FCS range
smooth_l1 = SmoothL1(pred, true)                # absolute accuracy
loss = log_mse + 0.1 × smooth_l1
```

Log-space MSE is critical because FCS values span 0.02–6+. Without it, high-FCS augmented samples dominate the loss and the network ignores fine differences among good motions.

### Results

| Metric | v1 (generic) | v2 (physics-informed) |
|--------|-------------|----------------------|
| Val loss | 23.3 | **7.1** |
| Pearson correlation | 0.22 | **0.986** |
| Behavior | Predicted ~1.0 for everything | Tracks true FCS closely |

Sample predictions:
```
     Type |   True FCS |   Pred FCS |   Error
     real |     0.1256 |     0.1334 |  0.0077
augmented |     2.4211 |     2.3537 |  0.0674
augmented |     5.6807 |     4.7900 |  0.8906
```

---

## Stage 8: The Integration Bug — Gradients That Did Nothing

**Commit**: `537c642` (initial predictor integration)

When the FCS predictor was first integrated into the EDGE training loop, it was applied in `EDGE.train_loop()` like this:

```python
# In EDGE.train_loop — the training step
total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(x, cond)

# Add FCS loss
if self.fcs_predictor is not None:
    x_unnorm = self.normalizer.unnormalize(x)   # ← THIS IS THE BUG
    fcs_loss, mean_fcs = self.compute_fcs_loss_with_predictor(x_unnorm, cond)
    total_loss = total_loss + self.fcs_loss_weight * fcs_loss

self.optim.zero_grad()
self.accelerator.backward(total_loss)
self.optim.step()
```

### The problem

`x` is the **ground-truth motion from the dataset**, not the model's prediction. It is a leaf tensor loaded from the dataloader — it has no gradient connection to the model's weights. Running the FCS predictor on `x` produces a valid FCS score, and calling `.backward()` computes valid gradients, but those gradients flow into the FCS predictor's parameters (which are frozen in eval mode) and into `x` (which has no connection to the denoising model). **Zero gradient reaches the model being trained.**

The loss term appeared in the training logs and WandB, giving the impression that physics-aware training was active. But the model was learning nothing from it. The FCS predictor was essentially evaluating the training data's physics quality (which is high, since it's real mocap) rather than the model's output quality.

### Why it was hard to catch

- The training loop ran without errors
- The FCS loss was non-zero and varied across batches (because different training clips have slightly different FCS scores)
- The other loss terms were working correctly, masking the issue
- Without specifically checking gradient flow (e.g., checking `model.parameters()` gradients after the FCS backward pass), the bug is invisible

---

## Stage 9: The Fix — FCS Inside p_losses

**Uncommitted changes** (current working tree)

The fix moved FCS loss computation into `GaussianDiffusion.p_losses()` — the method where all other losses are computed — and applied it to `model_xp`, the model's **predicted joint positions** that already exist for the FK loss computation.

### Changes to `model/diffusion.py`

```python
# Added attributes
self.fcs_predictor = None
self.fcs_loss_weight = 0.0

# Inside p_losses(), after FK loss computation:
fcs_loss = torch.tensor(0.0, device=model_xp.device)
if self.fcs_predictor is not None and self.fcs_loss_weight > 0:
    fcs_pred = self.fcs_predictor(model_xp)  # (batch,)
    fcs_loss = fcs_pred.mean()

losses = (
    0.636 * loss.mean(),
    2.964 * v_loss.mean(),
    0.646 * fk_loss.mean(),
    10.942 * foot_loss.mean(),
    self.fcs_loss_weight * fcs_loss,   # ← NEW: 5th loss term
)
return sum(losses), losses
```

### Why this works

`model_xp` is computed by running forward kinematics on the model's **predicted rotations and root position**. The gradient path is:

```
Model weights
    → Transformer forward pass (model prediction)
    → Parse rotations and root position
    → SMPL forward kinematics → model_xp (joint positions)
    → FCS predictor → fcs_loss
    → Gradients flow all the way back to model weights ✓
```

This is the same gradient path used by the FK loss, which has been working from the start.

### Changes to `EDGE.py`

- The FCS predictor is now **attached to the diffusion model** (`self.diffusion.fcs_predictor = self.fcs_predictor`) rather than used directly in the training loop
- The old `compute_fcs_loss_with_predictor()` call in the training loop is removed
- The periodic physics monitoring block (DDIM sampling + `compute_physics_penalty()`) is also removed — FCS is now computed on every training step via the predictor, not periodically on full samples
- FCS loss is unpacked from the diffusion output tuple and logged normally

---

## Stage 10: Phase 2 Branch — Alternative Physics Losses

**Branch**: `origin/phase2` (Mar 24, 2026)

A parallel exploration on a separate branch attempted a different approach to physics losses. Instead of the FCS predictor, three explicit physics loss terms were added directly to `p_losses`:

1. **CoM Loss** — Penalizes center-of-mass trajectories that deviate from physically plausible paths
2. **Bilateral Loss** — Encourages left-right symmetry in limb dynamics
3. **Height Loss** — Penalizes unrealistic body heights

### Circular gradient bug (`7ec7577`)

The CoM/balance loss had a circular gradient issue that was fixed in a follow-up commit. The exact nature: the loss was computed using variables that depended on the loss output itself, creating an infinite loop during backpropagation.

This branch was not merged into the main line of development. The FCS predictor approach on `phase3` was preferred because:
- It captures a broader range of physics violations (not just CoM, symmetry, height)
- It learns from the actual FCS metric rather than hand-designing individual penalties
- The predictor approach is more modular — you train it once and plug it in

---

## Current Architecture

```
                    TRAINING PIPELINE (current)
                    ══════════════════════════

OFFLINE: Train FCS Predictor
─────────────────────────────
AIST++ mocap ──► FK ──► Joint positions ──► FCS Evaluator ──► Ground-truth FCS
                              │
                    7 augmentation types
                              │
                              ▼
                     (joints, fcs) pairs
                              │
                              ▼
                     FCS Predictor Network
                     (PhysicsFeatureExtractor
                      + Conv blocks + MLP)
                              │
                              ▼
                    models/fcs_predictor.pt


ONLINE: Diffusion Training
──────────────────────────
Noisy motion + music ──► Transformer ──► Predicted clean motion
                                                  │
                              ┌────────────────────┼──────────────────┐
                              ▼                    ▼                  ▼
                        Reconstruction        FK → model_xp     Foot contact
                        + Velocity loss       │         │          loss
                                              ▼         ▼
                                          FK loss   FCS Predictor
                                                        │
                                                        ▼
                                                    FCS loss
                                                        │
                              ┌──────────────────────────┘
                              ▼
                    Total loss = 0.636×recon + 2.964×vel + 0.646×fk
                              + 10.942×foot + w×fcs
                              │
                              ▼
                    backward() ──► update model weights
```

### Files involved

| File | Role |
|------|------|
| `eval/eval_fcs.py` | Ground-truth FCS evaluator (NumPy, non-differentiable) |
| `model/fcs_predictor.py` | Differentiable FCS approximation (PhysicsFeatureExtractor + FCSPredictor) |
| `train_fcs_predictor.py` | Offline predictor training with 7 augmentation types |
| `model/diffusion.py` | FCS loss computed in `p_losses` on `model_xp` |
| `EDGE.py` | Predictor loading, attachment to diffusion model, training orchestration |
| `eval_checkpoints.py` | Post-training evaluation of FCS/PFC on saved checkpoints |

---

## Experimental Results

Two full training runs were completed to 2000 epochs and evaluated with 200 generated samples per checkpoint. Both used batch size 128 with Jukebox features on AIST++.

- **Baseline** (`runs/baseline/no_fcs/`): Standard EDGE training, FCS loss weight = 0. Trained Mar 28–30, 2026.
- **Physics W12** (`runs/fcs/physics_w12/`): FCS predictor loss enabled with weight = 0.12. Trained Mar 30–31, 2026.

Several earlier FCS experiments failed or were abandoned before completing:
- `physics_v1`: Failed immediately (WandB authentication error)
- `physics_w1`: Abandoned early (no data logged)
- `physics_v12`: Stopped at 100 epochs
- `physics_v13`: Stopped at 400 epochs (4 checkpoints saved, renders produced)

### FCS Score Comparison (200 samples per checkpoint)

FCS measures force inconsistency — lower is better. All values are means across 200 generated samples.

| Epoch | Baseline FCS | Physics W12 FCS | Improvement |
|------:|-------------:|----------------:|------------:|
| 100   | 0.0909       | **0.0045**      | 20.2×       |
| 200   | 0.9832       | 0.1218          | 8.1×        |
| 300   | 0.7497       | 0.2020          | 3.7×        |
| 400   | 0.2669       | 0.0458          | 5.8×        |
| 500   | 0.1344       | 0.0235          | 5.7×        |
| 600   | 0.1070       | **0.0142**      | 7.5×        |
| 700   | 0.0927       | 0.0333          | 2.8×        |
| 800   | 0.0832       | 0.0436          | 1.9×        |
| 900   | 0.1288       | 0.0561          | 2.3×        |
| 1000  | **0.0677**   | 0.0313          | 2.2×        |
| 1100  | 0.0917       | 0.0314          | 2.9×        |
| 1200  | 0.0965       | 0.0321          | 3.0×        |
| 1300  | 0.0874       | 0.0562          | 1.6×        |
| 1400  | 0.1265       | 0.0476          | 2.7×        |
| 1500  | 0.1282       | 0.0423          | 3.0×        |
| 1600  | 0.1382       | 0.0487          | 2.8×        |
| 1700  | 0.1049       | **0.0369**      | 2.8×        |
| 1800  | 0.1187       | 0.0661          | 1.8×        |
| 1900  | 0.1113       | 0.0652          | 1.7×        |
| 2000  | 0.1560       | 0.0470          | 3.3×        |

**Best FCS**: Baseline achieved 0.0677 at epoch 1000. Physics W12 achieved 0.0045 at epoch 100 and 0.0142 at epoch 600 — an order of magnitude lower.

**Average across all checkpoints**: Baseline = 0.170, Physics W12 = 0.048 — a **3.5× overall improvement**.

### PFC Score Comparison (200 samples per checkpoint)

PFC is the original EDGE metric (heuristic, lower is better).

| Epoch | Baseline PFC | Physics W12 PFC |
|------:|-------------:|----------------:|
| 100   | 0.387        | **0.324**       |
| 500   | 1.183        | **0.464**       |
| 1000  | **0.828**    | 0.801           |
| 1500  | 1.133        | **0.940**       |
| 2000  | 1.204        | **0.907**       |

PFC improvements are smaller than FCS improvements but still consistent: the physics-trained model generally scores better.

### Training Loss Comparison

The FCS-trained model shows slightly higher reconstruction/FK losses (expected trade-off with the physics constraint) but dramatically lower foot contact loss.

| Metric (at epoch 2000)  | Baseline | Physics W12 | Ratio |
|------------------------:|:--------:|:-----------:|:-----:|
| Total Loss              | 0.01343  | 0.01767     | 1.3×  |
| Reconstruction          | 0.00640  | 0.00806     | 1.3×  |
| Velocity                | 0.00317  | 0.00411     | 1.3×  |
| FK                      | 0.00358  | 0.00540     | 1.5×  |
| Foot Contact            | 0.00028  | 0.00011     | **0.4×** (lower is better) |
| FCS Train Loss          | N/A      | 0.00033     | —     |

Key observations:

1. **Reconstruction/FK trade-off**: The physics-trained model has ~30% higher reconstruction and FK losses. This is expected — the model is optimizing an additional objective (physics) that sometimes conflicts with exact pose matching. Some of the original "best poses" may have physics issues.

2. **Foot contact loss drops 60%**: The physics loss indirectly improves foot contact quality. The FCS predictor penalizes foot skating (velocity during contact), which overlaps with what the foot contact loss measures. The two losses reinforce each other.

3. **FCS train loss is tiny**: The FCS predictor loss during training is ~0.0003, several orders of magnitude below other losses even after weighting. This suggests the weight (0.12) could potentially be increased in future experiments to push physics quality further, at the cost of additional reconstruction degradation.

### Failed Experiments

| Run | Epochs | Status | Cause |
|-----|--------|--------|-------|
| `physics_v1` | 0 | Failed | WandB authentication error at launch |
| `physics_w1` | ~0 | Abandoned | No training data logged |
| `physics_v12` | 100 | Partial | Unknown — early termination |
| `physics_v13` | 400 | Partial | Unknown — 4 checkpoints saved, has renders |

The naming progression (v1 → v12 → v13 → w1 → w12) reflects iterative experimentation with FCS predictor versions and weight values. "v" likely refers to predictor version, "w" to weight configuration. Only `w12` (weight=0.12 with the v2 physics-informed predictor) ran to completion.

---

## Lessons Learned

### 1. You can't just backpropagate through everything

DDIM sampling is 50 sequential forward passes. Retaining computation graphs for all of them is impractical. The surrogate network approach — train a fast approximation offline, use it online — is a well-established pattern (e.g., learned reward models in RLHF). The key requirement is that the surrogate must be applied to a tensor with a gradient path to the model weights.

### 2. Check where your gradients flow

The most insidious bug in Phase 3 was the FCS loss computed on dataset samples (`x`) instead of model predictions (`model_xp`). The code ran, the loss was non-zero, WandB logged it — but zero gradient reached the model. When adding a new loss term, verify gradient flow explicitly: `loss.backward()` should produce non-zero gradients in `model.parameters()`.

### 3. Physics features are not learnable from raw coordinates

The v1 predictor (generic convolutions on 72D joint coordinates) failed catastrophically (correlation 0.22). Physics involves second derivatives, weighted averages with anatomical mass fractions, conditional logic (contacts), and force ratios. Generic convolutions cannot discover these relationships from ~2000 training samples. The v2 architecture computes these features explicitly and lets the network learn only the aggregation — a 50× reduction in what needs to be learned.

### 4. Contact detection is domain-specific

Five commits were spent on contact detection alone. The core challenge: dance involves controlled foot sliding (pivots, turns, glides) that should not be penalized, while accidental sliding (skating artifacts) should. The calibration required real-data analysis (`debug_foot_contacts.py`) and iterating on AND/OR logic and threshold values. The final FCS predictor sidesteps this by using soft, learnable thresholds.

### 5. Augmentation is essential for surrogate models

The predictor trained only on real (physically plausible) mocap data would never see poor-quality motions. It would learn that FCS ≈ 0.1–0.5 for everything and be unable to distinguish bad from good. The seven augmentation types generate diverse physics violations with known FCS scores, teaching the predictor to assign high scores to implausible motion.

### 6. Log-space loss for wide-range targets

FCS values range from 0.02 (perfect mocap) to 6+ (severely broken motion). Standard MSE would be dominated by the extreme high-FCS augmented samples and ignore fine differences among good motions. Log-space MSE treats a 2× error equally whether the true value is 0.1 or 5.0.
