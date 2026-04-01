# EDGE — Physics-Aware Dance Generation (Dissertation)

## Project Overview

This is an extension of the EDGE (Editable Dance GEneration) CVPR 2023 paper. The dissertation goal is to make the original diffusion-based dance generation model **more physically plausible** — dances that obey Newton's laws, don't have feet sliding through the floor, and produce forces consistent with biomechanics.

**Model**: Diffusion transformer that denoises 151D motion representations (24 SMPL joints × 6D rotations + root position + foot contacts) conditioned on 4800D Jukebox music features. Trained on AIST++ dataset (600 dance clips).

## Current State: Phase 3 Complete

### What exists

- **Baseline model** trained to 2000 epochs (`runs/baseline/no_fcs/`) — standard EDGE, no physics loss
- **Physics-trained model** (`runs/fcs/physics_w12/`) — with FCS predictor loss (weight=0.12), 3.5× better FCS scores
- **FCS predictor network** (`models/fcs_predictor.pt`) — learned physics proxy, correlation 0.986 with ground-truth FCS
- **Evaluation results** — 200-sample eval at every checkpoint for both models in `runs/*/eval_results.json`

### Key files

| File | Purpose |
|------|---------|
| `EDGE.py` | Main model class, training loop orchestration |
| `model/diffusion.py` | Diffusion process, `p_losses` method where all training losses are computed |
| `model/fcs_predictor.py` | Learned FCS approximation (PhysicsFeatureExtractor + conv head) |
| `model/model.py` | DanceDecoder transformer architecture |
| `train.py` | Training entry point |
| `train_fcs_predictor.py` | FCS predictor training with 7 augmentation types |
| `eval/eval_fcs.py` | Ground-truth FCS evaluator (NumPy, non-differentiable) |
| `eval_checkpoints.py` | Batch checkpoint evaluation |
| `args.py` | All CLI arguments for training and testing |
| `dataset/dance_dataset.py` | AIST++ data loading |

### Loss function (in `model/diffusion.py:p_losses`)

```python
losses = (
    0.636 * reconstruction,    # L2 on rotations/positions
    2.964 * velocity,          # L2 on temporal derivatives
    0.646 * fk,                # L2 on FK joint positions (model_xp)
    10.942 * foot_contact,     # Penalize sliding when contact predicted
    weight * fcs_predictor,    # Learned physics score on model_xp
)
```

All losses operate on the model's **predicted** clean motion (not the noisy input). `model_xp` = FK joint positions from predicted rotations — this is the tensor with gradient path to model weights, used by FK loss and FCS loss.

### Training infrastructure

- **Accelerate** for multi-GPU, **WandB** for logging, **EMA** for stable generation
- Checkpoints saved every 100 epochs to `runs/<exp>/weights/train-<epoch>.pt`
- Metrics logged to CSV + JSON in run directory

## Phase 4: Next Steps

**Read `doc/ideas.md` for the full plan.** The priority items are:

### Priority 1: Reintegrate Phase 2 Physics Losses

The `origin/phase2` branch (commits `6fd5e84`, `7ec7577`) added three explicit physics losses to `p_losses` that were never merged into the main line. The goal is to combine them with the FCS predictor loss.

**Three losses to add:**

1. **CoM Balance Loss** — Penalizes center of mass (horizontal) being far from the support center (mean position of contact feet). Uses SMPL joint mass fractions. **IMPORTANT**: `.detach()` the foot positions for the support center to prevent a circular gradient loop (this bug was found and fixed on phase2).

2. **Bilateral Foot Exclusivity Loss** — Product of left-foot velocity × right-foot velocity. Penalizes both feet sliding simultaneously.

3. **Foot Height During Contact Loss** — Penalizes feet hovering above ground while model predicts contact. Ground reference = min foot height per sequence, `.detach()`ed.

**Files to modify:**

| File | Change |
|------|--------|
| `model/diffusion.py` | Add `_SMPL_JOINT_MASSES` constant, 3 loss weight attributes in `__init__`, 3 loss computations after foot skate loss in `p_losses`, extend losses tuple from 5 → 8 |
| `EDGE.py` | Accept 3 new weight params in `__init__`, pass to diffusion constructor, unpack 8-element loss tuple, accumulate and log the 3 new losses |
| `args.py` | Add `--com_loss_weight`, `--bilateral_loss_weight`, `--foot_height_loss_weight` (default 0.0) |
| `train.py` | Pass the 3 new args to `EDGE()` constructor |

The full implementation exists on `origin/phase2` — reference commits `6fd5e84` (initial) and `7ec7577` (gradient bug fix). Do NOT cherry-pick blindly — the phase2 branch doesn't have the FCS predictor changes. Manually port the loss code into the current codebase.

**Experiments to run:**
1. Phase 2 losses only (no FCS predictor) — isolate their effect
2. Phase 2 losses + FCS predictor — do they stack?
3. Weight sweep for the 3 new losses

### After Priority 1

See `doc/ideas.md` for the full list including: higher FCS weight experiments, curriculum/warm-up, motion quality metrics (FID, diversity, beat alignment), real data FCS baseline, post-hoc optimization, physics-guided sampling.

## Documentation

- `doc/phase3_evolution.md` — Full narrative of Phase 3 development, trials/errors, and results
- `doc/ideas.md` — All ideas for Phase 4 with priority ranking
- `doc/fcs_predictor.md` — FCS predictor architecture, training, and integration details
- `doc/training_losses.md` — All 5 current loss terms explained
- `doc/eval_checkpoints.md` — How to evaluate saved checkpoints

## Key Lessons (avoid repeating mistakes)

1. **Check gradient flow**: When adding a new loss, verify it's computed on a tensor connected to the model weights (e.g., `model_xp`), NOT on dataset ground truth (`x`). The FCS predictor was initially run on `x` — loss appeared in logs but zero gradient reached the model.

2. **Detach fixed references**: In the CoM balance loss, `.detach()` the support center target. Otherwise gradients create a circular loop (moving feet changes the target, which moves feet again).

3. **Don't backprop through DDIM**: Retaining computation graphs for 50 sampling steps is infeasible. Use surrogate networks or operate on `p_losses` predictions instead.

4. **Physics features > generic convolutions**: The FCS predictor v1 (generic convs on raw joints) failed (correlation 0.22). v2 with explicit physics feature extraction works (0.986). Always encode known physics structure.
