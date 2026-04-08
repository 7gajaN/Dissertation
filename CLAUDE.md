# EDGE — Physics-Aware Dance Generation (Dissertation)

## Project Overview

This is an extension of the EDGE (Editable Dance GEneration) CVPR 2023 paper. The dissertation goal is to make the original diffusion-based dance generation model **more physically plausible** — dances that obey Newton's laws, don't have feet sliding through the floor, and produce forces consistent with biomechanics.

**Model**: Diffusion transformer that denoises 151D motion representations (24 SMPL joints × 6D rotations + root position + foot contacts) conditioned on 4800D Jukebox music features. Trained on AIST++ dataset (600 dance clips).

## Current State: Phase 4 Complete

### What exists

- **Baseline model** trained to 2000 epochs (`runs/baseline/no_fcs/`) — standard EDGE, no physics loss. FCS@2000 = 0.156, PFC@2000 = 1.204.
- **Phase 3 FCS predictor model** (`runs/fcs/physics_w12/`) — FCS predictor loss at weight 0.12. FCS@2000 = 0.047, PFC@2000 = 0.907.
- **Phase 4 combined models** in `runs/phase4/`:
  - `fcs_com_combined` — FCS w=0.12 + masked CoM w=0.05
  - `fcs_w1` — FCS predictor at higher weight 1.0 alone
  - `fcs_com_bilateral` — **best FCS** (0.013), FCS w=0.12 + CoM w=0.05 + Bilateral w=5.0
  - `fcs1_com_bilateral` — **best balance** (FCS 0.018, PFC 0.858), FCS w=1.0 + CoM w=0.05 + Bilateral w=5.0
  - `fcs1_com_bilat2` — looser-legs variant, FCS w=1.0 + CoM w=0.05 + Bilateral w=2.0 (FCS 0.028, PFC 0.936). Worse on metrics than the bilateral=5 sibling but trades physics for visible leg expressiveness — judge from renders.
- **FCS predictor network** (`models/fcs_predictor.pt`) — learned physics proxy, correlation 0.986 with ground-truth FCS
- **Real mocap baseline** — FCS 0.132, PFC 2.180 (from `eval_real_data_fcs.py` on 100 AIST++ test sequences). Best Phase 4 model is 10× better than real mocap on FCS, 2.5× better on PFC.
- **Evaluation results** in each run's `eval_results.json` (50 samples per checkpoint)
- **Side-by-side comparison renders** in `renders/comparison/` (baseline vs best FCS vs best balance)

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
    0.636 * reconstruction,                  # L2 on rotations/positions
    2.964 * velocity,                        # L2 on temporal derivatives
    0.646 * fk,                              # L2 on FK joint positions (model_xp)
    10.942 * foot_contact,                   # Penalize sliding when contact predicted
    fcs_loss_weight * fcs_predictor,         # Learned physics score on model_xp
    com_loss_weight * com_balance,           # Phase 4: CoM over support, masked by acceleration
    bilateral_loss_weight * bilateral,       # Phase 4: bilateral foot exclusivity, masked by contact
    foot_height_loss_weight * foot_height,   # Phase 4: feet at ground during predicted contact
)
```

All losses operate on the model's **predicted** clean motion (not the noisy input). `model_xp` = FK joint positions from predicted rotations — this is the tensor with gradient path to model weights, used by FK, FCS, CoM, bilateral and height losses.

The 3 Phase 4 losses default to weight 0.0 (disabled). Note: foot height loss did not converge at any tested weight and is currently unused. The CoM loss requires the acceleration mask (frames with `com_acc < 0.01`) to avoid suppressing dynamic dance moves.

### Training infrastructure

- **Accelerate** for multi-GPU, **WandB** for logging, **EMA** for stable generation
- Checkpoints saved every 100 epochs to `runs/<exp>/weights/train-<epoch>.pt`
- Metrics logged to CSV + JSON in run directory

## Phase 4 Result Summary

| Model | FCS | PFC | vs Real FCS | vs Real PFC |
|-------|-----|-----|-------------|-------------|
| Real mocap (ceiling) | 0.132 | 2.180 | 1.0× | 1.0× |
| Baseline (no physics) | 0.156 | 1.204 | 1.18× worse | 1.81× better |
| FCS predictor w=0.12 (Phase 3) | 0.047 | 0.907 | 2.81× better | 2.40× better |
| FCS w=1.0 | 0.019 | 1.152 | 6.95× better | 1.89× better |
| **FCS w=0.12 + CoM + Bilateral=5** (best FCS) | **0.013** | 1.040 | **10.2× better** | 2.10× better |
| **FCS w=1.0 + CoM + Bilateral=5** (best balance) | 0.018 | **0.858** | 7.33× better | **2.54× better** |
| FCS w=1.0 + CoM + Bilateral=2 (looser legs) | 0.028 | 0.936 | 4.71× better | 2.33× better |

Both best models surpass real mocap on both metrics. There is a stability vs expressiveness trade-off — physics models are visibly more stable but some larger expressive moves get dampened.

Full narrative in `doc/phase4_evolution.md`.

## Phase 5 (potential future work)

See `doc/ideas.md` for the full list. Items not pursued in Phase 4 that could be explored next:
- Curriculum / warm-up for FCS weight (not tried)
- Foot height loss with masking — the only Phase 4 loss that failed; an acceleration- or onset-masked variant might work
- Iterative FCS predictor refinement (retrain predictor on samples from current model)
- Physics-guided sampling at inference time
- Acceleration threshold tuning for CoM masking (currently 0.01, picked once and not tuned)
- The missing combined variant `FCS w=1.0 + CoM (no bilateral)`

## Documentation

- `doc/phase3_evolution.md` — Full narrative of Phase 3 development, trials/errors, and results
- `doc/phase4_evolution.md` — Full narrative of Phase 4: combined runs, bug fixes, CoM acceleration masking, real mocap baseline, qualitative findings
- `doc/ideas.md` — All ideas, including remaining Phase 5 candidates
- `doc/fcs_predictor.md` — FCS predictor architecture, training, and integration details
- `doc/training_losses.md` — All current loss terms explained
- `doc/eval_checkpoints.md` — How to evaluate saved checkpoints

## Training Commands

Always activate the conda env and launch with Accelerate:

```bash
conda activate edge
accelerate launch train.py --com_loss_weight 1.0 --epochs 500 --exp_name com_test --project runs/phase4
```

## Key Lessons (avoid repeating mistakes)

1. **Check gradient flow**: When adding a new loss, verify it's computed on a tensor connected to the model weights (e.g., `model_xp`), NOT on dataset ground truth (`x`). The FCS predictor was initially run on `x` — loss appeared in logs but zero gradient reached the model.

2. **Detach fixed references**: In the CoM balance loss, `.detach()` the support center target. Otherwise gradients create a circular loop (moving feet changes the target, which moves feet again).

3. **Don't backprop through DDIM**: Retaining computation graphs for 50 sampling steps is infeasible. Use surrogate networks or operate on `p_losses` predictions instead.

4. **Physics features > generic convolutions**: The FCS predictor v1 (generic convs on raw joints) failed (correlation 0.22). v2 with explicit physics feature extraction works (0.986). Always encode known physics structure.
