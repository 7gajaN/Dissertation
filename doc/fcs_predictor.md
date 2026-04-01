# FCS Predictor: Physics-Aware Differentiable Loss for Diffusion Training

## Problem

EDGE trains a diffusion model to generate dance from music. The standard training losses (reconstruction, velocity, FK, foot contact) encourage realistic motion but don't explicitly enforce physical plausibility. The Force Consistency Score (FCS) measures physical plausibility by checking whether ground reaction forces are consistent with observed center-of-mass dynamics, but it cannot be directly used as a training loss because:

1. FCS is computed on **complete generated sequences**, which require running the full DDIM sampling loop (50 steps) — too expensive to do every training step.
2. The iterative sampling loop is **not easily differentiable** — gradients can't flow back to update the model.
3. During diffusion training, the model predicts clean motion from a noisy input at a single timestep. FCS needs a coherent multi-frame sequence with meaningful dynamics.

## Solution: Learned Physics Proxy

Train a separate neural network (the FCS predictor) to estimate FCS scores from joint positions. This predictor is:
- **Differentiable** — gradients flow through it back to the diffusion model
- **Fast** — single forward pass, no sampling loop needed
- **Applied inside `p_losses`** — runs on the model's predicted joint positions (`model_xp`), which already exist for the FK loss computation

### Architecture (v2 — Physics-Informed)

The first version used generic temporal convolutions on raw joint positions and failed (collapsed to predicting the mean, correlation ~0.22). The problem: learning physics from scratch via convolutions is too hard.

The current version mirrors what FCS actually computes:

```
Joint Positions (B, S, 24, 3)
        │
        ▼
PhysicsFeatureExtractor (differentiable)
  ├── CoM position (weighted average of joints by segment mass)
  ├── CoM velocity, acceleration (finite differences)
  ├── Required ground reaction force (F = ma + mg)
  ├── Soft foot contact detection (sigmoid thresholds on height & velocity)
  ├── Available force (proportional to contacts)
  ├── Force deficit (ReLU of required - available)
  ├── Foot skating metric (velocity during contact)
  ├── Ground penetration (feet below ground)
  ├── Joint velocity/acceleration statistics
  └── CoM height
        │
        ▼
  14 physics features per frame (B, S-2, 14)
        │
        ▼
  Temporal Conv Blocks (3 layers, hidden_dim=128)
        │
        ▼
  Global Average Pooling → MLP → Softplus
        │
        ▼
  Scalar FCS prediction (B,)
```

Key design decisions:
- **Soft contact detection**: uses `sigmoid(10 * (threshold - value))` instead of hard thresholds, keeping gradients smooth
- **Learnable thresholds**: height and velocity thresholds for contact detection are `nn.Parameter`s, fine-tuned during training
- **Explicit force deficit**: the core FCS signal is computed directly, not learned — the network just learns how to aggregate it

### Training Data: Augmented Motions

The predictor must generalize from clean mocap to noisy diffusion outputs. Training only on real (physically plausible) data would leave it unable to evaluate bad motions.

Seven augmentation types generate motions with controlled physics violations:

| Augmentation | What it does | Physics violation |
|---|---|---|
| Joint noise | Gaussian jitter on joint positions | Jittery, unrealistic accelerations |
| Foot skating | Horizontal drift on grounded feet | Force during supposed contact |
| Trajectory corruption | Low-frequency sinusoidal perturbation | Unrealistic CoM accelerations |
| Temporal jitter | Randomly swap adjacent frames | Non-smooth velocity profiles |
| Gravity violation | Push feet below ground or lift body | Contact/no-contact inconsistency |
| Limb explosion | Scale limb segments outward | Impossible body proportions |
| Velocity spike | Sudden position jumps at random frames | Impossible forces |

Each augmented motion has its FCS recomputed with the actual FCS evaluator, so the predictor learns the true mapping. Augmented data is **regenerated every 20 epochs** so the predictor sees fresh corruptions throughout training.

### Training Results

| Metric | v1 (generic convs) | v2 (physics-informed) |
|---|---|---|
| Val loss | 23.3 | **7.1** |
| Correlation | 0.22 | **0.986** |
| Real data accuracy | Predicted ~1.0 for everything | Error ~0.1 |
| Augmented data accuracy | Random | Tracks true FCS closely |

Sample predictions (v2):
```
        Type |   True FCS |   Pred FCS |      Error
        real |     0.1256 |     0.1334 |     0.0077
   augmented |     2.4211 |     2.3537 |     0.0674
   augmented |     5.6807 |     4.7900 |     0.8906
```

### Loss Function

```python
log_mse = MSE(log(pred + eps), log(true + eps))  # Log-space for wide FCS range
smooth_l1 = SmoothL1(pred, true)                  # Direct accuracy
loss = log_mse + 0.1 * smooth_l1
```

Log-space MSE is critical because FCS values span 0.02–6+. Raw MSE would be dominated by high-FCS augmented samples.

## Integration with Diffusion Training

### Before (broken)

```python
# In EDGE.train_loop — computed on ground truth data x
x_unnorm = self.normalizer.unnormalize(x)  # x = dataset sample, NOT model output
fcs_loss = self.compute_fcs_loss_with_predictor(x_unnorm, cond)
total_loss = total_loss + weight * fcs_loss
# BUG: x has no gradient connection to model weights → fcs_loss contributes zero gradients
```

### After (fixed)

```python
# In GaussianDiffusion.p_losses — computed on model's predicted joint positions
# model_xp already exists (used for FK loss), has gradient path to model weights
if self.fcs_predictor is not None and self.fcs_loss_weight > 0:
    fcs_pred = self.fcs_predictor(model_xp)  # (batch,)
    fcs_loss = fcs_pred.mean()

losses = (
    0.636 * loss.mean(),       # reconstruction
    2.964 * v_loss.mean(),     # velocity
    0.646 * fk_loss.mean(),    # forward kinematics
    10.942 * foot_loss.mean(), # foot contact
    self.fcs_loss_weight * fcs_loss,  # physics (NEW)
)
```

Gradient flow:
```
noisy input → model prediction → FK → model_xp (joint positions) → FCS predictor → loss
                  ↑                                                                    ↓
             model weights  ←←←←←←←←←←←←  gradients flow correctly  ←←←←←←←←←←←←←←←←
```

## Files Changed

| File | Change |
|---|---|
| `model/fcs_predictor.py` | Complete rewrite: physics-informed architecture with `PhysicsFeatureExtractor`, log-space loss |
| `model/diffusion.py` | Added `fcs_predictor` and `fcs_loss_weight` attributes; FCS loss computed in `p_losses` on `model_xp` |
| `EDGE.py` | Attach predictor to diffusion model; removed broken FCS computation from training loop |
| `train_fcs_predictor.py` | Complete rewrite: 7 augmentation types, periodic regeneration, correlation tracking, early stopping |

## Usage

### Train the predictor
```bash
accelerate launch train_fcs_predictor.py \
    --max_train_samples 500 \
    --max_val_samples 186 \
    --epochs 200 \
    --aug_ratio 3 \
    --hidden_dim 128 \
    --num_layers 3
```

### Train diffusion model with physics loss
```bash
accelerate launch train.py \
    --fcs_predictor_path models/fcs_predictor.pt \
    --fcs_loss_weight 0.1 \
    --epochs 2000
```

### Evaluate checkpoints independently
```bash
python eval_checkpoints.py --run_dir runs/baseline/no_fcs --num_samples 50
```
