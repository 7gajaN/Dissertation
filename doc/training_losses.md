# EDGE Training Losses

## Overview

EDGE is a diffusion model that generates dance motion conditioned on music. During training, the model takes a real dance clip with added noise and predicts the original clean motion. Five loss terms compare this prediction against the ground truth.

## Diffusion Training Step

```
Real motion x (150 frames, 151 dims)
        │
        ▼
Add noise at random timestep t ──► Noisy motion x_t
                                        │
                                        ▼
                                Transformer(x_t, music, t)
                                        │
                                        ▼
                                Predicted clean motion x̂
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    Compare x̂ vs x  across 5 loss terms
```

The model predicts the **clean motion directly** (not the noise), so all losses operate on clean-looking sequences.

## Motion Representation

Each frame is 151 dimensions:
```
[4 foot contacts | 3 root position | 144 joint rotations (24 joints x 6D)]
```

## Loss Terms

### 1. Train Loss (Reconstruction) — weight: 0.636

```python
loss = MSE(model_output, target)
```

Direct L2 distance between predicted and real motion across all 147 motion dimensions (contacts split off separately). This is the fundamental "did you predict the right dance" signal.

### 2. Velocity Loss — weight: 2.964

```python
target_v = target[:, 1:] - target[:, :-1]      # ground truth frame-to-frame change
model_v  = model_out[:, 1:] - model_out[:, :-1] # predicted frame-to-frame change
v_loss = MSE(model_v, target_v)
```

Compares the **temporal derivatives** (velocity) of predicted vs real motion. Encourages smooth, temporally coherent motion. Without this, the model could predict correct individual poses but jittery transitions. Gets ~4.7x more weight than reconstruction — smoothness matters more than exact position matching.

### 3. FK Loss (Forward Kinematics) — weight: 0.646

```python
model_joints = SMPL_FK(predicted_rotations, predicted_root)   # (B, 150, 24, 3)
target_joints = SMPL_FK(target_rotations, target_root)        # (B, 150, 24, 3)
fk_loss = MSE(model_joints, target_joints)
```

Runs both predicted and real motions through the SMPL skeleton to get **3D joint positions**, then compares those. This matters because small rotation errors can cause large position errors — a 5-degree hip rotation error moves the entire leg, but a 5-degree wrist error barely matters. FK loss penalizes in the space that matters visually.

### 4. Foot Loss (Foot Contact) — weight: 10.942

```python
grounded = model_contact > 0.95          # frames where model says foot is planted
foot_velocity = foot_pos[t+1] - foot_pos[t]  # how much foot moves
foot_velocity[~grounded] = 0             # only penalize during contact
foot_loss = MSE(foot_velocity, zeros)    # planted feet should have zero velocity
```

When the model predicts a foot is **in contact with the ground**, that foot must not slide. Penalizes any horizontal movement of "planted" feet. Gets the highest weight (10.9x reconstruction) because foot skating is the most visually jarring artifact in dance generation.

Uses the **model's own** contact predictions, not ground truth — forcing self-consistency.

### 5. FCS Loss (Physics) — weight: configurable (default 0)

```python
fcs_pred = fcs_predictor(model_joints)  # learned physics score
fcs_loss = fcs_pred.mean()
```

A learned physics loss from the FCS predictor network. Penalizes motions where the predicted ground reaction forces are physically inconsistent with the observed center-of-mass dynamics. Only active when `--fcs_loss_weight > 0` and `--fcs_predictor_path` is provided. See `doc/fcs_predictor.md`.

## Total Loss

```python
total = 0.636 * train_loss + 2.964 * v_loss + 0.646 * fk_loss + 10.942 * foot_loss + w * fcs_loss
```

The weights (0.636, 2.964, 0.646, 10.942) were set by the original EDGE authors. The FCS weight is configurable and defaults to 0 (disabled).

## P2 Weighting

Losses 1–3 are additionally scaled by P2 weights per diffusion timestep. This reweights the loss so the model doesn't disproportionately focus on easy (low-noise) timesteps at the expense of hard (high-noise) ones.

## Evaluation-Only Metrics

These are not training losses — they're computed at checkpoints for monitoring:

- **FCS (Force Consistency Score)**: Physics-based evaluation. Checks if the ground reaction forces needed to produce the observed CoM accelerations are feasible given foot contacts. Lower = better. Uses `eval/eval_fcs.py`.

- **PFC (Physical Foot Contact)**: Simpler heuristic. `min_left_foot_velocity * min_right_foot_velocity * root_acceleration * 10000`. Lower = better. Used in the original EDGE paper.
