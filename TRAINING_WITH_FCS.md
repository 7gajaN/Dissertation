# Training EDGE with Force Consistency Score (FCS)

## Three Approaches to Use FCS in Training

### 🎯 Option 1: FCS as Validation Metric (✅ IMPLEMENTED)

**Status**: Ready to use - modifications already applied to EDGE.py

**What it does**:
- Monitors physics quality during training
- Logs FCS scores to WandB alongside other metrics
- Doesn't change the training loss
- Evaluates 4 generated samples every save interval

**How it works**:
```
Training Loop:
├─ Standard training with existing losses
│  ├─ Reconstruction loss
│  ├─ Velocity loss
│  ├─ FK loss
│  └─ Foot skate loss
│
└─ At save intervals (e.g., every 5 epochs):
   ├─ Generate validation samples
   ├─ Compute FCS scores
   └─ Log to WandB: "FCS Score: 0.234"
```

**Usage**:
```bash
# Train normally - FCS is automatically computed
python train.py \
    --exp_name my_experiment \
    --feature_type jukebox \
    --epochs 10000 \
    --save_interval 5
```

**WandB Dashboard will show**:
- Train Loss
- V Loss  
- FK Loss
- Foot Loss
- **FCS Score** ← Monitor this for physics quality

**Advantages**:
- ✅ No training slowdown (only at save intervals)
- ✅ Track physics improvement over epochs
- ✅ Compare different models by FCS
- ✅ Easy to implement (already done!)

---

### 🔬 Option 2: FCS as Training Loss (Advanced)

**Status**: Not implemented - requires careful design

**What it would do**:
- Use FCS directly in the loss function
- Model learns to minimize physics violations
- Potentially better physics at generation time

**Challenges**:
```
┌─────────────────────────────────────────────┐
│ Challenge 1: Computational Cost             │
│ - FCS requires CoM calculation (expensive)  │
│ - FK computation for all joints             │
│ - Would slow training significantly         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Challenge 2: Differentiability              │
│ - FCS uses numpy, max(), min() operations   │
│ - Need PyTorch version with gradients       │
│ - Contact detection is non-differentiable   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Challenge 3: Loss Balancing                 │
│ - FCS scale different from other losses     │
│ - Need to tune weighting carefully          │
│ - May conflict with other objectives        │
└─────────────────────────────────────────────┘
```

**If you want to implement this**:

1. **Create differentiable FCS** in [eval/eval_fcs_torch.py](eval/eval_fcs_torch.py):
```python
class DifferentiableFCS(nn.Module):
    """PyTorch version of FCS for use in loss function."""
    
    def forward(self, joint_positions, return_grad=True):
        # All operations must be PyTorch tensors
        # Replace numpy with torch
        # Use soft approximations for non-differentiable ops
        pass
```

2. **Add to diffusion loss** in [model/diffusion.py](model/diffusion.py#L514):
```python
# After line 514, add:
from eval.eval_fcs_torch import DifferentiableFCS

fcs_calculator = DifferentiableFCS()
fcs_loss = fcs_calculator(model_xp)  # model_xp is joint positions

# Update loss combination (line 514-520):
losses = (
    0.636 * loss.mean(),
    2.964 * v_loss.mean(),
    0.646 * fk_loss.mean(),
    10.942 * foot_loss.mean(),
    1.000 * fcs_loss.mean(),  # ← NEW: tune this weight
)
```

**Recommendation**: ⚠️ Start with Option 1, only pursue this if validation FCS shows it's necessary.

---

### 📊 Option 3: FCS-Guided Fine-Tuning

**Status**: Future work - train with existing loss, then fine-tune with FCS

**Strategy**:
```
Phase 1: Pre-train (Epochs 1-8000)
├─ Use existing losses (fast training)
├─ Monitor FCS as validation metric
└─ Get model to reasonable quality

Phase 2: Fine-tune (Epochs 8001-10000)
├─ Add FCS to loss with small weight
├─ Slower but improves physics
└─ Best of both worlds
```

**Implementation**:
```python
# In train_loop (EDGE.py), modify loss:
if epoch > 8000:
    # Add FCS-based regularization
    with torch.no_grad():
        fcs_penalty = compute_fcs_penalty(model_xp)
    # Include in total_loss with small weight
    total_loss += 0.1 * fcs_penalty
```

---

## Current Implementation Details

### What was modified in EDGE.py:

1. **Initialization** (after line 100):
```python
from eval.eval_fcs import ForceConsistencyEvaluator
self.fcs_evaluator = ForceConsistencyEvaluator(fps=30)
self.use_fcs = True
```

2. **New Method** `evaluate_fcs_on_batch()`:
- Generates samples from current model
- Runs FCS evaluation
- Returns mean score and individual scores

3. **Training Loop** (save interval):
- Evaluates FCS on 4 validation samples
- Logs to WandB
- Prints to console

### Monitoring FCS During Training

**Interpretation**:
- **FCS decreasing** → Model learning better physics ✅
- **FCS increasing** → Model forgetting physics ⚠️
- **FCS stable & low** → Good physics maintained ✅
- **FCS stable & high** → Not learning physics ⚠️

**Typical values**:
- Real AIST++ data: 0.2 - 0.5 (good physics)
- Early training: 1.5 - 3.0 (poor physics)
- Well-trained model: 0.5 - 1.0 (acceptable physics)
- Target: < 0.6 (competitive with real data)

---

## Running Training with FCS Monitoring

### Basic Training:
```bash
python train.py \
    --exp_name fcs_baseline \
    --feature_type jukebox \
    --epochs 10000 \
    --save_interval 5 \
    --batch_size 64
```

### Monitor in WandB:
```
1. Go to your WandB project
2. Look for run: "fcs_baseline"  
3. Charts will show:
   - Train Loss (should decrease)
   - FCS Score (should decrease if physics improving)
```

### Compare Different Configurations:
```bash
# Experiment 1: High foot loss weight
python train.py --exp_name high_foot_loss --epochs 10000

# Then modify diffusion.py line 518 to increase foot_loss weight:
# 20.0 * foot_loss.mean()  # instead of 10.942

# Experiment 2: Different architecture
python train.py --exp_name deeper_model --epochs 10000

# Then modify EDGE.py num_layers=12  # instead of 8

# Compare FCS scores in WandB!
```

---

## Validation: Does Your Model Have Good Physics?

### After Training, Evaluate FCS:

```bash
# Generate test samples
python EDGE.py --generate --checkpoint results/train-10000.pt

# Evaluate with FCS
python eval/eval_fcs.py \
    --motion_path generated_motions/ \
    --output results/fcs_evaluation.json

# Compare to real data
python eval/eval_fcs.py \
    --motion_path data/aist_test/ \
    --compare_path generated_motions/ \
    --output results/final_comparison/
```

### Success Criteria:
```
✅ Generated FCS < 0.8
✅ Generated FCS / Real FCS < 2.0 (within 2x of real data)
✅ FCS decreased during training (check WandB curve)
✅ No catastrophic violations (max per-frame < 3.0)
```

---

## Troubleshooting

### FCS not appearing in logs:
```
Check console for:
"FCS evaluator initialized for physics monitoring" ✅
or
"FCS evaluator not available - skipping physics validation" ❌

If ❌, check that eval/eval_fcs.py exists and numpy is installed
```

### FCS evaluation fails during training:
```
Common causes:
1. Joint positions have NaN values → Model diverged
2. Generated motion too short → Need at least 3 frames
3. GPU memory issue → Reduce num_samples in evaluate_fcs_on_batch()
```

### FCS scores seem too high/low:
```
Check:
1. Units: Motions should be in meters (not cm)
2. FPS: Should match your data (default 30fps)
3. Body mass: Default 70kg, adjust if needed
```

---

## Next Steps for Your Research

### Week 3-4: Validation
1. ✅ Train model with FCS monitoring enabled
2. ✅ Track FCS over epochs in WandB
3. ✅ Compare FCS of your model vs. baseline EDGE
4. ✅ Run final evaluation: real vs generated

### Phase 3: If FCS Shows Problems
1. Adjust loss weights (increase foot_loss weight)
2. Try different architectures
3. Consider adding FCS to training loss (Option 2)
4. Fine-tune with FCS guidance (Option 3)

---

## Summary

| Approach | Difficulty | Speed | Physics Quality |
|----------|-----------|-------|-----------------|
| **Option 1: Validation** | ✅ Easy | ✅ Fast | Good baseline |
| **Option 2: Training Loss** | ⚠️ Hard | ❌ Slow | Potentially better |
| **Option 3: Fine-tuning** | 🟡 Medium | 🟡 Medium | Best of both |

**Recommendation for Phase 2**: Use Option 1 (already implemented!) 

This gives you:
- ✅ Physics monitoring during training
- ✅ Comparable evaluation metric
- ✅ No slowdown in training
- ✅ Foundation for future improvements

Start training now and monitor the FCS curve! 📈
