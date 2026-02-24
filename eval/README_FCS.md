# Force Consistency Score (FCS) - Implementation Guide

## Overview

The Force Consistency Score (FCS) is a physics-based metric for evaluating dance motion quality. Unlike heuristic metrics like PFC, FCS validates whether the observed motions obey fundamental biomechanical principles.

## Key Innovation

**Core Physics Principle**: The ground reaction forces must be sufficient to produce the observed Center of Mass (CoM) accelerations.

$$\vec{F}_{ground} = m \cdot \vec{a}_{CoM}$$

The metric checks:
1. ✅ Can the feet in contact generate the required force?
2. ✅ Are the forces within biomechanical limits?
3. ✅ Is the force distribution physically plausible?

## Why FCS is Better Than PFC

### PFC Limitations
- ❌ **No physics model** - Just multiplies heuristics
- ❌ **Ignores mass distribution** - Doesn't consider CoM
- ❌ **Binary foot logic** - Misses single-foot support
- ❌ **No force validation** - Doesn't check if forces are possible
- ❌ **Arbitrary scaling** - Normalizes by max, hard to interpret

### FCS Advantages
- ✅ **Physics-based** - Uses F=ma and biomechanics
- ✅ **CoM modeling** - Anthropometric mass distribution
- ✅ **Force feasibility** - Validates actual force capacity
- ✅ **Interpretable** - Violations have physical meaning
- ✅ **Discriminative** - Should separate real from generated

## Implementation Details

### 1. Center of Mass Calculation
Uses SMPL anthropometric data (Winter, 2009):
- 14 body segments with realistic mass distributions
- Segment CoMs calculated from joint positions
- Whole-body CoM = weighted average of segment CoMs

### 2. Force Analysis
```python
# For each frame:
1. Calculate CoM acceleration (2nd derivative of position)
2. Required force: F_req = m * a_CoM + m * g (gravity)
3. Detect which feet are in contact (velocity < threshold)
4. Calculate maximum force feet can generate (biomechanical limits)
5. Force inconsistency = max(0, |F_req| - F_available)
```

### 3. Biomechanical Constraints
- **Max vertical force**: 3× body weight (running/jumping limit)
- **Max horizontal force**: Limited by friction (~0.5 coefficient)
- **Contact detection**: Foot velocity < 0.02 m/s

## Usage

### Basic Evaluation
```bash
python eval/eval_fcs.py --motion_path motions/generated/
```

### Compare Real vs Generated
```bash
python eval/eval_fcs.py \
    --motion_path data/aist_motions/ \
    --compare_path motions/generated/ \
    --output results/fcs_comparison/
```

### Limit Number of Samples
```bash
python eval/eval_fcs.py \
    --motion_path motions/ \
    --max_samples 100 \
    --output results/fcs_results.json
```

## Output Metrics

### FCS Score
- **Low score (< 0.5)**: Good physics, realistic motion
- **Medium score (0.5-1.5)**: Minor violations, acceptable
- **High score (> 1.5)**: Major violations, unrealistic physics

### Additional Metrics
- **Contact ratio**: % of time at least one foot is on ground
- **Average contacts**: Mean number of feet in contact
- **Per-frame violations**: Detailed violation timeline

### Comparison Output
```
COMPARISON ANALYSIS
================================
FCS Score Comparison:
  Real data:      0.234 ± 0.089
  Generated data: 1.456 ± 0.312
  Difference:     +1.222 (+522.2%)

Separability:
  FCS Cohen's d: 5.123 (large)
  PFC Cohen's d: 1.234 (large)
```

## Validation Criteria

For Phase 2 success, FCS should show:

1. **✅ Clear Separation**: Real data << Generated data
2. **✅ Statistical Significance**: Cohen's d > 0.8 (large effect)
3. **✅ Better Than PFC**: Higher discriminative power
4. **✅ Interpretable Failures**: Can explain why motions fail

## Files Structure

```
eval/
├── eval_fcs.py          # Main FCS implementation
├── eval_pfc.py          # Original PFC for comparison
└── README_FCS.md        # This file
```

## Python API

```python
from eval.eval_fcs import ForceConsistencyEvaluator

# Initialize evaluator
evaluator = ForceConsistencyEvaluator(fps=30, body_mass=70.0)

# Evaluate single motion
result = evaluator.evaluate_motion(joint_positions)  # (S, J, 3)

print(f"FCS Score: {result['fcs_score']:.4f}")
print(f"Contact ratio: {result['contact_ratio']:.2%}")

# Access detailed data
com_positions = result['com_positions']  # (S, 3)
violations = result['per_frame_violations']  # (S-2,)
```

## Next Steps for Phase 2

### Week 3-4: Validation
1. **Prepare datasets**:
   ```bash
   # Get AIST++ ground truth motions
   # Generate EDGE samples
   ```

2. **Run comparison**:
   ```bash
   python eval/eval_fcs.py \
       --motion_path data/aist_test/ \
       --compare_path motions/edge_generated/ \
       --output results/phase2_validation/
   ```

3. **Analyze results**:
   - Check for clear separation (FCS_generated >> FCS_real)
   - Verify statistical significance (p < 0.01)
   - Compare to PFC discriminative power
   - Identify and explain failure cases

4. **Iterate if needed**:
   - Tune biomechanical parameters
   - Refine contact detection threshold
   - Adjust force distribution model

## References

- Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement*
- SMPL: A Skinned Multi-Person Linear Model (Loper et al., 2015)
- AIST++ Dataset (Li et al., 2021)

## Troubleshooting

**Issue**: FCS scores are all very low
- Check if motions are in meters (not centimeters)
- Verify fps parameter matches motion data
- Check body_mass parameter (default 70kg)

**Issue**: No clear separation between real/generated
- Increase max_samples for better statistics
- Check if generated data is actually different
- Review per-frame violations for patterns

**Issue**: High computational cost
- Use --max_samples to limit evaluation
- Process in batches
- Consider parallelization for large datasets
