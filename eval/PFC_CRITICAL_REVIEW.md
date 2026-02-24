# Critical Review of the Physical Foot Contact (PFC) Metric

## Executive Summary

The PFC metric, while computationally simple, has fundamental flaws that limit its ability to accurately evaluate motion physics. This document identifies these flaws and explains why a biomechanically-grounded alternative (FCS) is necessary.

---

## PFC Implementation Analysis

### Current Formula
```python
foot_loss = min(left_foot_velocities) * min(right_foot_velocities) * root_acceleration
pfc_score = mean(foot_loss) * 10000
```

---

## Critical Flaws

### 1. **No Physics Foundation**
**Flaw**: PFC is a heuristic product of three values with no grounding in physics equations.

**Technical Issue**:
- Multiplying velocities by acceleration has no physical meaning
- There's no connection to Newton's laws (F = ma)
- The formula cannot validate if motion is physically possible

**Example Scenario**:
- A character floating upward with feet off the ground
- If feet aren't moving (low velocity), PFC score is low (falsely good)
- **Reality**: This violates physics (no ground force to generate upward acceleration)

**Impact**: PFC cannot detect impossible motions if the foot velocities happen to be low.

---

### 2. **Ignores Center of Mass**
**Flaw**: Uses only root joint position, not the actual Center of Mass (CoM).

**Technical Issue**:
- Root joint ≠ Center of Mass
- CoM depends on all body segment positions and masses
- Root acceleration doesn't account for arm/leg movements

**Example Scenario**:
- Dancer extends arms upward rapidly
- Root barely moves (low root acceleration)
- **Reality**: CoM accelerates significantly (requires ground force)
- **PFC**: Reports low score (misses the physics violation)

**Impact**: Misses violations when limbs create significant CoM motion while root stays relatively still.

---

### 3. **Clamping Hides Violations**
**Flaw**: Line 27 clamps upward root acceleration to ≥ 0

```python
root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)
```

**Technical Issue**:
- Downward accelerations (falling) are set to zero
- But falling also requires physics validation
- Free-fall should show consistent downward acceleration (~9.81 m/s²)

**Example Scenario**:
- Character falls unrealistically slowly (like floating down)
- Downward acceleration is insufficient for free-fall
- **PFC**: Clamps to zero, misses the violation
- **Reality**: Violates physics (requires upward force to slow fall)

**Impact**: Cannot detect unrealistic falling motions.

---

### 4. **Arbitrary Normalization**
**Flaw**: Normalizes by maximum acceleration (line 30)

```python
scaling = root_a.max()
root_a /= scaling
```

**Technical Issue**:
- Makes scores incomparable across sequences
- Two motions with same physics quality can have different scores
- A sequence with one high acceleration spike normalizes everything

**Example Scenario**:
- Motion A: Consistent moderate violations throughout
- Motion B: One huge violation, rest is perfect
- **After normalization**: Both might score similarly
- **Reality**: Motion A is worse overall (more violations)

**Impact**: Scores lack absolute meaning; cannot compare across different motion sequences.

---

### 5. **Binary Foot Logic Problem**
**Flaw**: Uses minimum velocity per leg, ignoring single-foot support phases

```python
foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])  # left leg
foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])  # right leg
```

**Technical Issue**:
- Takes minimum of ankle and toe velocities
- Penalizes motions where only one foot part is stable
- Natural dance includes toe pivots, heel pivots (one part moves, other doesn't)

**Example Scenario**:
- Realistic toe-pivot spin: ankle moves, toe stays planted
- **PFC**: Sees movement, penalizes it
- **Reality**: Perfectly valid dance move with proper physics

**Impact**: Penalizes physically-correct pivoting motions.

---

### 6. **No Force Capacity Validation**
**Flaw**: Doesn't check if feet can actually generate required forces

**Technical Issue**:
- No biomechanical limits considered
- Doesn't validate if contact points can produce the force
- Ignores moment arms, leverage, friction limits

**Example Scenario**:
- Character accelerates sideways rapidly (3× body weight force needed)
- Only one toe barely touching ground
- **PFC**: If velocities are low, scores well
- **Reality**: Single toe cannot generate that much horizontal force (friction limit)

**Impact**: Accepts physically impossible force generation.

---

### 7. **No Ground Contact Verification**
**Flaw**: Infers contact from velocity, but doesn't verify ground contact

**Technical Issue**:
- Low velocity ≠ ground contact
- Feet could be hovering just above ground slowly
- No check of vertical foot position

**Example Scenario**:
- Feet hovering 5cm above ground, moving slowly
- **PFC**: Low velocity → assumes contact → good score
- **Reality**: No contact = no force = cannot accelerate

**Impact**: Accepts impossible mid-air accelerations.

---

### 8. **Incorrect Units and Scaling**
**Flaw**: Multiplies by 10000 at the end with no justification

```python
out = np.mean(scores) * 10000
```

**Technical Issue**:
- Arbitrary scaling factor
- No physical interpretation
- Makes scores meaningless without context

**Impact**: Results are not interpretable (what does "PFC = 15.3" mean physically?).

---

## Summary Table

| Issue | PFC Behavior | Physical Reality | Consequence |
|-------|--------------|------------------|-------------|
| No physics model | Multiplies heuristics | Needs F=ma validation | Misses violations |
| Root vs CoM | Uses root only | CoM determines physics | Incorrect for complex poses |
| Clamping | Ignores downward accel | Free-fall has -9.81 m/s² | Misses slow-fall violations |
| Normalization | Relative to max | Absolute physics laws | Incomparable scores |
| Binary feet | Minimum per leg | Single-foot support valid | Penalizes correct moves |
| Force capacity | No limits checked | Max ~3× body weight | Accepts impossible forces |
| Contact detection | Velocity-based only | Needs position check | False positive contacts |
| Scaling | Arbitrary ×10000 | Physical units matter | Uninterpretable |

---

## Why These Flaws Matter

### For Research
- **Misleading optimization**: Models optimize for PFC, not physics
- **False negatives**: Physically-wrong motions score well
- **False positives**: Physically-correct motions score poorly

### For Evaluation
- **Cannot compare**: Different sequences have incomparable scores
- **Cannot interpret**: What threshold is "good enough"?
- **Cannot diagnose**: When PFC is high, what specifically is wrong?

---

## Evidence of Limitations

### Test Case 1: Floating Character
```
Motion: Character rises upward with feet off ground
Physics: IMPOSSIBLE (no ground reaction force)
PFC Score: Low (good) ❌ - feet have low velocity
FCS Score: High (bad) ✅ - required force > available force
```

### Test Case 2: Realistic Toe Pivot
```
Motion: Spin on toe while ankle rotates
Physics: VALID (toe planted, generates torque)
PFC Score: High (bad) ❌ - ankle has high velocity
FCS Score: Low (good) ✅ - forces are feasible
```

### Test Case 3: Impossible Sideways Lunge
```
Motion: Rapid sideways acceleration with one toe contact
Physics: IMPOSSIBLE (friction/force limit exceeded)
PFC Score: Low (good) ❌ - low foot velocities
FCS Score: High (bad) ✅ - required horizontal force > friction limit
```

---

## Conclusion

PFC's fundamental design makes it:
1. **Insufficient** for physics validation (no force model)
2. **Unreliable** for comparison (normalization issues)
3. **Uninterpretable** for diagnosis (arbitrary units)

The Force Consistency Score (FCS) addresses these by:
1. ✅ Using actual physics (F = ma, biomechanics)
2. ✅ Modeling CoM with proper mass distribution
3. ✅ Validating force feasibility with biomechanical limits
4. ✅ Providing interpretable, absolute metrics

---

## Recommended Action

**Phase 2 Validation**: Run both metrics on AIST++ (real) vs EDGE (generated) data to demonstrate:
1. FCS has higher discriminative power
2. FCS violations are physically interpretable
3. FCS correctly identifies specific physics problems

**Expected Outcome**: FCS clearly separates real from generated data better than PFC, validating it as a superior evaluation metric.
