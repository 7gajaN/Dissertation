# Training Metrics Documentation

## Overview

The EDGE training pipeline now automatically saves detailed metrics to files during training. This allows you to:
- Track training progress over time
- Compare different experiments
- Create custom visualizations
- Analyze training dynamics

## Files Generated

When you run training, two metric files are created in your experiment directory:

```
runs/fcs_experiment/
├── weights/
│   ├── train-50.pt
│   ├── train-100.pt
│   └── ...
├── training_metrics.csv       ← CSV format (spreadsheet-friendly)
└── training_metrics.json      ← JSON format (complete data)
```

## File Formats

### 1. CSV File (`training_metrics.csv`)

**Format**: Comma-separated values, easy to open in Excel/Google Sheets

**Columns**:
- `Epoch`: Training epoch number
- `Timestamp`: When the metric was recorded
- `Total_Loss`: Sum of all loss components
- `Train_Loss`: Reconstruction loss
- `V_Loss`: Velocity loss
- `FK_Loss`: Forward kinematics loss
- `Foot_Loss`: Foot skating loss
- `FCS_Score`: Force Consistency Score (physics quality)
- `Type`: Either "progress" (50-epoch) or "checkpoint" (save interval)

**Example**:
```csv
Epoch,Timestamp,Total_Loss,Train_Loss,V_Loss,FK_Loss,Foot_Loss,FCS_Score,Type
50,2026-02-24 19:30:45,0.012345,0.006123,0.003162,0.002530,0.000530,N/A,progress
100,2026-02-24 19:45:30,0.010234,0.005234,0.002987,0.001713,0.000300,0.823400,checkpoint
```

### 2. JSON File (`training_metrics.json`)

**Format**: Structured JSON with metadata and all metrics

**Structure**:
```json
{
  "experiment": "fcs_experiment",
  "total_epochs": 2000,
  "batch_size": 128,
  "feature_type": "jukebox",
  "completed": "2026-02-24 23:15:00",
  "metrics": [
    {
      "epoch": 50,
      "timestamp": "2026-02-24 19:30:45",
      "total_loss": 0.012345,
      "train_loss": 0.006123,
      "v_loss": 0.003162,
      "fk_loss": 0.00253,
      "foot_loss": 0.00053,
      "fcs_score": null,
      "type": "progress"
    },
    ...
  ]
}
```

## When Metrics Are Saved

| Event | Frequency | FCS Score | Type |
|-------|-----------|-----------|------|
| **Progress Update** | Every 50 epochs | ❌ No | `progress` |
| **Checkpoint** | Every `--save_interval` epochs | ✅ Yes | `checkpoint` |

## Using the Metrics

### 1. View in Spreadsheet

Open `training_metrics.csv` in Excel or Google Sheets:

```bash
# On Windows
start runs/fcs_experiment/training_metrics.csv

# On Linux
open runs/fcs_experiment/training_metrics.csv
```

Create custom charts, filter data, compare experiments.

### 2. Plot with Provided Script

Use the included `plot_metrics.py` script:

```bash
# Plot from JSON
python plot_metrics.py --json runs/fcs_experiment/training_metrics.json

# Plot from CSV
python plot_metrics.py --csv runs/fcs_experiment/training_metrics.csv

# Save plot instead of showing
python plot_metrics.py --json runs/fcs_experiment/training_metrics.json --output runs/fcs_experiment/

# Print summary only (no plot)
python plot_metrics.py --json runs/fcs_experiment/training_metrics.json --summary
```

**Generated Plots**:
- Total loss over time
- Individual loss components
- FCS score progression
- Loss distribution pie chart

### 3. Custom Analysis with Python

```python
import json
import pandas as pd

# Load JSON
with open('runs/fcs_experiment/training_metrics.json') as f:
    data = json.load(f)

metrics = data['metrics']
checkpoint_metrics = [m for m in metrics if m['type'] == 'checkpoint']

# Convert to DataFrame
df = pd.DataFrame(checkpoint_metrics)

# Analysis
print(f"Mean FCS: {df['fcs_score'].mean():.4f}")
print(f"Final Loss: {df['total_loss'].iloc[-1]:.6f}")
print(f"Loss reduction: {(df['total_loss'].iloc[0] - df['total_loss'].iloc[-1]):.6f}")

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['fcs_score'])
plt.xlabel('Epoch')
plt.ylabel('FCS Score')
plt.title('Physics Quality Over Training')
plt.show()
```

### 4. Load CSV in Python

```python
import pandas as pd

# Load CSV
df = pd.read_csv('runs/fcs_experiment/training_metrics.csv')

# Filter checkpoints only
checkpoints = df[df['Type'] == 'checkpoint']

# Plot all losses
checkpoints.plot(x='Epoch', y=['Train_Loss', 'V_Loss', 'FK_Loss', 'Foot_Loss'])
```

## Comparing Multiple Experiments

```python
import json
import matplotlib.pyplot as plt

# Load multiple experiments
exp1 = json.load(open('runs/exp1/training_metrics.json'))
exp2 = json.load(open('runs/exp2/training_metrics.json'))

# Extract FCS scores
fcs1 = [(m['epoch'], m['fcs_score']) for m in exp1['metrics'] if m['fcs_score']]
fcs2 = [(m['epoch'], m['fcs_score']) for m in exp2['metrics'] if m['fcs_score']]

# Plot comparison
plt.plot(*zip(*fcs1), label='Experiment 1')
plt.plot(*zip(*fcs2), label='Experiment 2')
plt.xlabel('Epoch')
plt.ylabel('FCS Score')
plt.legend()
plt.title('FCS Comparison')
plt.show()
```

## What to Monitor

### 1. **Total Loss** (should decrease)
- Indicates overall model improvement
- Target: Consistent decrease with plateau near end

### 2. **FCS Score** (should decrease)
- Lower = better physics quality
- **Good**: < 0.5 (similar to real data)
- **Acceptable**: 0.5 - 1.0
- **Poor**: > 1.5

### 3. **Loss Components Balance**
- Check if one component dominates (pie chart)
- Ideally all components contribute to improvement
- Sudden spikes may indicate instability

### 4. **Training Stability**
- Smooth curves = stable training
- Oscillations = may need learning rate adjustment
- Divergence = training failed, restart needed

## Troubleshooting

**Issue**: CSV file is empty or has only headers
- Check if training reached at least epoch 50
- Ensure `--save_interval` is set

**Issue**: FCS scores are all 0 or N/A
- FCS evaluator failed to initialize
- Check console output for error messages
- Verify numpy/tqdm are installed: `pip install numpy tqdm`

**Issue**: Plot script fails
- Install matplotlib: `pip install matplotlib pandas`
- Use `--summary` flag to skip plotting

**Issue**: Want more frequent metrics
- Modify line 335 in EDGE.py: change `epoch % 50` to `epoch % 25`
- More frequent = larger files but better resolution

## Tips

1. **Archive metrics**: Copy metrics files after each experiment for comparison
2. **Git ignore**: Metrics files can be large, consider adding to `.gitignore`
3. **Long training**: Metrics are saved incrementally, safe to check during training
4. **Disk space**: JSON file grows with training duration (~1KB per checkpoint)

## Example Workflow

```bash
# Start training
accelerate launch train.py --batch_size 128 --epochs 2000 --feature_type jukebox --exp_name fcs_v1

# During training, monitor progress
tail -f runs/fcs_v1/training_metrics.csv

# After training, create plots
python plot_metrics.py --json runs/fcs_v1/training_metrics.json --output runs/fcs_v1/

# Compare with baseline
python plot_metrics.py --json runs/baseline/training_metrics.json
python plot_metrics.py --json runs/fcs_v1/training_metrics.json

# Export for paper/presentation
# Open training_metrics_plot.png in runs/fcs_v1/
```
