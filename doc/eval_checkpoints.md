# Checkpoint Evaluation Script

## Purpose

`eval_checkpoints.py` evaluates FCS and PFC metrics on saved training checkpoints, **independently from the training loop**. This allows:

- Evaluating with more samples (50–200) for statistically reliable results vs the 4 samples used during training
- Running evaluation after training completes on all checkpoints at once
- Re-evaluating checkpoints without restarting training
- Producing a comparison table across all epochs

## Usage

```bash
# Evaluate all checkpoints in a run (50 samples each)
python eval_checkpoints.py --run_dir runs/baseline/no_fcs --num_samples 50

# Single checkpoint
python eval_checkpoints.py --checkpoint runs/baseline/no_fcs/weights/train-500.pt --num_samples 50

# Paper-ready evaluation (more samples for tighter confidence intervals)
python eval_checkpoints.py --run_dir runs/baseline/no_fcs --num_samples 200
```

## How It Works

1. Loads the test dataset once
2. For each checkpoint:
   - Loads model weights (EMA)
   - Generates N dance samples conditioned on random test music
   - Computes FCS and PFC on each generated sample
   - Reports mean, std, median
   - Frees GPU memory before next checkpoint
3. Prints summary table and saves results to JSON + CSV

## Output

### Console Table
```
 Epoch |   FCS Mean    FCS Std    FCS Med |   PFC Mean    PFC Std    PFC Med |    N
--------------------------------------------------------------------------------
   100 |     0.4521     0.2103     0.3891 |     8.2341     4.1230     6.9821 |   50
   200 |     0.0891     0.0412     0.0723 |     0.4231     0.2100     0.3521 |   50
   ...
```

### Files
- `eval_results.json` — full results with per-checkpoint statistics
- `eval_results.csv` — tabular format for plotting

## Sample Size Guidance

- **4 samples** (current training default): Too noisy for reliable comparison. FCS can swing 10x between checkpoints.
- **50 samples**: Good for tracking trends during development. Standard error ~14% of std.
- **200 samples**: Suitable for paper results. Standard error ~7% of std.
- **Full test set (186 clips)**: Most reliable, recommended for final evaluation.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--run_dir` | | Run directory containing `weights/` |
| `--checkpoint` | | Single checkpoint path |
| `--num_samples` | 50 | Samples to generate per checkpoint |
| `--feature_type` | jukebox | Audio feature type |
| `--data_path` | data/ | Dataset path |
| `--output` | `{run_dir}/eval_results.json` | Output file |
