# Model-sweep queue

Edit [`model_sweep_queue.csv`](model_sweep_queue.csv) to define paired comparisons of the finalized RealNVP and Transformer models.

- Set `run` to `YES` for rows that should run; leave it as `NO` to skip them.
- Each row represents one seed and trains both models.
- Change the shared training fields or the model-specific hyperparameters directly in the CSV.
- Results are saved under `results/model_comparison/<experiment_id>`.
- Summary metrics are appended to `results/model_comparison/model_sweep_results.csv`.

Run all enabled rows:

```powershell
python experiments\run_model_sweep.py
```

Run one row explicitly:

```powershell
python experiments\run_model_sweep.py --id EXP_MS_002
```

Use `--force` to intentionally rerun a completed row.
