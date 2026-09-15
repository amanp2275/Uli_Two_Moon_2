# Optuna hyperparameter search

This folder is independent of the existing manual experiment pipeline. It
uses the same persisted two-moons dataset and model implementations, but it
chooses parameters from validation NLL only. The test split is not read during
the search.

## Install

From the repository root:

```powershell
python -m pip install -r optuna_search/requirements.txt
```

## Run

Start with a small RealNVP check:

```powershell
python -m optuna_search.run_study --model real_nvp --n-trials 3 --epochs 40
```

Then run the useful search:

```powershell
python -m optuna_search.run_study --model real_nvp --n-trials 30 --epochs 200
```

Transformer search is supported too:

```powershell
python -m optuna_search.run_study --model transformer --n-trials 30 --epochs 200
```

Add `--device cuda` to require the GPU, or `--device cpu` to require CPU. An
interrupted command can be run again: the study resumes from its SQLite file.

## Results

Each model writes to
`optuna_search/results/<model>/<conditional-or-unconditional>/`:

- `best_parameters.json`: best parameters, validation NLL, and full resolved config.
- `trials.csv`: every completed, pruned, and failed trial.
- `manual_cross_check.csv`: one row matching `experiments/experiment_queue.csv`.
- `study.db`: resumable Optuna study.
- `optimization_history.png`, `parameter_importance.png`, and
  `parameter_slices.png`: diagnostics (when enough trials exist).

## Manual cross-check

After the search, open `manual_cross_check.csv`. Copy its single data row into
`experiments/experiment_queue.csv`, change its `experiment_id` if necessary,
and set `run=YES`. Then use the normal runner:

```powershell
python experiments/run_experiment.py --id OPTUNA_REAL_NVP_CHECK
```

The exported row defaults to `run=NO` so an expensive verification run never
starts accidentally. For the final comparison, increase `epochs` in that row
if you want to retrain the winning parameters with your full training budget.
