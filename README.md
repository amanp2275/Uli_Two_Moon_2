# Uli Two Moon Flows

Experiments with normalizing flows on the two-moons dataset.

The repository compares RealNVP and a Transformer-based flow in individual,
same-protocol, and model-specific tuned experiments.

## Project structure

- `dataset.py`: two-moons data generation and train/validation/test splitting.
- `dataset.py`: shared persisted dataset and split handling.
- `configs/`: shared training protocol and model-specific configurations.
- `models/`: model implementations with a common flow interface.
- `training/`: shared training, evaluation, plotting, and metric export.
- `experiments/`: individual and comparison experiment entry points.
- `results/`: outputs from the new experiment pipeline.

Training uses one persisted dataset file with disjoint train, validation, and
test splits. The validation split drives early stopping; checkpoint plots
include both train and held-out test loss curves.

## Run training

New experiment pipeline:

```powershell
python experiments/run_individual.py --model real_nvp
python experiments/run_individual.py --model transformer
python experiments/compare_same_params.py
python experiments/compare_tuned_params.py
```

Add `--unconditional` to the individual runner for an unconditional run.

Generated plots are intentionally ignored by Git. 

## Manual experiment runner

Edit only `experiments/experiment_queue.csv`. Each row is one experiment and
rows with `run=YES` are executed by:

```powershell
python experiments/run_experiment.py
```

To run one row explicitly, use `python experiments/run_experiment.py --id
EXP_001`. Completed IDs are skipped. Use `--force` only for an intentional
rerun; the previous output row and artifacts are preserved under a separate
run ID.

The output table is `results/experiment_results.csv`. It contains the queue
parameters, existing training/evaluation metrics, timing, Git commit, and
paths to the saved configuration, plots, and checkpoint. Experiment artifacts
are organized as `results/<experiment_type>/<experiment_id>/run_<number>/`.

## Temporary Telegram notification setup

> Cleanup prompt: Once this project is finished, remove all temporary Telegram
> notification code and configuration without changing the original experiment
> behavior. Remove `.github/hooks/telegram.json`, `.github/hooks/telegram.ps1`,
> `.vscode/tasks.json`, and `scripts/run_with_telegram.ps1`. Remove the Telegram
> notification imports, helper functions, and notification calls from
> `experiments/run_individual.py` and `experiments/run_experiment.py`. Also
> remove the Windows user environment variables `TELEGRAM_BOT_TOKEN` and
> `TELEGRAM_CHAT_ID`. Remove this temporary README section after cleanup.
