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

Add experiments to `experiments/experiment_config.json`. Each entry is run once
with the parameters written in that file:

```json
{
  "experiments": [
    {
      "experiment_id": "EXP_001",
      "experiment_type": "parameter_sweep",
      "model": "real_nvp",
      "parameters": {
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 100,
        "num_layers": 6,
        "hidden_features": 64
      },
      "seed": 42
    }
  ]
}
```

Run one entry with `python experiments/run_experiment.py --id EXP_001`, or
run all entries explicitly listed in the file with `--all`. Completed IDs are
skipped to prevent accidental reruns. Per-experiment artifacts are stored in
`results/<experiment_id>/`, and the persistent summary table is
`results/experiments.csv`.
