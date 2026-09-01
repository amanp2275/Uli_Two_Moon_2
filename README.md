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
