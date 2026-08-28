# Uli Two Moon Flows

Experiments with normalizing flows on the two-moons dataset.

## Project structure

- `dataset.py`: two-moons data generation and train/validation/test splitting.
- `configuration.py`: shared training configurations.
- `Transformer_flow.py`: transformer-based flow model components.
- `train.py`: shared transformer-flow training and plotting utilities.
- `real_nvp.py`: RealNVP model and training implementation.
- `transformer_flows/`: conditional and unconditional transformer-flow launchers.
- `real_nvp/`: conditional and unconditional RealNVP launchers.

Training uses one persisted dataset file with disjoint train, validation, and
test splits. The validation split drives early stopping; checkpoint plots
include both train and held-out test loss curves.

## Run training

From the project root:

```powershell
python transformer_flows/train_conditional.py
python transformer_flows/train_unconditional.py
python real_nvp/train_conditional.py
python real_nvp/train_unconditional.py
```

Generated plots are intentionally ignored by Git.
