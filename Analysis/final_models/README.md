# Final model results

This folder contains the selected final configurations and their key graphs.

| Model | Experiment | Configuration | NLL |
|---|---|---|---:|
| RealNVP | EXP_024 | 9 layers, 256 hidden features, learning rate 0.001, weight decay 0 | 0.2197 |
| Transformer | EXP_032 | 6 blocks, head dimension 64, expansion 2, learning rate 0.001, weight decay 0.0001 | 0.0747 |

Each model folder contains:

- `loss_curves.png` — training and validation behavior
- `samples_final.png` — final generated samples
- `final_summary.png` — final result summary
- `parameters.png` — model parameter visualization

The selected metrics are in [`final_models_metrics.csv`](final_models_metrics.csv).

The full experiment comparisons are available as separate, cleaner charts:

- [`transformer_nll_by_experiment.png`](transformer_nll_by_experiment.png)
- [`realnvp_nll_by_experiment.png`](realnvp_nll_by_experiment.png)
