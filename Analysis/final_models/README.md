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

## All training runs

For a single browsable view of every completed training run, open
[`../all_training_overview.html`](../all_training_overview.html). It includes
each run's final configuration, final metrics, parameter plot, and loss curves.
The combined loss-curve image is [`../all_training_loss_curves.png`](../all_training_loss_curves.png),
and the machine-readable table is [`../all_training_parameters.csv`](../all_training_parameters.csv).

Regenerate these files after adding new results with:

```bash
python Analysis/generate_training_overview.py
```

## Organized model plots

The model-specific plots are separated by model and sweep type:

- Parameter sweeps: [Transformer loss curves](../organized_plots/parameter_sweep/transformer_loss_curves_by_experiment_type.png), [RealNVP loss curves](../organized_plots/parameter_sweep/real_nvp_loss_curves_by_experiment_type.png)
- Model sweeps: [Transformer loss curves](../organized_plots/model_sweep/transformer_loss_curves_by_experiment_type.png), [RealNVP loss curves](../organized_plots/model_sweep/real_nvp_loss_curves_by_experiment_type.png)
- Matching final-parameter summaries are in the same folders, named `*_parameters_by_experiment_type.png`.

Regenerate them with:

```bash
python Analysis/generate_organized_model_plots.py
```
