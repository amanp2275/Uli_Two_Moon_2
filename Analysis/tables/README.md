# Organized experiment tables

Each CSV in this folder is one focused table derived from the complete results file at [`../experiment_results.csv`](../experiment_results.csv). The original columns are not modified; these files select only the parameters and metrics needed for each experiment group.

| File | Experiment group | Rows |
|---|---|---:|
| `learning_rate_sweep.csv` | Learning-rate comparison for both models | 8 |
| `weight_decay_sweep.csv` | Weight-decay comparison for both models | 8 |
| `realnvp_architecture_sweep.csv` | RealNVP depth and hidden-width experiments | 7 |
| `transformer_num_blocks_sweep.csv` | Transformer block-count experiments | 8 |
| `transformer_head_dimension_sweep.csv` | Transformer head-dimension experiments | 6 |
| `transformer_expansion_sweep.csv` | Transformer expansion-factor experiments | 7 |

Each table includes the relevant hyperparameters, run status, best epoch, validation loss, test loss, NLL, training time, and error information where available.

CSV files support one table per file. This folder therefore uses separate CSV files for separate headings while keeping `../experiment_results.csv` as the complete source dataset.
