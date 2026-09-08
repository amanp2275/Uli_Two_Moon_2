# Experiment MS-1: model sweep

This folder contains the comparison of the two finalized model configurations:

- `final_models/real_nvp` — finalized RealNVP configuration from EXP_024.
- `final_models/transformer` — finalized Transformer configuration from EXP_032.

The tuned comparison runner writes future outputs directly here. It does not create the `02_diff_params_same_results` grouping because this experiment is a model sweep, not a same-result parameter grouping.
