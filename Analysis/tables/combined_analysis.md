# Combined experiment analysis

The complete combined data table is available as [`all_experiment_tables.csv`](all_experiment_tables.csv). It contains one row per experiment and uses `experiment_group` to identify the relevant heading.

## Learning-rate sweep

Rows are identified by `experiment_group = Learning-rate sweep`.

## Weight-decay sweep

Rows are identified by `experiment_group = Weight-decay sweep`.

## RealNVP architecture sweep

Rows are identified by `experiment_group = RealNVP architecture sweep`.

## Transformer block-count sweep

Rows are identified by `experiment_group = Transformer block-count sweep`.

## Transformer head-dimension sweep

Rows are identified by `experiment_group = Transformer head-dimension sweep`.

## Transformer expansion sweep

Rows are identified by `experiment_group = Transformer expansion sweep`.

The combined CSV keeps all necessary hyperparameter and result columns in one table. Empty cells indicate parameters that do not apply to a particular experiment group.
