# Transformer NLL-fix comparison

Selected experiments: EXP_041, EXP_043, EXP_047, EXP_048, EXP_049, and EXP_050.
The figure uses the saved loss history and saved final sample panels from each run.

## Cross-check

The original Transformer loss formed `0.5 * z.pow(2).sum(dim=-1)` and then added the Gaussian constant once per point. Because the latent has two coordinates, the constant should be applied to both coordinates. The corrected implementation works elementwise, sums over dimensions `[1, 2]`, and divides by `z.size(1) * z.size(2)`.

The repository history shows that EXP_047/048 used the corrected coordinate-wise NLL with a unit-variance loss, but their trainer still called `model.reverse(...)` without the `latent_var` override; therefore generation used the learned `self.var`. EXP_049/050 changed the loss to use the learned per-position, per-coordinate `self.var` as well. So the claim that EXP_047/048 generated with fixed variance 0.1 or 1 is not supported by the recorded code commit.

## Results

| Experiment | Final test NLL | Best epoch | Parameters | NLL implementation |
|---|---:|---:|---:|---|
| EXP_041 | 0.161210 | 600 | 495,384 | Old NLL: Gaussian constant was added once per point, not once per coordinate. Loss ignored variance in the prior; generation used self.var. |
| EXP_043 | 0.184351 | 600 | 495,384 | Old NLL: Gaussian constant was added once per point, not once per coordinate. Loss ignored variance in the prior; generation used self.var. |
| EXP_047 | 0.541893 | 1000 | 495,384 | Coordinate-wise NLL fix: sums [sequence, coordinate] and normalizes by T*C. Loss uses unit variance; generation still uses self.var. |
| EXP_048 | 0.562462 | 980 | 495,384 | Coordinate-wise NLL fix: sums [sequence, coordinate] and normalizes by T*C. Loss uses unit variance; generation still uses self.var. |
| EXP_049 | 0.508822 | 960 | 495,384 | Coordinate-wise NLL fix plus learned self.var in the loss and generation (per-position, per-coordinate prior variance). |
| EXP_050 | 0.522751 | 960 | 495,384 | Coordinate-wise NLL fix plus learned self.var in the loss and generation (per-position, per-coordinate prior variance). |

Raw NLL values across these groups should be interpreted with care: EXP_041/043 use the under-normalized old objective, EXP_047/048 use the corrected unit-variance objective, and EXP_049/050 use the corrected learned-variance objective.

Source artifacts are linked in `experiment_summary.csv`; the registry also records the Git commit used for each run.
