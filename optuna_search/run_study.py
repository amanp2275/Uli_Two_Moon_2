"""Run resumable Optuna searches without modifying the normal experiment pipeline."""

import argparse
import gc
from pathlib import Path
import random
import sys

import numpy as np
import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import load_or_generate_two_moons
from optuna_search.reporting import export_results
from optuna_search.search_spaces import build_trial


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_trial(trial, args):
    """Train one sampled model and return its best validation NLL."""
    set_seed(args.seed)
    config, model = build_trial(
        args.model,
        trial,
        root=ROOT,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        conditional=args.conditional,
        evaluation_frequency=args.evaluation_frequency,
        early_stopping_patience=args.early_stopping_patience,
    )
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    splits = load_or_generate_two_moons(
        config.dataset_path,
        points_per_batch=config.points_per_batch,
        num_batches=config.dataset_num_batches,
        noise=config.noise,
        seed=config.seed,
    ).to(device)
    if splits.validation_X.size(0) == 0:
        raise ValueError("Optuna requires a non-empty validation split")

    mean = splits.train_X.mean(dim=(0, 1), keepdim=True)
    std = splits.train_X.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    train_x = (splits.train_X - mean) / std
    validation_x = (splits.validation_X - mean) / std
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    evaluation_step = 0

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = torch.randperm(train_x.size(0), device=device)
            for start in range(0, train_x.size(0), config.batch_size):
                indices = order[start:start + config.batch_size]
                labels = splits.train_labels[indices] if config.conditional else None
                optimizer.zero_grad(set_to_none=True)
                z, logdet = model(train_x[indices], labels)
                loss = model.get_loss(z, logdet)
                if not torch.isfinite(loss):
                    raise optuna.TrialPruned("non-finite training loss")
                loss.backward()
                optimizer.step()
                if hasattr(model, "update_prior"):
                    with torch.no_grad():
                        model.update_prior(z)

            if epoch % config.evaluation_frequency != 0 and epoch != config.epochs:
                continue
            model.eval()
            with torch.no_grad():
                labels = splits.validation_labels if config.conditional else None
                z, logdet = model(validation_x, labels)
                validation_loss = model.get_loss(z, logdet).item()
            if not np.isfinite(validation_loss):
                raise optuna.TrialPruned("non-finite validation loss")

            trial.report(validation_loss, evaluation_step)
            evaluation_step += 1
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
            else:
                stale += 1

            print(
                f"[trial {trial.number:03d}] epoch {epoch:03d}/{config.epochs} "
                f"validation_nll={validation_loss:.8f} best={best_loss:.8f}",
                flush=True,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()
            if stale >= config.early_stopping_patience:
                break

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("trainable_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
        return best_loss
    finally:
        del model, optimizer, splits, train_x, validation_x
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("real_nvp", "transformer"), required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200, help="maximum epochs per tuning trial")
    parser.add_argument("--timeout", type=int, default=None, help="optional total timeout in seconds")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None, help="for example cuda, cuda:0, or cpu")
    parser.add_argument("--conditional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluation-frequency", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--study-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials < 1 or args.epochs < 1 or args.evaluation_frequency < 1:
        raise SystemExit("n-trials, epochs, and evaluation-frequency must be positive")
    condition_name = "conditional" if args.conditional else "unconditional"
    output_dir = ROOT / "optuna_search" / "results" / args.model / condition_name
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"two_moons_{args.model}_{condition_name}"
    storage = f"sqlite:///{(output_dir / 'study.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    def objective(trial):
        return train_trial(trial, args)

    def save_after_trial(current_study, _trial):
        best_config = None
        completed = [t for t in current_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            set_seed(args.seed)
            best_config, _ = build_trial(
                args.model,
                optuna.trial.FixedTrial(current_study.best_trial.params),
                root=ROOT,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
                conditional=args.conditional,
                evaluation_frequency=args.evaluation_frequency,
                early_stopping_patience=args.early_stopping_patience,
            )
        export_results(current_study, output_dir, args.model, best_config)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[save_after_trial])
    save_after_trial(study, None)
    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        print(f"Best validation NLL: {study.best_value:.8f}")
        print(f"Best parameters: {study.best_params}")
        print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
