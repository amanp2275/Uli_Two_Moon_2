from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
import random

import numpy as np
import torch

from dataset import load_or_generate_two_moons
from .plotting import (
    save_loss_curves,
    save_metrics,
    save_parameter_table,
    save_samples_plot,
    save_summary_table,
    save_training_plot,
)
from .storage import config_dict, prepare_run_directory, raw_run_path, save_config


class TrainingResult(dict):
    """Dictionary-like result returned by the shared trainer."""


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model: torch.nn.Module, config, model_name: str) -> TrainingResult:
    _set_seed(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    splits = load_or_generate_two_moons(
        config.dataset_path,
        points_per_batch=config.points_per_batch,
        num_batches=config.dataset_num_batches,
        noise=config.noise,
        seed=config.seed,
    ).to(device)
    mean = splits.train_X.mean(dim=(0, 1), keepdim=True)
    std = splits.train_X.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    train_x = (splits.train_X - mean) / std
    validation_x = (splits.validation_X - mean) / std
    test_x = (splits.test_X - mean) / std
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir = prepare_run_directory(raw_run_path(config.output_dir, model_name, config))
    save_config(config, output_dir / "config.json")
    plot_parameters = {"model": model_name}
    if is_dataclass(config):
        plot_parameters.update(asdict(config))
    plot_parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in plot_parameters.items()
    }
    plot_parameters["total_trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    save_parameter_table(plot_parameters, output_dir / "parameters.png")
    train_losses, validation_losses, test_losses, eval_epochs = [], [], [], []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(train_x.size(0), device=device)
        batch_losses = []
        for start in range(0, train_x.size(0), config.batch_size):
            indices = order[start:start + config.batch_size]
            labels = splits.train_labels[indices] if config.conditional else None
            optimizer.zero_grad()
            z, logdet = model(train_x[indices], labels)
            loss = model.get_loss(z, logdet)
            loss.backward()
            optimizer.step()
            if hasattr(model, "update_prior"):
                with torch.no_grad():
                    model.update_prior(z)
            batch_losses.append(loss.item())
        train_losses.append(sum(batch_losses) / len(batch_losses))

        if epoch % config.evaluation_frequency != 0 and epoch != config.epochs:
            continue
        model.eval()
        with torch.no_grad():
            validation_labels = splits.validation_labels if config.conditional else None
            test_labels = splits.test_labels if config.conditional else None
            validation_z, validation_logdet = model(validation_x, validation_labels)
            test_z, test_logdet = model(test_x, test_labels)
            validation_loss = model.get_loss(validation_z, validation_logdet).item()
            test_loss = model.get_loss(test_z, test_logdet).item()
            generated_labels = test_labels[:config.plot_batches] if config.conditional else None
            generated = model.reverse(torch.randn_like(test_x[:config.plot_batches]), generated_labels)
            generated = generated * std + mean
        validation_losses.append(validation_loss)
        test_losses.append(test_loss)
        eval_epochs.append(epoch)
        print(
            f"[{model_name}] checkpoint epoch {epoch:03d}/{config.epochs}: "
            f"train_loss={train_losses[-1]:.6f}, "
            f"validation_loss={validation_loss:.6f}, "
            f"test_loss={test_loss:.6f}",
            flush=True,
        )
        save_training_plot(
            test_x[:config.plot_batches] * std + mean,
            splits.test_labels[:config.plot_batches], generated, generated_labels,
            train_losses, validation_losses, test_losses, eval_epochs,
            output_dir / f"epoch_{epoch:03d}.png",
        )
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_train_z, final_train_logdet = model(train_x, splits.train_labels if config.conditional else None)
        final_validation_z, final_validation_logdet = model(
            validation_x, splits.validation_labels if config.conditional else None
        )
        final_test_z, final_test_logdet = model(test_x, splits.test_labels if config.conditional else None)
        final_train_loss = model.get_loss(final_train_z, final_train_logdet).item()
        final_validation_loss = model.get_loss(final_validation_z, final_validation_logdet).item()
        final_test_loss = model.get_loss(final_test_z, final_test_logdet).item()
        final_generated_labels = splits.test_labels[:config.plot_batches] if config.conditional else None
        final_generated = model.reverse(torch.randn_like(test_x[:config.plot_batches]), final_generated_labels)
        final_generated = final_generated * std + mean
    save_samples_plot(
        splits.test_X[:config.plot_batches],
        splits.test_labels[:config.plot_batches],
        final_generated,
        final_generated_labels,
        output_dir / "samples_final.png",
    )
    save_loss_curves(
        train_losses,
        validation_losses,
        test_losses,
        eval_epochs,
        output_dir / "loss_curves.png",
    )
    final_summary = {
        "model": model_name,
        "train_loss": f"{final_train_loss:.6f}",
        "validation_loss": f"{final_validation_loss:.6f}",
        "test_loss": f"{final_test_loss:.6f}",
        "trainable_parameters": plot_parameters["total_trainable_parameters"],
    }
    save_summary_table(final_summary, output_dir / "final_summary.png")
    print(
        f"{model_name}: train_loss={final_train_loss:.6f}, "
        f"validation_loss={final_validation_loss:.6f}, test_loss={final_test_loss:.6f}, "
        f"trainable_parameters={plot_parameters['total_trainable_parameters']}"
    )
    metrics = {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test_loss_at_last_evaluation": test_losses[-1] if test_losses else None,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
        "final_test_loss": final_test_loss,
        "total_trainable_parameters": plot_parameters["total_trainable_parameters"],
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "test_losses": test_losses,
        "evaluation_epochs": eval_epochs,
        "config": config_dict(config),
    }
    # Keep this schema stable: final_test_loss is the canonical Test NLL.
    save_metrics(metrics, output_dir / "metrics.json")
    torch.save({"model_state_dict": model.state_dict(), "mean": mean.cpu(), "std": std.cpu()}, output_dir / "best_model.pt")
    return TrainingResult(metrics)
