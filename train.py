from pathlib import Path
from datetime import datetime
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons

from configuration import CONDITIONAL_CONFIG, TrainingConfig
from dataset import load_or_generate_two_moons
from Transformer_flow import Model


def get_two_moons(
	points_per_batch: int = 200,
	num_batches: int = 64,
	noise: float = 0.05,
	seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Generate two-moon points and labels from scikit-learn.

	Returns ``X`` with shape ``(B, T, C)`` and ``y`` with shape ``(B, T)``.
	``B`` is ``num_batches``, ``T`` is ``points_per_batch``, and ``C`` is 2
	for the x/y coordinates. Labels are 0 or 1 for the two moons.
	"""
	if points_per_batch < 1:
		raise ValueError("points_per_batch must be at least 1")
	if num_batches < 1:
		raise ValueError("num_batches must be at least 1")
	if noise < 0:
		raise ValueError("noise must be non-negative")
	if points_per_batch % 2:
		raise ValueError("points_per_batch must be even for balanced classes")

	# Asking sklearn for equal class counts makes every sequence balanced.
	points_per_class = points_per_batch // 2
	points, labels = make_moons(
		n_samples=(num_batches * points_per_class, num_batches * points_per_class),
		noise=noise,
		random_state=seed,
	)

	X = torch.from_numpy(points).reshape(num_batches, points_per_batch, 2).float() # this two is kind of confusing. I know what does it mean but how the tensor is gonna comprehend it? 
	y = torch.from_numpy(labels).reshape(num_batches, points_per_batch).long()
	# Shuffle each sequence so class is not tied to a fixed position range.
	generator = torch.Generator().manual_seed(0 if seed is None else seed)
	for batch_index in range(num_batches):
		order = torch.randperm(points_per_batch, generator=generator)
		X[batch_index] = X[batch_index, order]
		y[batch_index] = y[batch_index, order]
	return X, y


def save_plot(
	X: torch.Tensor,
	labels: torch.Tensor,
	generated: torch.Tensor,
	generated_labels: torch.Tensor | None,
	losses: list[float],
	test_losses: list[float],
	test_epochs: list[int],
	epoch: int,
	plot_dir: Path,
) -> None:
	"""Save real/generated point clouds and the loss curve for one checkpoint."""
	plot_dir.mkdir(parents=True, exist_ok=True)

	real_points = X.detach().cpu().reshape(-1, 2)
	real_labels = labels.detach().cpu().reshape(-1)
	generated_points = generated.detach().cpu().reshape(-1, 2)
	if generated_labels is not None:
		generated_labels = generated_labels.detach().cpu().reshape(-1)

	figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
	axes = axes.flatten()
	axes[0].scatter(
		real_points[:, 0],
		real_points[:, 1],
		c=real_labels,
		cmap="coolwarm",
		s=8,
		alpha=0.8,
	)
	axes[0].set_title("Real two moons")
	if generated_labels is None:
		axes[1].scatter(generated_points[:, 0], generated_points[:, 1], color="darkgreen", s=8, alpha=0.8)
		axes[1].set_title("Generated samples (unconditional)")
	else:
		axes[1].scatter(
			generated_points[:, 0],
			generated_points[:, 1],
			c=generated_labels,
			cmap="coolwarm",
			s=8,
			alpha=0.8,
		)
		axes[1].set_title("Generated samples (conditional)")
	axes[2].plot(range(1, len(losses) + 1), losses, color="black", label="Train")
	axes[2].plot(test_epochs, test_losses, color="tab:orange", label="Test")
	axes[2].set_title("Train/test loss")
	axes[2].set_xlabel("Epoch")
	axes[2].legend()
	if generated_labels is None:
		axes[3].axis("off")
		axes[3].text(0.5, 0.5, "No class labels\n(unconditional model)", ha="center", va="center")
	else:
		real_counts = torch.bincount(real_labels, minlength=2).numpy()
		generated_counts = torch.bincount(generated_labels, minlength=2).numpy()
		positions = torch.arange(2).numpy()
		width = 0.35
		axes[3].bar(positions - width / 2, real_counts, width, label="Real")
		axes[3].bar(positions + width / 2, generated_counts, width, label="Generated")
		axes[3].set_title("Class counts")
		axes[3].set_xticks(positions, ["Moon 0", "Moon 1"])
		axes[3].legend()

	for axis in axes[:2]:
		# Keep the same zoom at every checkpoint so moon size is comparable.
		axis.set_xlim(-1.75, 2.75)
		axis.set_ylim(-1.5, 1.5)
		axis.set_aspect("equal")
		axis.set_xlabel("x")
		axis.set_ylabel("y")
		axis.grid(alpha=0.2)

	figure.suptitle(f"Two-moon flow at epoch {epoch}")
	figure.text(
		0.99,
		0.01,
		datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		ha="right",
		va="bottom",
		fontsize=8,
		color="dimgray",
	)
	figure.tight_layout()
	figure.savefig(plot_dir / f"two_moons_epoch_{epoch:03d}.png", dpi=150)
	plt.close(figure)


def train_two_moons(config: TrainingConfig) -> tuple[Model, list[float]]:
	"""Train longer and evaluate with more samples for cleaner plots."""
	batch_size = config.batch_size
	points_per_batch = config.points_per_batch
	epochs = config.epochs
	plot_frequency = config.plot_frequency
	noise = config.noise
	seed = config.seed
	plot_dir = config.plot_dir
	updates_per_epoch = config.updates_per_epoch
	plot_batches = config.plot_batches
	test_batches = config.test_batches
	device = config.device
	conditional = config.conditional
	if batch_size < 1 or points_per_batch < 1 or epochs < 1:
		raise ValueError("batch_size, points_per_batch, and epochs must be at least 1")
	if plot_frequency < 1:
		raise ValueError("plot_frequency must be at least 1")
	if updates_per_epoch < 1 or plot_batches < 1 or test_batches < 1:
		raise ValueError("updates_per_epoch, plot_batches, and test_batches must be at least 1")

	# Prefer the GPU automatically, while allowing explicit CPU/GPU selection.
	selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	if selected_device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available")

	torch.manual_seed(seed)
	if selected_device.type == "cuda":
		torch.cuda.manual_seed_all(seed)
	model = Model(
		in_channels=config.in_channels,
		seq_length=points_per_batch,
		channels=config.channels,
		num_blocks=config.num_blocks,
		num_classes=2 if conditional else 0,
	).to(selected_device)
	optimizer = torch.optim.Adam(
		model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
	)
	losses: list[float] = []
	test_losses: list[float] = []
	test_epochs: list[int] = []
	plot_path = Path(plot_dir)
	splits = load_or_generate_two_moons(
		path=config.dataset_path,
		points_per_batch=points_per_batch,
		num_batches=config.dataset_num_batches,
		noise=noise,
		seed=seed,
		train_fraction=0.8,
		validation_fraction=0.1,
	)
	splits = splits.to(selected_device)
	data_mean = splits.train_X.mean(dim=(0, 1), keepdim=True)
	data_std = splits.train_X.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
	normalized_train_X = (splits.train_X - data_mean) / data_std
	normalized_validation_X = (splits.validation_X - data_mean) / data_std
	normalized_test_X = (splits.test_X - data_mean) / data_std
	train_sequences = normalized_train_X.size(0)
	if train_sequences < 1 or splits.validation_X.size(0) < 1 or splits.test_X.size(0) < 1:
		raise ValueError("dataset_num_batches must provide non-empty train, validation, and test splits")
	best_validation_loss = float("inf")
	best_state = deepcopy(model.state_dict())
	stale_evaluations = 0

	for epoch in range(1, epochs + 1):
		model.train()
		epoch_losses = []
		sequence_order = torch.randperm(train_sequences, device=selected_device)
		for start in range(0, train_sequences, batch_size):
			indices = sequence_order[start : start + batch_size]
			X = normalized_train_X[indices]
			labels = splits.train_labels[indices]
			optimizer.zero_grad()
			# In unconditional mode, the flow receives coordinates only.
			z, _, logdet = model(X, labels if conditional else None)
			loss = model.get_loss(z, logdet)
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				model.update_prior(z)
			epoch_losses.append(loss.item())

		losses.append(sum(epoch_losses) / len(epoch_losses))

		if epoch % plot_frequency == 0 or epoch == epochs:
			model.eval()
			with torch.no_grad():
				validation_z, _, validation_logdet = model(
					normalized_validation_X, splits.validation_labels if conditional else None
				)
				validation_loss = model.get_loss(validation_z, validation_logdet).item()
				test_z, _, test_logdet = model(
					normalized_test_X, splits.test_labels if conditional else None
				)
				test_losses.append(model.get_loss(test_z, test_logdet).item())
				test_epochs.append(epoch)
				plot_X = normalized_test_X[:plot_batches]
				plot_labels = splits.test_labels[:plot_batches]
				generated_labels = plot_labels if conditional else None
				generated = model.reverse(torch.randn_like(plot_X), generated_labels)
				generated = generated * data_std + data_mean
				save_plot(
					plot_X * data_std + data_mean,
					plot_labels,
					generated,
					generated_labels,
					losses,
					test_losses,
					test_epochs,
					epoch,
					plot_path,
				)
			if validation_loss < best_validation_loss:
				best_validation_loss = validation_loss
				best_state = deepcopy(model.state_dict())
				stale_evaluations = 0
			else:
				stale_evaluations += 1
				if stale_evaluations >= config.early_stopping_patience:
					print(f"Early stopping at epoch {epoch:03d}: validation loss stopped improving")
					break
			print(f"Epoch {epoch:03d}/{epochs}: loss={losses[-1]:.4f} ({selected_device})")

	model.load_state_dict(best_state)
	return model, losses


if __name__ == "__main__":
	train_two_moons(CONDITIONAL_CONFIG)
