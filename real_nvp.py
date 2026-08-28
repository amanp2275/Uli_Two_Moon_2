from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

import torch
from torch import nn

from dataset import load_or_generate_two_moons
from train import save_plot


@dataclass(frozen=True)
class RealNVPConfig:
	batch_size: int = 32
	points_per_batch: int = 500
	epochs: int = 300
	plot_frequency: int = 25
	noise: float = 0.05
	seed: int = 7
	plot_dir: str | Path = "real_nvp_plots"
	updates_per_epoch: int = 8
	dataset_num_batches: int = 512
	dataset_path: str | Path = Path(__file__).parent / "two_moons_splits.pt"
	plot_batches: int = 32
	test_batches: int = 32
	early_stopping_patience: int = 3
	weight_decay: float = 1e-4
	device: str | None = None
	conditional: bool = True
	num_layers: int = 8
	hidden_features: int = 128
	learning_rate: float = 1e-3


class CouplingNetwork(nn.Module):
	def __init__(self, input_features: int, output_features: int, hidden_features: int):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(input_features, hidden_features),
			nn.ReLU(),
			nn.Linear(hidden_features, hidden_features),
			nn.ReLU(),
			nn.Linear(hidden_features, output_features),
		)
		# Starting at the identity makes early optimization more stable.
		nn.init.zeros_(self.network[-1].weight)
		nn.init.zeros_(self.network[-1].bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)


class AffineCoupling(nn.Module):
	def __init__(self, mask: torch.Tensor, hidden_features: int, conditional: bool):
		super().__init__()
		self.register_buffer("mask", mask.view(1, 1, -1))
		condition_features = 2 if conditional else 0
		self.conditional = conditional
		self.transform = CouplingNetwork(2 + condition_features, 4, hidden_features)

	def _network_input(self, x: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
		if not self.conditional:
			return x
		if labels is None:
			label_input = torch.full((*x.shape[:2], 2), 0.5, device=x.device, dtype=x.dtype)
		else:
			label_input = torch.nn.functional.one_hot(labels, num_classes=2).to(x.dtype)
		return torch.cat((x, label_input), dim=-1)

	def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
		identity = x * self.mask
		log_scale, translation = self.transform(self._network_input(identity, labels)).chunk(2, dim=-1)
		log_scale = 2.0 * torch.tanh(log_scale / 2.0) * (1.0 - self.mask)
		translation = translation * (1.0 - self.mask)
		return identity + (1.0 - self.mask) * (x * log_scale.exp() + translation), log_scale.sum(dim=-1)

	def inverse(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
		identity = z * self.mask
		log_scale, translation = self.transform(self._network_input(identity, labels)).chunk(2, dim=-1)
		log_scale = 2.0 * torch.tanh(log_scale / 2.0) * (1.0 - self.mask)
		translation = translation * (1.0 - self.mask)
		return identity + (1.0 - self.mask) * ((z - translation) * (-log_scale).exp()), -log_scale.sum(dim=-1)


class RealNVP(nn.Module):
	def __init__(self, num_layers: int = 8, hidden_features: int = 128, conditional: bool = True):
		super().__init__()
		if num_layers < 1:
			raise ValueError("num_layers must be at least 1")
		layers = []
		for layer_index in range(num_layers):
			mask = torch.tensor([1.0, 0.0] if layer_index % 2 == 0 else [0.0, 1.0])
			layers.append(AffineCoupling(mask, hidden_features, conditional))
		self.layers = nn.ModuleList(layers)
		for layer_index in range(num_layers - 1):
			matrix = torch.linalg.qr(torch.randn(2, 2)).Q
			self.register_buffer(f"mixing_{layer_index}", matrix)

	def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
		logdet = torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype)
		for layer_index, layer in enumerate(self.layers):
			x, layer_logdet = layer(x, labels)
			logdet = logdet + layer_logdet
			if layer_index < len(self.layers) - 1:
				x = x @ getattr(self, f"mixing_{layer_index}")
		return x, logdet

	def reverse(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
		for layer_index in range(len(self.layers) - 1, -1, -1):
			if layer_index < len(self.layers) - 1:
				z = z @ getattr(self, f"mixing_{layer_index}").transpose(-1, -2)
			layer = self.layers[layer_index]
			z, _ = layer.inverse(z, labels)
		return z

	def get_loss(self, z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
		return 0.5 * z.pow(2).mean() - logdet.mean()


def train_two_moons(config: RealNVPConfig) -> tuple[RealNVP, list[float]]:
	if config.points_per_batch < 1 or config.batch_size < 1 or config.epochs < 1:
		raise ValueError("batch_size, points_per_batch, and epochs must be at least 1")
	if config.points_per_batch % 2:
		raise ValueError("points_per_batch must be even for balanced classes")
	if config.test_batches < 1:
		raise ValueError("test_batches must be at least 1")

	device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available")
	torch.manual_seed(config.seed)
	if device.type == "cuda":
		torch.cuda.manual_seed_all(config.seed)

	model = RealNVP(config.num_layers, config.hidden_features, config.conditional).to(device)
	optimizer = torch.optim.Adam(
		model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
	)
	splits = load_or_generate_two_moons(
		path=config.dataset_path,
		points_per_batch=config.points_per_batch,
		num_batches=config.dataset_num_batches,
		noise=config.noise,
		seed=config.seed,
		train_fraction=0.8,
		validation_fraction=0.1,
	).to(device)
	data_mean = splits.train_X.mean(dim=(0, 1), keepdim=True)
	data_std = splits.train_X.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
	normalized_train_X = (splits.train_X - data_mean) / data_std
	normalized_validation_X = (splits.validation_X - data_mean) / data_std
	normalized_test_X = (splits.test_X - data_mean) / data_std
	train_sequences = normalized_train_X.size(0)
	if train_sequences < 1 or splits.validation_X.size(0) < 1 or splits.test_X.size(0) < 1:
		raise ValueError("dataset_num_batches must provide non-empty train, validation, and test splits")
	losses: list[float] = []
	test_losses: list[float] = []
	test_epochs: list[int] = []
	best_validation_loss = float("inf")
	best_state = deepcopy(model.state_dict())
	stale_evaluations = 0

	for epoch in range(1, config.epochs + 1):
		model.train()
		order = torch.randperm(train_sequences, device=device)
		epoch_losses = []
		for start in range(0, train_sequences, config.batch_size):
			indices = order[start : start + config.batch_size]
			labels = splits.train_labels[indices]
			optimizer.zero_grad()
			z, logdet = model(normalized_train_X[indices], labels if config.conditional else None)
			loss = model.get_loss(z, logdet)
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss.item())
		losses.append(sum(epoch_losses) / len(epoch_losses))

		if epoch % config.plot_frequency == 0 or epoch == config.epochs:
			model.eval()
			with torch.no_grad():
				validation_z, validation_logdet = model(
					normalized_validation_X,
					splits.validation_labels if config.conditional else None,
				)
				validation_loss = model.get_loss(validation_z, validation_logdet).item()
				test_z, test_logdet = model(
					normalized_test_X, splits.test_labels if config.conditional else None
				)
				test_losses.append(model.get_loss(test_z, test_logdet).item())
				test_epochs.append(epoch)
				plot_X = normalized_test_X[: config.plot_batches]
				plot_labels = splits.test_labels[: config.plot_batches]
				generated_labels = plot_labels if config.conditional else None
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
					Path(config.plot_dir),
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
			print(f"Epoch {epoch:03d}/{config.epochs}: loss={losses[-1]:.4f} ({device})")

	model.load_state_dict(best_state)
	return model, losses


if __name__ == "__main__":
	train_two_moons(RealNVPConfig())