from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from train import get_two_moons, save_plot


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
	plot_batches: int = 32
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

	device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available")
	torch.manual_seed(config.seed)
	if device.type == "cuda":
		torch.cuda.manual_seed_all(config.seed)

	model = RealNVP(config.num_layers, config.hidden_features, config.conditional).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
	sequences_per_epoch = config.batch_size * config.updates_per_epoch
	all_X, all_labels = get_two_moons(config.points_per_batch, sequences_per_epoch, config.noise, config.seed)
	all_X, all_labels = all_X.to(device), all_labels.to(device)
	data_mean = all_X.mean(dim=(0, 1), keepdim=True)
	data_std = all_X.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
	normalized_X = (all_X - data_mean) / data_std
	losses: list[float] = []

	for epoch in range(1, config.epochs + 1):
		model.train()
		order = torch.randperm(sequences_per_epoch, device=device)
		epoch_losses = []
		for start in range(0, sequences_per_epoch, config.batch_size):
			indices = order[start : start + config.batch_size]
			labels = all_labels[indices]
			optimizer.zero_grad()
			z, logdet = model(normalized_X[indices], labels if config.conditional else None)
			loss = model.get_loss(z, logdet)
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss.item())
		losses.append(sum(epoch_losses) / len(epoch_losses))

		if epoch % config.plot_frequency == 0 or epoch == config.epochs:
			model.eval()
			with torch.no_grad():
				plot_X, plot_labels = get_two_moons(config.points_per_batch, config.plot_batches, config.noise, config.seed)
				plot_X, plot_labels = plot_X.to(device), plot_labels.to(device)
				generated_labels = plot_labels if config.conditional else None
				generated = model.reverse(torch.randn_like(plot_X), generated_labels)
				generated = generated * data_std + data_mean
				save_plot(plot_X, plot_labels, generated, generated_labels, losses, epoch, Path(config.plot_dir))
			print(f"Epoch {epoch:03d}/{config.epochs}: loss={losses[-1]:.4f} ({device})")

	return model, losses


if __name__ == "__main__":
	train_two_moons(RealNVPConfig())