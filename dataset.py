from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.datasets import make_moons


@dataclass(frozen=True)
class TwoMoonsSplits:
	train_X: torch.Tensor
	train_labels: torch.Tensor
	validation_X: torch.Tensor
	validation_labels: torch.Tensor
	test_X: torch.Tensor
	test_labels: torch.Tensor

	def save(self, path: str | Path) -> None:
		"""Save all dataset splits to a PyTorch file."""
		payload = {
			"train_X": self.train_X,
			"train_labels": self.train_labels,
			"validation_X": self.validation_X,
			"validation_labels": self.validation_labels,
			"test_X": self.test_X,
			"test_labels": self.test_labels,
		}
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(payload, path)

	@classmethod
	def load(cls, path: str | Path) -> "TwoMoonsSplits":
		"""Load dataset splits saved by :meth:`save`."""
		payload = torch.load(path, map_location="cpu", weights_only=True)
		required_keys = {
			"train_X",
			"train_labels",
			"validation_X",
			"validation_labels",
			"test_X",
			"test_labels",
		}
		if not required_keys.issubset(payload):
			missing = sorted(required_keys - payload.keys())
			raise ValueError(f"Dataset file is missing keys: {', '.join(missing)}")
		return cls(**{key: payload[key] for key in required_keys})

	def to(self, device: torch.device | str) -> "TwoMoonsSplits":
		return TwoMoonsSplits(
			self.train_X.to(device),
			self.train_labels.to(device),
			self.validation_X.to(device),
			self.validation_labels.to(device),
			self.test_X.to(device),
			self.test_labels.to(device),
		)


def generate_two_moons(
	points_per_batch: int = 200,
	num_batches: int = 64,
	noise: float = 0.05,
	seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Generate balanced two-moon sequences and their class labels."""
	if points_per_batch < 1:
		raise ValueError("points_per_batch must be at least 1")
	if num_batches < 1:
		raise ValueError("num_batches must be at least 1")
	if noise < 0:
		raise ValueError("noise must be non-negative")
	if points_per_batch % 2:
		raise ValueError("points_per_batch must be even for balanced classes")

	points_per_class = points_per_batch // 2
	points, labels = make_moons(
		n_samples=(num_batches * points_per_class, num_batches * points_per_class),
		noise=noise,
		random_state=seed,
	)
	points_tensor = torch.from_numpy(points).reshape(num_batches, points_per_batch, 2).float()
	labels_tensor = torch.from_numpy(labels).reshape(num_batches, points_per_batch).long()

	generator = torch.Generator().manual_seed(0 if seed is None else seed)
	for batch_index in range(num_batches):
		order = torch.randperm(points_per_batch, generator=generator)
		points_tensor[batch_index] = points_tensor[batch_index, order]
		labels_tensor[batch_index] = labels_tensor[batch_index, order]
	return points_tensor, labels_tensor


def split_two_moons(
	points: torch.Tensor,
	labels: torch.Tensor,
	train_fraction: float = 0.8,
	validation_fraction: float = 0.1,
	seed: int = 7,
) -> TwoMoonsSplits:
	"""Split two-moon sequences into disjoint train, validation, and test sets."""
	if points.ndim != 3 or labels.ndim != 2 or points.shape[:2] != labels.shape:
		raise ValueError("points must have shape (B, T, C) and labels shape (B, T)")
	if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
		raise ValueError("fractions must satisfy train > 0, validation >= 0, and train + validation < 1")
	if points.shape[0] < 3:
		raise ValueError("at least 3 sequences are required for three splits")

	num_sequences = points.shape[0]
	train_count = max(1, int(num_sequences * train_fraction))
	validation_count = int(num_sequences * validation_fraction)
	if train_count + validation_count >= num_sequences:
		validation_count = num_sequences - train_count - 1
	if validation_count < 0:
		raise ValueError("split fractions do not leave any sequences for the test set")

	generator = torch.Generator().manual_seed(seed)
	order = torch.randperm(num_sequences, generator=generator)
	train_indices = order[:train_count]
	validation_indices = order[train_count : train_count + validation_count]
	test_indices = order[train_count + validation_count :]

	return TwoMoonsSplits(
		points[train_indices],
		labels[train_indices],
		points[validation_indices],
		labels[validation_indices],
		points[test_indices],
		labels[test_indices],
	)


def load_or_generate_two_moons(
	path: str | Path | None = None,
	*,
	points_per_batch: int = 200,
	num_batches: int = 64,
	noise: float = 0.05,
	seed: int = 7,
	train_fraction: float = 0.8,
	validation_fraction: float = 0.1,
) -> TwoMoonsSplits:
	"""Load saved splits or generate and split a new two-moon dataset."""
	if path is not None and Path(path).exists():
		return TwoMoonsSplits.load(path)
	points, labels = generate_two_moons(points_per_batch, num_batches, noise, seed)
	return split_two_moons(points, labels, train_fraction, validation_fraction, seed)
