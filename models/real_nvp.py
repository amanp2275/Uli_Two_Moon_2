from math import log, pi

import torch
from torch import nn


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
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AffineCoupling(nn.Module):
    def __init__(self, mask: torch.Tensor, hidden_features: int, conditional: bool):
        super().__init__()
        self.register_buffer("mask", mask.view(1, 1, -1))
        self.conditional = conditional
        condition_features = 2 if conditional else 0
        self.transform = CouplingNetwork(2 + condition_features, 4, hidden_features)

    def _network_input(self, x: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        if not self.conditional:
            return x
        if labels is None:
            label_input = torch.full((*x.shape[:2], 2), 0.5, device=x.device, dtype=x.dtype)
        else:
            label_input = torch.nn.functional.one_hot(labels, num_classes=2).to(x.dtype)
        return torch.cat((x, label_input), dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        identity = x * self.mask
        log_scale, translation = self.transform(self._network_input(identity, labels)).chunk(2, dim=-1)
        log_scale = 2.0 * torch.tanh(log_scale / 2.0) * (1.0 - self.mask)
        translation = translation * (1.0 - self.mask)
        transformed = identity + (1.0 - self.mask) * (x * log_scale.exp() + translation)
        return transformed, log_scale.sum(dim=-1)

    def inverse(self, z: torch.Tensor, labels: torch.Tensor | None = None):
        identity = z * self.mask
        log_scale, translation = self.transform(self._network_input(identity, labels)).chunk(2, dim=-1)
        log_scale = 2.0 * torch.tanh(log_scale / 2.0) * (1.0 - self.mask)
        translation = translation * (1.0 - self.mask)
        original = identity + (1.0 - self.mask) * ((z - translation) * (-log_scale).exp())
        return original, -log_scale.sum(dim=-1)


class RealNVP(nn.Module):
    """RealNVP model with the repository-wide flow interface."""

    def __init__(self, num_layers: int = 8, hidden_features: int = 128, conditional: bool = True):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.layers = nn.ModuleList(
            [AffineCoupling(torch.tensor([1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]), hidden_features, conditional)
             for i in range(num_layers)]
        )
        for i in range(num_layers - 1):
            self.register_buffer(f"mixing_{i}", torch.linalg.qr(torch.randn(2, 2)).Q)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        logdet = torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype)
        for i, layer in enumerate(self.layers):
            x, layer_logdet = layer(x, y)
            logdet = logdet + layer_logdet
            if i < len(self.layers) - 1:
                x = x @ getattr(self, f"mixing_{i}")
        return x, logdet

    def reverse(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.layers) - 1:
                z = z @ getattr(self, f"mixing_{i}").transpose(-1, -2)
            z, _ = self.layers[i].inverse(z, y)
        return z

    def get_loss(self, z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        base_nll = 0.5 * z.pow(2).sum(dim=-1) + z.size(-1) * 0.5 * log(2.0 * pi)
        return (base_nll - logdet).mean() / z.size(-1)
