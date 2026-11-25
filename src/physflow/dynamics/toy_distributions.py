"""Two-dimensional test distributions with analytical perturbed scores."""

from __future__ import annotations

import math

import torch


class GaussianMixture2D:
    """A mixture of isotropic Gaussians with a shared standard deviation."""

    def __init__(self, means: torch.Tensor, weights: torch.Tensor, base_std: float = 0.15) -> None:
        if means.ndim != 2 or means.shape[1] != 2 or means.shape[0] == 0:
            raise ValueError("means must have shape (K, 2) with K > 0")
        if not torch.isfinite(means).all():
            raise ValueError("means must be finite")
        if weights.shape != means.shape[:1]:
            raise ValueError("weights must have shape (K,)")
        if not (torch.isfinite(weights).all() and (weights >= 0).all() and (weights > 0).any()):
            raise ValueError("weights must be finite, nonnegative, and contain a positive value")
        if not (base_std > 0 and math.isfinite(base_std)):
            raise ValueError("base_std must be finite and positive")
        self.means = means
        if weights.dtype != torch.float64:
            weights = weights.float()
        scaled_weights = weights / weights.max()
        self.weights = scaled_weights / scaled_weights.sum()
        self.base_std = base_std

    def sample(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw samples from the unperturbed mixture."""
        device = self.means.device if device is None else device
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")
        if num_samples == 0:
            return torch.empty((0, 2), device=device, dtype=dtype)

        weights = self.weights.to(device=device)
        components = torch.multinomial(weights, num_samples, replacement=True, generator=generator)
        means = self.means.to(device=device, dtype=dtype)
        noise = torch.randn((num_samples, 2), device=device, dtype=dtype, generator=generator)
        return means[components] + self.base_std * noise

    def _component_terms(
        self, x: torch.Tensor, sigma: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != 2 or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor of shape (B, 2)")
        # Keep exponentials and squared distances in at least single precision.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        noise = torch.as_tensor(sigma, device=x.device, dtype=dtype)
        if noise.ndim == 1 and noise.shape[0] in (1, x.shape[0]):
            noise = noise[:, None]
        elif noise.ndim != 0:
            raise ValueError("sigma must be scalar or have shape (B,)")

        variance = noise.square() + self.base_std**2
        means = self.means.to(device=x.device, dtype=dtype)
        log_weights = self.weights.log().to(device=x.device, dtype=dtype)
        offsets = x.to(dtype=dtype)[:, None, :] - means
        return offsets, variance, log_weights

    def log_prob(self, x: torch.Tensor, sigma: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Evaluate the log density after Gaussian perturbation of scale sigma."""
        offsets, variance, log_weights = self._component_terms(x, sigma)
        logits = log_weights - 0.5 * offsets.square().sum(dim=-1) / variance
        log_normalizer = math.log(2 * math.pi) + variance.log()
        log_density = torch.logsumexp(logits, dim=-1) - log_normalizer.squeeze(-1)
        return log_density.to(dtype=x.dtype)

    def score(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """Evaluate the gradient of the perturbed log density in closed form."""
        offsets, variance, log_weights = self._component_terms(x, sigma)
        # Scale extreme offsets before squaring, then subtract the common
        # distance to the closest nonzero-weight component before restoring scale.
        safe_limit = math.sqrt(torch.finfo(offsets.dtype).max / 4)
        scale = (offsets.abs().amax(dim=(1, 2), keepdim=True) / safe_limit).clamp_min(1)
        distances = (offsets / scale).square().sum(dim=-1)
        distances = distances.masked_fill(log_weights == -torch.inf, torch.inf)
        differences = distances - distances.amin(dim=-1, keepdim=True)
        scale = scale.squeeze(-1)
        logits = log_weights - (0.5 * differences / variance * scale) * scale
        posterior = logits.softmax(dim=-1)
        mean_offset = (posterior[..., None] * offsets).sum(dim=1)
        return (-mean_offset / variance).to(dtype=x.dtype)


def make_eight_gaussians(radius: float = 2.0, std: float = 0.15) -> GaussianMixture2D:
    """Build eight uniformly weighted modes spaced around a circle."""
    angles = torch.arange(8, dtype=torch.float64) * (math.tau / 8)
    means = radius * torch.stack((angles.cos(), angles.sin()), dim=-1)
    return GaussianMixture2D(means, torch.ones(8), base_std=std)


def make_checkerboard_gmm(grid_size: int = 4, std: float = 0.12) -> GaussianMixture2D:
    """Place Gaussian modes on alternating squares of a centered grid."""
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    indices = torch.arange(grid_size)
    rows, cols = torch.meshgrid(indices, indices, indexing="ij")
    occupied = (rows + cols) % 2 == 0
    means = torch.stack((cols[occupied], rows[occupied]), dim=-1) - (grid_size - 1) / 2
    return GaussianMixture2D(means, torch.ones(len(means)), base_std=std)
