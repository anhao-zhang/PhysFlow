"""External differentiable energies for steering score-driven samplers."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import torch

from physflow.samplers.base import ScoreFn


class EnergyGuidedScore:
    """Subtract a scaled energy gradient from a base score."""

    def __init__(
        self,
        base_score_fn: ScoreFn,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        scale: float = 1.0,
        max_grad_norm: float | None = 50.0,
        sigma_schedule_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        if max_grad_norm is not None and not (math.isfinite(max_grad_norm) and max_grad_norm >= 0):
            raise ValueError("max_grad_norm must be finite and nonnegative or None")
        self.base_score_fn = base_score_fn
        self.energy_fn = energy_fn
        self.scale = scale
        self.max_grad_norm = max_grad_norm
        self.sigma_schedule_fn = sigma_schedule_fn

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        score = self.base_score_fn(x, sigma)
        if self.scale == 0 or self.max_grad_norm == 0:
            return score

        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            energy = self.energy_fn(x_req).sum()
            if not energy.requires_grad:
                return score
            (grad,) = torch.autograd.grad(energy, x_req, allow_unused=True)
        if grad is None:
            return score

        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        grad = grad.detach().to(dtype=dtype)
        batch_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        if self.max_grad_norm is not None:
            flat = grad.reshape(x.shape[0], math.prod(x.shape[1:]))
            norm = torch.linalg.vector_norm(flat, dim=1)
            factor = self.max_grad_norm / norm.clamp_min(self.max_grad_norm)
            grad = grad * factor.reshape(batch_shape)

        if self.sigma_schedule_fn is not None:
            weight = self.sigma_schedule_fn(sigma).to(device=x.device, dtype=dtype)
            if weight.ndim == 1 and weight.shape[0] == x.shape[0]:
                weight = weight.reshape(batch_shape)
            grad = grad * weight
        return (score - self.scale * grad).to(dtype=x.dtype)


def _validate_stiffness(stiffness: float) -> None:
    if not (math.isfinite(stiffness) and stiffness >= 0):
        raise ValueError("stiffness must be finite and nonnegative")


class HalfSpacePotential:
    """Apply a quadratic penalty to positions on the undesired side of a plane."""

    def __init__(
        self,
        dim: int = 0,
        threshold: float = 0.0,
        side: Literal["positive", "negative"] = "positive",
        stiffness: float = 10.0,
    ) -> None:
        if dim not in (0, 1, -1, -2):
            raise ValueError("dim must index a coordinate of a 2D point")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if side not in ("positive", "negative"):
            raise ValueError("side must be 'positive' or 'negative'")
        _validate_stiffness(stiffness)
        self.dim = dim
        self.threshold = threshold
        self.side = side
        self.stiffness = stiffness

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        offset = x[:, self.dim] - self.threshold
        violation = (-offset if self.side == "positive" else offset).clamp_min(0)
        return (0.5 * self.stiffness) * violation.square()


class AnnulusPotential:
    """Apply a quadratic penalty to the distance from the target radius."""

    def __init__(self, radius: float = 2.0, stiffness: float = 5.0) -> None:
        if not (math.isfinite(radius) and radius >= 0):
            raise ValueError("radius must be finite and nonnegative")
        _validate_stiffness(stiffness)
        self.radius = radius
        self.stiffness = stiffness

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.vector_norm(x, dim=1)
        return (0.5 * self.stiffness) * (radius - self.radius).square()


class HarmonicWellPotential:
    """Apply a quadratic penalty to the distance from a center point."""

    def __init__(self, center: Sequence[float] = (0.0, 0.0), stiffness: float = 1.0) -> None:
        if len(center) != 2 or not all(math.isfinite(coord) for coord in center):
            raise ValueError("center must contain two finite coordinates")
        _validate_stiffness(stiffness)
        self.center = tuple(center)
        self.stiffness = stiffness

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        offset = x - x.new_tensor(self.center)
        return (0.5 * self.stiffness) * offset.square().sum(dim=1)
