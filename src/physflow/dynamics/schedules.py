"""Descending noise schedules for the EDM parameterization."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch


class NoiseSchedule(ABC):
    """Produce num_steps positive noise levels and a final zero."""

    @abstractmethod
    def get_sigmas(
        self,
        num_steps: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a strictly descending one-dimensional tensor."""


def _validate_range(sigma_min: float, sigma_max: float) -> None:
    if not (0 < sigma_min < sigma_max and math.isfinite(sigma_max)):
        raise ValueError("Expected finite noise levels with sigma_max > sigma_min > 0")


def _ramp(num_steps: int, dtype: torch.dtype) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if not dtype.is_floating_point:
        raise ValueError("Noise levels require a floating-point dtype")
    return torch.linspace(0, 1, num_steps, dtype=torch.float64, device="cpu")


def _finish_schedule(
    sigmas: torch.Tensor, device: torch.device | str | None, dtype: torch.dtype
) -> torch.Tensor:
    sigmas = torch.cat((sigmas, sigmas.new_zeros(1))).to(dtype=dtype)
    if not (torch.isfinite(sigmas).all() and (sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError("Requested noise levels are not strictly descending in this dtype")
    return sigmas.to(device=device or "cpu")


class KarrasSchedule(NoiseSchedule):
    """Space noise levels linearly in sigma raised to the power of 1/rho."""

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 10.0, rho: float = 7.0) -> None:
        _validate_range(sigma_min, sigma_max)
        if not (rho > 0 and math.isfinite(rho)):
            raise ValueError("rho must be finite and positive")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(
        self,
        num_steps: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ramp = _ramp(num_steps, dtype)
        start = ramp.new_tensor(self.sigma_max).pow(1 / self.rho)
        end = ramp.new_tensor(self.sigma_min).pow(1 / self.rho)
        sigmas = torch.lerp(start, end, ramp).pow(self.rho)
        sigmas[0] = self.sigma_max
        if num_steps > 1:
            sigmas[-1] = self.sigma_min
        return _finish_schedule(sigmas, device, dtype)


class LinearSchedule(NoiseSchedule):
    """Space noise levels linearly from sigma_max to sigma_min."""

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 10.0) -> None:
        _validate_range(sigma_min, sigma_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(
        self,
        num_steps: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ramp = _ramp(num_steps, dtype)
        sigmas = torch.lerp(ramp.new_tensor(self.sigma_max), ramp.new_tensor(self.sigma_min), ramp)
        return _finish_schedule(sigmas, device, dtype)
