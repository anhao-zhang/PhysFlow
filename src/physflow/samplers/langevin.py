"""Underdamped Langevin integration with exponential BAOAB updates."""

from __future__ import annotations

import math

import torch

from physflow.dynamics import NoiseSchedule

from .symplectic import _MomentumSampler


class UnderdampedLangevinSampler(_MomentumSampler):
    """Apply thermal velocity noise between the two halves of the BAOAB drift."""

    def __init__(
        self,
        schedule: NoiseSchedule | None = None,
        num_steps: int = 20,
        friction: float = 8.0,
        mass: float = 1.0,
        temperature: float = 0.01,
        noise_min_sigma: float = 0.05,
        noise_max_sigma: float = float("inf"),
    ) -> None:
        super().__init__(schedule, num_steps)
        if not (math.isfinite(friction) and friction > 0):
            raise ValueError("friction must be finite and positive")
        if not (math.isfinite(mass) and mass > 0):
            raise ValueError("mass must be finite and positive")
        if not (math.isfinite(temperature) and temperature >= 0):
            raise ValueError("temperature must be finite and nonnegative")
        if not (math.isfinite(noise_min_sigma) and noise_min_sigma >= 0):
            raise ValueError("noise_min_sigma must be finite and nonnegative")
        if not noise_max_sigma >= noise_min_sigma:
            raise ValueError("noise_max_sigma must be at least noise_min_sigma")
        self.friction = friction
        self.mass = mass
        self.temperature = temperature
        self.noise_min_sigma = noise_min_sigma
        self.noise_max_sigma = noise_max_sigma
        self._relaxation_rate = friction / mass

    def _drift(
        self,
        x: torch.Tensor,
        velocity: torch.Tensor,
        h: torch.Tensor,
        sigma_next: torch.Tensor,
        decay: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temperature == 0 or not self.noise_min_sigma <= sigma_next <= self.noise_max_sigma:
            return super()._drift(x, velocity, h, sigma_next, decay, generator)

        half_h = 0.5 * h
        x_mid = x + half_h * velocity
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        variance_factor = -torch.expm1(-2 * decay)
        eta = variance_factor.sqrt() * math.sqrt(self.temperature) * sigma_next
        noisy_velocity = velocity + eta * noise
        return x_mid + half_h * noisy_velocity, noisy_velocity
