"""Exponentially damped velocity-Verlet integration."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import torch

from physflow.dynamics import NoiseSchedule

from .base import BaseSampler, SamplerState, ScoreFn


class _MomentumSampler(BaseSampler):
    """Share the damped kicks, terminal Euler step, and cached endpoint force."""

    _relaxation_rate: float

    def init_state(
        self,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        x_init: torch.Tensor | None = None,
    ) -> SamplerState:
        state = super().init_state(shape, device, dtype, generator, x_init)
        return replace(state, v=torch.zeros_like(state.x))

    def _velocity(self, state: SamplerState, force: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(state.x) if state.v is None else state.v

    def _drift(
        self,
        x: torch.Tensor,
        velocity: torch.Tensor,
        h: torch.Tensor,
        sigma_next: torch.Tensor,
        decay: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x + h * velocity, velocity

    def _start_interval(
        self,
        x: torch.Tensor,
        velocity: torch.Tensor,
        force: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the initial kick and drift, leaving the endpoint kick pending."""
        h = sigma - sigma_next
        # Relax velocity in log-sigma time while integrating position in sigma.
        dtype = torch.float64 if sigma.dtype == torch.float64 else torch.float32
        current = sigma.to(dtype=dtype)
        following = sigma_next.to(dtype=dtype)
        delta_tau = torch.log1p((current - following) / following)
        decay = self._relaxation_rate * delta_tau
        kick_fraction = (-torch.expm1(-0.5 * decay)).to(dtype=x.dtype)
        velocity = torch.lerp(velocity, force, kick_fraction)
        x_next, velocity = self._drift(x, velocity, h, sigma_next, decay, generator)
        return x_next, velocity, kick_fraction

    def step(
        self,
        state: SamplerState,
        sigma_next: torch.Tensor,
        score_fn: ScoreFn,
        generator: torch.Generator | None = None,
    ) -> SamplerState:
        force, nfe = state.force, state.nfe
        if force is None:
            force = state.sigma * score_fn(state.x, state.sigma)
            nfe += 1

        if sigma_next == 0:
            return SamplerState(
                x=state.x + state.sigma * force,
                sigma=sigma_next,
                v=force,
                step_index=state.step_index + 1,
                nfe=nfe,
            )

        x_next, velocity, kick_fraction = self._start_interval(
            state.x, self._velocity(state, force), force, state.sigma, sigma_next, generator
        )
        force_next = sigma_next * score_fn(x_next, sigma_next)
        return SamplerState(
            x=x_next,
            sigma=sigma_next,
            v=torch.lerp(velocity, force_next, kick_fraction),
            step_index=state.step_index + 1,
            nfe=nfe + 1,
            force=force_next,
        )


class SymplecticSampler(_MomentumSampler):
    """Integrate momentum with velocity-Verlet steps and log-sigma damping."""

    def __init__(
        self,
        schedule: NoiseSchedule | None = None,
        num_steps: int = 20,
        damping: float = 8.0,
        mass: float = 1.0,
        init_velocity: Literal["zero", "drift"] = "zero",
    ) -> None:
        super().__init__(schedule, num_steps)
        if not (math.isfinite(damping) and damping > 0):
            raise ValueError("damping must be finite and positive")
        if not (math.isfinite(mass) and mass > 0):
            raise ValueError("mass must be finite and positive")
        if init_velocity not in ("zero", "drift"):
            raise ValueError("init_velocity must be 'zero' or 'drift'")
        self.damping = damping
        self.mass = mass
        self.init_velocity = init_velocity
        self._relaxation_rate = damping / mass

    def init_state(
        self,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        x_init: torch.Tensor | None = None,
    ) -> SamplerState:
        state = super().init_state(shape, device, dtype, generator, x_init)
        return replace(state, v=None) if self.init_velocity == "drift" else state

    def _prepare_state(self, state: SamplerState, score_fn: ScoreFn) -> SamplerState:
        if self.init_velocity == "drift":
            force = state.sigma * score_fn(state.x, state.sigma)
            return replace(state, v=force, force=force, nfe=state.nfe + 1)
        return state

    def _velocity(self, state: SamplerState, force: torch.Tensor) -> torch.Tensor:
        if state.step_index == 0 and self.init_velocity == "drift":
            return force
        return super()._velocity(state, force)
