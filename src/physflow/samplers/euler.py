"""First-order Euler integration of the EDM probability-flow ODE."""

from __future__ import annotations

from dataclasses import replace

import torch

from .base import BaseSampler, SamplerState, ScoreFn


class EulerSampler(BaseSampler):
    """Advance using the score at the start of each noise interval."""

    def step(
        self,
        state: SamplerState,
        sigma_next: torch.Tensor,
        score_fn: ScoreFn,
        generator: torch.Generator | None = None,
    ) -> SamplerState:
        derivative = -state.sigma * score_fn(state.x, state.sigma)
        x_next = state.x + (sigma_next - state.sigma) * derivative
        return replace(
            state, x=x_next, sigma=sigma_next, step_index=state.step_index + 1, nfe=state.nfe + 1
        )
