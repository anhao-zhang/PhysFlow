"""Second-order Heun integration of the EDM probability-flow ODE."""

from __future__ import annotations

from dataclasses import replace

import torch

from .base import BaseSampler, SamplerState, ScoreFn


class HeunSampler(BaseSampler):
    """Correct the Euler predictor at each nonzero noise level."""

    def step(
        self,
        state: SamplerState,
        sigma_next: torch.Tensor,
        score_fn: ScoreFn,
        generator: torch.Generator | None = None,
    ) -> SamplerState:
        delta = sigma_next - state.sigma
        derivative = -state.sigma * score_fn(state.x, state.sigma)
        x_next = state.x + delta * derivative
        nfe = state.nfe + 1

        if sigma_next > 0:
            derivative_next = -sigma_next * score_fn(x_next, sigma_next)
            x_next = state.x + (0.5 * delta) * (derivative + derivative_next)
            nfe += 1

        return replace(
            state,
            x=x_next,
            sigma=sigma_next,
            step_index=state.step_index + 1,
            nfe=nfe,
            force=None,
        )
