"""Annealed Langevin corrections for Euler or Heun predictions."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import torch

from physflow.dynamics import NoiseSchedule

from .base import BaseSampler, SamplerState, ScoreFn
from .euler import EulerSampler
from .heun import HeunSampler


class PredictorCorrectorSampler(BaseSampler):
    """Apply annealed MCMC corrections at positive noise levels at or above the cutoff."""

    def __init__(
        self,
        schedule: NoiseSchedule | None = None,
        num_steps: int = 20,
        predictor: Literal["euler", "heun"] = "euler",
        corrector_steps: int = 1,
        snr: float = 0.06,
        max_corrector_step_size: float = 0.05,
        corrector_min_sigma: float = 0.1,
    ) -> None:
        super().__init__(schedule, num_steps)
        if predictor not in ("euler", "heun"):
            raise ValueError("predictor must be 'euler' or 'heun'")
        if not isinstance(corrector_steps, int) or corrector_steps < 0:
            raise ValueError("corrector_steps must be a nonnegative integer")
        if not (math.isfinite(snr) and snr > 0):
            raise ValueError("snr must be finite and positive")
        if not (math.isfinite(max_corrector_step_size) and max_corrector_step_size > 0):
            raise ValueError("max_corrector_step_size must be finite and positive")
        if not (math.isfinite(corrector_min_sigma) and corrector_min_sigma >= 0):
            raise ValueError("corrector_min_sigma must be finite and nonnegative")
        self.predictor = predictor
        self.corrector_steps = corrector_steps
        self.snr = snr
        self.max_corrector_step_size = max_corrector_step_size
        self.corrector_min_sigma = corrector_min_sigma
        predictor_type = EulerSampler if predictor == "euler" else HeunSampler
        self._predictor = predictor_type(self.schedule, self.num_steps)

    def step(
        self,
        state: SamplerState,
        sigma_next: torch.Tensor,
        score_fn: ScoreFn,
        generator: torch.Generator | None = None,
    ) -> SamplerState:
        predicted = self._predictor.step(state, sigma_next, score_fn, generator)
        if self.corrector_steps == 0 or sigma_next == 0 or sigma_next < self.corrector_min_sigma:
            return predicted

        x = predicted.x
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        cap = self.max_corrector_step_size * sigma_next.to(dtype=dtype).square()
        norm_shape = (x.shape[0], math.prod(x.shape[1:]))
        for _ in range(self.corrector_steps):
            gradient = score_fn(x, sigma_next).to(dtype=dtype)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            noise = noise.to(dtype=dtype)
            mean_gradient_norm = torch.linalg.vector_norm(
                gradient.reshape(norm_shape), dim=1
            ).mean()
            mean_noise_norm = torch.linalg.vector_norm(noise.reshape(norm_shape), dim=1).mean()
            step_size = 2 * (self.snr * mean_noise_norm / (mean_gradient_norm + 1e-8)).square()
            step_size = step_size.clamp(max=cap)
            x = (x + step_size * gradient + (2 * step_size).sqrt() * noise).to(dtype=x.dtype)

        return replace(predicted, x=x, nfe=predicted.nfe + self.corrector_steps)
