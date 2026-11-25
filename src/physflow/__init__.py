"""Physics-inspired samplers and phase-space diagnostics."""

from .diagnostics import sliced_wasserstein_2d, trajectory_curvature
from .dynamics import (
    GaussianMixture2D,
    KarrasSchedule,
    LinearSchedule,
    NoiseSchedule,
    make_checkerboard_gmm,
    make_eight_gaussians,
)
from .samplers import BaseSampler, EulerSampler, HeunSampler, SamplerState, ScoreFn, Trajectory

__all__ = [
    "BaseSampler",
    "EulerSampler",
    "GaussianMixture2D",
    "HeunSampler",
    "KarrasSchedule",
    "LinearSchedule",
    "NoiseSchedule",
    "SamplerState",
    "ScoreFn",
    "Trajectory",
    "make_checkerboard_gmm",
    "make_eight_gaussians",
    "sliced_wasserstein_2d",
    "trajectory_curvature",
]
