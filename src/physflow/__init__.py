"""Physics-inspired samplers and phase-space diagnostics."""

from .diagnostics import kinetic_energy_trace, sliced_wasserstein_2d, trajectory_curvature
from .dynamics import (
    GaussianMixture2D,
    KarrasSchedule,
    LinearSchedule,
    NoiseSchedule,
    make_checkerboard_gmm,
    make_eight_gaussians,
)
from .guidance import AnnulusPotential, EnergyGuidedScore, HalfSpacePotential, HarmonicWellPotential
from .samplers import (
    BaseSampler,
    EulerSampler,
    HeunSampler,
    PredictorCorrectorSampler,
    SamplerState,
    ScoreFn,
    SymplecticSampler,
    Trajectory,
    UnderdampedLangevinSampler,
)

__all__ = [
    "AnnulusPotential",
    "BaseSampler",
    "EnergyGuidedScore",
    "EulerSampler",
    "GaussianMixture2D",
    "HalfSpacePotential",
    "HarmonicWellPotential",
    "HeunSampler",
    "KarrasSchedule",
    "LinearSchedule",
    "NoiseSchedule",
    "PredictorCorrectorSampler",
    "SamplerState",
    "ScoreFn",
    "SymplecticSampler",
    "Trajectory",
    "UnderdampedLangevinSampler",
    "kinetic_energy_trace",
    "make_checkerboard_gmm",
    "make_eight_gaussians",
    "sliced_wasserstein_2d",
    "trajectory_curvature",
]
