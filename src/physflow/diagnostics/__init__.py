"""Diagnostics for generated samples and trajectories."""

from .metrics import kinetic_energy_trace, sliced_wasserstein_2d, trajectory_curvature

__all__ = ["kinetic_energy_trace", "sliced_wasserstein_2d", "trajectory_curvature"]
