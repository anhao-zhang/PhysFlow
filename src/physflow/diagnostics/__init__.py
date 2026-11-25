"""Diagnostics for generated samples and trajectories."""

from .metrics import sliced_wasserstein_2d, trajectory_curvature

__all__ = ["sliced_wasserstein_2d", "trajectory_curvature"]
