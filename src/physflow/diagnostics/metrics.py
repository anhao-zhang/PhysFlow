"""Trajectory smoothness and two-dimensional sample distance metrics."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from physflow.samplers.base import Trajectory


def trajectory_curvature(traj: Trajectory) -> float:
    """Average the angles in radians between consecutive nonzero displacements."""
    if len(traj.xs) < 3 or traj.xs[0].numel() == 0:
        return 0.0

    positions = traj.stack_xs().detach()
    if positions.dtype != torch.float64:
        positions = positions.float()
    displacements = positions.reshape(*positions.shape[:2], -1).diff(dim=0)
    lengths = torch.linalg.vector_norm(displacements, dim=-1, keepdim=True)
    directions = displacements / torch.where(lengths > 0, lengths, 1.0)
    before, after = directions[:-1], directions[1:]
    valid = (lengths[:-1, :, 0] > 0) & (lengths[1:, :, 0] > 0)
    angles = 2 * torch.atan2(
        torch.linalg.vector_norm(after - before, dim=-1),
        torch.linalg.vector_norm(after + before, dim=-1),
    )
    return torch.where(valid, angles, 0.0).sum().div(valid.sum().clamp_min(1)).item()


def sliced_wasserstein_2d(
    samples: torch.Tensor, target: torch.Tensor, num_projections: int = 64, seed: int = 0
) -> float:
    """Average the one-dimensional Wasserstein-1 distance over random projections."""
    if num_projections < 1:
        raise ValueError("num_projections must be at least 1")
    for points in (samples, target):
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            raise ValueError("Each sample set must have shape (N, 2) with N > 0")

    samples_np = samples.detach().to(device="cpu", dtype=torch.float64).numpy()
    target_np = target.detach().to(device="cpu", dtype=torch.float64).numpy()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    angles = 2 * np.pi * torch.rand(num_projections, dtype=torch.float64, generator=generator)
    directions = torch.stack((angles.cos(), angles.sin()), dim=-1).numpy()
    projected_samples = np.einsum("ij,kj->ki", samples_np, directions)
    projected_target = np.einsum("ij,kj->ki", target_np, directions)
    distances = [
        wasserstein_distance(a, b) for a, b in zip(projected_samples, projected_target, strict=True)
    ]
    return float(np.mean(distances))
