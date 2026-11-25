from __future__ import annotations

import math

import pytest
import torch

from physflow import (
    BaseSampler,
    EulerSampler,
    HeunSampler,
    Trajectory,
    make_eight_gaussians,
    sliced_wasserstein_2d,
    trajectory_curvature,
)


@pytest.mark.parametrize(("sampler_type", "expected_nfe"), [(EulerSampler, 6), (HeunSampler, 11)])
def test_baseline_output_and_nfe(sampler_type: type[BaseSampler], expected_nfe: int) -> None:
    mixture = make_eight_gaussians()
    sampler = sampler_type(num_steps=6)
    calls: list[float] = []

    def score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        calls.append(sigma.item())
        return mixture.score(x, sigma)

    x_init = mixture.sample(16, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    samples, traj = sampler.sample(score, (16, 2), x_init=x_init.requires_grad_(), return_traj=True)
    xs = traj.stack_xs()
    assert samples.shape == (16, 2)
    assert samples.dtype == x_init.dtype
    assert not samples.requires_grad
    assert xs.shape == (7, 16, 2)
    assert not xs.requires_grad
    assert torch.isfinite(xs).all()
    assert len(traj.sigmas) == 7
    assert traj.sigmas[-1] == 0
    assert traj.vs is None
    assert traj.nfe == len(calls) == expected_nfe
    assert min(calls) > 0
    assert xs[0].equal(x_init)
    assert xs[-1].equal(samples)
    assert traj.xs[0].data_ptr() != x_init.data_ptr()
    assert traj.xs[-1].data_ptr() != samples.data_ptr()


@pytest.mark.parametrize("sampler_type", [EulerSampler, HeunSampler])
def test_reproducible_samples(sampler_type: type[BaseSampler]) -> None:
    mixture = make_eight_gaussians()
    sampler = sampler_type(num_steps=4)
    first = sampler.sample(mixture.score, (32, 2), generator=torch.Generator().manual_seed(17))
    second = sampler.sample(mixture.score, (32, 2), generator=torch.Generator().manual_seed(17))
    assert torch.equal(first, second)


def test_heun_one_step_uses_euler_on_final_interval() -> None:
    x_init = torch.ones(2, 3, 4)

    def score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        assert sigma.item() == 10.0
        return -x / 200

    samples, traj = HeunSampler(num_steps=1).sample(
        score, x_init.shape, x_init=x_init, return_traj=True
    )
    torch.testing.assert_close(samples, x_init * 0.5)
    assert traj.nfe == 1


def test_heun_improves_sliced_wasserstein_at_15_steps() -> None:
    mixture = make_eight_gaussians()
    shape = (4096, 2)
    target = mixture.sample(8192, generator=torch.Generator().manual_seed(123))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        euler = EulerSampler(num_steps=15).sample(
            mixture.score, shape, generator=torch.Generator().manual_seed(42)
        )
        heun = HeunSampler(num_steps=15).sample(
            mixture.score, shape, generator=torch.Generator().manual_seed(42)
        )
    finally:
        torch.set_num_threads(previous_threads)

    assert sliced_wasserstein_2d(heun, target) < sliced_wasserstein_2d(euler, target)


@pytest.mark.parametrize(
    ("points", "angle"),
    [
        ([(0, 0), (1, 0), (2, 0)], 0.0),
        ([(0, 0), (1, 0), (1, 1)], math.pi / 2),
        ([(0, 0), (1, 0), (0, 0)], math.pi),
        ([(0, 0), (0, 0), (1, 0)], 0.0),
    ],
)
def test_trajectory_curvature(points: list[tuple[int, int]], angle: float) -> None:
    xs = [torch.tensor([point], dtype=torch.float64) for point in points]
    assert trajectory_curvature(Trajectory(xs, [2.0, 1.0, 0.0])) == pytest.approx(angle)


def test_sliced_wasserstein_properties() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x + 1
    assert sliced_wasserstein_2d(x, x.flip(0)) == 0.0
    distance = sliced_wasserstein_2d(x, y, seed=13)
    assert distance > 0
    assert distance == sliced_wasserstein_2d(x, y, seed=13)
    assert distance == sliced_wasserstein_2d(y, x, seed=13)
