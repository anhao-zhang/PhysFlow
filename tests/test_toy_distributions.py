from __future__ import annotations

import math

import pytest
import torch

from physflow import GaussianMixture2D, make_checkerboard_gmm, make_eight_gaussians


@pytest.mark.parametrize("sigma", [0.0, 0.002, 0.3, 2.0, 10.0, torch.tensor([0.0, 0.3, 2.0])])
def test_closed_form_score_matches_log_density_gradient(sigma: torch.Tensor | float) -> None:
    mixture = GaussianMixture2D(
        means=torch.tensor([[0.0, 0.5], [1.5, 1.0], [-2.0, -1.5]]),
        weights=torch.tensor([1.0, 3.0, 0.0]),
    )
    x = torch.tensor([[2.5, 1.5], [-0.5, 1.0], [-3.5, 2.0]], requires_grad=True)
    (gradient,) = torch.autograd.grad(mixture.log_prob(x, sigma).sum(), x)
    score = mixture.score(x, sigma)
    assert torch.isfinite(score).all()
    torch.testing.assert_close(score, gradient, atol=1e-5, rtol=1e-5)


def test_perturbed_log_density_includes_normalization() -> None:
    mixture = GaussianMixture2D(torch.zeros(1, 2), torch.ones(1), base_std=0.5)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    sigma = 0.5
    variance = mixture.base_std**2 + sigma**2
    expected = -math.log(2 * math.pi * variance) - x.square().sum(dim=-1) / (2 * variance)
    torch.testing.assert_close(mixture.log_prob(x, sigma), expected)


def test_closed_form_score_stays_finite_in_far_tails() -> None:
    mixture = make_eight_gaussians()
    x = torch.tensor([[1e20, -1e20]])
    sigma = 1.0
    expected = -x / (mixture.base_std**2 + sigma**2)
    torch.testing.assert_close(mixture.score(x, sigma), expected)


def test_sampling_is_deterministic_and_respects_dtype() -> None:
    mixture = make_eight_gaussians(radius=3.0)
    first = mixture.sample(64, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    second = mixture.sample(64, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    assert first.shape == (64, 2)
    assert first.dtype == torch.float64
    assert torch.equal(first, second)
    torch.testing.assert_close(
        mixture.means.norm(dim=1), torch.full((8,), 3.0, dtype=mixture.means.dtype)
    )


def test_checkerboard_occupies_alternating_cells() -> None:
    mixture = make_checkerboard_gmm()
    assert mixture.means.shape == (8, 2)
    assert (mixture.means.sum(dim=1) % 2 == 1).all()
    assert mixture.weights.sum().item() == pytest.approx(1.0)
