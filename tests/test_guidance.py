from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from physflow import (
    AnnulusPotential,
    EnergyGuidedScore,
    HalfSpacePotential,
    HarmonicWellPotential,
    UnderdampedLangevinSampler,
    make_eight_gaussians,
)


def _zero_score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def test_guidance_inside_no_grad_uses_correct_sign_and_weight() -> None:
    def base_score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        return x + 1

    guided = EnergyGuidedScore(
        base_score,
        HarmonicWellPotential(center=(1.0, 2.0), stiffness=3.0),
        scale=0.5,
        sigma_schedule_fn=torch.square,
        max_grad_norm=None,
    )
    x = torch.tensor([[2.0, -1.0], [1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    sigma = x.new_tensor(2.0)
    with torch.no_grad():
        actual = guided(x, sigma)
        expected = x + 1 - 6 * (x - x.new_tensor([1.0, 2.0]))
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert x.grad is None


def test_guidance_clips_per_sample_before_scaling() -> None:
    guided = EnergyGuidedScore(_zero_score, HarmonicWellPotential(), scale=2, max_grad_norm=1.5)
    x = torch.tensor([[300.0, 400.0], [0.0, 0.5], [0.0, 0.0]])
    with torch.no_grad():
        shift = guided(x, x.new_tensor(1.0))
    torch.testing.assert_close(shift.norm(dim=1), torch.tensor([3.0, 1.0, 0.0]))
    torch.testing.assert_close(shift[0], -3 * torch.tensor([0.6, 0.8]))


def test_guidance_broadcasts_batch_weights_on_image_tensors() -> None:
    def energy(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x.square()).sum(dim=(1, 2, 3))

    x = torch.ones(2, 1, 3, 3, dtype=torch.float16)
    sigma = x.new_tensor([0.5, 2])
    guided = EnergyGuidedScore(_zero_score, energy, sigma_schedule_fn=torch.square)
    torch.testing.assert_close(guided(x, sigma), -sigma.square().reshape(2, 1, 1, 1) * x)


def test_guidance_accepts_constant_energy() -> None:
    def energy(x: torch.Tensor) -> torch.Tensor:
        return x.new_ones(x.shape[0])

    x = torch.ones(2, 2)
    guided = EnergyGuidedScore(_zero_score, energy)
    torch.testing.assert_close(guided(x, x.new_tensor(1)), torch.zeros_like(x))


@pytest.mark.parametrize(
    ("potential", "expected"),
    [
        (HalfSpacePotential(dim=1, threshold=1, stiffness=2), [1, 1, 16]),
        (HalfSpacePotential(dim=0, threshold=1, side="negative", stiffness=2), [0, 4, 0]),
        (AnnulusPotential(radius=2, stiffness=2), [4, 1, 1]),
        (HarmonicWellPotential(center=(1, -1), stiffness=2), [2, 5, 5]),
    ],
)
def test_builtin_potentials_return_batched_energy(
    potential: Callable[[torch.Tensor], torch.Tensor], expected: list[float]
) -> None:
    x = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, -3.0]], dtype=torch.float64)
    torch.testing.assert_close(potential(x), x.new_tensor(expected))


def test_half_space_potential_steers_eight_gaussians() -> None:
    mixture = make_eight_gaussians()
    sampler = UnderdampedLangevinSampler(num_steps=20)
    guided = EnergyGuidedScore(
        mixture.score,
        HalfSpacePotential(dim=0, threshold=0.0, side="positive", stiffness=15.0),
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        unguided, steered = (
            sampler.sample(score, (1024, 2), generator=torch.Generator().manual_seed(0))
            for score in (mixture.score, guided)
        )
    finally:
        torch.set_num_threads(previous)
    assert 0.4 < (unguided[:, 0] > -0.1).float().mean().item() < 0.6
    assert (steered[:, 0] > -0.1).float().mean().item() >= 0.9
    assert torch.isfinite(steered).all()
