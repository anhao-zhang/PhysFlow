from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Literal

import pytest
import torch

from physflow import (
    BaseSampler,
    EulerSampler,
    HeunSampler,
    LinearSchedule,
    PredictorCorrectorSampler,
    SamplerState,
    SymplecticSampler,
    Trajectory,
    UnderdampedLangevinSampler,
    kinetic_energy_trace,
    make_eight_gaussians,
    sliced_wasserstein_2d,
)

PHYSICS_SAMPLERS = (SymplecticSampler, UnderdampedLangevinSampler, PredictorCorrectorSampler)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("sampler_type", "kwargs", "parameter"),
    [
        (SymplecticSampler, {"damping": -1.0}, "damping"),
        (SymplecticSampler, {"mass": 0.0}, "mass"),
        (SymplecticSampler, {"init_velocity": "invalid"}, "init_velocity"),
        (UnderdampedLangevinSampler, {"friction": -1.0}, "friction"),
        (UnderdampedLangevinSampler, {"mass": -1.0}, "mass"),
        (UnderdampedLangevinSampler, {"temperature": -1.0}, "temperature"),
        (UnderdampedLangevinSampler, {"noise_min_sigma": -1.0}, "noise_min_sigma"),
        (UnderdampedLangevinSampler, {"noise_max_sigma": 0.0}, "noise_max_sigma"),
        (PredictorCorrectorSampler, {"predictor": "invalid"}, "predictor"),
        (PredictorCorrectorSampler, {"corrector_steps": -1}, "corrector_steps"),
        (PredictorCorrectorSampler, {"snr": -1.0}, "snr"),
        (PredictorCorrectorSampler, {"max_corrector_step_size": 0.0}, "max_corrector_step_size"),
        (PredictorCorrectorSampler, {"corrector_min_sigma": -1.0}, "corrector_min_sigma"),
    ],
)
def test_invalid_sampler_parameters(
    sampler_type: type[BaseSampler], kwargs: dict[str, object], parameter: str
) -> None:
    with pytest.raises(ValueError, match=parameter):
        sampler_type(**kwargs)


@pytest.mark.parametrize("num_steps", [1, 6])
@pytest.mark.parametrize(
    ("sampler_type", "kwargs", "expected_nfe_at_six_steps"),
    [
        (SymplecticSampler, {}, 6),
        (SymplecticSampler, {"init_velocity": "drift"}, 6),
        (UnderdampedLangevinSampler, {}, 6),
        # Three of the six destination sigmas are at or above the default cutoff.
        (PredictorCorrectorSampler, {"corrector_steps": 3}, 15),
        (PredictorCorrectorSampler, {"predictor": "heun", "corrector_steps": 2}, 17),
        (PredictorCorrectorSampler, {"corrector_steps": 3, "corrector_min_sigma": 0.0}, 21),
    ],
)
def test_nfe_counts_observed_score_calls(
    sampler_type: type[BaseSampler],
    kwargs: dict[str, object],
    expected_nfe_at_six_steps: int,
    num_steps: int,
) -> None:
    sampler = sampler_type(num_steps=num_steps, **kwargs)
    calls: list[float] = []

    def score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        calls.append(sigma.item())
        return -x / (1 + sigma.square())

    x_init = torch.ones(4, 2)
    _, traj = sampler.sample(
        score,
        x_init.shape,
        x_init=x_init,
        return_traj=True,
        generator=torch.Generator().manual_seed(0),
    )
    assert traj.nfe == len(calls) == (1 if num_steps == 1 else expected_nfe_at_six_steps)
    assert all(sigma > 0 for sigma in calls)
    if sampler_type is not PredictorCorrectorSampler:
        assert calls == traj.sigmas[:-1]


@pytest.mark.parametrize("sampler_type", PHYSICS_SAMPLERS)
def test_terminal_step_is_deterministic_euler(sampler_type: type[BaseSampler]) -> None:
    sampler = sampler_type(num_steps=1)
    x = torch.ones(3, 2, dtype=torch.float64)
    generator = torch.Generator().manual_seed(5)
    generator_state = generator.get_state()

    def score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        assert sigma.item() == 10.0
        return -0.005 * x

    samples, traj = sampler.sample(score, x.shape, x_init=x, generator=generator, return_traj=True)
    torch.testing.assert_close(samples, 0.5 * x)
    assert torch.equal(generator.get_state(), generator_state)
    if traj.vs is not None:
        torch.testing.assert_close(traj.vs[-1], -0.05 * x)


def test_drift_initialization_and_symplectic_cached_force() -> None:
    schedule = LinearSchedule(sigma_min=1, sigma_max=2)
    sampler = SymplecticSampler(schedule, num_steps=2, damping=2, init_velocity="drift")
    x_init = torch.tensor([[1.0, -0.5]], dtype=torch.float64)

    def score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return -0.25 * x

    state = sampler.init_state(x_init.shape, x_init=x_init)
    drift = state.sigma * score(x_init, state.sigma)
    stepped = sampler.step(state, state.sigma.new_tensor(1.0), score)
    x_expected = x_init + drift
    torch.testing.assert_close(stepped.x, x_expected)
    torch.testing.assert_close(stepped.force, score(x_expected, stepped.sigma))
    alpha = math.exp(-math.log(2))
    v_expected = alpha * drift + (1 - alpha) * stepped.force
    torch.testing.assert_close(stepped.v, v_expected)
    assert stepped.step_index == 1
    assert stepped.nfe == 2

    def unexpected_score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        pytest.fail("The final step should reuse the cached force")

    terminal = sampler.step(stepped, state.sigma.new_tensor(0.0), unexpected_score)
    assert terminal.nfe == stepped.nfe
    assert terminal.force is None
    torch.testing.assert_close(terminal.x, stepped.x + stepped.force)
    _, traj = sampler.sample(score, x_init.shape, x_init=x_init, return_traj=True)
    torch.testing.assert_close(traj.vs[0], drift)
    assert traj.stack_vs().shape == traj.stack_xs().shape


@pytest.mark.parametrize(
    "noise_kwargs",
    [
        {"temperature": 0.0},
        {"noise_min_sigma": 20.0},
        {"noise_min_sigma": 0.0, "noise_max_sigma": 0.0},
    ],
)
def test_langevin_reduces_to_symplectic_without_thermal_noise(
    noise_kwargs: dict[str, float],
) -> None:
    mixture = make_eight_gaussians()
    x_init = mixture.sample(16, torch.Generator().manual_seed(10), dtype=torch.float64)
    symplectic = SymplecticSampler(damping=2.5, mass=0.75)
    langevin = UnderdampedLangevinSampler(friction=2.5, mass=0.75, **noise_kwargs)
    outputs = [
        sampler.sample(mixture.score, x_init.shape, x_init=x_init, return_traj=True)
        for sampler in (symplectic, langevin)
    ]
    first, second = (traj for _, traj in outputs)
    torch.testing.assert_close(first.stack_xs(), second.stack_xs(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(first.stack_vs(), second.stack_vs(), rtol=1e-5, atol=1e-5)


def test_langevin_noise_is_applied_between_half_drifts_at_inclusive_boundary() -> None:
    sampler = UnderdampedLangevinSampler(
        friction=2, mass=2, temperature=0.25, noise_min_sigma=1, noise_max_sigma=1
    )
    x = torch.zeros(2, 2, dtype=torch.float64)
    state = SamplerState(x=x, sigma=x.new_tensor(2), v=torch.ones_like(x))

    def score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(2), dtype=x.dtype)
    next_state = sampler.step(
        state, x.new_tensor(1), score, generator=torch.Generator().manual_seed(2)
    )
    delta_tau = math.log(2)
    alpha = math.exp(-0.5 * delta_tau)
    amplitude = 0.5 * math.sqrt(1 - math.exp(-2 * delta_tau))
    torch.testing.assert_close(next_state.x, alpha + 0.5 * amplitude * noise)
    torch.testing.assert_close(next_state.v, alpha * (alpha + amplitude * noise))


@pytest.mark.parametrize(
    ("predictor", "baseline"), [("euler", EulerSampler), ("heun", HeunSampler)]
)
def test_disabling_corrector_matches_baseline(
    predictor: Literal["euler", "heun"], baseline: type[BaseSampler]
) -> None:
    mixture = make_eight_gaussians()
    pc = PredictorCorrectorSampler(predictor=predictor, corrector_steps=0, num_steps=4)
    expected, expected_traj = baseline(num_steps=4).sample(
        mixture.score, (16, 2), generator=torch.Generator().manual_seed(4), return_traj=True
    )
    actual, traj = pc.sample(
        mixture.score, (16, 2), generator=torch.Generator().manual_seed(4), return_traj=True
    )
    torch.testing.assert_close(actual, expected)
    assert traj.nfe == expected_traj.nfe


@pytest.mark.parametrize("step_cap", [0.01, 1e-6])
def test_corrector_uses_one_adaptive_step_size_for_the_image_batch(step_cap: float) -> None:
    sampler = PredictorCorrectorSampler(snr=0.5, max_corrector_step_size=step_cap)
    x = torch.zeros(2, 1, 2, 2, dtype=torch.float64)
    state = SamplerState(x=x, sigma=x.new_tensor(2))
    gradient = torch.zeros_like(x)
    gradient[1] = 100

    def score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return gradient if sigma == 1 else torch.zeros_like(x)

    actual = sampler.step(state, x.new_tensor(1), score, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(1), dtype=x.dtype)
    mean_noise_norm = (noise[0].norm().item() + noise[1].norm().item()) / 2
    step = min(2 * (0.5 * mean_noise_norm / (100 + 1e-8)) ** 2, step_cap)
    torch.testing.assert_close(actual.x, step * gradient + math.sqrt(2 * step) * noise)
    assert actual.nfe == 2


@pytest.mark.parametrize(("sigma_next", "correction_applied"), [(0.099, False), (0.1, True)])
def test_corrector_cutoff_controls_score_calls_and_random_draws(
    sigma_next: float, correction_applied: bool
) -> None:
    x = torch.zeros(2, 2, dtype=torch.float64)
    state = SamplerState(x=x, sigma=x.new_tensor(0.2))
    generator = torch.Generator().manual_seed(5)
    rng_before = generator.get_state()

    def score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    result = PredictorCorrectorSampler().step(state, x.new_tensor(sigma_next), score, generator)
    assert result.nfe == 1 + correction_applied
    assert torch.equal(generator.get_state(), rng_before) is not correction_applied
    assert torch.equal(result.x, x) is not correction_applied


@pytest.mark.parametrize("sampler_type", PHYSICS_SAMPLERS)
def test_seeded_stability_and_eight_gaussian_quality(sampler_type: type[BaseSampler]) -> None:
    mixture = make_eight_gaussians()
    sampler = sampler_type(num_steps=20)
    x, traj = sampler.sample(
        mixture.score, (1024, 2), generator=torch.Generator().manual_seed(0), return_traj=True
    )
    replay = sampler.sample(mixture.score, (1024, 2), generator=torch.Generator().manual_seed(0))
    target = mixture.sample(4096, torch.Generator().manual_seed(1))
    assert torch.equal(replay, x)
    assert torch.isfinite(traj.stack_xs()).all()
    assert sliced_wasserstein_2d(x, target) < 0.15

    velocities = traj.stack_vs()
    if sampler_type is PredictorCorrectorSampler:
        assert velocities is None
        with pytest.raises(ValueError, match="velocity"):
            kinetic_energy_trace(traj)
    else:
        assert velocities.shape == (21, 1024, 2)
        assert torch.isfinite(velocities).all()
        energy = kinetic_energy_trace(traj)
        assert len(energy) == 21
        assert energy[0] == 0
        assert all(math.isfinite(value) and value >= 0 for value in energy)


def test_kinetic_energy_averages_per_particle() -> None:
    velocity = torch.tensor([[[3.0, 4.0]], [[0.0, 5.0]]], dtype=torch.float16)
    traj = Trajectory(xs=[], sigmas=[], vs=[torch.zeros_like(velocity), velocity])
    assert kinetic_energy_trace(traj, mass=2) == [0.0, 25.0]
    assert Trajectory(xs=[], sigmas=[], vs=[]).stack_vs() is None
