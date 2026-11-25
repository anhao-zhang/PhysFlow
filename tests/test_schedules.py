from __future__ import annotations

import pytest
import torch

from physflow import KarrasSchedule, LinearSchedule, NoiseSchedule


@pytest.mark.parametrize("schedule_type", [KarrasSchedule, LinearSchedule])
@pytest.mark.parametrize("num_steps", [1, 2, 20])
def test_schedule_descends_to_zero(schedule_type: type[NoiseSchedule], num_steps: int) -> None:
    sigmas = schedule_type().get_sigmas(num_steps, device="cpu", dtype=torch.float64)
    assert sigmas.shape == (num_steps + 1,)
    assert sigmas.dtype == torch.float64
    assert sigmas.device.type == "cpu"
    assert sigmas[0].item() == 10.0
    assert sigmas[-1].item() == 0.0
    assert (sigmas[:-1] > sigmas[1:]).all()
    if num_steps > 1:
        assert sigmas[-2].item() == 0.002


def test_schedule_spacing() -> None:
    karras = KarrasSchedule(sigma_min=1.0, sigma_max=9.0, rho=2.0)
    linear = LinearSchedule(sigma_min=1.0, sigma_max=9.0)
    torch.testing.assert_close(karras.get_sigmas(3), torch.tensor([9.0, 4.0, 1.0, 0.0]))
    torch.testing.assert_close(linear.get_sigmas(3), torch.tensor([9.0, 5.0, 1.0, 0.0]))


@pytest.mark.parametrize("schedule_type", [KarrasSchedule, LinearSchedule])
@pytest.mark.parametrize(("sigma_min", "sigma_max"), [(0.0, 1.0), (2.0, 1.0), (1.0, float("inf"))])
def test_invalid_sigma_range(
    schedule_type: type[KarrasSchedule] | type[LinearSchedule], sigma_min: float, sigma_max: float
) -> None:
    with pytest.raises(ValueError, match="sigma_max > sigma_min"):
        schedule_type(sigma_min=sigma_min, sigma_max=sigma_max)


@pytest.mark.parametrize("rho", [0.0, -1.0, float("nan")])
def test_invalid_rho(rho: float) -> None:
    with pytest.raises(ValueError, match="rho"):
        KarrasSchedule(rho=rho)


@pytest.mark.parametrize("schedule", [KarrasSchedule(), LinearSchedule()])
def test_invalid_steps(schedule: NoiseSchedule) -> None:
    with pytest.raises(ValueError, match="num_steps"):
        schedule.get_sigmas(0)


def test_schedule_rejects_levels_that_round_to_the_same_value() -> None:
    schedule = LinearSchedule(sigma_min=1.0, sigma_max=1.0001)
    with pytest.raises(ValueError, match="strictly descending"):
        schedule.get_sigmas(2, dtype=torch.float16)
