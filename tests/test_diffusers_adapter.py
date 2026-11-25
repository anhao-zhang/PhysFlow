from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from physflow import (
    BaseSampler,
    EnergyGuidedScore,
    EulerSampler,
    HeunSampler,
    KarrasSchedule,
    PredictorCorrectorSampler,
    SymplecticSampler,
    UnderdampedLangevinSampler,
)
from physflow.adapters import (
    DiffusersScoreWrapper,
    PhysFlowDiffusersScheduler,
    to_diffusers_scheduler,
)

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = torch.tensor([0.99, 0.9, 0.6, 0.3, 0.1], dtype=torch.float64)
SIGMAS = ((1 - ALPHAS) / ALPHAS).sqrt()


class GaussianUNet:
    """Return exact VP predictions for a centered Gaussian data prior."""

    def __init__(self, prediction_type: str = "epsilon", data_variance: float = 0.3) -> None:
        self.prediction_type = prediction_type
        self.data_variance = data_variance
        self.calls: list[torch.Tensor] = []

    def __call__(self, z: torch.Tensor, t: torch.Tensor) -> SimpleNamespace:
        self.calls.append(t.detach().clone())
        alpha = ALPHAS.to(z)[t]
        if alpha.ndim:
            alpha = alpha[:, None, None, None]
        variance = alpha * self.data_variance + 1 - alpha
        denoised = z * alpha.sqrt() * self.data_variance / variance
        noise = z * (1 - alpha).sqrt() / variance
        if self.prediction_type == "epsilon":
            prediction = noise
        elif self.prediction_type == "sample":
            prediction = denoised
        else:
            prediction = alpha.sqrt() * noise - (1 - alpha).sqrt() * denoised
        return SimpleNamespace(sample=prediction)


@pytest.mark.parametrize("prediction", ["epsilon", "sample", "v_prediction"])
def test_wraps_exact_gaussian_predictions(
    prediction: Literal["epsilon", "sample", "v_prediction"],
) -> None:
    unet = GaussianUNet(prediction, data_variance=0.3)
    wrapper = DiffusersScoreWrapper(unet, ALPHAS, prediction_type=prediction)
    x = torch.linspace(-2, 2, 2 * 3 * 4 * 4, dtype=torch.float64).reshape(2, 3, 4, 4)
    sigma = SIGMAS[[0, 3]]
    score = wrapper(x, sigma)
    torch.testing.assert_close(
        score, -x / (0.3 + sigma[:, None, None, None].square()), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(unet.calls[0], torch.tensor([0, 3]))
    assert wrapper.sigma_min == SIGMAS[0].item()
    assert wrapper.sigma_max == SIGMAS[-1].item()


def test_continuous_timestep_lookup_clamps_and_scales_unet_input() -> None:
    x = torch.ones(3, 1, 2, 2, dtype=torch.float64)
    sigma = torch.stack((SIGMAS[0] / 4, (SIGMAS[1] * SIGMAS[2]).sqrt(), SIGMAS[-1] * 4))

    def unet(z: torch.Tensor, t: torch.Tensor, *, scale: float) -> torch.Tensor:
        assert scale == 3
        torch.testing.assert_close(t, x.new_tensor([0, 1.5, 4]))
        torch.testing.assert_close(z, x / (1 + sigma[:, None, None, None].square()).sqrt())
        return torch.zeros_like(z)

    wrapper = DiffusersScoreWrapper(
        unet, ALPHAS, continuous_timesteps=True, unet_kwargs={"scale": 3}
    )
    torch.testing.assert_close(wrapper(x, sigma), torch.zeros_like(x))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_small_sigma_epsilon_conversion_preserves_dtype(dtype: torch.dtype) -> None:
    def unet(z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor]:
        assert z.dtype == dtype
        return (torch.full_like(z, 0.125),)

    x = torch.ones(2, 3, 2, 2, dtype=dtype)
    sigma = x.new_tensor(0.002)
    wrapper = DiffusersScoreWrapper(unet, ALPHAS)
    expected = torch.full_like(x, -0.125 / sigma.item())
    torch.testing.assert_close(wrapper(x, sigma), expected)


def test_pipeline_configuration_and_clipped_denoised_prediction() -> None:
    def unet(z: torch.Tensor, t: torch.Tensor, *, value: float) -> torch.Tensor:
        return torch.full_like(z, value)

    pipe = SimpleNamespace(
        unet=unet,
        scheduler=SimpleNamespace(
            alphas_cumprod=ALPHAS,
            config={"prediction_type": "sample", "clip_sample": True, "clip_sample_range": 0.7},
        ),
    )
    x = torch.full((2, 3, 4, 4), -2.0)
    sigma = torch.tensor(0.5)
    wrapper = DiffusersScoreWrapper.from_pipeline(pipe, {"value": 3.0})
    torch.testing.assert_close(wrapper(x, sigma), torch.full_like(x, 2.7 / 0.25))


@pytest.mark.parametrize(
    ("sampler_type", "nfe"),
    [
        (EulerSampler, 6),
        (HeunSampler, 11),
        (SymplecticSampler, 6),
        (UnderdampedLangevinSampler, 6),
        (PredictorCorrectorSampler, 11),
    ],
)
def test_adapter_integrates_images_and_energy_guidance(
    sampler_type: type[BaseSampler], nfe: int
) -> None:
    unet = GaussianUNet()
    wrapper = DiffusersScoreWrapper(unet, ALPHAS)
    schedule = KarrasSchedule(sigma_min=wrapper.sigma_min, sigma_max=wrapper.sigma_max)
    sampler = sampler_type(schedule, num_steps=6)

    def energy(x: torch.Tensor) -> torch.Tensor:
        return x.square().mean(dim=(1, 2, 3))

    guided = EnergyGuidedScore(wrapper, energy)
    x, trajectory = sampler.sample(
        guided, (2, 3, 8, 8), generator=torch.Generator().manual_seed(42), return_traj=True
    )
    assert x.shape == (2, 3, 8, 8)
    assert torch.isfinite(trajectory.stack_xs()).all()
    assert trajectory.nfe == len(unet.calls) == nfe


@pytest.mark.parametrize("beta_schedule", ["linear", "scaled_linear", "squaredcos_cap_v2"])
def test_scheduler_factory_retains_config_and_training_schedule(beta_schedule: str) -> None:
    diffusers = pytest.importorskip("diffusers")
    base = diffusers.DDPMScheduler(
        num_train_timesteps=100,
        prediction_type="v_prediction",
        beta_schedule=beta_schedule,
        clip_sample_range=0.5,
    )
    sampler = SymplecticSampler(
        KarrasSchedule(rho=5), damping=2.75, mass=0.4, init_velocity="drift"
    )
    scheduler = to_diffusers_scheduler(sampler, base.config)
    assert isinstance(scheduler, diffusers.SchedulerMixin)
    assert isinstance(scheduler, diffusers.ConfigMixin)
    assert scheduler.config.sampler_type == "symplectic"
    assert scheduler.config.prediction_type == "v_prediction"
    assert scheduler.config.beta_schedule == beta_schedule
    assert scheduler.config.clip_sample_range == 0.5
    assert scheduler.config.damping == 2.75
    assert scheduler.config.mass == 0.4
    assert scheduler.config.rho == 5
    assert scheduler.config.init_velocity == "drift"
    assert len(scheduler) == 100 and scheduler.order == 1
    torch.testing.assert_close(scheduler.alphas_cumprod, base.alphas_cumprod)
    from_config = DiffusersScoreWrapper(GaussianUNet(), base.config)
    from_alphas = DiffusersScoreWrapper(GaussianUNet(), base.alphas_cumprod)
    assert from_config.sigma_min == pytest.approx(from_alphas.sigma_min)
    assert from_config.sigma_max == pytest.approx(from_alphas.sigma_max)
    reloaded = PhysFlowDiffusersScheduler.from_config(scheduler.config)
    assert reloaded.config.damping == 2.75
    assert reloaded.init_noise_sigma == scheduler.init_noise_sigma


@pytest.mark.parametrize(
    "sampler_type", [EulerSampler, SymplecticSampler, UnderdampedLangevinSampler]
)
def test_scheduler_matches_native_sampler_and_resets(sampler_type: type[BaseSampler]) -> None:
    pytest.importorskip("diffusers")
    # Only eight training points for many inference steps deliberately gives duplicate timesteps.
    original = sampler_type(num_steps=12)
    scheduler = to_diffusers_scheduler(original, {"num_train_timesteps": 8})
    scheduler.set_timesteps(original.num_steps)
    assert scheduler.sigmas.shape == (13,) and scheduler.sigmas[-1] == 0
    assert scheduler.init_noise_sigma == scheduler.sigmas[0].item()
    assert scheduler.timesteps.unique().numel() < scheduler.timesteps.numel()
    x_init = torch.randn((2, 3, 4, 4), generator=torch.Generator().manual_seed(8))

    def unet(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 0.2 * z + 0.01 * t

    wrapper = DiffusersScoreWrapper(unet, scheduler.alphas_cumprod)
    sampler = sampler_type(KarrasSchedule(wrapper.sigma_min, wrapper.sigma_max), num_steps=12)
    expected = sampler.sample(
        wrapper, x_init.shape, x_init=x_init, generator=torch.Generator().manual_seed(3)
    )
    first_timestep = scheduler.timesteps[0]
    first_prediction = unet(scheduler.scale_model_input(x_init, first_timestep), first_timestep)
    scheduler.step(first_prediction, first_timestep, x_init)
    if sampler_type is not EulerSampler:
        assert scheduler._velocity is not None
    replays = []
    for _ in range(2):
        scheduler.set_timesteps(original.num_steps)
        assert scheduler.step_index == 0 and scheduler._velocity is None
        x = x_init
        generator = torch.Generator().manual_seed(3)
        for i, t in enumerate(scheduler.timesteps):
            scaled = scheduler.scale_model_input(x, t)
            torch.testing.assert_close(scaled, x / (1 + scheduler.sigmas[i].square()).sqrt())
            (x,) = scheduler.step(unet(scaled, t), t, x, generator=generator, return_dict=False)
        assert scheduler.step_index == len(scheduler.timesteps)
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x, expected)
        replays.append(x)
    assert torch.equal(*replays)


def test_scheduler_is_a_drop_in_for_ddpm_pipeline_in_vp_space() -> None:
    diffusers = pytest.importorskip("diffusers")
    unet = diffusers.UNet2DModel(
        sample_size=4,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(4,),
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
        norm_num_groups=1,
        add_attention=False,
    )
    pipe = diffusers.DDPMPipeline(
        unet=unet, scheduler=diffusers.DDPMScheduler(num_train_timesteps=8)
    )
    pipe.scheduler = to_diffusers_scheduler("symplectic", pipe.scheduler.config, sample_space="vp")
    images = pipe(
        num_inference_steps=2, output_type="np", generator=torch.Generator().manual_seed(3)
    ).images
    assert images.shape == (1, 4, 4, 3)
    assert all(map(math.isfinite, images.ravel()))


def test_importing_without_optional_diffusers() -> None:
    code = """
import sys
import torch
import physflow
assert 'diffusers' not in sys.modules
sys.modules['diffusers'] = None
from physflow.adapters import DiffusersScoreWrapper, PhysFlowDiffusersScheduler
wrapper = DiffusersScoreWrapper(lambda z, t: torch.zeros_like(z), {'num_train_timesteps': 10})
x = torch.randn(1, 3, 2, 2)
assert torch.equal(wrapper(x, torch.tensor(0.5)), torch.zeros_like(x))
try:
    PhysFlowDiffusersScheduler()
except ImportError as exc:
    assert "pip install -e '.[diffusers]'" in str(exc)
else:
    raise AssertionError('missing optional-dependency error')
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=60)


def test_synthetic_benchmark_smoke(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_diffusers.py",
            "--mode",
            "synthetic",
            "--steps",
            "6",
            "--num-images",
            "2",
            "--device",
            "cpu",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=os.environ | {"HF_HUB_OFFLINE": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert "| Heun | 6 | 11 |" in result.stdout
    assert "| Underdamped Langevin | 6 | 6 |" in result.stdout
    assert (tmp_path / "phase3_diffusers_comparison.png").read_bytes().startswith(b"\x89PNG\r\n")
