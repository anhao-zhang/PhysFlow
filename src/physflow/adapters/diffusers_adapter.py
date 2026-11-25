"""Bridge VP diffusion models and PhysFlow's EDM coordinates."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar

import torch

from physflow.dynamics import KarrasSchedule
from physflow.samplers import (
    BaseSampler,
    EulerSampler,
    SymplecticSampler,
    UnderdampedLangevinSampler,
)

try:
    from diffusers import ConfigMixin, DDPMScheduler, SchedulerMixin
    from diffusers.configuration_utils import register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerOutput
except ModuleNotFoundError as exc:
    if exc.name != "diffusers":
        raise

    class ConfigMixin:
        pass

    class SchedulerMixin:
        pass

    DDPMScheduler = None
    _Init = TypeVar("_Init", bound=Callable[..., None])

    def register_to_config(init: _Init) -> _Init:
        return init


PredictionType = Literal["epsilon", "sample", "v_prediction"]
SamplerType = Literal["euler", "symplectic", "underdamped_langevin"]


def _validate_prediction(prediction_type: str, clip_sample_range: float) -> None:
    if prediction_type not in ("epsilon", "sample", "v_prediction"):
        raise ValueError(f"Unsupported prediction_type: {prediction_type!r}")
    if not (math.isfinite(clip_sample_range) and clip_sample_range > 0):
        raise ValueError("clip_sample_range must be finite and positive")


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _alphas_from_config(config: Mapping[str, Any]) -> torch.Tensor:
    if "alphas_cumprod" in config:
        return torch.as_tensor(config["alphas_cumprod"])
    trained_betas = config.get("trained_betas")
    if trained_betas is not None:
        betas = torch.as_tensor(trained_betas, dtype=torch.float32)
    else:
        count = config.get("num_train_timesteps", 1000)
        start = config.get("beta_start", 0.0001)
        end = config.get("beta_end", 0.02)
        schedule = config.get("beta_schedule", "linear")
        if schedule == "linear":
            betas = torch.linspace(start, end, count, dtype=torch.float32)
        elif schedule == "scaled_linear":
            betas = torch.linspace(math.sqrt(start), math.sqrt(end), count, dtype=torch.float32)
            betas = betas.square()
        elif schedule == "squaredcos_cap_v2":
            t = torch.linspace(0, 1, count + 1, dtype=torch.float64)
            signal = ((t + 0.008) * (math.pi / 2 / 1.008)).cos().square()
            betas = (1 - signal[1:] / signal[:-1]).clamp(max=0.999).float()
        else:
            raise ValueError(f"Unsupported beta_schedule: {schedule!r}")
    return (1 - betas).cumprod(dim=0)


def _training_sigmas(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    alphas = alphas_cumprod.detach().to(device="cpu", dtype=torch.float64)
    if alphas.ndim != 1 or not alphas.numel() or not ((alphas > 0) & (alphas < 1)).all():
        raise ValueError("alphas_cumprod must be a nonempty vector with values strictly in (0, 1)")
    if not (alphas[1:] < alphas[:-1]).all():
        raise ValueError("alphas_cumprod must be strictly decreasing")
    return ((1 - alphas) / alphas).sqrt()


def _lookup_timesteps(
    sigma: torch.Tensor, log_sigmas: torch.Tensor, continuous: bool
) -> torch.Tensor:
    if log_sigmas.numel() == 1:
        timestep = torch.zeros_like(sigma)
    else:
        query = sigma.log().clamp(min=log_sigmas[0], max=log_sigmas[-1])
        upper = torch.searchsorted(log_sigmas, query.contiguous()).clamp(1, log_sigmas.numel() - 1)
        lower = upper - 1
        weight = (query - log_sigmas[lower]) / (log_sigmas[upper] - log_sigmas[lower])
        timestep = lower + weight
    return timestep if continuous else timestep.round().long()


def _vp_scale(sigma: torch.Tensor) -> torch.Tensor:
    return torch.hypot(sigma, torch.ones_like(sigma))


def _edm_force(
    x: torch.Tensor,
    sigma: torch.Tensor,
    model_output: torch.Tensor,
    prediction_type: str,
    clip_sample: bool,
    clip_sample_range: float,
) -> torch.Tensor:
    if prediction_type == "sample":
        denoised = model_output
    else:
        if prediction_type == "epsilon":
            force = -model_output
        else:
            scale = _vp_scale(sigma)
            force = -(x * (sigma / scale) + model_output) / scale
        if not clip_sample:
            return force
        denoised = x + sigma * force
    if clip_sample:
        denoised = denoised.clamp(-clip_sample_range, clip_sample_range)
    return (denoised - x) / sigma


class DiffusersScoreWrapper:
    """Evaluate a VP UNet as an EDM score using cumulative alphas or a VP schedule mapping."""

    def __init__(
        self,
        unet: Callable[..., Any],
        alphas_cumprod: torch.Tensor | Mapping[str, Any],
        prediction_type: PredictionType = "epsilon",
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        continuous_timesteps: bool = False,
        unet_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _validate_prediction(prediction_type, clip_sample_range)
        self.unet = unet
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.continuous_timesteps = continuous_timesteps
        self.unet_kwargs = dict(unet_kwargs or {})
        if isinstance(alphas_cumprod, Mapping):
            alphas_cumprod = _alphas_from_config(alphas_cumprod)
        sigmas = _training_sigmas(alphas_cumprod)
        self._sigma_min = sigmas[0].item()
        self._sigma_max = sigmas[-1].item()
        self._log_sigmas = sigmas.log()
        self._tables: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @classmethod
    def from_pipeline(
        cls, pipe: Any, unet_kwargs: dict[str, Any] | None = None
    ) -> DiffusersScoreWrapper:
        """Use the model and VP training schedule of a Diffusers pipeline."""
        config = pipe.scheduler.config
        if not isinstance(config, Mapping):
            config = vars(config)
        return cls(
            pipe.unet,
            pipe.scheduler.alphas_cumprod,
            prediction_type=config.get("prediction_type", "epsilon"),
            clip_sample=config.get("clip_sample", False),
            clip_sample_range=config.get("clip_sample_range", 1.0),
            unet_kwargs=unet_kwargs,
        )

    @property
    def sigma_min(self) -> float:
        return self._sigma_min

    @property
    def sigma_max(self) -> float:
        return self._sigma_max

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        dtype = _work_dtype(x)
        sigma = torch.as_tensor(sigma, device=x.device, dtype=dtype)
        if sigma.ndim > 1 or (sigma.ndim == 1 and sigma.numel() not in (1, x.shape[0])):
            raise ValueError("sigma must be scalar or have shape (B,)")
        if not (torch.isfinite(sigma) & (sigma > 0)).all():
            raise ValueError("sigma must be finite and positive")
        key = (x.device, dtype)
        if key not in self._tables:
            self._tables[key] = self._log_sigmas.to(device=x.device, dtype=dtype)
        timestep = _lookup_timesteps(sigma, self._tables[key], self.continuous_timesteps)
        if sigma.ndim == 1:
            sigma = sigma.reshape(-1, *((1,) * (x.ndim - 1)))

        work_x = x.to(dtype=dtype)
        output = self.unet(
            (work_x / _vp_scale(sigma)).to(dtype=x.dtype), timestep, **self.unet_kwargs
        )
        model_output = output[0] if isinstance(output, tuple) else getattr(output, "sample", output)
        if not isinstance(model_output, torch.Tensor) or model_output.shape != x.shape:
            raise ValueError("The UNet prediction must be a tensor with the input shape")
        force = _edm_force(
            work_x,
            sigma,
            model_output.to(dtype=dtype),
            self.prediction_type,
            self.clip_sample,
            self.clip_sample_range,
        )
        return (force / sigma).to(dtype=x.dtype)


class PhysFlowDiffusersScheduler(SchedulerMixin, ConfigMixin):
    """One-model-call Euler or momentum scheduler for VP-trained models.

    The default sample space is EDM, for pipelines that call scale_model_input.
    Set sample_space="vp" for DDPMPipeline, which passes its state to the UNet directly.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        sampler_type: SamplerType = "symplectic",
        damping: float = 3.0,
        mass: float = 1.0,
        temperature: float = 0.01,
        rho: float = 7.0,
        sample_space: Literal["edm", "vp"] = "edm",
        init_velocity: Literal["zero", "drift"] = "zero",
        noise_min_sigma: float = 0.05,
        noise_max_sigma: float = math.inf,
    ) -> None:
        if DDPMScheduler is None:
            raise ImportError("Install optional dependency via: pip install -e '.[diffusers]'")
        _validate_prediction(prediction_type, clip_sample_range)
        if beta_schedule not in ("linear", "scaled_linear", "squaredcos_cap_v2"):
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule!r}")
        if sample_space not in ("edm", "vp"):
            raise ValueError("sample_space must be 'edm' or 'vp'")
        self.alphas_cumprod = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        ).alphas_cumprod
        self._sigmas_train = _training_sigmas(self.alphas_cumprod)
        self._schedule = KarrasSchedule(
            sigma_min=self._sigmas_train[0].item(), sigma_max=self._sigmas_train[-1].item(), rho=rho
        )
        self._momentum: SymplecticSampler | UnderdampedLangevinSampler | None
        if sampler_type == "euler":
            self._momentum = None
        elif sampler_type == "symplectic":
            self._momentum = SymplecticSampler(
                damping=damping, mass=mass, init_velocity=init_velocity
            )
        elif sampler_type == "underdamped_langevin":
            self._momentum = UnderdampedLangevinSampler(
                friction=damping,
                mass=mass,
                temperature=temperature,
                noise_min_sigma=noise_min_sigma,
                noise_max_sigma=noise_max_sigma,
            )
        else:
            raise ValueError(f"Unsupported sampler_type: {sampler_type!r}")
        self.num_inference_steps: int | None = None
        self.sigmas = torch.empty(0)
        self.timesteps = torch.empty(0, dtype=torch.long)
        self._step_index = 0
        self._velocity: torch.Tensor | None = None
        self._pending_kick: torch.Tensor | None = None

    @property
    def init_noise_sigma(self) -> float:
        if self.config.sample_space == "vp":
            return 1.0
        return self.sigmas[0].item() if self.sigmas.numel() else self._sigmas_train[-1].item()

    @property
    def step_index(self) -> int:
        return self._step_index

    def __len__(self) -> int:
        return self.config.num_train_timesteps

    def set_timesteps(
        self, num_inference_steps: int, device: torch.device | str | None = None
    ) -> None:
        sigmas = self._schedule.get_sigmas(num_inference_steps)
        timesteps = _lookup_timesteps(sigmas[:-1].double(), self._sigmas_train.log(), False)
        self.sigmas = sigmas.to(device=device)
        self.timesteps = timesteps.to(device=device)
        self.num_inference_steps = num_inference_steps
        self._step_index = 0
        self._velocity = None
        self._pending_kick = None

    def _current_sigma(self, timestep: float | torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps() before denoising")
        if self._step_index >= self.num_inference_steps:
            raise ValueError("The schedule has finished. Call set_timesteps() to start a new run")
        if timestep != self.timesteps[self._step_index]:
            raise ValueError("timestep must match the current entry of scheduler.timesteps")
        return self.sigmas[self._step_index].to(device=sample.device, dtype=_work_dtype(sample))

    def scale_model_input(
        self, sample: torch.Tensor, timestep: float | torch.Tensor
    ) -> torch.Tensor:
        sigma = self._current_sigma(timestep, sample)
        if self.config.sample_space == "vp":
            return sample
        return (sample.to(dtype=sigma.dtype) / _vp_scale(sigma)).to(dtype=sample.dtype)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple[torch.Tensor]:
        sigma = self._current_sigma(timestep, sample)
        sigma_next = self.sigmas[self._step_index + 1].to(sigma)
        x = sample.to(dtype=sigma.dtype)
        if self.config.sample_space == "vp":
            x = x * _vp_scale(sigma)
        force = _edm_force(
            x,
            sigma,
            model_output.to(dtype=sigma.dtype),
            self.config.prediction_type,
            self.config.clip_sample,
            self.config.clip_sample_range,
        )
        if self._momentum is None or sigma_next == 0:
            x_next = x + (sigma - sigma_next) * force
            self._velocity = None
            self._pending_kick = None
        else:
            velocity = self._velocity
            if velocity is None:
                velocity = force if self.config.init_velocity == "drift" else torch.zeros_like(x)
            else:
                # The current model call supplies the previous interval's endpoint force.
                velocity = torch.lerp(velocity, force, self._pending_kick)
            x_next, self._velocity, self._pending_kick = self._momentum._start_interval(
                x, velocity, force, sigma, sigma_next, generator
            )
        self._step_index += 1
        if self.config.sample_space == "vp":
            x_next = x_next / _vp_scale(sigma_next)
        x_next = x_next.to(dtype=sample.dtype)
        return SchedulerOutput(prev_sample=x_next) if return_dict else (x_next,)


def to_diffusers_scheduler(
    sampler: BaseSampler | str, base_config: Mapping[str, Any], **overrides: Any
) -> PhysFlowDiffusersScheduler:
    """Build a PhysFlow scheduler from a VP scheduler config and a sampler or its name."""
    keys = (
        "num_train_timesteps",
        "beta_start",
        "beta_end",
        "beta_schedule",
        "prediction_type",
        "clip_sample",
        "clip_sample_range",
    )
    config = {key: base_config[key] for key in keys if key in base_config}
    if isinstance(sampler, str):
        config["sampler_type"] = sampler
    elif isinstance(sampler, EulerSampler):
        config["sampler_type"] = "euler"
    elif isinstance(sampler, SymplecticSampler):
        config.update(
            sampler_type="symplectic",
            damping=sampler.damping,
            mass=sampler.mass,
            init_velocity=sampler.init_velocity,
        )
    elif isinstance(sampler, UnderdampedLangevinSampler):
        config.update(
            sampler_type="underdamped_langevin",
            damping=sampler.friction,
            mass=sampler.mass,
            temperature=sampler.temperature,
            noise_min_sigma=sampler.noise_min_sigma,
            noise_max_sigma=sampler.noise_max_sigma,
        )
    else:
        raise ValueError("Diffusers scheduler supports euler, symplectic, and underdamped_langevin")
    if isinstance(sampler, BaseSampler) and isinstance(sampler.schedule, KarrasSchedule):
        config["rho"] = sampler.schedule.rho
    return PhysFlowDiffusersScheduler(**(config | overrides))
