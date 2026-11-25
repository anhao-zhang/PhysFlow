"""Common sampler state, trajectory storage, and integration loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch

from physflow.dynamics.schedules import KarrasSchedule, NoiseSchedule

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SamplerState:
    x: torch.Tensor
    sigma: torch.Tensor
    v: torch.Tensor | None = None
    step_index: int = 0
    nfe: int = 0


@dataclass
class Trajectory:
    xs: list[torch.Tensor]
    sigmas: list[float]
    vs: list[torch.Tensor] | None = None
    nfe: int = 0

    def stack_xs(self) -> torch.Tensor:
        """Stack position snapshots in (time, batch, *dims) order."""
        return torch.stack(self.xs)


class BaseSampler(ABC):
    """Integrate a score-driven trajectory along a descending noise schedule."""

    def __init__(self, schedule: NoiseSchedule | None = None, num_steps: int = 20) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        self.schedule = KarrasSchedule() if schedule is None else schedule
        self.num_steps = num_steps

    def init_state(
        self,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        x_init: torch.Tensor | None = None,
    ) -> SamplerState:
        """Use supplied positions or draw a Gaussian at the initial noise level."""
        if x_init is not None and x_init.shape != shape:
            raise ValueError("x_init must match shape")
        if device is None:
            device = x_init.device if x_init is not None else "cpu"
        if dtype is None:
            dtype = x_init.dtype if x_init is not None else torch.float32
        sigma = self.schedule.get_sigmas(self.num_steps, device=device, dtype=dtype)[0]

        if x_init is None:
            x = torch.randn(shape, device=device, dtype=dtype, generator=generator) * sigma
        else:
            x = x_init.to(device=device, dtype=dtype)
        return SamplerState(x=x, sigma=sigma)

    @abstractmethod
    def step(
        self,
        state: SamplerState,
        sigma_next: torch.Tensor,
        score_fn: ScoreFn,
        generator: torch.Generator | None = None,
    ) -> SamplerState:
        """Advance one noise interval and update the cumulative NFE count."""

    @torch.no_grad()
    def sample(
        self,
        score_fn: ScoreFn,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        x_init: torch.Tensor | None = None,
        return_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Trajectory]:
        """Run the sampler and optionally record positions and momenta on their device."""
        state = self.init_state(shape, device, dtype, generator, x_init)
        sigmas = self.schedule.get_sigmas(self.num_steps, state.x.device, state.x.dtype)
        trajectory = None
        if return_traj:
            trajectory = Trajectory(
                xs=[state.x.detach().clone()],
                sigmas=sigmas.tolist(),
                vs=None if state.v is None else [state.v.detach().clone()],
            )

        for sigma_next in sigmas[1:]:
            state = self.step(state, sigma_next, score_fn, generator)
            if trajectory is not None:
                trajectory.xs.append(state.x.detach().clone())
                if trajectory.vs is not None and state.v is not None:
                    trajectory.vs.append(state.v.detach().clone())

        if trajectory is not None:
            trajectory.nfe = state.nfe
            return state.x, trajectory
        return state.x
