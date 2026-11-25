"""Compare PhysFlow samplers on a VP-trained UNet at low NFE."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Literal

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from physflow import (
    EulerSampler,
    HeunSampler,
    KarrasSchedule,
    SymplecticSampler,
    Trajectory,
    UnderdampedLangevinSampler,
    kinetic_energy_trace,
    trajectory_curvature,
)
from physflow.adapters import DiffusersScoreWrapper


class SyntheticUNet:
    """Return the VP conditional mean of the noise for a unit Gaussian data prior."""

    def __init__(self, alphas: torch.Tensor) -> None:
        self.noise_scale = (1 - alphas).sqrt()

    def __call__(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.noise_scale[t] * z


def load_model(
    mode: Literal["pretrained", "synthetic"], model_id: str, device: torch.device
) -> tuple[DiffusersScoreWrapper, tuple[int, int, int], torch.dtype]:
    if mode == "synthetic":
        betas = torch.linspace(0.0001, 0.02, 1000, device=device)
        alphas = (1 - betas).cumprod(dim=0)
        return DiffusersScoreWrapper(SyntheticUNet(alphas), alphas), (3, 8, 8), torch.float32

    try:
        from diffusers import DDPMPipeline
    except ModuleNotFoundError as exc:
        if exc.name != "diffusers":
            raise
        raise ImportError("Install optional dependency via: pip install -e '.[diffusers]'") from exc

    pipe = DDPMPipeline.from_pretrained(model_id).to(device)
    pipe.unet.eval()
    config = pipe.unet.config
    size = config.sample_size
    height, width = (size, size) if isinstance(size, int) else size
    return (
        DiffusersScoreWrapper.from_pipeline(pipe),
        (config.in_channels, height, width),
        pipe.unet.dtype,
    )


@dataclass
class ComparisonRow:
    sampler: str
    steps: int
    nfe: int
    elapsed: float
    images: torch.Tensor
    turning_angles: list[float]
    kinetic_energy: list[float]


def benchmark(
    wrapper: DiffusersScoreWrapper,
    image_shape: tuple[int, int, int],
    steps: list[int],
    seeds: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> list[ComparisonRow]:
    schedule = KarrasSchedule(sigma_min=wrapper.sigma_min, sigma_max=wrapper.sigma_max)
    initial_noise = torch.stack(
        [
            torch.randn(
                image_shape,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            for seed in seeds
        ]
    )
    x_init = initial_noise * wrapper.sigma_max
    with torch.no_grad():
        wrapper(x_init, x_init.new_tensor(wrapper.sigma_max))
    samplers = {
        "Euler": EulerSampler,
        "Heun": HeunSampler,
        "Symplectic": SymplecticSampler,
        # The image signal is in [-1, 1]; reserve thermal noise for that scale.
        "Underdamped Langevin": partial(UnderdampedLangevinSampler, noise_max_sigma=1.0),
    }
    rows = []
    print("| Sampler | Steps | NFE | Curvature (rad) | Mean KE | Wall Time (s) |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for num_steps in steps:
        for name, sampler_type in samplers.items():
            sampler = sampler_type(schedule=schedule, num_steps=num_steps)
            generator = torch.Generator(device=device).manual_seed(seeds[-1] + 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            images, trajectory = sampler.sample(
                wrapper, x_init.shape, x_init=x_init, generator=generator, return_traj=True
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - start

            energy = kinetic_energy_trace(trajectory) if trajectory.vs is not None else []
            curvature = trajectory_curvature(trajectory)
            xs = list(trajectory.stack_xs().cpu().unbind())
            angles = [
                trajectory_curvature(
                    Trajectory(xs[i - 1 : i + 2], trajectory.sigmas[i - 1 : i + 2])
                )
                for i in range(1, num_steps)
            ]
            if not torch.isfinite(images).all() or not all(map(math.isfinite, angles + energy)):
                raise RuntimeError(f"{name} produced nonfinite output at {num_steps} steps")
            mean_ke = f"{sum(energy) / len(energy):.5f}" if energy else "—"
            print(
                f"| {name} | {num_steps} | {trajectory.nfe} | {curvature:.5f} | "
                f"{mean_ke} | {elapsed:.3f} |",
                flush=True,
            )
            rows.append(
                ComparisonRow(
                    name, num_steps, trajectory.nfe, elapsed, images.cpu(), angles, energy
                )
            )
    return rows


def plot_comparison(
    rows: list[ComparisonRow], seeds: list[int], title: str, destination: Path
) -> None:
    grid_height = max(1.1 * len(rows), 4)
    figure = Figure(figsize=(2.1 + 1.25 * len(seeds), grid_height + 3.2), layout="constrained")
    FigureCanvasAgg(figure)
    outer = figure.add_gridspec(2, 1, height_ratios=[grid_height, 2.6])
    grid = outer[0].subgridspec(len(rows), len(seeds) + 1, width_ratios=[1.6, *([1] * len(seeds))])
    figure.suptitle(title, fontweight="bold")

    for i, row in enumerate(rows):
        label = figure.add_subplot(grid[i, 0])
        label.set_axis_off()
        label.text(
            0.98,
            0.5,
            f"{row.sampler}\n{row.steps} steps · {row.nfe} NFE\n{row.elapsed:.2f} s",
            ha="right",
            va="center",
            fontsize=8,
        )
        images = (row.images.float() / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        for j, (image, seed) in enumerate(zip(images, seeds, strict=True)):
            ax = figure.add_subplot(grid[i, j + 1])
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"Seed {seed}", fontsize=9)
            ax.imshow(image.squeeze(-1).numpy(), interpolation="nearest")

    diagnostics = outer[1].subgridspec(1, 2)
    angle_ax = figure.add_subplot(diagnostics[0])
    energy_ax = figure.add_subplot(diagnostics[1])
    max_steps = max(row.steps for row in rows)
    colors = ("#28689e", "#943d55", "#447447", "#a25a28")
    max_step_rows = {row.sampler: row for row in rows if row.steps == max_steps}
    for row, color in zip(max_step_rows.values(), colors, strict=True):
        angle_ax.plot(
            range(1, len(row.turning_angles) + 1),
            row.turning_angles,
            color=color,
            label=row.sampler,
        )
        if row.kinetic_energy:
            energy_ax.plot(
                range(len(row.kinetic_energy)), row.kinetic_energy, color=color, label=row.sampler
            )
    angle_ax.set_title(f"Turning angle · {max_steps} steps")
    angle_ax.set_ylabel("Radians")
    energy_ax.set_title(f"Mean kinetic energy · {max_steps} steps")
    for ax in (angle_ax, energy_ax):
        ax.set_xlabel("Denoising step")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    figure.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.08)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pretrained", "synthetic"), default="pretrained")
    parser.add_argument("--model-id", default="google/ddpm-cifar10-32")
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 15, 20])
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.num_images < 1 or any(step < 1 for step in args.steps):
        parser.error("--num-images and all --steps values must be positive")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    wrapper, shape, dtype = load_model(args.mode, args.model_id, device)
    seeds = list(range(args.seed, args.seed + args.num_images))
    rows = benchmark(wrapper, shape, args.steps, seeds, device, dtype)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / "phase3_diffusers_comparison.png"
    title = args.model_id if args.mode == "pretrained" else "Synthetic VP model"
    plot_comparison(rows, seeds, title, destination)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
