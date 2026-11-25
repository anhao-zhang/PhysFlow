"""Compare five PhysFlow samplers and demonstrate half-space energy guidance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from physflow import (
    EnergyGuidedScore,
    EulerSampler,
    HalfSpacePotential,
    HeunSampler,
    PredictorCorrectorSampler,
    SymplecticSampler,
    UnderdampedLangevinSampler,
    make_checkerboard_gmm,
    make_eight_gaussians,
    sliced_wasserstein_2d,
    trajectory_curvature,
)

SAMPLERS = {
    "Euler": EulerSampler,
    "Heun": HeunSampler,
    "Symplectic": SymplecticSampler,
    "Underdamped Langevin": UnderdampedLangevinSampler,
    "Predictor-Corrector": PredictorCorrectorSampler,
}
STEP_COUNTS = (8, 12, 20, 35)


@dataclass(frozen=True)
class BenchmarkResult:
    nfe: int
    swd: float
    curvature: float


def run_benchmarks(seed: int, num_samples: int) -> dict[str, list[BenchmarkResult]]:
    results: dict[str, list[BenchmarkResult]] = {name: [] for name in SAMPLERS}
    datasets = {
        "Eight Gaussians": make_eight_gaussians(),
        "Checkerboard": make_checkerboard_gmm(),
    }
    print("| Dataset | Sampler | Steps | NFE | SWD | Curvature | Time (ms) |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    for dataset, mixture in datasets.items():
        target = mixture.sample(4 * num_samples, generator=torch.Generator().manual_seed(seed + 1))
        for name, sampler_type in SAMPLERS.items():
            for steps in STEP_COUNTS:
                sampler = sampler_type(num_steps=steps)
                generator = torch.Generator().manual_seed(seed)
                start = perf_counter()
                samples, trajectory = sampler.sample(
                    mixture.score, (num_samples, 2), generator=generator, return_traj=True
                )
                elapsed = 1000 * (perf_counter() - start)
                swd = sliced_wasserstein_2d(samples, target, seed=seed)
                curvature = trajectory_curvature(trajectory)
                print(
                    f"| {dataset} | {name} | {steps} | {trajectory.nfe} | "
                    f"{swd:.5f} | {curvature:.5f} | {elapsed:.2f} |"
                )
                if dataset == "Eight Gaussians":
                    results[name].append(BenchmarkResult(trajectory.nfe, swd, curvature))
    return results


def plot_nfe(results: dict[str, list[BenchmarkResult]], destination: Path) -> None:
    figure = Figure(figsize=(10.2, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    swd_ax, curvature_ax = figure.subplots(1, 2)
    colors = ("#28689e", "#943d55", "#447447", "#a25a28", "#66569a")
    markers = ("o", "s", "^", "D", "v")
    for (name, points), color, marker in zip(results.items(), colors, markers, strict=True):
        nfe = [point.nfe for point in points]
        swd_ax.plot(nfe, [point.swd for point in points], color=color, marker=marker, label=name)
        curvature_ax.plot(nfe, [point.curvature for point in points], color=color, marker=marker)
    for ax in (swd_ax, curvature_ax):
        ax.set_xlabel("Score evaluations (NFE)")
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    swd_ax.set_ylabel("Sliced Wasserstein distance")
    curvature_ax.set_ylabel("Mean turning angle (rad)")
    figure.suptitle("Eight Gaussians")
    swd_ax.legend(frameon=False, fontsize=8)
    figure.savefig(destination, dpi=180)


def _guidance_weight(sigma: torch.Tensor) -> torch.Tensor:
    return (1 + sigma.square()).reciprocal()


def plot_guidance(seed: int, num_samples: int, destination: Path) -> None:
    mixture = make_eight_gaussians()
    sampler = UnderdampedLangevinSampler(num_steps=20)
    potential = HalfSpacePotential(stiffness=15.0)
    guided_score = EnergyGuidedScore(mixture.score, potential, sigma_schedule_fn=_guidance_weight)
    figure = Figure(figsize=(8.4, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    for ax, (label, score, color) in zip(
        axes,
        (("Unguided", mixture.score, "#28689e"), ("Half-space guidance", guided_score, "#447447")),
        strict=True,
    ):
        samples = sampler.sample(
            score, (num_samples, 2), generator=torch.Generator().manual_seed(seed)
        )
        fraction = (samples[:, 0] >= 0).float().mean().item()
        points = samples.numpy()
        ax.axvspan(0, 3, color="#447447", alpha=0.08)
        ax.axvline(0, color="#696969", linestyle="--", linewidth=0.8)
        ax.scatter(points[:, 0], points[:, 1], s=8, color=color, alpha=0.5, linewidths=0)
        ax.set(
            title=f"{label}\n{fraction:.1%} of samples with $x_1 \\geq 0$",
            xlabel="$x_1$",
            xlim=(-3, 3),
            ylim=(-3, 3),
            aspect="equal",
        )
    axes[0].set_ylabel("$x_2$")
    figure.suptitle("Underdamped Langevin · Eight Gaussians")
    figure.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.08)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4096)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")

    # Tiny CPU operations spend less time waiting for workers with a single thread.
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = run_benchmarks(args.seed, args.num_samples)
    plot_nfe(results, args.out_dir / "phase2_nfe_benchmark.png")
    plot_guidance(args.seed, args.num_samples, args.out_dir / "phase2_energy_guidance.png")


if __name__ == "__main__":
    main()
