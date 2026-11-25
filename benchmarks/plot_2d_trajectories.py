"""Plot the Euler and Heun baselines on the eight-Gaussian benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from physflow import (
    EulerSampler,
    HeunSampler,
    make_eight_gaussians,
    sliced_wasserstein_2d,
    trajectory_curvature,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    # The small CPU tensors run faster without thread-pool overhead.
    torch.set_num_threads(1)
    num_samples = 1024
    mixture = make_eight_gaussians()
    target = mixture.sample(4 * num_samples, generator=torch.Generator().manual_seed(123))

    figure = Figure(figsize=(9.2, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    figure.suptitle("Eight-Gaussian benchmark · 20 noise steps")
    for ax, sampler, color in zip(
        axes,
        (EulerSampler(num_steps=20), HeunSampler(num_steps=20)),
        ("#28689e", "#943d55"),
        strict=True,
    ):
        samples, traj = sampler.sample(
            mixture.score,
            (num_samples, 2),
            generator=torch.Generator().manual_seed(42),
            return_traj=True,
        )
        swd = sliced_wasserstein_2d(samples, target)
        curvature = trajectory_curvature(traj)
        name = type(sampler).__name__.removesuffix("Sampler")
        print(f"{name:5s}  NFE = {traj.nfe:2d}  SWD = {swd:.5f}  Curvature = {curvature:.5f} rad")

        paths = traj.stack_xs()[:, :36].numpy()
        points = samples.numpy()
        ax.plot(paths[..., 0], paths[..., 1], color=color, alpha=0.22, linewidth=0.75)
        ax.scatter(points[:, 0], points[:, 1], s=8, color=color, alpha=0.45, linewidths=0)
        ax.set(
            title=f"{name}\nNFE {traj.nfe} · SWD {swd:.4f} · Curvature {curvature:.3f} rad",
            xlabel="$x_1$",
            xlim=(-3, 3),
            ylim=(-3, 3),
            aspect="equal",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("$x_2$")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / "week1_2d_baselines.png"
    figure.savefig(destination, dpi=160)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
