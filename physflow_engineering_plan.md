# PhysFlow Engineering Specification & Implementation Plan

> **Purpose**: Hand-off specification for implementing **PhysFlow** (`import physflow`), a lightweight, modular Python library for physics-inspired generative samplers (Symplectic, Underdamped Langevin, Predictor-Corrector) and phase-space diagnostics for Diffusion and Flow Matching models.
> **Execution Strategy**: Implement **Phase 1 (Week 1)** first and verify all acceptance tests before proceeding to Phases 2–4.

## 1. Project Identity & Standards

- **Repository Name**: `PhysFlow`
- **Python Package Name**: `physflow` (installed via `src/physflow`)
- **Python Version**: `>=3.10` (tested on 3.11)
- **Core Stack**: `torch`, `numpy`, `scipy`, `matplotlib`
- **Dev & Quality Stack**: `pytest`, `ruff`, GitHub Actions CI
- **Optional Integrations (Phases 3–4)**: `diffusers` ([Hugging Face Diffusers](https://github.com/huggingface/diffusers)), `gradio` ([Gradio](https://github.com/gradio-app/gradio))
- **Code Style**:
  - Strict PEP 8 with full Python type hints (`from __future__ import annotations`).
  - All tensors must preserve `device` and `dtype` from inputs (CPU and CUDA compatible).
  - Deterministic reproducibility via `torch.Generator` wherever randomness is used.
  - No hardcoded absolute paths; scripts save outputs to configurable `--out-dir` (default `outputs/`).

## 2. Repository Directory Layout

```text
PhysFlow/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── physflow/
│       ├── __init__.py
│       ├── dynamics/
│       │   ├── __init__.py
│       │   ├── schedules.py          # Phase 1: Karras EDM & Linear sigma schedules
│       │   └── toy_distributions.py  # Phase 1: Analytical score 2D GMM / Ring benchmarks
│       ├── samplers/
│       │   ├── __init__.py
│       │   ├── base.py               # Phase 1: SamplerState, Trajectory, BaseSampler
│       │   ├── euler.py              # Phase 1: First-order PF-ODE Euler baseline
│       │   ├── heun.py               # Phase 1: Second-order EDM Heun baseline
│       │   ├── symplectic.py         # Phase 2: Symplectic / Leapfrog momentum sampler
│       │   ├── langevin.py           # Phase 2: Underdamped / Critically-damped Langevin
│       │   └── predictor_corrector.py# Phase 2: Annealed Langevin PC sampler
│       ├── guidance/
│       │   ├── __init__.py
│       │   └── potential.py          # Phase 2: External energy E(x) & constraint wrapper
│       ├── diagnostics/
│       │   ├── __init__.py
│       │   └── metrics.py            # Phase 1-2: Trajectory curvature, energy, 2D Wasserstein
│       └── adapters/
│           ├── __init__.py
│           └── diffusers_adapter.py  # Phase 3: Diffusers SchedulerMixin compatibility
├── benchmarks/
│   ├── plot_2d_trajectories.py       # Phase 1: 2D phase-space evolution figure/GIF
│   └── benchmark_2d_nfe.py           # Phase 2: NFE vs. Wasserstein-2 & curvature table
├── demo/
│   └── app.py                        # Phase 4: Dual-tab Gradio interactive app
└── tests/
    ├── test_schedules.py             # Phase 1
    ├── test_toy_distributions.py     # Phase 1
    ├── test_baselines.py             # Phase 1
    ├── test_physics_samplers.py      # Phase 2
    └── test_diffusers_adapter.py     # Phase 3
```

## 3. Unified Mathematical Convention (EDM Parameterization)

To ensure every sampler, toy benchmark, and future [Diffusers](https://github.com/huggingface/diffusers) adapter shares one consistent mathematical convention, adopt the **EDM noise-level parameterization** from [Karras et al. (2022)](https://arxiv.org/abs/2206.00364) ([NVlabs/edm](https://github.com/nvlabs/edm)):

- **Perturbation Kernel**: $p_\sigma(x) = \int p_{\text{data}}(x_0)\,\mathcal{N}(x; x_0, \sigma^2 I)\,dx_0$, where $\sigma \in [\sigma_{\min}, \sigma_{\max}]$.
- **Score Function Interface**:
  `ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
  Takes `(x, sigma)` where `x` has shape `(B, *dims)` and `sigma` is a scalar tensor or shape `(B,)` broadcastable to `x`, returning $\nabla_x \log p_\sigma(x)$ with the same shape as `x`.
- **Probability Flow ODE (PF-ODE)**:
  $$\frac{dx}{d\sigma} = -\sigma\,\nabla_x \log p_\sigma(x)$$
  Equivalently, with denoiser $D_\theta(x; \sigma) = x + \sigma^2 \nabla_x \log p_\sigma(x)$, $\frac{dx}{d\sigma} = \frac{x - D_\theta(x; \sigma)}{\sigma}$.
- **Karras Noise Schedule** ($\sigma_0 = \sigma_{\max} > \sigma_1 > \dots > \sigma_{N-1} = \sigma_{\min}$, $\sigma_N = 0$):
  $$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}\left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right)\right)^\rho, \quad i = 0, \dots, N-1$$
  Default parameters: `sigma_min = 0.002`, `sigma_max = 80.0` (or `10.0` for 2D toy distributions), `rho = 7.0`.

## 4. Phase 1 Detailed Specification (Week 1 — Implement Now)

Implement all files below cleanly so Phase 1 passes `pytest` and `ruff` with zero errors.

### 4.1 Packaging & CI

1. **`pyproject.toml`**:
   - Build backend: `setuptools>=68` (`setuptools.build_meta`) with `src` layout (`packages.find` where `["src"]`).
   - Project name: `physflow`, version: `0.1.0`, `requires-python = ">=3.10"`.
   - Core dependencies: `torch>=2.0`, `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`.
   - Optional dependency groups:
     - `dev`: `pytest>=7.4`, `ruff>=0.4`
     - `diffusers`: `diffusers>=0.27`, `accelerate>=0.27`
     - `demo`: `gradio>=4.0`
   - Configure `[tool.ruff]` (`line-length = 100`, `target-version = "py310"`, rules `["E", "F", "I", "UP", "B"]`) and `[tool.pytest.ini_options]` (`testpaths = ["tests"]`).
2. **`.github/workflows/ci.yml`**:
   - Trigger on `push` and `pull_request` to `main`.
   - Setup Python 3.11, install CPU PyTorch + `pip install -e ".[dev]"`, run `ruff check src tests benchmarks` and `pytest -v`.
3. **Package `__init__.py` files**:
   - Export public classes cleanly from `physflow`, `physflow.dynamics`, `physflow.samplers`, `physflow.diagnostics`.

### 4.2 `src/physflow/dynamics/schedules.py`

Define an abstract base class `NoiseSchedule` and two concrete implementations:

- `NoiseSchedule(ABC)`:
  - `get_sigmas(self, num_steps: int, device=None, dtype=torch.float32) -> torch.Tensor`: returns a 1D tensor of length `num_steps + 1` in strictly descending order ending with `0.0`.
- `KarrasSchedule(sigma_min: float = 0.002, sigma_max: float = 10.0, rho: float = 7.0)`:
  - Implements the Karras formula above with input validation (`sigma_max > sigma_min > 0`, `rho > 0`, `num_steps >= 1`).
- `LinearSchedule(sigma_min: float = 0.002, sigma_max: float = 10.0)`:
  - Linearly spaced from `sigma_max` to `sigma_min`, appended with `0.0`.

### 4.3 `src/physflow/dynamics/toy_distributions.py`

Implement exact, training-free 2D analytical score benchmarks so samplers can be evaluated without neural network approximation error:

- `GaussianMixture2D`:
  - Parameters: `means: Tensor (K, 2)`, `weights: Tensor (K,)` (normalized), `base_std: float = 0.15`.
  - Under Gaussian perturbation of std $\sigma$, the marginal density is an exact GMM with component variance $\sigma_{\text{eff}}^2 = \text{base\_std}^2 + \sigma^2$.
  - `sample(self, num_samples: int, generator: torch.Generator | None = None, device=None, dtype=torch.float32) -> torch.Tensor`
  - `log_prob(self, x: torch.Tensor, sigma: torch.Tensor | float = 0.0) -> torch.Tensor` (use `torch.logsumexp` for numerical stability)
  - `score(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor`: compute exact analytical $\nabla_x \log p_\sigma(x)$ in closed form (posterior-weighted mean shift $-\sum_k w_k(x) \frac{x - \mu_k}{\sigma_{\text{eff}}^2}$), avoiding unnecessary autograd overhead and guaranteeing zero `NaN`s.
- Factory helper functions:
  - `make_eight_gaussians(radius: float = 2.0, std: float = 0.15) -> GaussianMixture2D`: 8 equally spaced modes on a circle of given radius.
  - `make_checkerboard_gmm(grid_size: int = 4, std: float = 0.12) -> GaussianMixture2D`: multi-modal 2D grid approximation with sharp multimodal structure.

### 4.4 `src/physflow/samplers/base.py`

Design clean, extensible dataclasses and base sampler that already support auxiliary momentum `v` (for Phase 2 Symplectic and Underdamped Langevin samplers):

- `@dataclass class SamplerState`:
  - `x: torch.Tensor`
  - `sigma: torch.Tensor`
  - `v: torch.Tensor | None = None` (auxiliary velocity/momentum for physics samplers)
  - `step_index: int = 0`
  - `nfe: int = 0` (cumulative number of score function evaluations)
- `@dataclass class Trajectory`:
  - `xs: list[torch.Tensor]` (detached CPU/device snapshots of `x`)
  - `sigmas: list[float]`
  - `vs: list[torch.Tensor] | None = None`
  - `nfe: int = 0`
  - Helper method `stack_xs() -> torch.Tensor` returning shape `(T, B, *dims)`.
- `class BaseSampler(ABC)`:
  - `__init__(self, schedule: NoiseSchedule | None = None, num_steps: int = 20)`
  - `init_state(self, shape: tuple[int, ...], device, dtype, generator: torch.Generator | None = None, x_init: torch.Tensor | None = None) -> SamplerState`: initializes $x_0 \sim \mathcal{N}(0, \sigma_{\max}^2 I)$ unless `x_init` is provided.
  - `@abstractmethod def step(self, state: SamplerState, sigma_next: torch.Tensor, score_fn: ScoreFn, generator: torch.Generator | None = None) -> SamplerState`
  - `sample(self, score_fn: ScoreFn, shape: tuple[int, ...], ..., return_traj: bool = False) -> torch.Tensor | tuple[torch.Tensor, Trajectory]`: runs the schedule loop under `@torch.no_grad()`, increments `nfe`, optionally records `Trajectory`.

### 4.5 `src/physflow/samplers/euler.py` & `heun.py`

- **`EulerSampler(BaseSampler)`**:
  - Computes $d_i = -\sigma_i \cdot \text{score\_fn}(x_i, \sigma_i)$, updates $x_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) d_i$.
  - Increments `nfe` by `1` per step.
- **`HeunSampler(BaseSampler)`** (EDM 2nd-order predictor-corrector):
  - Predictor: $d_i = -\sigma_i \cdot \text{score\_fn}(x_i, \sigma_i)$, $\tilde{x}_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) d_i$.
  - If $\sigma_{i+1} > 0$: evaluates $d'_{i+1} = -\sigma_{i+1} \cdot \text{score\_fn}(\tilde{x}_{i+1}, \sigma_{i+1})$ and updates $x_{i+1} = x_i + \frac{1}{2}(\sigma_{i+1} - \sigma_i)(d_i + d'_{i+1})$, incrementing `nfe` by `2`.
  - On the final step ($\sigma_{i+1} == 0$): skips the second evaluation (just like [EDM](https://arxiv.org/abs/2206.00364)), incrementing `nfe` by `1`.

### 4.6 `src/physflow/diagnostics/metrics.py` (Phase 1 Core)

- `trajectory_curvature(traj: Trajectory) -> float`: average discrete turning angle / second-difference norm along sampled trajectories to quantify path smoothness.
- `sliced_wasserstein_2d(samples: torch.Tensor, target: torch.Tensor, num_projections: int = 64, seed: int = 0) -> float`: fast, dependency-light 2D distribution distance using `scipy.stats.wasserstein_distance` over random unit projections (works on CPU/GPU tensors cleanly).

### 4.7 Verification: Tests & First Visual Script

1. **`tests/test_schedules.py`**: verifies descending monotonicity, exact length `num_steps + 1`, terminal zero, and invalid parameter errors.
2. **`tests/test_toy_distributions.py`**: verifies closed-form `score(x, sigma)` matches `torch.autograd.grad(log_prob(x, sigma).sum(), x)` within `1e-5` across multiple $\sigma$ values.
3. **`tests/test_baselines.py`**:
   - Verifies output shapes, zero `NaN`/`Inf`, exact `nfe` counts (`N` for Euler, `2N - 1` for Heun).
   - Verifies deterministic reproducibility with `torch.Generator`.
   - Verifies numerical accuracy: on `make_eight_gaussians()`, `HeunSampler` achieves lower `sliced_wasserstein_2d` error than `EulerSampler` at `num_steps=15`.
4. **`benchmarks/plot_2d_trajectories.py`**:
   - Runs `EulerSampler` and `HeunSampler` on 8-Gaussians (`num_samples=1024`, `num_steps=20`), prints `NFE`, `SWD` (Sliced Wasserstein Distance), and `Curvature` to stdout, and saves a clean side-by-side comparison plot to `outputs/week1_2d_baselines.png`.

### 4.8 Phase 1 Acceptance Checklist

- [ ] `pip install -e ".[dev]"` succeeds in the `physflow` environment.
- [ ] `ruff check src tests benchmarks` passes with 0 warnings/errors.
- [ ] `pytest -v` passes 100% of tests.
- [ ] `python benchmarks/plot_2d_trajectories.py` runs in under 5 seconds on CPU and outputs `outputs/week1_2d_baselines.png` showing 8 crisp Gaussian modes.

## 5. Roadmap for Phases 2–4 (Architecture Context Only — Do Not Overbuild in Phase 1)

| Phase | Target Modules | Key Engineering Goals |
| :--- | :--- | :--- |
| **Phase 2 (Week 2)** | `samplers/symplectic.py`, `samplers/langevin.py`, `samplers/predictor_corrector.py`, `guidance/potential.py`, `benchmarks/benchmark_2d_nfe.py` | Implement Leapfrog/Symplectic momentum integrator, exponential-integrator Underdamped Langevin sampler (tunable friction $\gamma$), Annealed Langevin PC sampler, and external energy guidance wrapper $-\lambda \nabla_x E(x)$; generate NFE vs. SWD & curvature benchmark tables. |
| **Phase 3 (Week 3)** | `adapters/diffusers_adapter.py`, `tests/test_diffusers_adapter.py` | Wrap PhysFlow samplers into a [Diffusers](https://github.com/huggingface/diffusers)-compatible `SchedulerMixin` adapter; run same-seed low-NFE (10/15/20 steps) comparison on a lightweight pretrained diffusion checkpoint. |
| **Phase 4 (Week 4)** | `demo/app.py`, `README.md` | Build dual-tab [Gradio](https://github.com/gradio-app/gradio) interactive app (Tab 1: 2D phase-space sandbox with live animations & curvature plots; Tab 2: pretrained image comparison), export README hero GIFs, and tag `v0.1.0`. |
