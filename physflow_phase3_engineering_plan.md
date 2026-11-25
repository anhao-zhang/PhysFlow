# PhysFlow Phase 3 Engineering Specification & Implementation Plan

> **Purpose**: Hand-off specification for **Phase 3 (Week 3)** of **PhysFlow** (`import physflow`). Connects PhysFlow's verified physics samplers and diagnostics to the [Hugging Face Diffusers](https://github.com/huggingface/diffusers) ecosystem and delivers a real pretrained diffusion image benchmark.
> **Scope**: Implement the VP-DDPM $\leftrightarrow$ EDM coordinate bridge (`DiffusersScoreWrapper`), a drop-in Diffusers scheduler (`PhysFlowDiffusersScheduler` + `to_diffusers_scheduler`), offline unit tests for CI, and a same-seed low-NFE comparison script on [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32).

## 1. Mathematical Bridge: VP-DDPM $\leftrightarrow$ EDM $\sigma(t)$

Standard [Diffusers](https://github.com/huggingface/diffusers) checkpoints (e.g., [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) and Stable Diffusion UNets) are trained under a Variance-Preserving (VP) discrete schedule with cumulative products $\bar{\alpha}_t = \prod_{s=0}^t (1 - \beta_s)$ for $t \in \{0, \dots, T-1\}$:
$$z_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

Following [Karras et al. (2022)](https://arxiv.org/abs/2206.00364) ([NVlabs/edm](https://github.com/nvlabs/edm)), rescaling $x_{\text{edm}} = z_t / \sqrt{\bar{\alpha}_t}$ maps VP-DDPM exactly into PhysFlow's EDM noise-level parameterization:
$$x_{\text{edm}} = x_0 + \sigma_t\,\epsilon, \qquad \sigma_t = \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}}, \qquad z_t = \frac{x_{\text{edm}}}{\sqrt{\sigma_t^2 + 1}}$$

### Exact Conversion from Model Output to Denoised $x_0$ & EDM Score

Given `x_edm`, scalar noise level $\sigma > 0$, and UNet output $m = f_\theta(z_t, t)$:

| `prediction_type` | Denoised Prediction $\hat{x}_0$ | EDM Score $\nabla_x \log p_\sigma(x_{\text{edm}})$ |
| :--- | :--- | :--- |
| `"epsilon"` | $\hat{x}_0 = x_{\text{edm}} - \sigma\,m$ | $\frac{\hat{x}_0 - x_{\text{edm}}}{\sigma^2} = -\frac{m}{\sigma}$ |
| `"sample"` | $\hat{x}_0 = m$ | $\frac{m - x_{\text{edm}}}{\sigma^2}$ |
| `"v_prediction"` | $\hat{x}_0 = \frac{x_{\text{edm}} - \sigma\sqrt{\sigma^2 + 1}\,m}{1 + \sigma^2}$ | $\frac{\hat{x}_0 - x_{\text{edm}}}{\sigma^2}$ |

### Continuous Timestep Lookup ($\sigma \mapsto t$)

- Precompute the training noise table $\sigma_{\text{train}}[t] = \sqrt{(1 - \bar{\alpha}_t) / \bar{\alpha}_t}$ for $t = 0, \dots, T-1$.
- Given any query $\sigma$, map $\log \sigma$ to continuous/nearest discrete timestep $t \in [0, T-1]$ via monotonic linear interpolation on $\log \sigma_{\text{train}}$ (clamped to `[0, T - 1]`). Pass integer or float timesteps matching the UNet's expectation (default integer `round(t)` for discrete DDPM UNets, configurable via `continuous_timesteps: bool = False`).

## 2. Files to Create or Update in Phase 3

```text
src/physflow/
├── __init__.py                             # Keep core import independent of optional `diffusers`
└── adapters/
    ├── __init__.py                         # Lazy/guarded exports for Diffusers adapter API
    └── diffusers_adapter.py                # NEW: DiffusersScoreWrapper, PhysFlowDiffusersScheduler, to_diffusers_scheduler

benchmarks/
└── benchmark_diffusers.py                  # NEW: Pretrained CIFAR-10 (or synthetic) same-seed low-NFE visual benchmark

tests/
└── test_diffusers_adapter.py               # NEW: Fully offline unit tests (mock UNet, zero HF downloads in CI)

.github/workflows/ci.yml                    # Update if needed: keep core CI fast; optionally test adapter with lightweight mock
```

## 3. Detailed Module Specifications

### 3.1 Optional Dependency Isolation

- Importing `physflow` must **never** require `diffusers` to be installed.
- Inside `src/physflow/adapters/diffusers_adapter.py`, `DiffusersScoreWrapper` should work with **pure PyTorch** (accepting any callable UNet plus a config dict or `alphas_cumprod` tensor), while `PhysFlowDiffusersScheduler` inherits from `diffusers.SchedulerMixin` and `diffusers.ConfigMixin` when `diffusers` is available (raising a clear `ImportError("Install optional dependency via: pip install -e '.[diffusers]'")` only if the user instantiates `PhysFlowDiffusersScheduler` without `diffusers` installed).

### 3.2 `DiffusersScoreWrapper` (`src/physflow/adapters/diffusers_adapter.py`)

Wraps any Diffusers-style UNet (e.g., `pipe.unet`) into a standard PhysFlow `ScoreFn` so **all 5 PhysFlow samplers** (`Euler`, `Heun`, `Symplectic`, `UnderdampedLangevin`, `PredictorCorrector`) and `EnergyGuidedScore` work directly on 4D image/latent tensors `(B, C, H, W)`:

- **Constructor**:
  ```python
  class DiffusersScoreWrapper:
      def __init__(
          self,
          unet: nn.Module,
          alphas_cumprod: torch.Tensor,
          prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
          clip_sample: bool = False,
          clip_sample_range: float = 1.0,
          continuous_timesteps: bool = False,
          unet_kwargs: dict[str, Any] | None = None,
      ) -> None: ...
  ```
- **Convenience Classmethod**:
  - `DiffusersScoreWrapper.from_pipeline(pipe, unet_kwargs=None) -> DiffusersScoreWrapper`: extracts `pipe.unet`, `pipe.scheduler.alphas_cumprod`, `prediction_type`, `clip_sample`, and `clip_sample_range` directly from a loaded Diffusers pipeline.
  - Provide helper property `sigma_min` and `sigma_max` derived from `alphas_cumprod` so users can build a matched `KarrasSchedule(sigma_min=wrapper.sigma_min, sigma_max=wrapper.sigma_max)`.
- **`__call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor`**:
  1. Broadcast `sigma` to shape `(B, 1, 1, 1)` (preserving `x.device` and `x.dtype`).
  2. Scale input $z = x / \sqrt{\sigma^2 + 1}$.
  3. Map $\sigma \mapsto t$ using `log(sigma)` interpolation against `log(sigmas_train)` (cached on `x.device`).
  4. Invoke `out = self.unet(z, t, **self.unet_kwargs)`; extract `out.sample` if the return object has a `.sample` attribute, otherwise use `out` directly.
  5. Convert `out` to $\hat{x}_0$ per Section 1; if `self.clip_sample` is `True`, clamp $\hat{x}_0$ to `[-clip_sample_range, clip_sample_range]`.
  6. Return score $= (\hat{x}_0 - x) / \sigma^2$.

### 3.3 `PhysFlowDiffusersScheduler` & `to_diffusers_scheduler`

A genuine drop-in [Diffusers](https://github.com/huggingface/diffusers) scheduler for single-NFE-per-step PhysFlow integrators (`"euler"`, `"symplectic"`, `"underdamped_langevin"`), enabling one-line integration into standard pipelines (`pipe.scheduler = to_diffusers_scheduler(...)`):

- **Class**: `PhysFlowDiffusersScheduler(SchedulerMixin, ConfigMixin)`
  - `@register_to_config` parameters:
    - `num_train_timesteps: int = 1000`
    - `beta_start: float = 0.0001`
    - `beta_end: float = 0.02`
    - `beta_schedule: str = "linear"` (support `"linear"`, `"scaled_linear"`, `"squaredcos_cap_v2"`)
    - `prediction_type: str = "epsilon"`
    - `clip_sample: bool = False`
    - `clip_sample_range: float = 1.0`
    - `sampler_type: Literal["euler", "symplectic", "underdamped_langevin"] = "symplectic"`
    - `damping: float = 3.0` (used as `damping` for symplectic / `friction` for underdamped Langevin)
    - `mass: float = 1.0`
    - `temperature: float = 0.01` (used for `underdamped_langevin`)
    - `rho: float = 7.0`
  - **Required Diffusers interface**:
    - `order = 1`
    - `init_noise_sigma`: returns `float(self.sigmas[0])` after `set_timesteps()` (or `float(self._sigmas_train[-1])` before `set_timesteps()`).
    - `set_timesteps(self, num_inference_steps: int, device=None)`: builds Karras `self.sigmas` of length `num_inference_steps + 1` ending in `0.0`, maps `self.sigmas[:-1]` to `self.timesteps`, and **resets** internal `_step_index = 0` and `_velocity = None` (so consecutive pipeline runs never leak momentum state!).
    - `scale_model_input(self, sample: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor`: scales `sample / sqrt(sigma_i^2 + 1)`.
    - `step(self, model_output, timestep, sample, generator=None, return_dict=True) -> SchedulerOutput | tuple`:
      - Converts `model_output` to $\hat{x}_0$ and force $F_i = \sigma_i \cdot \frac{\hat{x}_0 - x_i}{\sigma_i^2} = \frac{\hat{x}_0 - x_i}{\sigma_i}$.
      - Advances one step using the configured physics integrator (`euler`, `symplectic`, or `underdamped_langevin` with log-noise-time damping $\Delta\tau_i = \ln(\sigma_i / \sigma_{i+1})$ and exact final-step Euler fallback at $\sigma_{i+1} == 0$).
      - Increments `_step_index` and returns `SchedulerOutput(prev_sample=x_next)` (or `(x_next,)` if `return_dict=False`).
- **Helper Factory**:
  - `to_diffusers_scheduler(sampler: BaseSampler | str, base_config: Mapping[str, Any], **overrides) -> PhysFlowDiffusersScheduler`: extracts `num_train_timesteps`, `beta_start`, `beta_end`, `beta_schedule`, `prediction_type`, `clip_sample`, `clip_sample_range` from `pipe.scheduler.config` and maps a PhysFlow sampler instance (or string name) directly to a configured `PhysFlowDiffusersScheduler`.

### 3.4 `benchmarks/benchmark_diffusers.py`

A reproducible CLI script that works on both GPU and CPU:
- **CLI flags**:
  - `--mode {pretrained,synthetic}` (default `pretrained`; `synthetic` runs offline with a tiny mock UNet for quick smoke tests)
  - `--model-id` (default `google/ddpm-cifar10-32`)
  - `--steps 10 15 20` (default `[10, 15, 20]`)
  - `--num-images 4` (number of fixed-seed samples per row)
  - `--seed 42`
  - `--device auto` (`cuda` if available else `cpu`)
  - `--out-dir outputs`
- **What it runs & outputs**:
  1. Loads `DDPMPipeline.from_pretrained(model_id)` (using `DiffusersScoreWrapper.from_pipeline(pipe)`).
  2. Generates identical-initial-noise image batches across `EulerSampler`, `HeunSampler`, `SymplecticSampler`, and `UnderdampedLangevinSampler` at each step count in `--steps`.
  3. Prints a clean Markdown table to stdout: `Sampler | Steps | NFE | Curvature (rad) | Mean KE | Wall Time (s)`.
  4. Saves **`outputs/phase3_diffusers_comparison.png`**: a crisp visual grid comparing generated CIFAR-10 images across samplers at low NFE (same initial noise seeds per column, sampler name + NFE + wall time labeled on each row), plus a compact diagnostic plot of trajectory curvature / kinetic energy across denoising steps.

## 4. Phase 3 Unit Tests (`tests/test_diffusers_adapter.py`)

All tests must run **100% offline** (zero Hugging Face Hub downloads) using a lightweight synthetic PyTorch UNet:

1. **Exact VP $\leftrightarrow$ EDM Math Tests**:
   - Verify `DiffusersScoreWrapper` recovers exact analytical score for all 3 `prediction_type` modes (`"epsilon"`, `"sample"`, `"v_prediction"`) when given an exact closed-form Gaussian denoiser mock UNet (tolerance `1e-5`).
   - Verify `clip_sample=True` clamps $\hat{x}_0$ cleanly.
2. **Full-Sampler 4D Tensor Integration**:
   - Run `EulerSampler`, `HeunSampler`, `SymplecticSampler`, `UnderdampedLangevinSampler`, `PredictorCorrectorSampler`, and `EnergyGuidedScore` with `DiffusersScoreWrapper` on 4D image tensors `(2, 3, 8, 8)`. Verify output shape, finiteness, and exact NFE counts.
3. **`PhysFlowDiffusersScheduler` Drop-in Tests** (skip gracefully with `pytest.importorskip("diffusers")` only if `diffusers` is absent, and install `diffusers` in local dev / optional CI check):
   - Verify `to_diffusers_scheduler(SymplecticSampler(...), base_config)` preserves `prediction_type`, `beta_schedule`, and sampler hyperparameters.
   - Verify `set_timesteps()` resets momentum state so two back-to-back runs with the same `torch.Generator` produce identical tensors.
   - Verify `scale_model_input()` divides by $\sqrt{\sigma^2 + 1}$ and `step()` runs a full denoising loop to completion with zero `NaN`/`Inf`.
4. **Offline Benchmark Smoke Test**:
   - Verify `benchmarks/benchmark_diffusers.py --mode synthetic --steps 6 --num-images 2` exits with code 0 and writes `phase3_diffusers_comparison.png`.

## 5. Phase 3 Acceptance Checklist

- [ ] `pip install -e ".[dev,diffusers]"` succeeds cleanly.
- [ ] `ruff check src tests benchmarks` passes with 0 errors.
- [ ] `pytest -v` passes 100% of Phase 1 + Phase 2 + Phase 3 tests.
- [ ] `python benchmarks/benchmark_diffusers.py --mode pretrained --steps 10 15 20 --num-images 4` runs on `google/ddpm-cifar10-32` (on your lab GPU) and saves `outputs/phase3_diffusers_comparison.png` with clearly recognizable CIFAR-10 images across all 4 samplers.
- [ ] GitHub Actions CI remains green after pushing Phase 3 to `main`.
