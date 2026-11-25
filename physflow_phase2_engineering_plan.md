# PhysFlow Phase 2 Engineering Specification & Implementation Plan

> **Purpose**: Hand-off specification for **Phase 2 (Week 2)** of **PhysFlow** (`import physflow`). Builds directly on the verified Phase 1 foundation (`schedules.py`, `toy_distributions.py`, `base.py`, `euler.py`, `heun.py`, `metrics.py`).
> **Scope**: Implement the three core physics-inspired samplers (`SymplecticSampler`, `UnderdampedLangevinSampler`, `PredictorCorrectorSampler`), the composable external energy guidance wrapper (`EnergyGuidedScore`), phase-space energy diagnostics, comprehensive tests, and the multi-sampler NFE benchmark & visualization suite.

## 1. Non-Negotiable Compatibility & Numerical Invariants

1. **Backward Compatibility**: All 40 existing Phase 1 tests in `tests/` must continue to pass unchanged.
2. **Unified EDM Force Convention**: Keep [Karras et al. (2022)](https://arxiv.org/abs/2206.00364) ([NVlabs/edm](https://github.com/nvlabs/edm)) parameterization. Define positive step size $h_i = \sigma_i - \sigma_{i+1} > 0$ and the effective probability-flow force:
   $$F(x, \sigma) = \sigma\,\nabla_x \log p_\sigma(x)$$
   So the PF-ODE in positive integration progress $ds = -d\sigma > 0$ is $\frac{dx}{ds} = F(x, \sigma)$.
3. **Final-Step Stability ($\sigma_{N} = 0$)**: Never evaluate `score_fn(x, 0.0)`. On the terminal interval where $\sigma_{i+1} == 0$, all physics samplers must perform a clean deterministic Euler step $x_N = x_{N-1} + h_{N-1} F(x_{N-1}, \sigma_{N-1})$ with no extra thermal noise injection, ensuring samples land sharply on the data manifold.
4. **Device, Dtype, & Reproducibility**: Every new sampler and guidance wrapper must preserve `device` and `dtype` (CPU/CUDA) and accept `torch.Generator` for exact deterministic reproducibility.
5. **Exact NFE Accounting**: Track `state.nfe` accurately for every call to `score_fn` (including FSAL caching where applicable and every MCMC corrector evaluation).

## 2. Files to Create or Update in Phase 2

```text
src/physflow/
├── __init__.py                         # Export new samplers, guidance, and diagnostics
├── samplers/
│   ├── __init__.py                     # Export SymplecticSampler, UnderdampedLangevinSampler, PredictorCorrectorSampler
│   ├── base.py                         # Minor optional extension: cache `last_force` on SamplerState if helpful
│   ├── symplectic.py                   # NEW: Conformal damped Velocity-Verlet / Leapfrog sampler
│   ├── langevin.py                     # NEW: Exponential-integrator BAOAB Underdamped Langevin sampler
│   └── predictor_corrector.py          # NEW: Annealed Langevin Predictor-Corrector sampler
├── guidance/
│   ├── __init__.py                     # Export EnergyGuidedScore & built-in 2D demo potentials
│   └── potential.py                    # NEW: Composable external energy E(x) guidance wrapper
└── diagnostics/
    ├── __init__.py                     # Export kinetic_energy_trace
    └── metrics.py                      # Add kinetic_energy_trace(traj, mass=1.0)

benchmarks/
└── benchmark_2d_nfe.py                 # NEW: 5-sampler NFE benchmark + energy guidance visual

tests/
├── test_physics_samplers.py            # NEW: Unit & convergence tests for all 3 physics samplers
└── test_guidance.py                    # NEW: Gradient correctness & steering tests for EnergyGuidedScore
```

## 3. Detailed Module Specifications

### 3.1 `src/physflow/samplers/base.py` (Minimal Extension)

- Ensure `SamplerState` supports optional cached force `force: torch.Tensor | None = None` (default `None`) so FSAL (First-Same-As-Last) momentum integrators can reuse $F(x_{i+1}, \sigma_{i+1})$ on the next step without double-counting NFE.
- Ensure `Trajectory` records `vs: list[torch.Tensor]` whenever `state.v is not None`, and add helper `stack_vs() -> torch.Tensor | None` returning shape `(T, B, *dims)` (or `None` if `vs` is empty/`None`).

### 3.2 `src/physflow/samplers/symplectic.py` — `SymplecticSampler(BaseSampler)`

Implements a **Conformal Damped Velocity-Verlet (Symplectic) Momentum Sampler** for the damped second-order system:
$$\frac{dx}{ds} = v, \qquad \frac{dv}{ds} = \frac{\gamma}{m}\big(F(x, \sigma) - v\big)$$
Notice that as $\gamma / m \to \infty$, $v \to F(x, \sigma)$ and the dynamics smoothly recover the exact PF-ODE. Using exponential damping makes the scheme **unconditionally stable** for any step size $h_i = \sigma_i - \sigma_{i+1} > 0$.

- **Constructor**:
  `__init__(self, schedule: NoiseSchedule | None = None, num_steps: int = 20, damping: float = 3.0, mass: float = 1.0, init_velocity: Literal["zero", "drift"] = "zero")`
  - Validate `damping > 0`, `mass > 0`, and `init_velocity in {"zero", "drift"}`.
- **Initialization (`init_state`)**:
  - Initialize $x_0$ via `super().init_state(...)`.
  - Set `v = torch.zeros_like(x_0)` when `init_velocity == "zero"`. (If `"drift"`, initialize `v` on the first step to $F(x_0, \sigma_0)$.)
- **Step Update (`step`)** with $h = \sigma_i - \sigma_{i+1} > 0$ and $\alpha = \exp\left(-\frac{\gamma h}{2m}\right)$:
  1. Obtain $F_i$: reuse `state.force` if available; otherwise evaluate $F_i = \sigma_i \cdot \text{score\_fn}(x_i, \sigma_i)$ and increment `nfe += 1`.
  2. If `step_index == 0` and `init_velocity == "drift"`, set $v_i = F_i$.
  3. **Final step ($\sigma_{i+1} == 0$)**:
     $$x_{i+1} = x_i + h\,F_i, \qquad v_{i+1} = F_i, \qquad \text{force} = \text{None}$$
  4. **Interior step ($\sigma_{i+1} > 0$)**:
     - Half-step damped kick: $v_{i+1/2} = \alpha\,v_i + (1 - \alpha)\,F_i$
     - Full-step drift: $x_{i+1} = x_i + h\,v_{i+1/2}$
     - Evaluate $F_{i+1} = \sigma_{i+1} \cdot \text{score\_fn}(x_{i+1}, \sigma_{i+1})$, increment `nfe += 1` (cache as `state.force` for FSAL reuse on step $i+1$).
     - Half-step damped kick: $v_{i+1} = \alpha\,v_{i+1/2} + (1 - \alpha)\,F_{i+1}$
- **Expected NFE**: For `num_steps = N` ($N \ge 2$), total NFE is exactly `N` (due to FSAL reuse: step 0 evaluates $F_0$ and $F_1$, steps $1 \dots N-2$ evaluate 1 new force each, and final step $N-1$ reuses $F_{N-1}$ with 0 new evaluations!).

### 3.3 `src/physflow/samplers/langevin.py` — `UnderdampedLangevinSampler(BaseSampler)`

Implements an **Exponential-Integrator BAOAB Underdamped Langevin Sampler** in $(x, v)$ phase space (inspired by [CLD-SGM](https://arxiv.org/abs/2112.07068), adapted to be training-free on standard EDM score models).

- **Constructor**:
  `__init__(self, schedule: NoiseSchedule | None = None, num_steps: int = 20, friction: float = 3.0, mass: float = 1.0, temperature: float = 0.05, noise_min_sigma: float = 0.05, noise_max_sigma: float = float("inf"))`
  - Validate `friction > 0`, `mass > 0`, `temperature >= 0`, `noise_min_sigma >= 0`.
- **Step Update (`step`)** with $h = \sigma_i - \sigma_{i+1} > 0$, $\alpha = \exp\left(-\frac{\gamma h}{2m}\right)$:
  1. Obtain $F_i$ via FSAL cache (or evaluate and increment `nfe += 1`).
  2. **Final step ($\sigma_{i+1} == 0$)**: deterministic Euler step $x_{i+1} = x_i + h\,F_i$, $v_{i+1} = F_i$.
  3. **Interior step ($\sigma_{i+1} > 0$)** (symmetric exponential BAOAB):
     - **B (half-kick + half-damp)**: $v_{1/2} = \alpha\,v_i + (1 - \alpha)\,F_i$
     - **A (half-drift)**: $x_{1/2} = x_i + \frac{h}{2}\,v_{1/2}$
     - **O (exact OU thermal noise)**: if `temperature > 0` and `noise_min_sigma <= sigma_next <= noise_max_sigma`:
       $$\eta = \sqrt{1 - \exp\left(-\frac{2\gamma h}{m}\right)} \cdot \sqrt{T}\,\sigma_{i+1}, \qquad v'_{1/2} = v_{1/2} + \eta\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
       (sampled using `generator`). Otherwise $v'_{1/2} = v_{1/2}$.
     - **A (half-drift)**: $x_{i+1} = x_{1/2} + \frac{h}{2}\,v'_{1/2}$
     - Evaluate $F_{i+1} = \sigma_{i+1} \cdot \text{score\_fn}(x_{i+1}, \sigma_{i+1})$, increment `nfe += 1`, cache as `state.force`.
     - **B (half-kick + half-damp)**: $v_{i+1} = \alpha\,v'_{1/2} + (1 - \alpha)\,F_{i+1}$.
- **Key Property**: When `temperature == 0.0`, this reduces cleanly to `SymplecticSampler` (within floating-point tolerance) — a great invariant to verify in unit tests!

### 3.4 `src/physflow/samplers/predictor_corrector.py` — `PredictorCorrectorSampler(BaseSampler)`

Implements an **Annealed Langevin MCMC Predictor-Corrector Sampler**:

- **Constructor**:
  `__init__(self, schedule: NoiseSchedule | None = None, num_steps: int = 20, predictor: Literal["euler", "heun"] = "euler", corrector_steps: int = 1, snr: float = 0.12, max_corrector_step_size: float = 0.25)`
  - Validate `predictor in {"euler", "heun"}`, `corrector_steps >= 0`, `snr > 0`, `max_corrector_step_size > 0`.
- **Step Update (`step`)**:
  1. **Predictor**: Advance $x_i \to x_{\text{pred}}$ from $\sigma_i$ to $\sigma_{i+1}$ using Euler (1 NFE) or Heun (2 NFE when $\sigma_{i+1} > 0$, 1 NFE when $\sigma_{i+1} == 0$).
  2. **Corrector**: Only when $\sigma_{i+1} > 0$ and `corrector_steps > 0`, repeat `corrector_steps` times:
     - $g = \text{score\_fn}(x, \sigma_{i+1})$, increment `nfe += 1`.
     - Sample $z \sim \mathcal{N}(0, I)$ with `generator`.
     - Compute per-sample adaptive step size:
       $$\epsilon_{\text{step}} = \text{clamp}\left(2 \left(\text{snr} \cdot \frac{\|z\|_2}{\|g\|_2 + 10^{-8}}\right)^2,\; \max=\text{max\_corrector\_step\_size} \cdot \sigma_{i+1}^2\right)$$
       (norm computed over non-batch dimensions so it generalizes cleanly to 2D and image tensors).
     - Update: $x \leftarrow x + \epsilon_{\text{step}}\,g + \sqrt{2\,\epsilon_{\text{step}}}\,z$.

### 3.5 `src/physflow/guidance/potential.py` — `EnergyGuidedScore` & Built-in Potentials

Provides a clean wrapper so **any** PhysFlow sampler can be steered by an external physical potential $E(x)$:
$$\tilde{s}(x, \sigma) = s(x, \sigma) - w(\sigma)\,\nabla_x E(x)$$

- **`EnergyGuidedScore`**:
  - `__init__(self, base_score_fn: ScoreFn, energy_fn: Callable[[torch.Tensor], torch.Tensor], scale: float = 1.0, max_grad_norm: float | None = 50.0, sigma_schedule_fn: Callable[[torch.Tensor], torch.Tensor] | None = None)`
  - `__call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor`:
    - **Crucial implementation detail**: `BaseSampler.sample()` executes under `@torch.no_grad()`. Therefore, compute $\nabla_x E(x)$ inside `with torch.enable_grad():` on `x_req = x.detach().requires_grad_(True)`, sum `energy_fn(x_req)` to a scalar, call `torch.autograd.grad(energy.sum(), x_req)[0]`, optionally clip per-sample L2 norm to `max_grad_norm`, and detach before combining with `base_score_fn(x, sigma)`.
- **Built-in 2D Demo Potentials** (callable classes returning shape `(B,)`):
  - `HalfSpacePotential(dim: int = 0, threshold: float = 0.0, side: Literal["positive", "negative"] = "positive", stiffness: float = 10.0)`: quadratic barrier penalizing violation of $x_{\text{dim}} \ge \text{threshold}$ (for `"positive"`).
  - `AnnulusPotential(radius: float = 2.0, stiffness: float = 5.0)`: $E(x) = \frac{1}{2} k (\|x\|_2 - R)^2$.
  - `HarmonicWellPotential(center: Sequence[float] = (0.0, 0.0), stiffness: float = 1.0)`: $E(x) = \frac{1}{2} k \|x - c\|_2^2$.

### 3.6 `src/physflow/diagnostics/metrics.py` (Addition)

- Add `kinetic_energy_trace(traj: Trajectory, mass: float = 1.0) -> list[float]`:
  - Returns `[float(0.5 * mass * (v.flatten(1).pow(2).sum(dim=1)).mean().item()) for v in traj.vs]` when `traj.vs` is present; raises `ValueError` if `traj.vs is None`.

### 3.7 `benchmarks/benchmark_2d_nfe.py`

A clean, reproducible CLI script (`--out-dir outputs`, `--seed 0`, `--num-samples 1024`):
1. Evaluates all 5 samplers (`EulerSampler`, `HeunSampler`, `SymplecticSampler`, `UnderdampedLangevinSampler`, `PredictorCorrectorSampler`) across `num_steps in [8, 12, 20, 35]` on `make_eight_gaussians()` and `make_checkerboard_gmm()`.
2. Prints a clean Markdown summary table (`Dataset | Sampler | Steps | NFE | SWD | Curvature | Time (ms)`).
3. Saves **`outputs/phase2_nfe_benchmark.png`**: a 2-panel publication-style plot showing **NFE vs. SWD** and **NFE vs. Trajectory Curvature** across all 5 samplers on 8-Gaussians.
4. Saves **`outputs/phase2_energy_guidance.png`**: a side-by-side 2D comparison of unguided vs. `HalfSpacePotential`-guided `UnderdampedLangevinSampler` on 8-Gaussians (demonstrating particles cleanly steered into the right half-plane $x_1 \ge 0$, with the fraction of samples in $x_1 \ge 0$ printed in the title).

## 4. Phase 2 Unit Tests

1. **`tests/test_physics_samplers.py`**:
   - Parameter validation tests (negative damping/friction/mass/temperature/snr raise `ValueError`).
   - Exact NFE count verification for `SymplecticSampler` (`N` for $N \ge 2$), `UnderdampedLangevinSampler` (`N` for $N \ge 2$), and `PredictorCorrectorSampler` (`N + (N - 1) * corrector_steps` for Euler predictor).
   - Zero-temperature equivalence: `UnderdampedLangevinSampler(temperature=0.0, friction=g, mass=m)` matches `SymplecticSampler(damping=g, mass=m)` to within `1e-5` on identical initial conditions.
   - Deterministic reproducibility via `torch.Generator` for all 3 samplers (including stochastic Langevin and PC).
   - Distribution quality & stability: zero `NaN`/`Inf`, and all 3 physics samplers achieve `sliced_wasserstein_2d < 0.15` on `make_eight_gaussians()` at `num_steps=20`.
   - `Trajectory.vs` and `kinetic_energy_trace()` return non-negative finite energy values of length `num_steps + 1` for momentum samplers.
2. **`tests/test_guidance.py`**:
   - Verifies `EnergyGuidedScore` works correctly inside `torch.no_grad()` without raising autograd errors.
   - Verifies gradient clipping (`max_grad_norm`) bounds the guidance shift norm.
   - Verifies steering effect: sampling `make_eight_gaussians()` with `HalfSpacePotential(dim=0, threshold=0.0, side="positive", stiffness=15.0)` yields `>= 90%` of samples with $x_1 > -0.1$ (vs. ~50% unguided).

## 5. Phase 2 Acceptance Checklist

- [ ] `ruff check src tests benchmarks` passes with 0 errors.
- [ ] `pytest -v` passes 100% of Phase 1 + Phase 2 tests.
- [ ] `python benchmarks/benchmark_2d_nfe.py` runs cleanly on CPU, prints the 5-sampler Markdown benchmark table, and saves both `outputs/phase2_nfe_benchmark.png` and `outputs/phase2_energy_guidance.png`.
- [ ] GitHub Actions CI passes on `main` after pushing Phase 2.
