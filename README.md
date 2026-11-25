# PhysFlow
Physics-inspired samplers (Symplectic, Underdamped Langevin, Predictor-Corrector) and phase-space diagnostics for diffusion &amp; flow matching models.

## Diffusers integration

Install the optional integration and run the same-seed CIFAR-10 comparison.

```sh
pip install -e ".[diffusers]"
python benchmarks/benchmark_diffusers.py --mode pretrained --steps 10 15 20 --num-images 4
```

The benchmark saves `outputs/phase3_diffusers_comparison.png`. It applies Langevin thermal noise at
`sigma <= 1`, near the scale of the normalized image signal. Use `--mode synthetic` for an offline
run with a Gaussian mock model.

Use `DiffusersScoreWrapper.from_pipeline(pipe)` to call a VP-trained UNet from any PhysFlow
sampler. Use the wrapper's `sigma_min` and `sigma_max` when constructing the noise schedule.
For a direct scheduler replacement in `DDPMPipeline`, use its VP state convention.

```python
from diffusers import DDPMPipeline
from physflow.adapters import to_diffusers_scheduler

pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
pipe.scheduler = to_diffusers_scheduler("symplectic", pipe.scheduler.config, sample_space="vp")
images = pipe(num_inference_steps=20).images
```

The default scheduler state is in EDM coordinates for pipelines that call `scale_model_input`
and initialize noise with `init_noise_sigma`.
