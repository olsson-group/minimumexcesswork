# MEW-OG

Minimum-Excess-Work Observable Guidance for diffusion models.

This package implements the MEW-OG workflow for the 1D Prinz potential: sample a biased DDPM prior, estimate maximum-entropy reweighting lambdas from observables, and use them to guide diffusion sampling toward the target distribution.

## Install

From this directory:

```bash
pip install -e .
```

Python 3.9+ is required. Dependencies are declared in `pyproject.toml`.

## Run the Prinz Pipeline

Use the bundled pipeline script:

```bash
./scripts/run_prinz_pipeline.sh
```

It runs the complete workflow:

1. Sample the configured biased Prinz DDPM.
2. Generate unbiased Prinz reference samples.
3. Fit a 4-component GMM observable.
4. Fit reweighting lambdas.
5. Train and evaluate MEW-OG guided sampling.

Default outputs are written to `results/prinz_pipeline/`, including:

| Path | Description |
| --- | --- |
| `biased_ddpm_samples.h5` | Samples from the biased DDPM prior |
| `unbiased_prinz_samples.h5` | Reference Prinz samples |
| `gmm_params.npy` | Fitted observable parameters |
| `reweighting/reweighting-results.h5` | Lambdas, weights, and trajectories |
| `mew_og/results-0.h5` | Guided sampling results |
| `mew_og/density-observable-comparison-0.{pdf,png}` | Evaluation plots |

## Configuration

The pipeline uses:

- `mew_og/config/prinz_mew_og.json` for DDPM sampling and MEW-OG training.
- `mew_og/config/prinz_reweighting.json` for lambda fitting.

`prinz_mew_og.json` points to the OGGM-trained biased Prinz checkpoint at `trained_models/toy/prinz-potential/biased-model/model.pth.tar` and to the pipeline artifacts under `results/prinz_pipeline/`.

## Useful Scripts

The pipeline script is the recommended entry point. Individual steps can also be run directly when debugging or changing one stage:

```bash
python scripts/sample_ddpm_prinz.py --config mew_og/config/prinz_mew_og.json
python scripts/generate_prinz_trajectory.py --output results/prinz_pipeline/unbiased_prinz_samples.h5
python scripts/fit_gmm_observable.py --input results/prinz_pipeline/unbiased_prinz_samples.h5 --output results/prinz_pipeline/gmm_params.npy
python scripts/fit_lambdas_prinz.py --config mew_og/config/prinz_reweighting.json
python scripts/train_mew_og_prinz.py --config mew_og/config/prinz_mew_og.json
```

`scripts/train_ddpm_prinz.py` is available for training a MEW-OG-format DDPM checkpoint, but the default Prinz pipeline uses the pretrained OGGM checkpoint instead.

## Package Layout

- `mew_og/data`: Prinz potential and data helpers.
- `mew_og/models`: score networks, beta schedules, and OGGM checkpoint compatibility.
- `mew_og/observables`: GMM observable functions.
- `mew_og/reweighting`: maximum-entropy lambda fitting.
- `mew_og/guidance`: MEW-OG augmenter and guided model.
- `mew_og/samplers`: VP-SDE sampling.
- `scripts`: command-line entry points for the Prinz workflow.

## License

MIT License
