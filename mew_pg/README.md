
<p align="center">
  <img src="assets/emu.png" alt="BioEmu logo" width="200"/>
</p>

# Path Guidance for BioEmu

**Minimum Excess Work Guidance for Score-Based Sampling with Experimental Data or Sparse Restraints**

This repository extends [BioEmu](https://github.com/microsoft/bioemu) with *path guidance* — a method that steers the BioEmu diffusion process toward conformational transition states while penalising solutions that require large deviations from the unguided trajectory.

> **Relationship to BioEmu.** The core generative model, SDE definitions, denoiser, and sampling infrastructure are taken directly from the [BioEmu repository](https://github.com/microsoft/bioemu) with minor modifications. This repository adds the guidance layer on top.

---

## Overview

Standard BioEmu samples from the equilibrium ensemble of a protein. Path guidance biases sampling toward a user-defined target region (here: conformational transition states) by:

1. Running a **forward probability-flow ODE** from a set of seed structures to obtain reference trajectories.
2. Optimising **guidance parameters** via Bayesian (GP) optimisation such that the reverse ODE / SDE produces structures that (a) are classified as transition states and (b) incur minimal excess work relative to the unguided path.
3. Drawing a large batch of samples with the best-found parameters.

Transition-state membership is determined by a classifier built from reference MD data: torsion-angle features → TICA → KMeans micro-states → MSM → PCCA macro-states → committor probability.

---

## Installation

### 1. Install BioEmu

Follow the [BioEmu installation instructions](https://github.com/microsoft/bioemu). The simplest route:

```bash
pip install bioemu
```

> On first use, BioEmu will set up [ColabFold](https://github.com/sokrypton/ColabFold) in a separate virtual environment for MSA generation. Set `BIOEMU_COLABFOLD_DIR` to control where it is installed.

### 2. Install additional dependencies

Path guidance requires several packages not bundled with BioEmu:

```bash
pip install deeptime scikit-learn scikit-optimize scipy
```

### 3. Clone this repository

```bash
git clone <this-repo-url>
cd bioemu-path-guidance
pip install -e .
```

---

## Prerequisites: data files

Path guidance requires two protein-specific data files that must be prepared in advance.

### `feat_ref.npz` — reference torsion features

This file encodes the conformational landscape of your protein from reference MD trajectories. It must contain a single array under the key `feat_ref` of shape `(n_frames, n_features)`, where features are sine/cosine encodings of backbone φ/ψ torsion angles.

To generate it from reference MD data stored as BioEmu-format NPZ files (keys `pos`, `node_orientations`):

```python
import numpy as np
from src.bioemu.transition_states import build_feature_matrix
import torch

# Load and concatenate your reference trajectory NPZ files
positions = ...       # (n_frames, n_residues, 3)  in nm
orientations = ...    # (n_frames, n_residues, 3, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat_ref = build_feature_matrix(positions, orientations, device=device)
np.savez("feat_ref.npz", feat_ref=feat_ref)
```

### `init.npz` — seed structures

The guidance optimisation starts from an initial set of structures. These should be structures in (or near) the region of interest. The file must contain arrays under the keys `pos` (nm) and `node_orientations`.

---

## Running path guidance

```bash
python run_path_guidance.py \
    --sequence DTYKLVIVLNGTTFTYTTEAVDAATAEKVFKQYANDAGVDGEWTYDAATKTFTVTE \
    --init_npz /path/to/init.npz \
    --feat_ref_npz /path/to/feat_ref.npz \
    --save_dir results/my_run \
    --gamma 0.02 \
    --N 100 \
    --batch_size 256 \
    --n_calls 50 \
    --seed 42
```

| Argument | Description | Default |
|---|---|---|
| `--sequence` | Amino acid sequence (one-letter code) | required |
| `--init_npz` | Path to seed structure NPZ | required |
| `--feat_ref_npz` | Path to reference feature NPZ | required |
| `--save_dir` | Output directory | `path_guidance_results` |
| `--gamma` | Weight of excess-work penalty in the objective | `0.1` |
| `--N` | Number of ODE integration steps | `100` |
| `--batch_size` | Structures per GP evaluation call | `100` |
| `--n_calls` | Number of GP optimisation calls | `50` |
| `--seed` | Random seed for reproducibility | `None` |

Results are written to `{save_dir}/{gamma}_{batch_size}_{N}_.../{`:
- `samples.npz` — sampled positions and orientations
- `params.txt` — optimised guidance parameters
- `transition_classifier_result.txt` — fraction of samples classified as transition states

---

## Repository structure

```
run_path_guidance.py  
src/bioemu/
    path_guidance.py          # GP optimisation loop and main()
    transition_states.py      # TICA/MSM/committor-based transition classifier
    guidance_sampling.py      # Guided reverse ODE
    sample_utils.py           # Shared utilities (chemgraph, integration helpers)
    sample.py                 # Unguided BioEmu sampling (upstream)
    ...                       # Remaining BioEmu infrastructure (unchanged)
```

---

## Acknowledgements

This work builds directly on **BioEmu** by Lewis et al. The generative model, SDE definitions, denoiser architecture, and sampling code are taken from the [BioEmu repository](https://github.com/microsoft/bioemu); only the guidance layer (`path_guidance.py`, `transition_states.py`, `guidance_sampling.py`, and associated scripts) is new. We are grateful to the BioEmu authors for releasing their code and model weights.

The `openfold` subdirectory is copied from [OpenFold](https://github.com/aqlaboratory/openfold) with minor modifications, as in the original BioEmu repository.

---

## Citation

If you use this code, please cite both this work and the original BioEmu paper:

```bibtex
@article{mew2026,
author = {Kolloff, Christopher and H{\"o}ppe, Tobias and Angelis, Emmanouil and Schreiner, Mathias Jacob and Bauer, Stefan and Dittadi, Andrea and Olsson, Simon},
title = {Minimum-Excess-Work Guidance: Score-Based Sampling with Experimental Data or Sparse Restraints},
journal = {Journal of Chemical Theory and Computation},
volume = {22},
number = {11},
pages = {5838-5848},
year = {2026},
doi = {10.1021/acs.jctc.6c00080},
    note ={PMID: 42150797},
 URL = { https://doi.org/10.1021/acs.jctc.6c00080},
eprint = {https://doi.org/10.1021/acs.jctc.6c00080}

}
```

```bibtex
@article{bioemu2025,
  title={Scalable emulation of protein equilibrium ensembles with generative deep learning},
  author={Lewis, Sarah and Hempel, Tim and Jim{\'e}nez-Luna, Jos{\'e} and Gastegger, Michael and Xie, Yu and Foong, Andrew YK and Satorras, Victor Garc{\'\i}a and Abdin, Osama and Veeling, Bastiaan S and Zaporozhets, Iryna and Chen, Yaoyi and Yang, Soojung and Foster, Adam E. and Schneuing, Arne and Nigam, Jigyasa and Barbero, Federico and Stimper Vincent and  Campbell, Andrew and Yim, Jason and Lienen, Marten and Shi, Yu and Zheng, Shuxin and Schulz, Hannes and Munir, Usman and Sordillo, Roberto and Tomioka, Ryota and Clementi, Cecilia and No{\'e},  Frank},
  journal={Science},
  pages={eadv9817},
  year={2025},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.adv9817}
}
```
