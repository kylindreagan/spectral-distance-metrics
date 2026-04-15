# Eigenspace Perturbations and Spectral Distance Metrics in Non‑Rigid Shape Retrieval
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

This repository contains the reference implementation and supporting code for the Senior Study thesis:

> **"Eigenspace Perturbations and Spectral Distance Metrics in Non‑Rigid Shape Retrieval"**  
> *Kylind Reagan – Maryville College, 2026*

The work surveys, implements, and evaluates spectral shape descriptors based on the Laplace–Beltrami operator and its discretizations. It includes:
- Computation of Laplace–Beltrami eigenpairs on triangle meshes  
- The **Heat Kernel Signature (HKS)** and its **Scale‑Invariant** variant (SI‑HKS)  
- The **Wave Kernel Signature (WKS)**  
- Shape retrieval experiments on the SHREC 2011 benchmark  
- Visualisations using t‑SNE and spectral clustering

# Obtaining the SHREC 2011 dataset
The experiments rely on the **SHREC 2011 Non‑rigid 3D Shape Retrieval benchmark.** You can download it from the [SHREC website](https://www.shrec.net/) or from the [original track page](https://www.nist.gov/itl). Place the extracted meshes in data/SHREC2011 and ensure the directory structure matches the expected layout (flat). Visualizations all used ind_visual_showcase.py on T280.off and T593.off.

# Core Functions

All descriptor computations follow the derivations presented in **Chapters 2.4, 3, 4 and 5** of the thesis.

## Laplace-Beltrami Operator

```python
from src.ShapeDNA.laplaceBeltramiShape import lb_eigenvalues

evals, evecs = lb_eigenvalues(mesh, k=200, return_eigenvectors=True)
```

Implements the cotangent Laplacian and solves the generalized eigenvalue problem using ARCPACK (shift‑invert mode for smallest eigenvalues).

## Heat Kernel Signature (HKS)
```python
from src.HKS.heatKernelSignatures import compute_hks

hks, times = compute_hks(evals, evecs, n_times=50)
```

## Scale Invariant Heat Kernel Signature (SIHKS)
```python
from src.HKS.scaleInvariantHKS import compute_sihks_norm

sihks, scaled_times = compute_scale_invariant_hks(evals, evecs, n_times=50)
```

```python
from src.HKS.sihkseignorm import compute_scale_invariant_hks

sihks, scaled_times = compute_scale_invariant_hks(evals, evecs, n_times=50)
```

## Wave Kernel Signature (WKS)
```python
from src.WKS.waveKernelSignatures import compute_wks

wks, energies = compute_wks(evals, evecs, n_energies=100, sigma=6.0)
```

# Citation
If you use this code in your research, please cite the thesis:
```BibTeX
@article{Reagan2026Eigenspace,
  author  = {Kylind Reagan},
  title   = {Eigenspace Perturbations and Spectral Distance Metrics in Non-Rigid Shape Retrieval},
  school  = {Maryville College},
  year    = {2026},
  month   = {May}
}
```

# License
The code in this repository is released under the MIT License.
The thesis text and figures are © Kylind Reagan, 2026 – All Rights Reserved.

# Acknowledgements
Special thanks to Dr. Barbara Johnson and Dr. Jesse Smith for their guidance throughout this project. This work builds upon the foundational contributions of Sun et al. (HKS), Aubry et al. (WKS), and Bronstein & Kokkinos (SI‑HKS).

---

*"Look for the geometry in the humming of the strings and the music in the spacing of the spheres"*  
— Pythagoras

---
