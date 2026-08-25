[![TACE](https://img.shields.io/pypi/v/tace?style=for-the-badge&label=TACE)](https://pypi.org/project/tace/)
[![Docs](https://img.shields.io/readthedocs/tace?style=for-the-badge&label=docs)](https://tace.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Matbench Discovery](https://img.shields.io/badge/Matbench%20Discovery-SOTA-brightgreen?style=for-the-badge)](https://matbench-discovery.materialsproject.org/)

# Tensor Atomic/Edge Cluster Expansion (TACE/TECE)

TACE is designed with physical priors and strong inductive biases to enhance extrapolation capability. 
It performs Atomic Cluster Expansion and Edge Cluster Expansion based on spherical tensors 
or irreducible Cartesian tensors, with an optional attention architecture.

## Cartesian Architecture

<img src="fig/cartesian_arch.png" width="100%" align="center">

## Spherical/SO(2) Architecture

The architecture of the spherical model is largely the same as that of the Cartesian space. 
For details on the SO(2) component, please refer to our paper and code.

## Wigner6j/O(2) Architecture

The current implementation is still subject to change, and backward compatibility is not guaranteed. Please refer to our paper and code for details.

## Documentation

[TACE DOCS](https://tace.readthedocs.io/en/latest/index.html)

## SOTA Foundation Model

[TACE FOUNDATION](https://github.com/xvzemin/tace-foundations)

Default Ranking on Matbench as of July 8, 2026

<img src="fig/matbench_tece_rra.png" width="100%" align="center">

## Install, Train and Tutorial

The docs contain a complete tutorial. 

We also provide complete input files and example scripts for training, ASE,
TorchSim, LAMMPS, and other workflows in the
[TACE examples](https://github.com/xvzemin/tace/tree/main/example).

```bash
# Minimal install and training example
git clone https://github.com/xvzemin/tace
cd tace
pip install . # or pip install tace
cd example/train
tace-train -cn tace.yaml
```

## Fine-tuning

- ✅ Full-parameter.

- ✅ Freeze-parameter.

- ✅ LoRA.

## Overview

Currently, the officially supported properties include:

- Energy
- Forces (conservative | direct)
- Hessian (conservative, predict only)
- Stress (conservative | direct)
- Virials (conservative | direct)
- Charges (lagrangian or uniform_distribution)
- Dipole moment (conservative | direct)
- Polarization (conservative, multi-value for PBC systems)
- Polarizability (conservative | direct)
- Born effective charges (conservative, under electric field)
- Atomic stresses (conservative, predict only)
- Atomic virials (conservative, predict only)
- Absolute final collinear magmoms
- Noncollinear magnetic forces (SOC, full O(3))

For embedding property, we support:

- fidelity_idx (different computational levels)
- charges
- total charge
- electric field
- initial noncollinear magmoms (SOC, full O(3))

## Plugins

TACE currently supports the following plugin:

- **TACE-LES** (Latent Ewald Summation)
- **TACE-QEq** (Lagrangian)
- **mTACE**    (Magnetic, Spin-Orbit Coupling)

## Interfaces

- ✅ Supports integration with [ASE Calculator](https://wiki.fysik.dtu.dk/ase/).

- ✅ Supports integration with [LAMMPS-ML-IAP](https://github.com/lammps/lammps).

- ✅ Supports integration with [TorchSim](https://torchsim.github.io/torch-sim/).

- ✅ Supports integration with [NVIDIA NValCHEMI](https://github.com/NVIDIA/nvalchemi-toolkit).

- ✅ Supports integration with [OpenMM-ML](https://github.com/openmm/openmm-ml) (OpenMM-ML -> ASE -> TACE).

- ✅ Supports integration with [USPEX](https://uspex-team.org/)
  (USPEX -> LAMMPS-ML-IAP -> TACE) (Python=3.9).

## Contact

For bugs or feature requests, please use the
[TACE issue](https://github.com/xvzemin/tace/issues).

<!-- For usage discussions, you can join the TACE community through QQ or Discord.

<table>
  <tr>
    <th>QQ</th>
    <th>Discord</th>
  </tr>
  <tr>
    <td align="center">
      <img src="fig/qq.jpg" alt="TACE QQ QR code" width="260">
    </td>
    <td align="center">
      <img src="fig/discord.jpg" alt="TACE Discord QR code" width="260">
    </td>
  </tr>
</table> -->

## Citing

If you use TACE, please cite our papers:

```bibtex
@misc{xu2026spectralspatialtensoratomiccluster,
   title={Spectral/Spatial Tensor Atomic Cluster Expansion with Universal Embeddings in Cartesian Space}, 
   author={Zemin Xu and Wenbo Xie and P. Hu},
   year={2026},
   eprint={2509.14961},
   archivePrefix={arXiv},
   primaryClass={stat.ML},
   url={https://arxiv.org/abs/2509.14961}, 
}

@misc{xu2026edgeclusterexpansionradial,
   title={Edge Cluster Expansion with Radial Rotary Attention for Interatomic Potentials}, 
   author={Zemin Xu and Wenbo Xie and P. Hu},
   year={2026},
   eprint={2607.10664},
   archivePrefix={arXiv},
   primaryClass={stat.ML},
   url={https://arxiv.org/abs/2607.10664}, 
}
```

If you use Local O(2) Frame or Generalized Wigner-6j Convlution, please cite our papers: 

```bibtex
@misc{xu2026completeo3interactionswigner6j,
   title={Complete O(3) Interactions from Wigner-6j Recoupling to Local O(2) Frames}, 
   author={Zemin Xu and Peijun Hu and Wenbo Xie},
   year={2026},
   eprint={2608.16592},
   archivePrefix={arXiv},
   primaryClass={physics.chem-ph},
   url={https://arxiv.org/abs/2608.16592}, 
}
```

If you use cartnn, Cartesian-3j, cMACE, cNequIP, cAllegro, please cite our papers:

```bibtex
@inproceedings{xu2026a,
   title={A Cartesian-3j Framework for Machine Learning Interatomic Potentials},
   author={Zemin Xu and Chenyu Wu and Wenbo Xie and Peijun Hu},
   booktitle={Forty-third International Conference on Machine Learning},
   year={2026},
   url={https://openreview.net/forum?id=9ZWK6gneWq}
}
```

## Development

Install the development tools and run the same formatting and lint checks used by
pre-commit:

```bash
pip install -e ".[dev]"
ruff check --fix tace
ruff format tace
ruff check tace
ruff format --check tace
```

## License

The TACE code is published and distributed under the MIT License.
