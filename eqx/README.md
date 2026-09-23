# EquivariantX

EquivariantX provides real O(2) representations, equivariant PyTorch
operators, and transformations between global O(3) and local O(2) features.
Irreps include time-reversal parity. Features use a flattened `ir_mul`
layout, with the multiplicity axis last within each irrep entry.

The library is bundled with TACE and can also be installed independently.
See the [tutorials](https://tace.readthedocs.io/en/latest/equivariantx/tutorials.html)
and [API reference](https://tace.readthedocs.io/en/latest/equivariantx/api.html).

## Installation

EquivariantX currently supports installation from source only and does not
require TACE. A standalone package release is planned once the library is
fully mature.

```bash
git clone https://github.com/xvzemin/tace.git
pip install ./tace/eqx
```

The default operators use PyTorch on CPU and CUDA. The base dependencies
are PyTorch, `e3nn`, and `opt_einsum_fx`.

For fused CUDA convolutions, install the optional backend:

```bash
pip install './tace/eqx[cuda]'
```

## Example

```python
import torch
from eqx import o2

linear = o2.Linear("8x0e + 4x1m", "4x0e + 2x1m")
features = linear.irreps_in.randn(32, -1)
output = linear(features)
```

## Citation

If you use the local O(2) method or its global O(3)/local O(2)
conversion, please cite:

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
