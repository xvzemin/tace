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

## Package organization

`o2` defines representations and individual operators. `conv` contains fused
implementations of specific convolution architectures. `kernels` contains
shared geometry kernels and CUDA compilation support, independent of any
convolution architecture.

```text
eqx/
├── o2/
│   ├── irreps.py                 # Representation metadata
│   ├── linear.py                 # O(2) Linear
│   ├── gate.py                   # Activation and Gate
│   ├── tensor_product.py         # O(2) TensorProduct
│   ├── asymmetric_contraction.py # Many-body contractions
│   ├── circular_harmonics.py     # Circular harmonics
│   ├── rotation_matrix.py        # Axis alignment
│   ├── wigner.py                 # WignerD interface and PyTorch implementation
│   ├── local_frame.py            # Restriction and feature rotations
│   └── o3_tensor_product.py      # Aligned O(3) TensorProduct
├── conv/
│   ├── __init__.py               # Public convolution classes
│   ├── contraction.py            # Shared recursive transpose rule
│   ├── graph.py                  # Shared graph ordering
│   ├── o3/
│   │   ├── convolution.py        # O3TensorProductConv and PyTorch reference
│   │   ├── cuda.py               # Path scheduling and CUDA execution
│   │   └── codegen.py            # Sparse CG and fused radial contractions
│   └── o2_o3/
│       ├── convolution.py        # O2O3TensorProductConv and reference adjoints
│       ├── geometry.py           # Recursive direction derivatives
│       ├── schedule.py           # Path grouping and radial workspaces
│       ├── cuda.py               # CUDA execution plans
│       ├── codegen.py            # Tensor-product CUDA source generation
│       └── direction_codegen.py  # Mixed-derivative CUDA source generation
└── kernels/
    ├── cuda.py                   # NVRTC compilation, caching and launches
    ├── codegen.py                # Shared geometry source generation
    ├── wigner.py                 # Packed Wigner construction and derivatives
    ├── quaternion.py             # Quaternion polynomial kernels
    └── csrc/runtime.cpp          # Model-independent CUDA runtime
```

### Convolution interfaces

| Class | Implementation directory | Operation | Status |
|---|---|---|---|
| `O3TensorProductConv` | `conv/o3/` | Sparse O(3) CGTP with fused radial projection | Implemented |
| `O2O3TensorProductConv` | `conv/o2_o3/` | O(3) CGTP through aligned-frame sparse coupling | Implemented |
| `UvO2TensorProductConv` | `conv/uv_o2/` | Channel-mixing O(2) Linear → Gate → Linear | Planned |
| `UuO2TensorProductConv` | `conv/uu_o2/` | Channelwise O(2) Linear convolution | Planned |

The planned directories and classes are not created until their operators are
implemented. Each architecture owns its reference operation, derivatives and
CUDA schedule; shared geometry and compilation are reused without inheriting
from another convolution implementation. A nonlinear gated convolution must
differentiate its activations and cannot reuse the multilinear CGTP transpose
rule unchanged.

Use `from eqx.conv import O2O3TensorProductConv` and
`from eqx.kernels import wigner_D`. CUDA compilation remains lazy. File
organization does not change instruction ordering, weights, normalization or
derivative rules.

`O3TensorProductConv` accepts a tensor product defining `uvu` instructions,
including repeated output irreps and different multiplicities. Features use
flattened `ir_mul` order. Supply node features, edge attributes, radial features,
the final radial projection, and edge indices to `forward`. CUDA fuses the
projection with sparse CG contractions and graph reduction without allocating
edge messages or projected edge weights. Transposed programs support force
training and recursive higher derivatives. The PyTorch reference remains
available on CPU or with `backend="torch"`. No CUDA compiler is loaded at import.

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
