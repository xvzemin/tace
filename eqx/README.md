# EquivariantX

EquivariantX (EQX) provides PyTorch-native O(2) and Cartesian O(2)/O(3) operators, e3nn-compatible
O(3)/O(2) frame conversion, and fused CUDA convolutions. Time-reversal labels
are optional.

## Installation

Install from source without installing TACE:

```bash
git clone https://github.com/xvzemin/tace.git
pip install ./tace/eqx
```

Native operations require **PyTorch >= 2.4** and **e3nn >= 0.4.4**, not PyG
or custom CUDA extensions. For fused kernels:

```bash
pip install './tace/eqx[cuda]'
```

A CUDA toolkit is required; set `CUDA_HOME` if necessary. Generated kernels
compile on first use and are cached. A standalone package release is planned.
With e3nn 0.4.x, import `eqx` before `e3nn.o3`. Global time-odd irreps require
the time-reversal e3nn extension.

## Native operations and frames

```python
import torch
from eqx import o2

linear = o2.Linear("8x0e + 4x1m", "4x0e + 2x1m")
features = linear.irreps_in.randn(32, -1)
output = linear(features)
```

O(2) features use flattened `ir_mul` storage: `(..., ir.dim, mul)` within
each irrep entry. Transpose each entry when converting from e3nn's `mul_ir`
storage. `WignerD` builds rotations; `LocalFrame` applies rotations and
reflection-basis changes. Use `WignerD(method="recursive")` for a purely
PyTorch construction, including on GPU.

See the [tutorial](https://tace.readthedocs.io/en/latest/equivariantx/tutorials.html)
for runnable Linear, Gate, TensorProduct, and frame-conversion examples.

## Fused convolutions

![Four convolution interfaces](docs/source/_static/convolutions.svg)

The CGTP interfaces preserve the supplied paths, weights, and normalization.
The aligned CGTP requires natural-parity, time-even harmonic edge inputs with
one channel per degree. Native Uu/Uv convolutions parameterize O(2) maps
directly. CUDA kernels support force training and higher derivatives.

| Module | Role |
|---|---|
| `eqx.o2` | Native O(2) operators and O(3)/O(2) frames |
| `eqx.co3` | Cartesian O(3) irreps, harmonics, Linear, Gate, and tensor products in `mul_ir` layout |
| `eqx.co2` | Cartesian O(2) operators and coordinate-free harmonic CGTP in `mul_ir` layout |
| `eqx.conv` | General fused convolutions and graph attention |
| `eqx.o3` | Element-dependent Linear and Gate in `mul_ir` layout |
| `eqx.ace` | Atomic cluster expansions |
| `eqx.models` | TACE fusion and MACE, NequIP, SevenNet, Prophet conversion |
| `eqx.kernels` | Shared geometry, compilation, and launch support |

See the [convolution guide](https://tace.readthedocs.io/en/latest/equivariantx/convolutions.html)
for fusion boundaries and the [API](https://tace.readthedocs.io/en/latest/equivariantx/api.html)
for parameters. Model converters use `implementation="o3"` by default;
`implementation="o2"` selects the equivalent aligned tensor product.
Install the consuming model package separately or through the `mace`,
`nequip`, or `sevennet` extras. Prophet is installed from its source repository.
See [model integration](docs/source/models.rst) for usage and restrictions.

## Cartesian O(3)

`co3` stores each degree-`l` tensor as `(..., mul, 3**l)` before flattening.
Iteration yields `(mul, ir)`. The symmetric traceless subspace has `2*l+1`
independent coordinates; `ChangeOfBasis` converts to and from spherical
features through orthonormal path matrices. Tensor products retain all
requested delta/epsilon paths and their independent weights. `project=False`
defers output projection past sums and channel-linear maps, but not past
nonlinearities or subsequent tensor products.

See the [Cartesian tutorial](docs/source/cartesian.rst) for normalization,
operators, and equivalent-model conversion.

## Cartesian O(2)

`co2` provides two-dimensional STF tensors and conversions to compact O(2)
features. `co2.O3TensorProduct` uses transverse Cartesian restriction to
evaluate harmonic CGTP paths without explicit local frames. Paths, weights,
and normalization are preserved, including conversions from `co3.TensorProduct`.
These are PyTorch reference operators, not fused CUDA kernels. See the
[Cartesian O(2) tutorial](docs/source/cartesian_o2.rst) for examples and limits.

## Tests

From the repository root:

```bash
pip install -e './eqx[dev]'
pytest eqx/tests
```

EQX tests do not require TACE. CUDA tests require the CUDA build dependencies
and a GPU; model integration tests require the corresponding model package.
Run TACE integration tests with `pytest tests`,
or both suites with `pytest tests eqx/tests`.

## Citation

If you use the local O(2) method or its global O(3)/local O(2) conversion,
please cite:

```bibtex
@misc{eqx,
  title={Through the Looking-Glass: Efficient Parity-Complete Learning via Local O(2) Frames}, 
  author={Zemin Xu and Peijun Hu and Wenbo Xie},
  year={2026},
  eprint={2608.16592},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2608.16592}, 
}
```
