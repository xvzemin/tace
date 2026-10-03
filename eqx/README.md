# EquivariantX

EquivariantX (EQX) provides spherical and Cartesian equivariant operators,
frame conversions, and fused CUDA convolutions.

## Installation

Install from source without installing TACE:

```bash
git clone https://github.com/xvzemin/tace.git
pip install ./tace/eqx
```

Requires PyTorch >= 2.4, e3nn >= 0.4.4, and SciPy >= 1.15.
For fused kernels, install `pip install './tace/eqx[cuda]'` and provide a CUDA
toolkit. Kernels compile on first use and are cached. A standalone release is planned.

With e3nn 0.4.x, import `eqx` before `e3nn.o3`; supported degrees depend on its
CG table. Global time-odd irreps require the
[time-reversal e3nn extension](https://github.com/xvzemin/e3nn/tree/time-reversal).

## API

| Module | API |
|---|---|
| `eqx.o3` | [Spherical O(3)](https://tace.readthedocs.io/en/latest/equivariantx/api/o3.html) |
| `eqx.co3` | [Cartesian O(3)](https://tace.readthedocs.io/en/latest/equivariantx/api/co3.html) |
| `eqx.co2` | [Cartesian O(2)](https://tace.readthedocs.io/en/latest/equivariantx/api/co2.html) |
| `eqx.o2` | [Spherical O(2)](https://tace.readthedocs.io/en/latest/equivariantx/api/o2.html) |
| `eqx.nn` | [Nonlinearities](https://tace.readthedocs.io/en/latest/equivariantx/api/nn.html) |
| `eqx.conv` | [Fused convolutions](https://tace.readthedocs.io/en/latest/equivariantx/api/convolutions.html) |
| `eqx.ace` | [Cluster expansions](https://tace.readthedocs.io/en/latest/equivariantx/api/ace.html) |
| `eqx.models` | [Model integration](https://tace.readthedocs.io/en/latest/equivariantx/api/models.html) |

[Tools](https://tace.readthedocs.io/en/latest/equivariantx/api/tools.html) cover
basis conversions, rotations, tensor decomposition, and module utilities.

```python
from eqx import o2

linear = o2.Linear("8x0e + 4x1m", "4x0e + 2x1m")
features = linear.irreps_in.randn(32, -1)
output = linear(features)
```

Spherical O(2) uses flattened `ir_mul` storage; Cartesian operators use `mul_ir`.
See each operator's API for input layouts, weight shapes, and backend support.

## Tests

From the repository root:

```bash
pip install -e './eqx[dev]'
pytest eqx/tests
```

CUDA tests require a GPU and CUDA build dependencies. Model integration tests
require the corresponding model packages. EQX tests do not require TACE.

## License

[Apache License 2.0](LICENSE.md).

## Citation

If you use the local O(2) method or its global O(3)/local O(2) conversion,
please cite:

```bibtex
@misc{xu2026lookingglassefficientparitycompletelearning,
  title={Through the Looking-Glass: Efficient Parity-Complete Learning via Local $O(2)$ Frames}, 
  author={Zemin Xu and Wenbo Xie},
  year={2026},
  eprint={2608.16592},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2608.16592}, 
}
```
