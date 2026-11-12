# EquivariantX

EquivariantX (EQX) provides spherical and Cartesian equivariant operators
using PyTorch, with optional fused CUDA implementations.

## Installation

Install from source without installing TACE:

```bash
git clone https://github.com/xvzemin/tace.git
pip install ./tace/equivariantx
```

Requires PyTorch >= 2.4, e3nn >= 0.4.4, and SciPy >= 1.15.
For fused kernels, install `pip install './tace/equivariantx[cuda]'` and provide a CUDA
toolkit. Kernels compile on first use and are cached.

Eligible CUDA row-product programs automatically replay CUDA Graphs, including
their registered derivatives. Shape buckets and shared temporary storage limit
capture overhead for changing batch sizes. Set `EQX_USE_CUDA_GRAPH=0` to compare
against ordinary CUDA execution; CPU execution is unchanged.

Import the package as:

```python
import equivariantx as eqx
```

With e3nn 0.4.x, import `equivariantx` before `e3nn.o3`; supported degrees depend on its
CG table. `o2` and `co2` support time-reversal labels directly. Time-odd
`o3` and `co3` irreps require the
[time-reversal e3nn extension](https://github.com/xvzemin/e3nn/tree/time-reversal).

## API

| Module | API |
|---|---|
| `equivariantx.o3` | [Spherical O(3)](https://tace.readthedocs.io/en/latest/equivariantx/api/o3.html) |
| `equivariantx.co3` | [Cartesian O(3)](https://tace.readthedocs.io/en/latest/equivariantx/api/co3.html) |
| `equivariantx.co2` | [Cartesian O(2)](https://tace.readthedocs.io/en/latest/equivariantx/api/co2.html) |
| `equivariantx.o2` | [Spherical O(2)](https://tace.readthedocs.io/en/latest/equivariantx/api/o2.html) |
| Rotations, basis conversions, decomposition and module utilities | [Tools](https://tace.readthedocs.io/en/latest/equivariantx/api/tools.html) |

Built-in modules: [neural network modules](https://tace.readthedocs.io/en/latest/equivariantx/api/nn.html),
[convolutions](https://tace.readthedocs.io/en/latest/equivariantx/api/convolutions.html),
and [many-body expansions](https://tace.readthedocs.io/en/latest/equivariantx/api/mbe.html).
[Model adapters](https://tace.readthedocs.io/en/latest/equivariantx/api/models.html)
convert supported pretrained architectures.

Spherical O(2) uses flattened `ir_mul` storage; Cartesian operators use `mul_ir`.
See the API for parameters and examples.

## Tests

From the repository root:

```bash
pip install -e './equivariantx[dev]'
pytest equivariantx/tests
```

CUDA tests require a GPU and CUDA build dependencies. Model integration tests
require the corresponding model packages. EQX tests do not require TACE.

## License

[Apache License 2.0](LICENSE.md).

## Citation

For a comprehensive citation of EquivariantX, please refer to our latest
manuscript unifying the four representations, listed below. It is not yet
published.

| Use | Reference |
|---|---|
| `o3`: spherical O(3) | [e3nn](https://github.com/e3nn/e3nn) |
| `co3`: Cartesian O(3) | We recommend citing [Cartesian TACE](#cartesian-tace) and [ICTP](https://github.com/nec-research/ictp). |
| `o2`: spherical O(2), including global/local conversion | [Local O(2)](#local-o2) |
| `co2`: Cartesian O(2) | [EquivariantX](#equivariantx-unpublished) |
| Any fused kernel | We recommend citing [EquivariantX](#equivariantx-unpublished) and [Sobek](https://github.com/Cortex-Discovery/sobek). |

### EquivariantX (unpublished)

Zemin Xu and Wenbo Xie. *EquivariantX: Unifying Spherical and Cartesian O(3)
and O(2) Representations with High-Performance Fused Kernels.*

### Cartesian TACE

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
```

### Local O(2)

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
