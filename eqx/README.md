# EquivariantX

EquivariantX (EQX) provides:

- **PyTorch-native O(2) operations:** real irreps, linear maps, tensor products,
  normalized activations, gates, harmonics, and many-body contractions.
- **Global O(3) ↔ local O(2) conversion:** e3nn-compatible representation
  metadata, Wigner rotations, restriction, and reflection-basis changes.
- **CUDA-fused O(3)/O(2) convolutions:** graph gathering, angular operations,
  radial weighting, and aggregation, with derivatives for force training.

O(2) irreps can include time-reversal parity. Native O(2) operations and frame
conversions use flattened `ir_mul` features: each irrep entry is stored as
`(..., ir.dim, mul)` before flattening. e3nn feature tensors use a different
layout and must be transposed within each entry before frame conversion.

See the [tutorials](https://tace.readthedocs.io/en/latest/equivariantx/tutorials.html),
[convolution guide](https://tace.readthedocs.io/en/latest/equivariantx/convolutions.html),
and [API reference](https://tace.readthedocs.io/en/latest/equivariantx/api.html).

## Installation

EQX is bundled with TACE but can be installed independently, without installing
TACE. Source installation is currently supported; a separate package release is
planned once the library is mature.

```bash
git clone https://github.com/xvzemin/tace.git
pip install ./tace/eqx
```

The base dependencies are PyTorch >= 2.4, e3nn >= 0.4.4, and opt_einsum_fx.
PyG, torch-scatter, Triton, and a CUDA compiler are not required for native
PyTorch operations.

With e3nn 0.4.x and recent PyTorch, import `eqx` before `e3nn.o3`. EQX loads
the packaged e3nn constants in a scoped safe-loading context, without changing
`torch.load` defaults. Available angular degrees follow the installed e3nn
coefficient tables. Global time-odd irreps require the time-reversal e3nn
extension; ordinary spatial operations work with upstream e3nn.

For CUDA fusion, install the optional build dependencies:

```bash
pip install './tace/eqx[cuda]'
```

A CUDA toolkit is also required. Set `CUDA_HOME` if it is not detected.
The C++ launcher and generated CUDA kernels compile lazily on first use and
are cached. Installing or importing EQX does not compile them.

## Native O(2) operations

```python
import torch
from eqx import o2

linear = o2.Linear("8x0e + 4x1m", "4x0e + 2x1m")
features = linear.irreps_in.randn(32, -1)
output = linear(features)
assert output.shape == (32, linear.irreps_out.dim)
```

`o2.Linear`, `o2.UuLinear`, `o2.TensorProduct`, `o2.Activation`, and `o2.Gate`
operate directly on PyTorch tensors. They do not require the fused backend.
`o2.WignerD(method="recursive")` also runs entirely in PyTorch. Its default
`method="auto"` selects the optional quaternion CUDA kernel for supported CUDA
inputs and the PyTorch construction otherwise.

## Convolutions

| Interface in `eqx.conv` | Operation |
|---|---|
| `O3TensorProductConv` | Sparse O(3) CGTP, retaining the supplied paths and normalization |
| `O2O3TensorProductConv` | Equivalent harmonic-edge O(3) CGTP evaluated in aligned O(2) frames |
| `UuO2TensorProductConv` | One externally weighted channelwise O(2) linear map |
| `UvO2TensorProductConv` | O(2) Linear → Gate → Linear, optionally with edge features and attention |

The first two preserve e3nn tensor-product instructions and weights after
feature-layout conversion. The aligned form requires multiplicity-one,
time-even, natural-parity spherical harmonics as the second input. It is not a
replacement for a tensor product with an arbitrary second feature tensor.
Native uu/uv convolutions instead parameterize local O(2) maps directly.

CGTP and uu CUDA contractions avoid full edge messages and recompute bounded
radial workspaces for backward. The uv convolution retains radial weights and
local GEMM operands while fusing surrounding rotations, gates, and aggregation.
These are distinct fusion strategies, not a claim that every edge tensor is
eliminated. See the convolution guide for supported backends and examples.

## Module organization

| Module | Responsibility |
|---|---|
| `eqx.o2` | Native O(2) operators and O(3)/O(2) frame conversion |
| `eqx.conv` | General fused O3, O2-O3, uu-O2, and uv-O2 graph convolutions |
| `eqx.kernels` | Shared geometry kernels, CUDA compilation, and launch support |
| `eqx.o3` | Element-dependent linear maps using e3nn `mul_ir` layouts |
| `eqx.ace` | Atomic cluster expansions, independent of convolutions |
| `eqx.models.tace` | TACE-specific interaction and bilinear ACE fusion |
| `eqx.models.mace` | MACE conversion for ASE inference and training |
| `eqx.utils` | Shared utilities, including cached operator metadata parsing |

Native coupling formulas and Wigner indexing are shared within `o2`.
Convolutions share graph ordering, radial projection, expression programs,
and derivative rules. CUDA source generation and execution stay separate from
the operator interfaces. TACE owns its model definition and checkpoint
migration. The MACE adapter replaces operators in an existing model without
modifying MACE source code.

## MACE integration

Install MACE separately, or include the optional dependency during source
installation:

```bash
pip install './tace/eqx[mace,cuda]'
```

```python
from eqx.models.mace import convert_mace_to_eqx
from mace.calculators import MACECalculator

# Load original weights and choose dtype/device before conversion.
model = convert_mace_to_eqx(model)
calculator = MACECalculator(models=model, device="cuda", default_dtype="float32")
```

The default leaves the other operators unchanged. Pass `enable_cueq=True` to
convert them through MACE's cuEquivariance interface; do not enable a second
backend conversion in `MACECalculator`. CPU execution uses PyTorch.

For training, convert before creating the optimizer and call the returned model
with `training=True`. Energy, force and stress losses retain their gradients.
Use a separate model for ASE, whose calculator disables parameter gradients.
See the convolution guide for training and checkpoint examples.

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
