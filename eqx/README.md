# EQX

EQX is a self-contained library for equivariant PyTorch operators. Its current
focus is complete real O(2) representation theory together with the conversion
between global O(3) features and local O(2) features.

The public O(2) API provides:

- `Irrep` and `Irreps` for complete real O(2) representations;
- `Linear` with `uv` and `uu` channel paths;
- `Gate` for equivariant scalar and tensor nonlinearities;
- `TensorProduct` with `u1u`, `uuu`, and `uvw` paths;
- `CircularHarmonics` for native two-dimensional inputs;
- `LocalFrame` and `WignerD` for global O(3)/local O(2) conversion;
- `AsymmetricContraction` for higher-order O(2) contractions.

EQX is under active development. Compatibility between development versions is
not currently guaranteed.

## Complete real O(2) representations

The one-dimensional scalar irreps are `0e` and `0o`. The positive-order real
irreps are two-dimensional and are written as `1m`, `2m`, and so on:

```python
from eqx import o2

irreps_o2 = o2.Irreps("4x0e + 2x0o + 3x1m + 2x2m")
```

`0e` and `0o` are distinguished by reflection parity. A positive-order `lm`
stores its real cosine-like and sine-like components together. O(2)-equivariant
linear maps may mix channels and copies of the same irrep, but never mix
different O(2) irreps.

Standalone O(2) operators use tensors with shape
`(..., irreps_o2.dim, channels)`. Missing output irreps are zero padded by
`Linear`.

## Linear–Gate–Linear in a local O(2) frame

An O(3) irrep with degree `l` restricts to local orders `0, ..., l`. Its local
order-zero representation is `0e` when `p * (-1)**l == 1` and `0o` otherwise.
Every positive order is represented by `1m`, `2m`, and so on. Consequently,
both polar and pseudo O(3) features can be processed without discarding parity.

The following example gathers global O(3) node features onto edges, rotates
them into local O(2) frames, applies a basic `Linear -> Gate -> Linear` network,
and rotates the resulting edge messages back to global O(3):

```python

import torch
from e3nn import o3

from eqx import o2

num_nodes = 8
num_edges = 20
channels = 2
lmax = 3
mmax = 2

# Every listed O(3) irrep has the same channel multiplicity.
irreps_o3 = o3.Irreps(
    "2x0e + 2x0o + 2x1o + 2x1e + "
    "2x2e + 2x2o + 2x3o + 2x3e"
)
node_feats = torch.randn(num_nodes, irreps_o3.dim)

edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_vectors = torch.randn(num_edges, 3)

wigner_d = o2.WignerD(lmax=lmax, mmax=mmax)
D, D_inv = wigner_d.get_wigner(edge_vectors)

frame = o2.LocalFrame(
    irreps_o3,
    lmax=lmax,
    mmax=mmax,
    layout="mul_ir",
)
irreps_o2 = frame.local_irreps

nonlinearity = o2.Gate(
    irreps_o2,
    act_0e=torch.nn.SiLU(),
    act_0o=torch.nn.Tanh(),  # An activation on 0o must be odd.
    act_lm=torch.nn.Sigmoid(),
)
linear_up = o2.Linear(
    irreps_o2,
    nonlinearity.irreps_in,
    channels_in=channels,
    channels_out=channels,
)
linear_down = o2.Linear(
    nonlinearity.irreps_out,
    irreps_o2,
    channels_in=channels,
    channels_out=channels,
    bias=False,
)

source = edge_index[0]
node_feats_o2 = frame.to_local(node_feats[source], D)
node_feats_o2 = linear_up.forward_grouped(node_feats_o2)
node_feats_o2 = nonlinearity.forward_grouped(node_feats_o2)
node_feats_o2 = linear_down.forward_grouped(node_feats_o2)
edge_messages = frame.to_global(node_feats_o2, D_inv)

assert edge_messages.shape == (num_edges, irreps_o3.dim)
```

`frame.to_local` returns a tuple ordered exactly as `irreps_o2`. Each tensor
has shape `(..., irrep.dim, multiplicity * channels)`. The `forward_grouped`
methods preserve this representation, and `frame.to_global` restores the
flattened O(3) layout selected by `layout`.

The example uses the default flattened `mul_ir` layout. `layout="ir_mul"` is
also supported.

Setting `mmax < lmax` retains only local orders `0 <= m <= mmax`.

## Native O(2) inputs

`CircularHarmonics` constructs O(2) features directly from two-dimensional
vectors:

```python
import torch

from eqx import o2

vectors_2d = torch.randn(32, 2)
circular_harmonics = o2.CircularHarmonics(m_max=3)
edge_attrs = circular_harmonics(vectors_2d)

assert circular_harmonics.irreps_out == o2.Irreps("0e + 1m + 2m + 3m")
assert edge_attrs.shape == (32, circular_harmonics.irreps_out.dim)
```

With `normalize=True`, the output depends only on direction. With
`normalize=False`, order `m` is homogeneous of degree `m` in the input vector.

## Citation

If you use the complete local O(2) method or its global O(3)/local O(2)
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
