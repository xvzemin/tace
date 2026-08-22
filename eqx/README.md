# EQX

EQX provides self-contained equivariant operators. Its current public API
focuses on complete real O(2) representations, linear layers, gates, tensor
products, asymmetric contractions, circular harmonics, Wigner-D matrices, and
conversion between global O(3) features and local O(2) features.

## Basic O(2) operations

O(2) scalar irreps are written as `0e` and `0o`. Every positive order is a
two-dimensional real irrep written as `1m`, `2m`, and so on. The component and
channel axes remain separate:

```python
import torch

from eqx import o2

irreps = o2.Irreps("0e + 0o + 2x1m")
features = torch.randn(8, irreps.dim, 16)

linear = o2.Linear(irreps, irreps, channels_in=16, channels_out=32)
output = linear(features)
assert output.shape == (8, irreps.dim, 32)
```

## Global O(3) and local O(2)

`WignerD` constructs an edge-aligned real Wigner-D transform directly from a
three-dimensional edge vector. `O3O2Layout` then rotates global O(3) features,
groups equal local O(2) irreps into contiguous blocks, and keeps the parent O(3)
degree and parity metadata required by the inverse transform.

For an O(3) irrep of degree `l` and parity `p`, restriction to the local frame
produces orders `0, 1, ..., l`. Its order-zero block is `0e` when
`p * (-1)**l == 1` and `0o` otherwise. Positive orders use `1m`, `2m`, and so
on.

The following example performs an O(2)-equivariant linear operation in each
edge frame and converts the result back to the global O(3) layout:

```python
import torch
from e3nn import o3

from eqx import o2

num_edges = 8
channels = 2
lmax = 2
mmax = 2

# Every O(3) irrep uses the same channel multiplicity in this example.
irreps_o3 = o3.Irreps("2x0e + 2x1o + 2x2e")
edge_features = torch.randn(num_edges, irreps_o3.dim)
edge_vectors = torch.randn(num_edges, 3)

wigner_d = o2.WignerD(lmax=lmax, mmax=mmax)
D, D_inv = wigner_d.get_wigner(edge_vectors)

layout = o2.O3O2Layout(irreps_o3, lmax=lmax, mmax=mmax)
local_blocks = layout(edge_features, D)

local_linear = o2.Linear(
    layout.local_irreps,
    layout.local_irreps,
    channels_in=channels,
)
local_output = local_linear.forward_grouped(local_blocks)
global_output = layout.inverse(local_output, D_inv)

assert global_output.shape == edge_features.shape
```

Setting `mmax < lmax` retains only local orders `0 <= m <= mmax`. The inverse
uses degree-wise rescaling to preserve average feature variance after this
truncation.

EQX is under active development, and compatibility between development
versions is not currently guaranteed.

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
