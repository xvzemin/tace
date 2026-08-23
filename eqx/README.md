# EQX

EQX is a self-contained library for equivariant PyTorch operators. Its current
focus is complete real O(2) representation theory, conversion between global
O(3) and local O(2) features, and Cartesian O(3) operators.

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
channels = 64
lmax = 3
mmax = 2

# Every listed O(3) irrep has the same channel multiplicity.
irreps_o3 = o3.Irreps(
    "64x0e + 64x0o + 64x1o + 64x1e + "
    "64x2e + 64x2o + 64x3o + 64x3e"
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

## Cartesian O(3)

The `co3` module represents a rank-`l` irreducible Cartesian
tensor with `3**l` stored components. Both polar tensors and pseudotensors are
supported through the explicit parity label `p` in `0e`, `0o`, `1o`, `1e`,
and so on. For an orthogonal matrix $Q\in O(3)$, the representation acts as

\[
\eta_{l,p}=\frac{1-p(-1)^l}{2},\qquad
T'=[\det(Q)]^{\eta_{l,p}}Q^{\otimes l}T.
\]

The tensor product contains the two Cartesian coupling branches. The
Kronecker-delta branch contracts `k` index pairs,

\[
(A\otimes_k^\delta B)_{\boldsymbol i\boldsymbol j}
=3^{-k/2}\sum_{\boldsymbol a}
A_{\boldsymbol i\boldsymbol a}B_{\boldsymbol a\boldsymbol j},
\qquad l_3=l_1+l_2-2k,
\]

while the Levi-Civita branch additionally couples one index from each input,

\[
(A\otimes_k^\epsilon B)_{\boldsymbol i w\boldsymbol j}
=(2\,3^k)^{-1/2}\sum_{\boldsymbol a,u,v}
A_{\boldsymbol i u\boldsymbol a}\epsilon_{wuv}
B_{\boldsymbol a v\boldsymbol j},
\qquad l_3=l_1+l_2-2k-1.
\]

Both branches obey `p3 = p1 * p2`. `Linear` uses dense UV channel mixing,
`Gate` uses invariant `0e` gates for Cartesian tensors, and
`CartesianHarmonics` constructs symmetric traceless Cartesian features.
The fixed tensors are also available as `co3.delta()` and
`co3.levi_civita()`.

### Cartesian convolution

A Cartesian convolution first gathers node features onto edges and couples
them to Cartesian harmonics in `u1u` mode. Every tensor-product path is kept
as a separate output copy, and `project=False` avoids projecting these edge
messages. After aggregation, a UV `Linear` compresses the paths and the result
is projected once at node level:

```python
import torch

from eqx import co3

num_nodes = 16
num_edges = 48
channels = 64
lmax = 2

irreps_node = co3.Irreps("0e + 0o + 1o + 1e + 2e + 2o")
cartesian_harmonics = co3.CartesianHarmonics(lmax, irreps_in="1o")
irreps_edge = cartesian_harmonics.irreps_out

# Project each ambient Cartesian block before using it as an irrep.
node_feats = torch.cat(
    [
        co3.project(torch.randn(num_nodes, irrep.dim, channels), irrep.l)
        for irrep in irreps_node.expanded()
    ],
    dim=1,
)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_vectors = torch.randn(num_edges, 3)
edge_attrs = cartesian_harmonics(edge_vectors).unsqueeze(-1)

# Retain one output irrep copy for every allowed tensor-product path.
irreps_paths = []
paths = []
for input1_index, irrep1 in enumerate(irreps_node.expanded()):
    for input2_index, irrep2 in enumerate(irreps_edge.expanded()):
        for irrep_out in irrep1 * irrep2:
            if irrep_out.l <= lmax:
                output_index = len(irreps_paths)
                irreps_paths.append(irrep_out)
                paths.append((output_index, input1_index, input2_index))
irreps_paths = co3.Irreps(irreps_paths)

tensor_product = co3.TensorProduct(
    irreps_node,
    irreps_edge,
    irreps_paths,
    channels_in1=channels,
    channels_in2=1,
    channels_out=channels,
    project=False,
    path_mode="u1u",
    path=paths,
)

source, target = edge_index
raw_edge_messages = tensor_product(node_feats[source], edge_attrs)
raw_node_messages = raw_edge_messages.new_zeros(
    num_nodes,
    irreps_paths.dim,
    channels,
)
raw_node_messages.index_add_(0, target, raw_edge_messages)

# Compress all paths of the same (l, p) type before projection.
linear_down = co3.Linear(
    irreps_paths,
    irreps_node,
    channels,
    channels,
    bias=False,
)
node_messages = linear_down(raw_node_messages)
node_messages = torch.cat(
    [
        co3.project(node_messages[..., block_slice, :], irrep.l)
        for irrep, block_slice in zip(
            irreps_node.expanded(),
            irreps_node.expanded_slices(),
        )
    ],
    dim=-2,
)
```

Because the projection is linear, projecting after aggregation is equivalent
to projecting every edge message before aggregation. Projection must be
completed before treating the result as an irreducible feature in a subsequent
nonlinear operation.

### Cartesian many-body expansion

For a node-level many-body expansion, node features are coupled directly with
`project=True`. The output is immediately an irreducible Cartesian:

```python
node_tensor_product = co3.TensorProduct(
    irreps_node,
    irreps_node,
    irreps_node,
    channels_in1=channels,
    channels_in2=channels,
    channels_out=channels,
    project=True,
    path_mode="uuu",
)
node_product = node_tensor_product(node_feats, node_feats)
```

The `project` argument is mandatory. Use `project=False` only when a linear
operation such as path compression or aggregation is intentionally placed
before projection; otherwise use `project=True`.

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

If you use Cartesian O(3) method please cite:

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
