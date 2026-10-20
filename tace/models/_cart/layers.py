"""Cartesian operators with the parameter layout of spherical TACE."""

import torch
from e3nn import o3
from e3nn.nn import Gate

from eqx import co3
from eqx.ace import TACE
from eqx.conv.co3 import CartesianTensorProductConv, Linear
from eqx.o3 import ElementLinear, MoEElementLinear
from tace.utils.env import acceleration_enabled
from tace.utils.torch_scatter import scatter_sum

from .._e3nn.basis_change import _SymmetricRankTwoBasisChange
from .._e3nn.fused import O3ScatterTensorProduct, UuuTensorProduct
from .._e3nn.layer_norm import EquivariantMergeLayerNorm
from .._e3nn.les import TACELES
from .._e3nn.node import SphericalTensorNodeEmbedding
from .._e3nn.prod import BilinearMoEACE, CgtpACE
from ..layout import LayoutTransform
from ..linear import e3nnElementLinear, e3nnLinear, e3nnMoEElementLinear


class RankTwoReadout(torch.nn.Module):
    """Combine a scalar trace and an STF tensor into a Cartesian matrix."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "zero", torch.eye(3, dtype=torch.float64) / 3**0.5, persistent=False
        )

    def forward(self, scalar, tensor):
        return scalar[:, None, None] * self.zero + tensor.reshape(-1, 3, 3)


def cartesian_tensor_product(module, *, project=True):
    """Build a Cartesian product with unchanged normalized instructions."""
    return co3.TensorProduct(
        module.irreps_in1,
        module.irreps_in2,
        module.irreps_out,
        [
            (
                p.i_in1,
                p.i_in2,
                p.i_out,
                p.connection_mode,
                p.has_weight,
                p.path_weight**2,
            )
            for p in module.instructions
        ],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
        shared_weights=module.shared_weights,
        project=project,
    )


class CartesianScatterTensorProduct(torch.nn.Module):
    """Contract Cartesian edge tensors and aggregate before STF projection.

    Parameters
    ----------
    tensor_product : o3.TensorProduct
        Weighted ``uvu`` paths and their normalization.
    """

    def __init__(self, tensor_product):
        super().__init__()
        self.tp = cartesian_tensor_product(tensor_product, project=False)
        self.irreps_in1 = self.tp.irreps_in1
        self.irreps_in2 = self.tp.irreps_in2
        self.irreps_out = self.tp.irreps_out
        self.weight_numel = self.tp.weight_numel
        self.compact_output = bool(acceleration_enabled("eqx", kernel="conv"))
        self.eqx_tp = CartesianTensorProductConv(
            self.tp,
            normalize=False,
            symmetric_inputs=True,
            compact_output=True,
        )

    def forward(self, features, edge_attrs, weights, edge_index):
        messages = self.tp(features[edge_index[0]], edge_attrs, weights)
        output = scatter_sum(messages, edge_index[1], dim=0, dim_size=features.shape[0])
        return (
            output.index_select(-1, self.eqx_tp.output_pack_index)
            if self.compact_output
            else output
        )

    def forward_stream(
        self,
        node_feats,
        edge_attrs,
        radial,
        projection,
        edge_index,
        edge_cutoff,
        edge_vector=None,
        radial_network=None,
    ):
        """Fuse the radial network, tensor contractions and graph reduction."""
        if edge_cutoff is not None and edge_vector is None:
            edge_attrs = edge_attrs * edge_cutoff
        return self.eqx_tp(
            node_feats,
            edge_attrs,
            radial,
            projection,
            edge_index,
            vectors=edge_vector,
            amplitudes=edge_cutoff,
            radial_network=radial_network,
        )


def linear_layout(module):
    """Replace angular storage without changing learned channel weights."""
    old = module.linear
    module.irreps_in = co3.Irreps(module.irreps_in)
    module.irreps_out = co3.Irreps(module.irreps_out)
    module.linear = co3.Linear(
        module.irreps_in,
        module.irreps_out,
        instructions=[(p.i_in, p.i_out) for p in old.instructions],
        internal_weights=False,
        shared_weights=old.shared_weights,
    )
    if isinstance(module, e3nnElementLinear):
        module.eqx_linear = ElementLinear(module.linear)
    indices = torch.zeros(module.irreps_out.dim, dtype=torch.long)
    mask = torch.zeros(module.irreps_out.dim, dtype=torch.float64)
    offset = 0
    for (mul, ir), section in zip(module.irreps_out, module.irreps_out.slices()):
        if ir.is_scalar():
            indices[section] = torch.arange(offset, offset + mul)
            mask[section] = 1
            offset += mul
    module._bias_index = indices
    module._bias_mask = mask
    return module


def cartesian_modules(module):
    """Construct Cartesian operators while retaining the model's parameters."""
    if isinstance(module, (e3nnLinear, e3nnElementLinear)):
        return linear_layout(module)
    if isinstance(module, e3nnMoEElementLinear):
        for name in (
            "irreps_in",
            "irreps_out",
            "expert_irreps_in",
            "expert_irreps_out",
        ):
            setattr(module, name, co3.Irreps(getattr(module, name)))
        module.linear = co3.Linear(
            module.expert_irreps_in,
            module.expert_irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        module.eqx_linear = MoEElementLinear(module.linear, module.num_experts)
        for side, irreps in (
            ("input", module.irreps_in),
            ("output", module.irreps_out),
        ):
            setattr(module, f"_{side}_slices", list(irreps.slices()))
            setattr(module, f"_{side}_dims", [ir.dim for _, ir in irreps])
        return module
    if isinstance(module, _SymmetricRankTwoBasisChange):
        return RankTwoReadout()
    if isinstance(module, EquivariantMergeLayerNorm):
        old_dim = module.dim
        module.dim = sum(3**l for l in module.ls)
        if module.affine:
            module.expand_index = torch.cat(
                [
                    torch.full((3**l,), i, dtype=torch.long)
                    for i, l in enumerate(module.ls)
                ]
            )
        if module.normalization == "component":
            module.balance_degree_weight = torch.cat(
                [
                    torch.full(
                        (3**l,),
                        1 / ((2 * l + 1) * len(module.ls))
                        if module.std_balance_degrees
                        else 1 / old_dim,
                        dtype=torch.float64,
                    )
                    for l in module.ls
                ]
            ).unsqueeze(0)
            module.std_balance_degrees = True
        return module
    if isinstance(module, Gate):
        gate = co3.Gate(
            module.act_scalars.irreps_in,
            [None] * len(module.act_scalars.acts),
            module.act_gates.irreps_in,
            [None] * len(module.act_gates.acts),
            module.irreps_gated,
        )
        # Reuse normalized activations rather than normalizing them twice.
        gate.act_scalars = module.act_scalars
        gate.act_gates = module.act_gates
        gate.mul = co3.ElementwiseTensorProduct(
            gate.irreps_gated, module.act_gates.irreps_out, project=False
        )
        gate.irreps_out = (
            co3.Irreps(module.act_scalars.irreps_out) + gate.mul.irreps_out
        )
        return gate
    if isinstance(module, O3ScatterTensorProduct):
        return CartesianScatterTensorProduct(module.tp)
    if isinstance(module, UuuTensorProduct):
        module.tp = cartesian_tensor_product(module.tp)
        module.irreps_in1 = module.tp.irreps_in1
        module.irreps_in2 = module.tp.irreps_in2
        module.irreps_out = module.tp.irreps_out
        if hasattr(module, "fused_tp"):
            del module.fused_tp
        return module
    if isinstance(module, o3.SphericalHarmonics):
        return co3.CartesianHarmonics(
            module.irreps_out, module.normalize, module.normalization
        )
    if isinstance(module, LayoutTransform):
        module.irreps = co3.Irreps(module.irreps)
        module.dims = [ir.dim for _, ir in module.irreps]
        module.num_components = sum(module.dims)
        module.dim = module.irreps.dim
        module._setup_indices()
        return module
    if isinstance(module, CgtpACE) and hasattr(module, "eqx_ace"):
        del module.eqx_ace
    if isinstance(module, BilinearMoEACE):
        module.use_eqx = False
        for name in list(module._modules):
            if name.startswith("eqx_"):
                delattr(module, name)
    for name, child in list(module.named_children()):
        setattr(module, name, cartesian_modules(child))
    if isinstance(module, CgtpACE) and module.aces:
        module.eqx_ace = TACE(
            [ace.tp for ace in module.aces],
            [coef.linear for coef in module.coefs],
            backend="cuda" if module.use_eqx else "torch",
        )
    if hasattr(module, "rejector") and isinstance(
        module.rejector, CartesianScatterTensorProduct
    ):
        if module.rejector.compact_output:
            module.linear_down.linear = Linear(
                module.linear_down.irreps_in,
                module.linear_down.irreps_out,
                module.rejector.eqx_tp.harmonic_output_index,
                internal_weights=False,
                shared_weights=True,
            )
        else:
            module.linear_down.linear.projection = co3.Projector(
                module.linear_down.irreps_out
            )
    for name, value in list(vars(module).items()):
        if isinstance(value, o3.Irreps):
            setattr(module, name, co3.Irreps(value))
    if isinstance(module, SphericalTensorNodeEmbedding):
        module.angular_sections = [
            slice((3**ir.l - 1) // 2, (3 ** (ir.l + 1) - 1) // 2)
            for _, ir in module.irreps_out
        ]
        module.use_wigner = False
        if hasattr(module, "frame"):
            del module.frame
    if isinstance(module, BilinearMoEACE) and module.use_bilinear_gate:
        module._double_hidden_slices = list(module.irreps_double_hidden.slices())
        module._hidden_ir_dims = [ir.dim for _, ir in module.irreps_hidden]
    if isinstance(module, TACELES):
        module.rank_two_dim = 9
    return module
