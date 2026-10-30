"""Cartesian operators with the parameter layout of spherical TACE."""

import torch
from e3nn import o3
from e3nn.nn import Gate

from eqx import co3
from eqx.ace.cartesian import CartesianBilinearACE, CartesianTACE
from eqx.co3 import ElementLinear, MoEElementLinear
from eqx.conv.co3 import CartesianTensorProductConv, Linear
from tace.utils.env import acceleration_enabled
from tace.utils.torch_scatter import scatter_sum

from .._e3nn.basis_change import _SymmetricRankTwoBasisChange
from .._e3nn.fused import (
    O3TensorProductConv,
    UuO2TensorProductConv,
    UuuTensorProduct,
    UvO2TensorProduct,
    UvuTensorProduct,
    Wigner6jTensorProductConv,
)
from .._e3nn.layer_norm import EquivariantMergeLayerNorm
from .._e3nn.les import TACELES
from .._e3nn.node import SphericalTensorNodeEmbedding
from .._e3nn.o2 import MagneticUvO2TensorProductConv, UvO2TensorProductConv
from .._e3nn.prod import BilinearMoEACE, CgtpACE
from .._e3nn.readout import DirectHessianReadout
from .._e3nn.residual import ProjectDown, ProjectUp
from .._e3nn.tece_oam_rra import Convolution
from ..angular import SphericalHarmonics
from ..layout import LayoutTransform
from ..linear import e3nnElementLinear, e3nnLinear, e3nnMoEElementLinear
from .local import (
    MagneticUvConvolution,
    RadialRotaryConvolution,
    UuConvolution,
    UvConvolution,
    UvTensorProduct,
)


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


class CartesianTensorProductConvLayer(torch.nn.Module):
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
        self.compact_output = True
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


class CartesianWigner6jConvolution(torch.nn.Module):
    """Recouple Cartesian node tensors before the edge convolution."""

    def __init__(self, reference):
        super().__init__()
        product = reference.eqx_tp
        self.tp13 = cartesian_tensor_product(product.tp13)
        self.tp132 = cartesian_tensor_product(product.tp132)
        self.conv13 = CartesianTensorProductConv(self.tp13)
        self.eqx_tp = CartesianTensorProductConv(self.tp132)
        for name in ("irreps_in1", "irreps_in2", "irreps_in3", "irreps_out"):
            setattr(self, name, co3.Irreps(getattr(reference, name)))
        self.weight_numel = reference.weight_numel
        self.magnetic_weight_numel = reference.magnetic_weight_numel
        self.coefficient_values = product.coefficient_values
        for name in ("weight_index1", "weight_index2", "coefficients"):
            self.register_buffer(name, getattr(product, name).clone(), persistent=False)

    def forward(self, x1, x2, x3, weight1, weight2, edge_index):
        weight1 = torch.cat((weight1, torch.ones_like(weight1[..., :1])), -1)
        weight2 = torch.cat((weight2, torch.ones_like(weight2[..., :1])), -1)
        weight = (
            weight1.index_select(-1, self.weight_index1)
            * weight2.index_select(-1, self.weight_index2)
            * self.coefficients
        )
        if x1.is_cuda and acceleration_enabled("eqx", kernel="conv"):
            features = self.conv13(
                x1, x3, x1.new_empty((1, 0)), x1.new_empty((0, 0)), None
            )
            return self.eqx_tp(
                features,
                x2,
                weight,
                weight.new_empty((0, self.tp132.weight_numel)),
                edge_index,
            )
        features = self.tp13(x1, x3)
        message = self.tp132(features[edge_index[0]], x2, weight)
        return scatter_sum(message, edge_index[1], dim=0, dim_size=x1.shape[0])

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.coefficients = self.coefficients.new_tensor(self.coefficient_values)
        return self


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
    if isinstance(module, UuO2TensorProductConv):
        return UuConvolution(module)
    if isinstance(module, UvO2TensorProductConv):
        return UvConvolution(module)
    if isinstance(module, MagneticUvO2TensorProductConv):
        return MagneticUvConvolution(module)
    if isinstance(module, UvO2TensorProduct):
        return UvTensorProduct(module)
    if isinstance(module, Wigner6jTensorProductConv):
        return CartesianWigner6jConvolution(module)
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
        if module.scalar_index is not None:
            old_offset = offset = 0
            for l in module.ls:
                if old_offset == module.scalar_index:
                    module.scalar_index = offset
                    break
                old_offset += 2 * l + 1
                offset += 3**l
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
            backend="cuda" if acceleration_enabled("eqx", kernel="linear") else "torch",
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
        gate.metadata = gate.build_program()
        return gate
    if isinstance(module, O3TensorProductConv):
        return CartesianTensorProductConvLayer(module.tp)
    if isinstance(module, (UuuTensorProduct, UvuTensorProduct)):
        module.tp = cartesian_tensor_product(module.tp)
        module.irreps_in1 = module.tp.irreps_in1
        module.irreps_in2 = module.tp.irreps_in2
        module.irreps_out = module.tp.irreps_out
        if hasattr(module, "fused_tp"):
            del module.fused_tp
        if hasattr(module, "eqx_tp"):
            del module.eqx_tp
        return module
    if isinstance(module, (o3.SphericalHarmonics, SphericalHarmonics)):
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
        for name in list(module._modules):
            if name.startswith("eqx_"):
                delattr(module, name)
    for name, child in list(module.named_children()):
        setattr(module, name, cartesian_modules(child))
    if isinstance(module, Convolution):
        module.__class__ = RadialRotaryConvolution
        module.setup_cartesian()
    if isinstance(module, CgtpACE) and module.aces:
        module.eqx_ace = CartesianTACE(
            [ace.tp for ace in module.aces],
            [coef.linear for coef in module.coefs],
            backend="cuda" if module.use_eqx else "torch",
        )
    if isinstance(module, BilinearMoEACE) and module.use_eqx:
        coef = module.coefs[1]
        module.eqx_ace = CartesianBilinearACE(
            module.aces[0].tp,
            coef.linear,
            getattr(coef, "num_experts", 1),
            shared_linear=(
                module.shared_coefs[1].linear
                if hasattr(module, "shared_coefs")
                else None
            ),
        )
    if hasattr(module, "rejector") and isinstance(
        module.rejector, CartesianTensorProductConvLayer
    ):
        module.linear_down.linear = Linear(
            module.linear_down.irreps_in,
            module.linear_down.irreps_out,
            module.rejector.eqx_tp.harmonic_output_index,
            internal_weights=False,
            shared_weights=True,
            output_basis="cartesian",
            backend="cuda" if acceleration_enabled("eqx", kernel="conv") else "torch",
        )
    for name, value in list(vars(module).items()):
        if isinstance(value, o3.Irreps):
            setattr(module, name, co3.Irreps(value))
    if isinstance(module, (ProjectDown, ProjectUp)):
        irreps_in, irreps_out = module.irreps_in, module.irreps_out
        if isinstance(module, ProjectDown):
            irreps_in, irreps_out = irreps_out, irreps_in
        module.idx = ProjectUp._get_indices(irreps_in, irreps_out)
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
    if isinstance(module, DirectHessianReadout):
        module.component_dims = (1, 3, 9)
        module.basis = torch.cat(
            [
                module.basis[..., l**2 : (l + 1) ** 2] @ co3.path_matrix(l).T
                for l in range(3)
            ],
            -1,
        )
    return module
