"""Residual node updates from edge cluster expansions."""

import math

import torch
from e3nn import o3

from eqx import o2
from eqx.conv import EceO2TensorProductConv

from ...utils.env import acceleration_enabled
from ...utils.torch_scatter import scatter_sum
from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import MLP
from ..radial import RadialBasis
from .node import NODE_EMBEDDING
from .tace import e3nnTACE


class EceO2Interaction(torch.nn.Module):
    """Apply an element-weighted edge expansion and a residual update.

    Parameters
    ----------
    irreps : o3.Irreps
        Node representation, with a common channel multiplicity.
    lmax, mmax : int
        Radial path degree and retained local order.
    num_elements, num_channel : int
        Element count and expansion multiplicity.
    radial_basis : dict
        Radial MLP widths and bias setting.
    correlation : int
        Maximum polynomial degree of the edge expansion.
    algorithm : {"recursive", "dense"}
        Contraction order, sharing the same parameters and basis.
    asymmetric : bool
        Use independent factors instead of powers of one feature tensor.
    avg_num_neighbors : float
        Normalization of the neighbor sum.
    """

    def __init__(
        self,
        irreps,
        lmax,
        mmax,
        num_elements,
        num_channel,
        radial_basis,
        correlation,
        algorithm,
        asymmetric,
        avg_num_neighbors,
    ):
        super().__init__()
        self.irreps_in = self.irreps_out = o3.Irreps(irreps)
        self.num_channel = num_channel
        self.num_inputs = correlation if asymmetric else 1
        self.asymmetric = asymmetric
        self.normalization = math.sqrt(avg_num_neighbors)
        self.use_eqx = bool(acceleration_enabled("eqx", kernel="conv"))
        self.reshape_in = LayoutTransform(
            irreps, layout_in="flatten_mul_ir", layout_out="flatten_ir_mul"
        )
        self.reshape_out = LayoutTransform(
            irreps, layout_in="flatten_ir_mul", layout_out="flatten_mul_ir"
        )
        self.frame_in = o2.LocalFrame(irreps, mmax)
        self.frame_out = o2.LocalFrame(irreps, mmax, reverse=True)
        local = self.frame_in.irreps_out
        paired = o2.Irreps((ir, 2 * mul) for ir, mul in local)
        parity = any(ir.p != (-1) ** ir.l for _, ir in self.irreps_in)
        intermediate = o3.Irreps(
            [
                (num_channel, (ell, p))
                for ell in range(lmax + 1)
                for p in ((1, -1) if parity else ((-1) ** ell,))
            ]
        )
        radial_out = o2.LocalFrame.restrict(intermediate, mmax).filter(
            keep=lambda ir_mul: paired.count(ir_mul.ir) > 0
        )
        self.radial_linear = o2.UuLinear(
            paired, radial_out, num_channel, path_mode="expand"
        )
        hidden = o2.Irreps((ir, num_channel) for ir, _ in radial_out)
        self.linear_up = o2.Linear(
            self.radial_linear.irreps_out,
            o2.Irreps((ir, mul * self.num_inputs) for ir, mul in hidden),
        )
        contraction = (
            o2.AsymmetricContraction if asymmetric else o2.SymmetricContraction
        )
        self.contraction = contraction(
            hidden,
            o2.Irreps((ir, num_channel) for ir, _ in local),
            correlation,
            algorithm=algorithm,
        )
        self.linear_down = o2.Linear(self.contraction.irreps_out, local)
        self.source_weight = torch.nn.Parameter(
            torch.randn(num_elements, self.contraction.weight_numel)
        )
        self.target_weight = torch.nn.Parameter(torch.randn_like(self.source_weight))
        self.edge_info = MLP(
            [
                radial_basis["num_radial_basis"],
                *radial_basis["hidden"],
                self.radial_linear.weight_numel,
            ],
            bias=radial_basis["bias"],
        )
        self.eqx_tp = EceO2TensorProductConv(
            self.frame_in,
            self.radial_linear,
            self.linear_up,
            self.contraction,
            self.linear_down,
            self.frame_out,
        )

    def set_algorithm(self, algorithm):
        """Select dense or recursive contraction without changing parameters."""
        self.eqx_tp.set_algorithm(algorithm)

    def forward(
        self, node_feats, node_type, radial, edge_index, wigner, wigner_inv, cutoff
    ):
        features = self.reshape_in(node_feats)
        source, target = edge_index
        source_weight = self.source_weight[node_type]
        target_weight = self.target_weight[node_type]
        if self.use_eqx and features.is_cuda:
            for layer in self.edge_info.mlp[:-1]:
                radial = layer(radial)
            projection = self.edge_info.mlp[-1]
            weight = projection.get_weight()
            bias = projection.bias
            if bias is None:
                bias = weight.new_zeros(weight.shape[1])
            message = self.eqx_tp(
                features,
                radial,
                weight,
                bias,
                self.linear_up.weight,
                self.linear_down.weight,
                source_weight,
                target_weight,
                edge_index,
                wigner,
                wigner_inv,
                cutoff,
            )
        else:
            # Rotate the two endpoints together; concatenate matching channels.
            local = self.frame_in.to_local(features[edge_index].movedim(0, 1), wigner)
            paired = torch.cat(
                [
                    part.transpose(-3, -2).flatten(-3)
                    for part in (
                        x.reshape(x.shape[0], 2, ir.dim, mul)
                        for x, (ir, mul) in zip(
                            local.split(
                                [entry.dim for entry in self.frame_in.irreps_out], -1
                            ),
                            self.frame_in.irreps_out,
                        )
                    )
                ],
                dim=-1,
            )
            features = self.linear_up(
                self.radial_linear(paired, self.edge_info(radial))
            )
            if self.asymmetric:
                inputs = [
                    torch.cat(
                        [
                            x.reshape(
                                x.shape[0], ir.dim, self.num_inputs, self.num_channel
                            )[..., i, :].flatten(-2)
                            for x, (ir, _) in zip(
                                features.split(
                                    [entry.dim for entry in self.linear_up.irreps_out],
                                    -1,
                                ),
                                self.linear_up.irreps_out,
                            )
                        ],
                        -1,
                    )
                    for i in range(self.num_inputs)
                ]
            else:
                inputs = features
            features = self.contraction(
                inputs, source_weight[source] * target_weight[target]
            )
            message = (
                self.frame_out.to_global(self.linear_down(features), wigner_inv)
                * cutoff
            )
            message = scatter_sum(message, target, dim=0, dim_size=node_feats.shape[0])
        return node_feats + self.reshape_out(message) / self.normalization


class TECERepresentation(torch.nn.Module):
    """Build node descriptors with transient edge cluster expansions."""

    def __init__(
        self,
        num_layers,
        atomic_numbers,
        cutoff,
        avg_num_neighbors,
        mmax,
        Lmax,
        lmax,
        num_channel,
        node_embedding,
        radial_basis,
        atomic_basis,
        parity,
        invariant_property,
        equivariant_property,
        **kwargs,
    ):
        super().__init__()
        if invariant_property or equivariant_property:
            raise ValueError("TECE currently accepts positions and element types only.")
        if node_embedding["type"] not in ("linear", "tensor"):
            raise ValueError("TECE supports linear and tensor node embeddings.")
        self.use_dens = self.use_time_reversal = self.use_magnetic_interaction = False
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.long)
        )
        self.radial_basis = RadialBasis(
            cutoff=cutoff,
            num_basis=radial_basis["num_radial_basis"],
            radial_basis=radial_basis["radial_basis"],
            cutoff_fn=radial_basis["cutoff_fn"],
            polynomial_cutoff=radial_basis["polynomial_cutoff"],
            distance_transform=radial_basis["distance_transform"],
            trainable=radial_basis["trainable"],
            apply_cutoff=False,
            gaussian_width=radial_basis["gaussian_width"],
        )
        self.node_embedding = NODE_EMBEDDING[node_embedding["type"]](
            len(atomic_numbers),
            radial_basis["num_radial_basis"],
            0,
            num_channel,
            Lmax,
            lmax,
            avg_num_neighbors,
        )
        self.irreps_out = o3.Irreps(
            [
                (num_channel, (ell, p))
                for ell in range(Lmax + 1)
                for p in ((1, -1) if parity else ((-1) ** ell,))
            ]
        )
        self.embedding_linear = e3nnLinear(
            self.node_embedding.irreps_out, self.irreps_out
        )
        self.irreps_outs = [self.irreps_out] * num_layers
        self.angular_basis = (
            o3.SphericalHarmonics(
                list(range(lmax + 1)), normalize=True, normalization="component"
            )
            if node_embedding["type"] == "tensor"
            else None
        )
        self.wigner = o2.WignerD(min(mmax, Lmax), Lmax)
        self.interactions = torch.nn.ModuleList(
            [
                EceO2Interaction(
                    self.irreps_out,
                    lmax,
                    mmax,
                    len(atomic_numbers),
                    num_channel,
                    radial_basis,
                    atomic_basis.get("correlation", 3),
                    atomic_basis.get("algorithm", "recursive"),
                    atomic_basis["use_asymmetric_contraction"],
                    avg_num_neighbors,
                )
                for _ in range(num_layers)
            ]
        )

    def set_algorithm(self, algorithm):
        """Change the expansion algorithm in every interaction."""
        for interaction in self.interactions:
            interaction.set_algorithm(algorithm)

    def forward(self, data, graph):
        if graph.lmp:
            raise NotImplementedError(
                "TECE does not yet support distributed LAMMPS inference."
            )
        node_type = graph.node_type
        if node_type is None:
            node_type = data["node_attrs"].argmax(-1)
        radial, cutoff = self.radial_basis(
            graph.edge_length,
            data["node_attrs"],
            data["edge_index"],
            self.atomic_numbers,
            node_type=node_type,
        )
        wigner, wigner_inv = self.wigner(graph.edge_vector)
        angular = (
            self.angular_basis(graph.edge_vector)
            if self.angular_basis is not None
            else None
        )
        features = self.embedding_linear(
            self.node_embedding(
                data["node_attrs"],
                radial,
                data["edge_index"],
                angular,
                cutoff,
                wigner,
                wigner_inv,
            )
        )
        descriptors = []
        for interaction in self.interactions:
            features = interaction(
                features,
                node_type,
                radial,
                data["edge_index"],
                wigner,
                wigner_inv,
                cutoff,
            )
            descriptors.append(features)
        return {
            "descriptors": descriptors,
            "uie_feats": None,
            "noise_mask_tensor": None,
            "dens_batch_mask_tensor": None,
            "magnetic_radial_basis": None,
        }


class TECE(e3nnTACE):
    """TACE readouts with residual O(2) edge cluster expansions.

    Parameters follow ``e3nnTACE``. ``atomic_basis`` selects ``correlation``,
    ``algorithm`` and ``use_asymmetric_contraction``. There is no node product.
    """

    representation_cls = TECERepresentation

    def __init__(self, **kwargs):
        atomic_basis = dict(kwargs.get("atomic_basis", {}))
        atomic_basis.setdefault("use_asymmetric_contraction", False)
        kwargs["atomic_basis"] = atomic_basis
        if set(kwargs.get("target_property", ("energy",))) - {
            "energy",
            "forces",
            "stress",
            "virials",
        }:
            raise ValueError("TECE supports energy, forces, stress and virials.")
        if kwargs.get("long_range", {}).get("les", {}).get("enable", False):
            raise ValueError("TECE does not support long-range interactions.")
        super().__init__(**kwargs)
