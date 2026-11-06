################################################################################
# Authors: Ilyes Batatia
# Symmetric contraction algorithm presented in the MACE paper
################################################################################

import math
from typing import Union

import torch
from e3nn import o3

from equivariantx import o2
from equivariantx.ace import BilinearContraction
from equivariantx.conv.attention import graph_softmax
from equivariantx.models.tace.tece_oam_rra import LocalSplit
from equivariantx.o2.clebsch_gordan import clebsch_gordan_product
from equivariantx.ops.channel_product import local_product
from equivariantx.ops.rotation import rotate
from tace.utils.env import acceleration_enabled, select_acceleration

from ..lammps import Graph
from ..layout import LayoutTransform
from ..linear import (
    IndexedFeatures,
    e3nnElementLinear,
    e3nnLinear,
    e3nnMoEElementLinear,
    has_lora,
    radial_features,
)
from ..mlp import ACTIVATION, get_scaled_activation
from .base import Product
from .dropout import GraphDropPath
from .fused import UuuTensorProduct
from .inter import O3CgtpInteraction
from .paths import SymmetricProductPaths


class Convolution(torch.nn.Module):
    """Channel-mixing local convolution with optional ECE and rotary attention.

    Parameters
    ----------
    mmax, lmax : int
        Maximum local order and angular degree.
    num_channel, num_head : int
        Feature channels and attention heads. ``num_head`` divides ``num_channel``.
    edge_ace_hidden : int
        Channels in the edge product expansion.
    num_radial_basis : int
        Input width of the rotary phase and bias projection.
    gate_m0 : bool
        Gate order-zero features instead of activating them directly.
    use_asymmetric_contraction, use_radial_rotary_attention : bool
        Enable the edge product expansion and rotary attention, respectively.
    reshape_in, reshape_out : torch.nn.Module
        Node layout transforms. The output transform provides ``inverse``.
    scalar_act, tensor_act : torch.nn.Module
        Already-scaled scalar and gate activations, used without renormalization.

    Notes
    -----
    Linear maps, gating, radial multiplication and edge products use native
    ``equivariantx.o2`` operators. Checkpoint loading packs the earlier order-wise
    matrices into their native weight layout. Rotary phases preserve the
    original SO(2) model; this is not an O(3)-equivariant replacement for RRA.
    """

    def __init__(
        self,
        mmax,
        lmax,
        num_channel,
        num_head,
        edge_ace_hidden,
        num_radial_basis,
        gate_m0,
        use_asymmetric_contraction,
        use_radial_rotary_attention,
        reshape_in,
        reshape_out,
        scalar_act,
        tensor_act,
    ):
        super().__init__()
        self.mmax, self.lmax = mmax, lmax
        self.num_channel, self.num_head = num_channel, num_head
        self.edge_ace_hidden = edge_ace_hidden
        if num_channel % num_head:
            raise ValueError("num_head must divide num_channel.")
        self.num_channel_per_head = num_channel // num_head
        self.use_asymmetric_contraction = use_asymmetric_contraction
        self.use_radial_rotary_attention = use_radial_rotary_attention
        self.gate_m0 = gate_m0
        self.reshape_in, self.reshape_out = reshape_in, reshape_out
        self.scalar_act, self.tensor_act = scalar_act, tensor_act
        self.attention_eps = 1e-16
        n = lmax + 1
        hidden = edge_ace_hidden if use_asymmetric_contraction else num_channel
        orders = [o2.Irrep(m, 1 if m == 0 else 0) for m in range(mmax + 1)]
        node_irreps = o2.Irreps([(ir, (n - ir.m) * num_channel) for ir in orders])
        self.irreps_out = node_irreps
        self.irreps_in = o2.Irreps([(ir, 2 * mul) for ir, mul in node_irreps])
        self.irreps_hidden = o2.Irreps(
            [
                (ir, (n if use_asymmetric_contraction else n - ir.m) * hidden)
                for ir in orders
            ]
        )
        self.num_components = sum(n - m for m in range(mmax + 1))
        self.weight_numel = self.num_components * 2 * num_channel
        self.radial_product = o2.TensorProduct(
            self.irreps_in,
            [(o2.Irrep(0, 1), mul) for _, mul in self.irreps_in],
            self.irreps_in,
            [(i, i, i, "uuu", False) for i in range(len(orders))],
        )
        expand = [
            j
            for m in range(mmax + 1)
            for _ in range(1 if m == 0 else 2)
            for j in range(
                sum(n - k for k in range(m)), sum(n - k for k in range(m + 1))
            )
        ]
        self._eqx_weight_metadata = repr(
            (
                2 * num_channel,
                (len(expand), self.num_components, len(expand)),
                tuple(((i, j, i), 1.0) for i, j in enumerate(expand)),
            )
        )
        self.num_gates = (
            sum(mul for ir, mul in self.irreps_hidden if gate_m0 or ir.m) // hidden
        )
        scalar_irreps = [] if gate_m0 else [self.irreps_hidden[0]]
        gated_irreps = self.irreps_hidden if gate_m0 else self.irreps_hidden[1:]
        # The caller supplies the checkpoint's activation scaling. Gate performs
        # only the native channel-wise product, avoiding a second normalization.
        self.nonlinearity = o2.Gate(
            scalar_irreps,
            [None] * len(scalar_irreps),
            [(o2.Irrep(0, 1), self.num_gates * hidden)] if self.num_gates else [],
            [None] if self.num_gates else [],
            gated_irreps,
        )
        widths = [
            mul * (2 if use_asymmetric_contraction else 1)
            for _, mul in self.irreps_hidden
        ]
        widths[0] += self.num_gates * hidden
        self.linear_up = o2.Linear(
            self.irreps_in, list(zip(orders, widths)), biases=True
        )
        self.linear_down = o2.Linear(self.irreps_hidden, node_irreps, biases=True)

        first, second = [], []
        offset = 0
        for m, width in enumerate(widths):
            size = self.irreps_hidden[m].mul // hidden
            for real in range(orders[m].dim):
                start = (
                    offset
                    + real * (width // hidden)
                    + (self.num_gates if m == 0 else 0)
                )
                first.extend(range(start, start + size))
                if use_asymmetric_contraction:
                    second.extend(range(start + size, start + 2 * size))
            offset += orders[m].dim * width // hidden
        self.register_buffer("feature_indices", torch.tensor(first), persistent=False)
        self.register_buffer(
            "product_indices", torch.tensor(second, dtype=torch.long), persistent=False
        )
        self.projection_slices = (
            (0, self.num_gates * hidden),
            (self.num_gates * hidden, n * hidden),
        )
        rows = []
        for role, (start, size) in enumerate(self.projection_slices, 1):
            for i in range(size // hidden):
                indices = [start // hidden + i, -1, -1]
                indices[role] = i
                rows.append((tuple(indices), 1.0))
        self._eqx_split_metadata = repr(
            (hidden, (offset, self.num_gates, n), tuple(rows))
        )

        if use_asymmetric_contraction:
            instructions = []
            for m, ir in enumerate(orders):
                paths = [
                    (i, j)
                    for i in range(mmax + 1)
                    for j in range(i + 1)
                    if ir in orders[i] * orders[j]
                ]
                # Convert unnormalized complex products to normalized O(2) products.
                scale = len(paths) ** -0.5
                instructions.extend(
                    (i, j, m, "uuu", True, scale**2 * (2 if i and j else 1))
                    for i, j in paths
                )
            self.ece = o2.TensorProduct(
                self.irreps_hidden,
                self.irreps_hidden,
                self.irreps_hidden,
                instructions,
                path_normalization="none",
                internal_weights=False,
                shared_weights=False,
            )
            self.linear_coefs = o2.Linear(
                [self.irreps_in[0]],
                [(o2.Irrep(0, 1), self.ece.weight_numel)],
                biases=True,
            )
            rows = []
            for i, index in enumerate(first):
                rows.append(((index, -1, -1, -1, -1, i), 1.0))
                if gate_m0 or i >= n:
                    m, component = divmod(i - n, 2 * n) if i >= n else (-1, i)
                    gate = (
                        (m + (1 if gate_m0 else 0)) * n + component % n if i >= n else i
                    )
                    rows.append(((index, -1, gate, -1, -1, i), 1.0))
                else:
                    rows.append(((-1, -1, -1, -1, i, i), 1.0))
            slices = self.irreps_hidden.slices()
            for path, ins in enumerate(self.ece.instructions):
                ir1, ir2, ir_out = (
                    orders[i] for i in (ins.i_in1, ins.i_in2, ins.i_out)
                )
                cg = clebsch_gordan_product(
                    torch.eye(ir1.dim, dtype=torch.float64),
                    ir1,
                    torch.eye(ir2.dim, dtype=torch.float64),
                    ir2,
                    ir_out,
                )
                for out, a, b in cg.nonzero().tolist():
                    coefficient = cg[out, a, b].item() * ins.path_weight
                    for degree in range(n):
                        rows.append(
                            (
                                (
                                    first[
                                        slices[ins.i_in1].start // hidden
                                        + a * n
                                        + degree
                                    ],
                                    second[
                                        slices[ins.i_in2].start // hidden
                                        + b * n
                                        + degree
                                    ],
                                    -1,
                                    path * n + degree,
                                    -1,
                                    slices[ins.i_out].start // hidden
                                    + out * n
                                    + degree,
                                ),
                                coefficient,
                            )
                        )
            self._eqx_update_metadata = repr(
                (
                    hidden,
                    (
                        offset,
                        offset,
                        self.num_gates,
                        self.ece.weight_numel // hidden,
                        n,
                        len(first),
                    ),
                    tuple(rows),
                )
            )

        if use_radial_rotary_attention:
            self.query_proj = o2.Linear(node_irreps, node_irreps, biases=True)
            self.key_proj = o2.Linear(node_irreps, node_irreps, biases=True)
            self.radial_proj = o2.Linear(
                [(o2.Irrep(0, 1), num_radial_basis)],
                [(o2.Irrep(0, 1), 2 * num_head)],
                biases=True,
            )
            torch.nn.init.zeros_(self.radial_proj.weight)
            self.attention_scale = (
                self.num_channel_per_head * (self.irreps_hidden.dim // hidden)
            ) ** -0.5
            self.temperature_min, self.temperature_max = 0.25, 4.0
            self.temperature_logit = torch.nn.Parameter(
                torch.full((num_head,), math.log(0.75 / 3.0))
            )
        local_degrees = tuple(
            ell
            for m in range(mmax + 1)
            for _ in range(1 if m == 0 else 2)
            for ell in range(m, n)
        )
        self._eqx_rotation_in = self._eqx_rotation_out = None
        if hasattr(reshape_in, "irreps") and hasattr(reshape_out, "irreps"):
            self._eqx_rotation_in = repr(
                (
                    tuple(ir.l for _, ir in reshape_in.irreps for _ in range(ir.dim)),
                    local_degrees,
                )
            )
            self._eqx_rotation_out = repr(
                (
                    local_degrees,
                    tuple(ir.l for _, ir in reshape_out.irreps for _ in range(ir.dim)),
                )
            )
        from equivariantx.models.tace.tece_oam_rra.program import metadata

        self._eqx_metadata = metadata(self)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in (
            "linear_up",
            "linear_down",
            "linear_coefs",
            "query_proj",
            "key_proj",
        ):
            if prefix + name + ".m0_rlinear.weight" not in state_dict:
                continue
            module = getattr(self, name)
            weights, biases = [], []
            names = (
                (name, "linear_glu")
                if name == "linear_up" and self.use_asymmetric_contraction
                else (name,)
            )
            for ir, _ in module.irreps_in:
                suffix = "m0_rlinear" if ir.m == 0 else f"ms_clinear.{ir.m - 1}.fc"
                matrices = [
                    state_dict.pop(prefix + layer + "." + suffix + ".weight")
                    for layer in names
                ]
                weights.append(torch.cat(matrices, dim=0).T.contiguous().flatten())
                if ir.m == 0:
                    biases.extend(
                        state_dict.pop(prefix + layer + "." + suffix + ".bias")
                        for layer in names
                    )
            state_dict[prefix + name + ".weight"] = torch.cat(weights)
            state_dict[prefix + name + ".bias"] = torch.cat(biases)
        radial_key = prefix + "radial_proj.weight"
        if radial_key in state_dict and state_dict[radial_key].ndim == 2:
            state_dict[radial_key] = state_dict[radial_key].T.contiguous().flatten()
        for name in ("scalar_act", "tensor_act"):
            old = prefix + "nonlinearity." + name + "."
            for key in list(state_dict):
                if key.startswith(old):
                    state_dict[prefix + name + "." + key[len(old) :]] = state_dict.pop(
                        key
                    )
        # Native parameter-free operators have empty persistent weight buffers.
        state_keys = self.state_dict().keys()
        for name, value in self.named_buffers():
            if value.numel() == 0 and name in state_keys:
                state_dict.setdefault(prefix + name, value)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def rotary_score(self, query, key, radial_basis, fused=False):
        batch, heads, channels = (
            query.shape[0],
            self.num_head,
            self.num_channel_per_head,
        )
        radial_bias, radial_phase = self.radial_proj(radial_basis).split(heads, dim=-1)
        radial_phase = math.pi * radial_phase.tanh()
        if fused:
            from equivariantx.ops.rotary import rotary_product

            orders = torch.arange(self.mmax + 1, device=query.device, dtype=query.dtype)
            angle = radial_phase[:, None, :] * orders[None, :, None]
            score = rotary_product(
                query,
                key,
                torch.stack((angle.cos(), angle.sin()), dim=-1),
                self.lmax,
                self.mmax,
                heads,
            )
        else:
            n = self.lmax + 1
            score = (
                query[:, :n].reshape(batch, n, heads, channels)
                * key[:, :n].reshape(batch, n, heads, channels)
            ).sum((1, 3))
            offset = n
            for m in range(1, self.mmax + 1):
                n = self.lmax + 1 - m
                q = query[:, offset : offset + 2 * n].reshape(
                    batch, 2, n, heads, channels
                )
                k = key[:, offset : offset + 2 * n].reshape(
                    batch, 2, n, heads, channels
                )
                angle = (m * radial_phase)[:, None, :, None]
                real = angle.cos() * k[:, 0] - angle.sin() * k[:, 1]
                imag = angle.sin() * k[:, 0] + angle.cos() * k[:, 1]
                score = score + (q[:, 0] * real + q[:, 1] * imag).sum((1, 3))
                offset += 2 * n
        temperature = (
            self.temperature_min
            + (self.temperature_max - self.temperature_min)
            * self.temperature_logit.sigmoid()
        )
        return score * self.attention_scale * temperature + radial_bias

    def forward(
        self,
        x,
        conv_weights,
        edge_index,
        cutoff,
        wigner,
        wigner_inv,
        radial_basis,
        stage="",
        fused=False,
    ):
        """Rotate, update and aggregate node features along the supplied edges.

        ``conv_weights`` has shape ``(edges, weight_numel)``. Setting
        ``stage='score_value'`` returns tile scores and unweighted global values
        for streaming attention; the default returns aggregated node features.
        ``fused`` selects equivalent CUDA implementations where available.
        """
        nodes, edges = x.shape[0], edge_index.shape[1]
        x = self.reshape_in(x)
        message = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
        fused = fused and x.is_cuda and x.dtype in (torch.float32, torch.float64)
        sparse_rotation = (
            fused
            and self._eqx_rotation_in is not None
            and self._eqx_rotation_out is not None
        )
        message = self.align(message, wigner, sparse_rotation)
        if self.use_radial_rotary_attention:
            key = self.key_proj(message[..., : self.num_channel].flatten(1)).view(
                edges, message.shape[1], self.num_channel
            )
            query = self.query_proj(
                message[..., self.num_channel :].flatten(1)
            ).view_as(key)
            score = self.rotary_score(query, key, radial_basis, fused)
        message = (
            local_product(self._eqx_weight_metadata, message, conv_weights)
            if fused
            else self.radial_product(message.flatten(1), conv_weights)
        )
        packed = self.linear_up(message)
        hidden = (
            self.edge_ace_hidden
            if self.use_asymmetric_contraction
            else self.num_channel
        )
        if self.use_asymmetric_contraction:
            coefs = self.scalar_act(
                self.linear_coefs(message[:, : self.irreps_in[0].mul])
            )
        gate, scalar = (
            LocalSplit.apply(packed, self.projection_slices, self._eqx_split_metadata)
            if fused
            else tuple(
                packed.narrow(1, start, size) for start, size in self.projection_slices
            )
        )
        if fused and self.use_asymmetric_contraction:
            message = local_product(
                self._eqx_update_metadata,
                packed,
                packed,
                self.tensor_act(gate),
                coefs,
                self.scalar_act(scalar),
            )
        else:
            packed = packed.view(edges, self.linear_up.irreps_out.dim // hidden, hidden)
            features = packed.index_select(1, self.feature_indices).flatten(1)
            gate_input = (
                torch.cat((self.tensor_act(gate), features), dim=-1)
                if self.gate_m0
                else torch.cat(
                    (
                        self.scalar_act(scalar),
                        self.tensor_act(gate),
                        features[:, scalar.shape[1] :],
                    ),
                    dim=-1,
                )
            )
            message = self.nonlinearity(gate_input)
            if self.use_asymmetric_contraction:
                other = packed.index_select(1, self.product_indices).flatten(1)
                message = features + message + self.ece(features, other, coefs)
        message = self.linear_down(message).view(
            edges, self.irreps_out.dim // self.num_channel, self.num_channel
        )
        if stage == "score_value":
            value = self.unalign(message, wigner_inv, sparse_rotation)
            return score, value
        if self.use_radial_rotary_attention:
            alpha = graph_softmax(
                score, edge_index[1], nodes, cutoff, self.attention_eps, fused=fused
            )
            if cutoff is not None:
                alpha = alpha * cutoff
            message = (
                message.view(
                    edges, message.shape[1], self.num_head, self.num_channel_per_head
                )
                * alpha[:, None, :, None]
            ).reshape_as(message)
        elif cutoff is not None:
            message = message * cutoff.unsqueeze(-1)
        message = self.unalign(message, wigner_inv, False)
        output = message.new_zeros(
            (nodes, message.shape[1], self.num_channel)
        ).index_add(0, edge_index[1], message)
        return self.reshape_out.inverse(output)

    def align(self, features, wigner, fused):
        """Express gathered features in the local representation."""
        return (
            rotate(features, wigner, self._eqx_rotation_in)
            if fused
            else torch.bmm(wigner, features)
        )

    def unalign(self, features, wigner_inv, fused):
        """Return edge messages to the node representation."""
        return (
            rotate(features, wigner_inv, self._eqx_rotation_out)
            if fused
            else torch.bmm(wigner_inv, features)
        )


class UvSO2Interaction(O3CgtpInteraction):
    """
    An interaction module based on native O(2) linear maps,
    Edge Cluster Expansion and Radial Rotary Complex Attention.

    This interaction block add nonlinearity to the message.
    """

    @property
    def use_eqx(self) -> bool:
        return self.rejector._eqx_metadata is not None and bool(
            acceleration_enabled("eqx", kernel="conv")
        )

    def _prepare_setup(self) -> None:
        super()._prepare_setup()
        if self.parity is True or any(
            ir.p != (-1) ** ir.l
            for irreps in (self.irreps_in, self.irreps_out)
            for _, ir in irreps
        ):
            raise ValueError("UvSO2Interaction does not support O(3).")
        if self.irreps_in.lmax <= 0:
            raise ValueError(
                "UvSO2Interaction requires irreps_in.lmax > 0. Use it after "
                "the first layer or provide a node embedding with l > 0."
            )
        if self.edge_nonlinear is None:
            raise ValueError("UvSO2Interaction requires edge_nonlinear to be set.")
        self.scatter_norm_type = None

    def _build_rejector(self) -> torch.nn.Module:
        if self.scalar_act is None:
            scalar_act = "silu"
        elif isinstance(self.scalar_act, str):
            scalar_act = self.scalar_act
        elif isinstance(self.scalar_act, list) and len(self.scalar_act) == 2:
            if not all(isinstance(name, str) for name in self.scalar_act):
                raise TypeError("scalar_act entries must be strings.")
            scalar_act = self.scalar_act[0]
        else:
            raise TypeError(
                "scalar_act must be None, a string, or a list of two strings "
                "for 0e and 0o."
            )
        tensor_act = self.tensor_act or (
            "silu" if self.use_asymmetric_contraction else "sigmoid"
        )
        if not isinstance(tensor_act, str):
            raise TypeError("tensor_act must be None or a string for tensor gates.")
        return Convolution(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            num_radial_basis=self.num_radial_basis,
            num_head=self.num_head,
            edge_ace_hidden=self.edge_ace_hidden,
            gate_m0=False,
            use_asymmetric_contraction=self.use_asymmetric_contraction,
            use_radial_rotary_attention=self.use_radial_rotary_attention,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(
                o3.Irreps([(self.num_channel, ir) for _, ir in self.irreps_out])
            ),
            scalar_act=get_scaled_activation(scalar_act),
            tensor_act=get_scaled_activation(tensor_act),
        )

    def _linear_down_irreps_in(self) -> o3.Irreps:
        return o3.Irreps([(self.num_channel, ir) for _, ir in self.irreps_out])

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_types: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_node_info: Union[
            tuple[torch.Tensor, torch.Tensor], None
        ] = None,
        magnetic_edge_attrs: Union[torch.Tensor, None] = None,
        graph: Union[Graph, None] = None,
    ) -> torch.Tensor:
        if (
            node_feats.is_cuda
            and self.use_radial_rotary_attention
            and self.use_eqx
            and acceleration_enabled("eqx", kernel="conv")
        ):
            from equivariantx.models.tace.tece_oam_rra.interaction import stream

            radial = radial_features(edge_feats)
            last = self.edge_info.mlp[-1]
            return stream(
                self.rejector,
                node_feats,
                radial,
                last.get_weight(),
                last.bias,
                edge_index,
                edge_cutoff,
                edge_wigner,
                edge_wigner_inv,
                edge_radial_basis,
                radial_network=self.edge_info.mlp[:-1],
            )
        return self.rejector(
            node_feats,
            self.edge_info(edge_feats),
            edge_index,
            edge_cutoff,
            edge_wigner,
            edge_wigner_inv,
            edge_radial_basis,
            fused=acceleration_enabled("eqx", kernel="conv"),
        )


class BilinearMoEACE(Product):
    """ACE with optional bilinear gates, coefficient experts and a shared expert."""

    def _setup(self):

        self.scale = 1.0 / math.sqrt(2.0)
        self.use_eqx = bool(
            self.use_bilinear_gate
            and select_acceleration("eqx", "eqt", kernel="product") == "eqx"
        )

        for_coefs = {
            "irreps_out": self.irreps_coefs_out,
            "bias": self.use_bias,
        }
        if self.agnostic:
            coefs_cls = e3nnLinear
        else:
            for_coefs["num_elements"] = self.num_elements
            coefs_cls = e3nnElementLinear
            if self.num_expert > 1:
                coefs_cls = e3nnMoEElementLinear
                for_coefs["num_experts"] = self.num_expert

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        if self.use_shared_expert and self.num_expert > 1:
            self.shared_coefs = torch.nn.ModuleList()
        self.coefs.append(
            coefs_cls(
                o3.Irreps(
                    [(self.num_hidden_channel, ir) for _, ir in self.irreps_hidden]
                ).simplify(),
                **for_coefs,
            )
        )
        if hasattr(self, "shared_coefs"):
            self.shared_coefs.append(
                e3nnLinear(
                    o3.Irreps(
                        [(self.num_hidden_channel, ir) for _, ir in self.irreps_hidden]
                    ).simplify(),
                    self.irreps_coefs_out,
                    bias=self.use_bias,
                )
            )

        self.irreps_base = self.irreps_hidden
        if self.use_bilinear_gate:
            assert self.correlation == 2
            self.nonlinearity = ACTIVATION[self.scalar_act]()
            self.irreps_base = self.irreps_hidden + o3.Irreps(
                [(self.num_hidden_channel, o3.Irrep("0e"))]
            )

        product_in1 = self.irreps_hidden
        symmetric_paths = (
            SymmetricProductPaths(product_in1)
            if self.correlation > 2 and not self.use_bilinear_gate
            else None
        )

        for nu in range(2, self.correlation + 1):
            this_ace = UuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_base,
                irreps_out=self.irreps_tp_out_list[nu - 2],
                l1l2=self.l1l2,
                trainable=self.use_bilinear_gate,
                identical_inputs=nu == 2 and not self.use_bilinear_gate,
                warning=self.correlation > 2 and self.layer == 0,
                use_fused=self.correlation > 2 and not self.use_time_reversal,
                symmetric_paths=symmetric_paths,
                use_eqt=False if self.use_eqx else None,
            )
            self.aces.append(this_ace)
            self.coefs.append(
                coefs_cls(
                    o3.Irreps(
                        [(self.num_hidden_channel, ir) for _, ir in this_ace.irreps_out]
                    ).simplify(),
                    **for_coefs,
                )
            )
            if hasattr(self, "shared_coefs"):
                self.shared_coefs.append(
                    e3nnLinear(
                        o3.Irreps(
                            [
                                (self.num_hidden_channel, ir)
                                for _, ir in this_ace.irreps_out
                            ]
                        ).simplify(),
                        self.irreps_coefs_out,
                        bias=self.use_bias,
                    )
                )
            product_in1 = this_ace.irreps_out

        if self.use_eqx:
            coef = self.coefs[1]
            self.eqx_ace = BilinearContraction(
                self.aces[0].tp, coef.linear, getattr(coef, "num_experts", 1),
                shared_linear=self.shared_coefs[1].linear if hasattr(self, "shared_coefs") else None,
            )

        if self.use_bilinear_gate:
            self._ace_gate_slices = []
            gate_offset = 0
            for ace in self.aces:
                self._ace_gate_slices.append(
                    slice(gate_offset, gate_offset + ace.weight_numel)
                )
                gate_offset += ace.weight_numel
            self.num_ace_gate_weights = gate_offset

            self.irreps_double_hidden = o3.Irreps(
                [(2 * mul, ir) for mul, ir in self.irreps_hidden]
            )
            self.irreps_linear_up = (
                o3.Irreps(f"{self.num_ace_gate_weights}x0e") + self.irreps_double_hidden
            )
            self.linear_up = e3nnLinear(
                self.irreps_in,
                self.irreps_linear_up,
                bias=self.use_bias,
            )
            self._double_hidden_slices = list(self.irreps_double_hidden.slices())
            self._hidden_muls = [mul for mul, _ in self.irreps_hidden]
            self._hidden_ir_dims = [ir.dim for _, ir in self.irreps_hidden]
        else:
            self.linear_up = (
                e3nnLinear(
                    self.irreps_in,
                    self.irreps_hidden,
                    bias=self.use_bias,
                )
                if self.num_channel != self.num_hidden_channel
                else torch.nn.Identity()
            )

        self.linear = e3nnLinear(
            o3.Irreps(
                [(self.num_hidden_channel, ir) for _, ir in self.irreps_coefs_out]
            ),
            self.irreps_out,
            bias=self.use_bias,
        )

        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

    def _merge_shared_expert(
        self,
        grouped: torch.Tensor,
        shared: torch.Tensor,
    ) -> torch.Tensor:
        return (grouped + shared) * self.scale

    def _bilinear_coefficients(
        self, kernel, coef, features, base, gates, node_type
    ):
        elements = getattr(coef, "num_elements", 1)
        experts = getattr(coef, "num_experts", 1)
        weight = (
            torch.cat([w.reshape(elements, experts, -1) for w in coef.weight], dim=-1)
            if coef.use_matrix_weight
            else coef.weight.reshape(elements, experts, -1)
        )
        types = (
            node_type
            if hasattr(coef, "num_elements")
            else torch.zeros(
                features.shape[0], device=features.device, dtype=torch.long
            )
        )
        shared = self.shared_coefs[1] if hasattr(self, "shared_coefs") else None
        shared_weight = None
        if shared is not None:
            shared_weight = (
                torch.cat([w.flatten() for w in shared.weight])
                if shared.use_matrix_weight else shared.weight
            )
        output = kernel(features, base, gates, weight, types, shared_weight)
        if shared is not None and shared.bias is not None:
            output = output + shared.bias.index_select(-1, shared._bias_index) * shared._bias_mask
        if coef.bias is None:
            return output
        if hasattr(coef, "num_experts"):
            outputs = [
                output[:, sl].reshape(features.shape[0], experts, mul, dim)
                for sl, mul, dim in zip(
                    coef._output_slices, coef._output_expert_muls, coef._output_dims
                )
            ]
            bias = coef.bias[types]
            for i, start, end in coef._bias_out_indices:
                outputs[i] = outputs[i] + bias[:, :, start:end].unsqueeze(-1)
            return torch.cat([value.flatten(1) for value in outputs], dim=-1)
        bias = (
            coef.bias.reshape(elements, -1)[node_type]
            if hasattr(coef, "num_elements")
            else coef.bias
        )
        return output + bias.index_select(-1, coef._bias_index) * coef._bias_mask

    def _linear_up_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        x = self.linear_up(x)
        if not self.use_bilinear_gate:
            return x, None, None

        ace_weights = self.nonlinearity(x[:, : self.num_ace_gate_weights])
        double_features = x[:, self.num_ace_gate_weights :]

        node_fields = []
        base_fields = []
        for tensor_slice, mul, ir_dim in zip(
            self._double_hidden_slices,
            self._hidden_muls,
            self._hidden_ir_dims,
        ):
            field = double_features[:, tensor_slice].reshape(
                x.shape[0],
                2 * mul,
                ir_dim,
            )
            node_fields.append(field[:, :mul].reshape(x.shape[0], -1))
            base_fields.append(field[:, mul:].reshape(x.shape[0], -1))

        node_feats = torch.cat(node_fields, dim=-1)
        base_feats = torch.cat(base_fields, dim=-1)
        base_feats = torch.cat(
            (
                base_feats,
                base_feats.new_ones(base_feats.shape[0], self.num_hidden_channel),
            ),
            dim=-1,
        )
        return node_feats, base_feats, ace_weights

    def forward(
        self,
        node_feats: torch.Tensor,
        node_type: torch.Tensor,
        sc: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        if self.agnostic:
            for_coefs = {}
        else:
            for_coefs = {"node_type": node_type}

        node_feats, base_feats, ace_weights = self._linear_up_features(node_feats)

        corr_feats = {
            1: node_feats,
        }

        outs = self.coefs[0](corr_feats[1], **for_coefs)
        shared_outs = (
            self.shared_coefs[0](corr_feats[1])
            if hasattr(self, "shared_coefs")
            else None
        )

        for nu in range(2, self.correlation + 1):
            if (
                hasattr(self, "eqx_ace")
                and node_feats.is_cuda
                and not has_lora(self.coefs[nu - 1])
                and not (
                    shared_outs is not None and has_lora(self.shared_coefs[nu - 1])
                )
            ):
                gates = ace_weights[:, self._ace_gate_slices[nu - 2]]
                outs = outs + self._bilinear_coefficients(
                    self.eqx_ace,
                    self.coefs[nu - 1],
                    node_feats,
                    base_feats,
                    gates,
                    node_type,
                )
                continue
            if self.use_bilinear_gate:
                corr_feats[nu] = self.aces[nu - 2](
                    corr_feats[nu - 1],
                    base_feats,
                    ace_weights[:, self._ace_gate_slices[nu - 2]],
                )
            else:
                corr_feats[nu] = self.aces[nu - 2](corr_feats[nu - 1], node_feats)
            outs = outs + self.coefs[nu - 1](corr_feats[nu], **for_coefs)
            if shared_outs is not None:
                shared_outs = shared_outs + self.shared_coefs[nu - 1](corr_feats[nu])

        if shared_outs is not None:
            outs = self._merge_shared_expert(outs, shared_outs)

        outs = self.linear(outs)

        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)

        if sc is not None:
            outs = outs + sc

        return outs
