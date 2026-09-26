"""TECE local updates composed from native O(2) operators."""

import math

import torch

from eqx import o2
from eqx.conv.attention import graph_softmax
from eqx.conv.models.tace.tece_oam_rra import LocalSplit
from eqx.kernels.channel_product import local_product
from eqx.kernels.rotation import rotate
from eqx.o2._clebsch_gordan import clebsch_gordan_product


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
    ``eqx.o2`` operators. Checkpoint loading packs the earlier order-wise
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
                # Older ECE used unnormalized complex products and a per-output
                # path factor stored in the construction dtype.
                scale = torch.tensor(len(paths) ** -0.5).item()
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
        from eqx.conv.models.tace.tece_oam_rra.program import metadata

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
            from eqx.kernels.rotary import rotary_product

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
        message = (
            rotate(message, wigner, self._eqx_rotation_in)
            if sparse_rotation
            else torch.bmm(wigner, message)
        )
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
            value = (
                rotate(message, wigner_inv, self._eqx_rotation_out)
                if sparse_rotation
                else torch.bmm(wigner_inv, message)
            )
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
        message = torch.bmm(wigner_inv, message)
        output = message.new_zeros(
            (nodes, message.shape[1], self.num_channel)
        ).index_add(0, edge_index[1], message)
        return self.reshape_out.inverse(output)
