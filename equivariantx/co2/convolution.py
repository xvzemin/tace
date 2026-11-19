"""Equivariant convolutions in planar Cartesian coordinates."""

import torch

from ..ops import config, resolve_backend
from ..ops.common.cartesian_frame import analysis, build_plan, uu_paths
from ..ops.common.gate import gate_program
from ..ops.common.linear import linear_program
from ..ops.common.radial import materialize, radial_reference
from ..ops.common.weights import IndexedProjection, prepare_indexed
from ..ops.precision import validate_module_precision
from ..ops.torch.co2 import gate as torch_gate
from ..ops.torch.co2 import linear as torch_linear
from ..ops.torch.co2 import local_frame as torch_frame


class CartesianConvolution(torch.nn.Module):
    """Edge program shared by channelwise and channel-mixing convolutions."""

    def operands(self):
        return ()

    def forward(
        self,
        features,
        weights,
        edge_index,
        rotation,
        cutoff,
        *,
        radial_network=None,
        projection=None,
        edge_pairs=None,
    ):
        """Sum edge contributions at the target nodes.

        Parameters
        ----------
        features : torch.Tensor
            Cartesian node features of shape ``(nodes, frame_in.input_dim)``.
        weights : torch.Tensor or tuple
            Per-edge weights, or radial inputs to the supplied network.
            Partitioned inputs contain ``(tensor, storage)`` pairs with
            ``edge``, ``source``, ``target`` or ``shared`` storage.
        edge_index : torch.Tensor
            Int64 source and target indices of shape ``(2, edges)``.
        rotation, cutoff : torch.Tensor
            Global-to-local rotations ``(edges, 3, 3)`` and cutoffs ``(edges, 1)``.
        radial_network : torch.nn.Module, optional
            Sequential scalar radial network.
        projection : torch.Tensor, optional
            Final radial projection in (input, output) layout, optionally
            with the affine bias in its last row.
        edge_pairs : equivariantx.conv.EdgePairs, optional
            Supply invariant radial inputs per undirected pair. Geometry and messages remain directed.

        Returns
        -------
        torch.Tensor
            Cartesian node features of shape ``(nodes, frame_out.input_dim)``.
        """
        if edge_pairs is not None:
            from ..ops.common.edge_pairs import reciprocal_weights

            weights, projection, radial_network = reciprocal_weights(
                self,
                edge_pairs,
                weights,
                projection,
                radial_network,
                edge_index,
                local=True,
            )
            if projection is not None and not projection.shape[0]:
                projection = None
        backend = self.backend if features.is_cuda else "torch"
        if isinstance(radial_network, IndexedProjection):
            weights, radial_network = prepare_indexed(
                self.plan, weights, radial_network, edge_index, projection
            )
            projection = None
        if backend == "torch" and config.TORCH_REFERENCE:
            validate_module_precision(
                self.plan,
                (features, weights, rotation, cutoff, projection, self.operands()),
            )
            w = materialize(weights, edge_index)
            if radial_network is not None:
                w = radial_reference(radial_network, w)
            if projection is not None:
                affine = w.shape[1] + 1 == projection.shape[0]
                w = w @ (projection[:-1] if affine else projection)
                if affine:
                    w = w + projection[-1]
            x = self.reference_local(features, w, edge_index, rotation)
            msg = torch_frame.to_global(self.frame_out, x, rotation) * cutoff
            return features.new_zeros(
                (features.shape[0], self.frame_out.input_dim)
            ).index_add(0, edge_index[1], msg)
        return self.plan(
            features,
            weights,
            edge_index,
            rotation,
            cutoff,
            self.operands(),
            radial_network=radial_network,
            projection=projection,
        )


class UuTensorProductConv(CartesianConvolution):
    """Apply a radial channelwise planar Cartesian map and sum over incoming edges.

    Parameters
    ----------
    frame_in, frame_out : co2.LocalFrame
        Direct Cartesian restriction and synthesis.
    linear : co2.UuLinear
        Local channelwise map with summed paths.
    radial_mode : {"full", "last"}, optional
        Stream the complete radial network or only its final projection.
        The latter may reuse a bounded wide projection during CUDA training.
    backend : {None, "torch", "cuda"}, optional
        Execution provider.

    Examples
    --------
    >>> from equivariantx import co2
    >>> frame = co2.LocalFrame("2x0e+2x1o")
    >>> linear = co2.UuLinear(frame.irreps_out, frame.irreps_out, 2)
    >>> conv = co2.UuTensorProductConv(frame, linear, frame)
    """

    def __init__(
        self, frame_in, linear, frame_out, *, backend=None, radial_mode="full"
    ):
        super().__init__()
        if (
            frame_in.method != "direct"
            or frame_out.method != "direct"
            or linear.path_mode != "sum"
            or frame_in.irreps_out != linear.irreps_in
            or frame_out.irreps_out != linear.irreps_out
        ):
            raise ValueError(
                "Expected matching direct Cartesian frames and summed UuLinear paths."
            )
        self.frame_in, self.linear, self.frame_out = frame_in, linear, frame_out
        self.backend = resolve_backend(backend)
        self.weight_numel = linear.weight_numel
        self.radial_mode = radial_mode
        self.plan = build_plan(self)

    def local_program(self, p, source, maps, weights, slot):
        return uu_paths(p, self.linear, source, weights)

    def reference_local(self, features, weight, edge_index, rotation):
        local = torch_frame.to_local(self.frame_in, features[edge_index[0]], rotation)
        return torch_linear.uu_reference(self.linear, local, weight)


class UvTensorProductConv(CartesianConvolution):
    """Mix both endpoints with a gated planar Cartesian channel network.

    Parameters
    ----------
    frame_in, frame_out : co2.LocalFrame
        Direct Cartesian restriction and synthesis.
    up, down : co2.Linear
        Shared local channel maps. Within each irrep, the input to ``up``
        concatenates target copies followed by source copies.
    gate : co2.Gate
        Nonlinearity between the local channel maps.
    hidden : sequence of (co2.Linear, co2.Gate), optional
        Additional channel maps and gates applied before ``down``.
    radial_mode : {"full", "last"}, optional
        Stream the complete radial network or only its final projection.
        The latter may reuse a bounded wide projection during CUDA training.
    backend : {None, "torch", "cuda"}, optional
        Execution provider.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co2
    >>> frame = co2.LocalFrame("2x0e")
    >>> gate = co2.Gate("2x0e", [torch.tanh], "", [], "")
    >>> conv = co2.UvTensorProductConv(frame, co2.Linear("4x0e", "2x0e"),
    ...     gate, co2.Linear("2x0e", "2x0e"), frame)
    """

    def __init__(
        self,
        frame_in,
        up,
        gate,
        down,
        frame_out,
        *,
        hidden=(),
        backend=None,
        radial_mode="full",
    ):
        super().__init__()
        blocks = ((up, gate), *hidden)
        following = [linear for linear, _ in blocks[1:]] + [down]
        if (
            frame_in.method != "direct"
            or frame_out.method != "direct"
            or [(2 * mul, ir) for mul, ir in frame_in.irreps_out] != list(up.irreps_in)
            or any(linear.irreps_out != act.irreps_in for linear, act in blocks)
            or any(
                act.irreps_out != layer.irreps_in
                for (_, act), layer in zip(blocks, following)
            )
            or down.irreps_out != frame_out.irreps_out
            or any(
                not layer.internal_weights
                or not layer.shared_weights
                or not isinstance(layer.projection, torch.nn.Identity)
                for layer in (up, *following)
            )
        ):
            raise ValueError(
                "Expected matching direct frames and shared Cartesian linear maps."
            )
        self.backend = resolve_backend(backend)
        self.frame_in, self.frame_out = frame_in, frame_out
        self.up, self.gate, self.down = up, gate, down
        self.hidden = torch.nn.ModuleList(
            torch.nn.ModuleList((linear, act)) for linear, act in blocks[1:]
        )
        self.weight_numel = sum(mul for mul, ir in up.irreps_in)
        self.radial_mode = radial_mode
        self.plan = build_plan(self)

    def operands(self):
        return tuple(
            value
            for layer in (self.up, *(block[0] for block in self.hidden), self.down)
            for value in (layer.weight, layer.bias)
        )

    def local_program(self, p, source, maps, weights, slot):
        target = analysis(
            p, self.frame_in, p.input(0, "target", self.frame_in.input_dim), maps
        )
        offset = 0
        values = []
        for s, t, (mul, ir) in zip(source, target, self.up.irreps_in):
            dim = ir.dim
            factors = p.gather(
                weights, (offset + c for c in range(mul) for _ in range(dim))
            )
            value = p.binary("mul", p.concatenate([t, s]), factors)
            values.append(p.transpose(value, mul, dim))
            offset += mul

        def channel_map(layer, parts, slot):
            weight = p.input(slot, "shared", layer.weight_numel)
            bias = p.input(slot + 1, "shared", layer.bias_numel)
            values = linear_program(p, parts, weight, bias, layer.layout)
            return [
                p.transpose(v, ir.dim, mul)
                for v, (mul, ir) in zip(values, layer.irreps_out)
            ]

        def split(value, layer):
            return [
                p.transpose(p.slice(value, s.start, mul * ir.dim), mul, ir.dim)
                for (mul, ir), s in zip(layer.irreps_in, layer.irreps_in.slices())
            ]

        def block(linear, act, parts, slot):
            value = p.concatenate(channel_map(linear, parts, slot))
            (value,) = p.compose(gate_program(act, cartesian=True), (value,))
            return value

        value = block(self.up, self.gate, values, slot)
        slot += 2
        for linear, act in self.hidden:
            value = block(linear, act, split(value, linear), slot)
            slot += 2
        return channel_map(self.down, split(value, self.down), slot)

    def reference_local(self, features, weight, edge_index, rotation):
        frame = self.frame_in
        source = torch_frame.to_local(frame, features[edge_index[0]], rotation)
        target = torch_frame.to_local(frame, features[edge_index[1]], rotation)
        values = []
        for (mul, ir), section, w in zip(
            self.up.irreps_in,
            frame.irreps_out.slices(),
            weight.split([m for m, _ in self.up.irreps_in], -1),
        ):
            pair = torch.cat([target[:, section], source[:, section]], -1)
            values.append(
                (pair.unflatten(-1, (mul, ir.dim)) * w[..., None]).flatten(-2)
            )
        value = torch_linear.linear(
            self.up, torch.cat(values, -1), self.up.weight, self.up.bias
        )
        value = torch_gate.evaluate(self.gate, value)
        for linear, act in self.hidden:
            value = torch_linear.linear(linear, value, linear.weight, linear.bias)
            value = torch_gate.evaluate(act, value)
        return torch_linear.linear(self.down, value, self.down.weight, self.down.bias)
