"""Independent radial networks with convolution path ordering."""

from itertools import accumulate
from typing import NamedTuple

import torch

from ..ops import config, resolve_backend
from ..ops.common.grouped_radial import (
    affine_chain,
    affine_shape,
    affine_weight,
    chain_tiles,
    eager_rows,
    linear_outputs,
    pack_matrices,
    states,
)
from ..ops.common.radial import radial_layers, radial_program, radial_reference
from ..utils.layout import permute


def grouped_chain(branches):
    schedules = tuple(affine_chain(layers) for layers in branches)
    if not schedules or not schedules[0] or any(s != schedules[0] for s in schedules):
        return ()
    steps = schedules[0]
    if (
        len(steps) < 2
        or steps[-1][0] != len(branches[0]) - 1
        or steps[-1][1] is not None
    ):
        return ()
    for i, _ in steps[:-1]:
        shapes = [affine_shape(layers[i]) for layers in branches]
        if len({s[1] for s in shapes}) != 1 or (i and len(set(shapes)) != 1):
            return ()
    return steps


def join_columns(values, order, inverse):
    out = torch.cat(tuple(values), dim=-1)
    return out if order is None else permute(out, order, inverse)


def grouped_program(inputs, program, value, sections, networks, context, outputs):
    if context:
        tail = program.slice(value, program.size(value) - context, context)
    groups = []
    for (offset, width, expected), network in zip(sections, networks):
        x = program.slice(value, offset, width)
        if context:
            x = program.concatenate((x, tail))
        y = radial_program(network, inputs, program, x)
        if program.size(y) != expected:
            raise ValueError("Radial branch output width does not match its paths.")
        groups.append(y)
    return (
        program.concatenate(
            tuple(program.slice(groups[i], start, width) for i, start, width in outputs)
        )
        if outputs
        else program.constant(0, 0)
    )


def grouped_reference(value, sections, networks, context, outputs):
    groups = []
    for (offset, width, expected), network in zip(sections, networks):
        x = value.narrow(-1, offset, width)
        if context:
            x = torch.cat((x, value[:, -context:]), -1)
        y = radial_reference(network, x)
        if y.shape[-1] != expected:
            raise ValueError("Radial branch output width does not match its paths.")
        groups.append(y)
    if not outputs:
        return value[:, :0]
    return torch.cat(
        [groups[i].narrow(-1, start, width) for i, start, width in outputs], -1
    )


class GroupedProjection(NamedTuple):
    """Final branch projections of a grouped radial network."""

    sections: tuple
    networks: tuple
    outputs: tuple
    mode: str = "full"
    order: torch.Tensor | None = None
    inverse: torch.Tensor | None = None

    def forward(self, value):
        if not self.sections:
            return value[:, :0]
        groups = [
            radial_reference(network, value.narrow(-1, start, n))
            for (start, n, _), network in zip(self.sections, self.networks)
        ]
        return join_columns(groups, self.order, self.inverse)

    def to_edge_program(self, inputs, program, value):
        return grouped_program(
            inputs, program, value, self.sections, self.networks, 0, self.outputs
        )


class GroupedRadialMLP(torch.nn.Module):
    """Generate convolution weights from independent groups of radial features.

    Parameters
    ----------
    in_features : sequence of int
        Width of each contiguous input group.
    networks : sequence of torch.nn.Module
        One radial network per used group, in ascending input-group order.
    paths : sequence of (int, int)
        Input group and output width for each path in convolution order.
        Each network emits its own paths in this order.
    context_features : int, optional
        Number of trailing input features supplied to every network.
    backend : {None, "torch", "cuda"}, optional
        Execution backend. CPU inputs use PyTorch.

    Examples
    --------
    >>> from torch import nn
    >>> radial = GroupedRadialMLP([3, 2], [nn.Linear(3, 4), nn.Linear(2, 1)],
    ...                          [(0, 2), (1, 1), (0, 2)])
    >>> radial(torch.randn(7, 5)).shape
    torch.Size([7, 5])

    Pass the module as ``radial_network`` with a projection of shape
    ``(0, weight_numel)`` to evaluate it inside a full-mode convolution.
    """

    def __init__(
        self, in_features, networks, paths, *, context_features=0, backend=None
    ):
        super().__init__()
        self.backend = resolve_backend(backend)
        self.in_features = tuple(in_features)
        self.context_features = context_features
        if any(n < 1 for n in in_features) or context_features < 0:
            raise ValueError("Invalid grouped radial input widths.")
        paths = tuple(paths)
        self.active = tuple(sorted({group for group, _ in paths}))
        if any(not 0 <= i < len(in_features) for i in self.active):
            raise ValueError("Radial path group is outside the input layout.")
        if len(networks) != len(self.active):
            raise ValueError("Provide one network per active radial group.")
        self.branches = torch.nn.ModuleList(networks)
        # Only the static module graph is cached, never parameter views or values.
        self.stages = tuple(radial_layers(network) for network in networks)
        self.chain_steps = grouped_chain(self.stages)
        self.workspace_features = sum(
            getattr(
                layer,
                "out_features",
                getattr(layer, "out_dim", getattr(layer, "h_out", 0)),
            )
            for network in networks
            for layer in radial_layers(network)
        )
        offsets = tuple(accumulate(in_features, initial=0))
        self.register_buffer("output_order", None, persistent=False)
        self.register_buffer("inverse_order", None, persistent=False)
        self.set_paths(paths)
        self.sections = tuple(
            (offsets[i], in_features[i], n)
            for i, n in zip(self.active, self.branch_widths)
        )
        if self.chain_steps:
            for (_, width, output), layers in zip(self.sections, self.stages):
                if affine_shape(layers[0])[0] != width + self.context_features:
                    raise ValueError(
                        "Radial branch input width does not match its group."
                    )
                if affine_shape(layers[-1])[1] != output:
                    raise ValueError(
                        "Radial branch output width does not match its paths."
                    )

    def set_paths(self, paths):
        """Set the order of path blocks without changing the branch parameters."""
        offsets = [0] * len(self.active)
        outputs = []
        for group, width in paths:
            if width < 1:
                raise ValueError("Radial path widths must be positive.")
            i = self.active.index(group)
            if outputs and outputs[-1][0] == i:
                _, start, n = outputs.pop()
                outputs.append((i, start, n + width))
            else:
                outputs.append((i, offsets[i], width))
            offsets[i] += width
        if hasattr(self, "branch_widths") and self.branch_widths != tuple(offsets):
            raise ValueError(
                "A path reordering cannot change the branch output widths."
            )
        self.outputs = tuple(outputs)
        self.branch_widths = tuple(offsets)
        self.out_features = sum(offsets)
        starts = tuple(accumulate(offsets, initial=0))
        columns = [
            starts[i] + k
            for i, start, width in outputs
            for k in range(start, start + width)
        ]
        if columns == list(range(self.out_features)):
            self.output_order = self.inverse_order = None
        else:
            first = next(self.parameters(), None)
            self.output_order = torch.tensor(
                columns, device=None if first is None else first.device
            )
            self.inverse_order = self.output_order.argsort()

    def forward(self, value):
        if (
            value.ndim != 2
            or value.shape[1] != sum(self.in_features) + self.context_features
        ):
            raise ValueError("Input width does not match grouped radial layout.")
        if self.chain_steps and (
            not config.TORCH_REFERENCE or (self.backend == "cuda" and value.is_cuda)
        ):
            return self.packed(value)
        return grouped_reference(
            value, self.sections, self.branches, self.context_features, self.outputs
        )

    def to_edge_program(self, inputs, program, value):
        if program.size(value) != sum(self.in_features) + self.context_features:
            raise ValueError("Input width does not match grouped radial layout.")
        return grouped_program(
            inputs,
            program,
            value,
            self.sections,
            self.branches,
            self.context_features,
            self.outputs,
        )

    def stream_rows(self, value, backend, mode="full"):
        """Evaluate pair rows in bounded tiles with a batched force-loss adjoint."""
        if not self.chain_steps or not eager_rows(value):
            return None
        return self.packed(value, backend=backend, mode=mode)

    def packed(self, value, *, hidden_only=False, backend=None, mode=None):
        """Apply independent affine stages in group-major layout."""
        backend = self.backend if backend is None else backend
        native = backend == "cuda" and eager_rows(value)
        xs = [value.narrow(-1, start, n) for start, n, _ in self.sections]
        if self.context_features:
            xs = [torch.cat((x, value[:, -self.context_features :]), -1) for x in xs]
        width = max(x.shape[1] for x in xs)
        x = torch.stack(
            [
                v
                if v.shape[1] == width
                else torch.nn.functional.pad(v, (0, width - v.shape[1]))
                for v in xs
            ]
        )
        params = []
        no_bias = x.new_empty(0)
        scales = tuple(scale for _, scale in self.chain_steps[:-1])
        hidden = 0
        for index, _ in self.chain_steps[:-1]:
            layers = [s[index] for s in self.stages]
            matrix = pack_matrices(layers, width if index == 0 else None)
            params.append(matrix)
            biases = [getattr(layer, "bias", None) for layer in layers]
            params.append(
                torch.stack(
                    [
                        matrix.new_zeros(matrix.shape[2]) if b is None else b
                        for b in biases
                    ]
                )[:, None, :]
                if any(b is not None for b in biases)
                else no_bias
            )
            hidden += len(layers) * matrix.shape[2]
        final = [s[-1] for s in self.stages]
        if mode is None:
            plan = scales, self.branch_widths, native, False
            h, _, _ = states(plan, (x, *params), save=False)
            if hidden_only:
                return torch.cat(h.unbind(0), -1)
        if len(final) > 1 and len(set(self.branch_widths)) == 1:
            params.append(pack_matrices(final))
        else:
            params.extend(affine_weight(layer) for layer in final)
        if mode is None:
            y = linear_outputs(plan, h, (x, *params))
        else:
            y = chain_tiles(x, params, scales, self.branch_widths, native, mode, hidden)
        biases = [getattr(layer, "bias", None) for layer in final]
        if any(b is not None for b in biases):
            y = y + torch.cat(
                [
                    y.new_zeros(n) if b is None else b
                    for n, b in zip(self.branch_widths, biases)
                ]
            )
        return (
            y
            if self.output_order is None
            else permute(y, self.output_order, self.inverse_order)
        )

    def final_projection(self, widths, layers):
        """Return a deferred projection for precomputed branch hidden features."""
        if len(widths) != len(layers) or len(widths) != len(self.active):
            raise ValueError("Expected one final affine map per active radial group.")
        sections = tuple(zip(accumulate(widths, initial=0), widths, self.branch_widths))
        return GroupedProjection(
            sections,
            tuple(layers),
            self.outputs,
            "full",
            self.output_order,
            self.inverse_order,
        )
