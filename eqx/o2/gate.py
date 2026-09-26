################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Callable, NamedTuple, Optional, Sequence

import torch
from e3nn.math import normalize2mom

from ._layout import _Permute
from .irreps import Irrep, Irreps, IrrepsLike


class Activation(torch.nn.Module):
    """Apply normalized scalar activations to O(2) features.

    Parameters
    ----------
    irreps_in : Irreps, str, or sequence
        Input representation. Non-``None`` activations are valid only for
        order-zero entries.
    acts : sequence of callable or None
        One activation per input entry. ``None`` leaves that entry unchanged.
        Each activation is rescaled to unit second moment for standard-normal
        inputs. Its parity determines the output reflection and time parity.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        acts: Sequence[Optional[Callable]],
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        if len(self.irreps_in) != len(acts):
            raise ValueError(
                "Irreps and activation counts do not match: "
                f"{len(self.irreps_in)} != {len(acts)}."
            )

        normalized_acts = [
            normalize2mom(act) if act is not None else None for act in acts
        ]
        irreps_out = []
        for (ir, mul), act in zip(self.irreps_in, normalized_acts):
            if act is None:
                irreps_out.append((ir, mul))
                continue
            if ir.m != 0:
                raise ValueError("Activation functions can only act on scalars.")
            if ir.is_invariant_scalar():
                irreps_out.append((ir, mul))
                continue
            reference = next(act.parameters(), None)
            if reference is None:
                reference = next(act.buffers(), None)
            kwargs = {}
            if reference is not None:
                kwargs["device"] = reference.device
                if reference.is_floating_point():
                    kwargs["dtype"] = reference.dtype
            values = torch.linspace(0.0, 10.0, 256, **kwargs)
            training = act.training
            act.eval()
            try:
                with torch.no_grad():
                    positive = act(values)
                    negative = act(-values)
            finally:
                act.train(training)
            scale = torch.maximum(
                torch.ones((), device=positive.device, dtype=positive.dtype),
                torch.maximum(positive.abs().max(), negative.abs().max()),
            )
            even = (positive - negative).abs().max() <= 1.0e-5 * scale
            odd = (positive + negative).abs().max() <= 1.0e-5 * scale
            activation_parity = 1 if even else -1 if odd else 0
            if activation_parity == 0 and (ir.p == -1 or ir.t == -1):
                raise ValueError(
                    "An activation acting on a reflection- or time-reversal-odd "
                    "scalar must be either even or odd."
                )
            if activation_parity == -1:
                output_parity = ir.p
                output_time_parity = ir.t
            else:
                output_parity = 1
                output_time_parity = 1
            irreps_out.append((Irrep(0, output_parity, output_time_parity), mul))

        self.irreps_out = Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(normalized_acts)
        self._slices = self.irreps_in.slices()
        self._sizes = tuple(ir_mul.dim for ir_mul in self.irreps_in)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the configured activations.

        Parameters
        ----------
        features : torch.Tensor
            Input with shape ``(..., irreps_in.dim)`` in flattened ``ir_mul`` order.

        Returns
        -------
        torch.Tensor
            Activated features with shape ``(..., irreps_out.dim)``.
        """
        if features.ndim < 1 or features.size(-1) != self.irreps_in.dim:
            raise ValueError(
                "Activation feature trailing dimension must be "
                f"{self.irreps_in.dim}, got {tuple(features.shape)}."
            )
        outputs = []
        for values, act in zip(features.split(self._sizes, dim=-1), self.acts):
            outputs.append(values if act is None else act(values))
        if outputs:
            return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        return features[..., :0]

    def __repr__(self) -> str:
        acts = "".join("x" if act is not None else " " for act in self.acts)
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"


class _GatePath(NamedTuple):
    i_gate: int
    gate_start: int
    i_gated: int
    gated_start: int
    mul: int
    ir_gate: Irrep
    ir_gated: Irrep
    ir_out: Irrep


@torch.compiler.allow_in_graph
class _GateExpand(torch.autograd.Function):
    """Broadcast gates and sum component gradients in the backward pass."""

    generate_vmap_rule = True

    @staticmethod
    def forward(
        gates, index, output_index, output_inverse, gate_index, gate_inverse, layout
    ):
        return gates.index_select(-1, index)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, index, output_index, output_inverse, gate_index, gate_inverse, layout = (
            inputs
        )
        ctx.save_for_backward(output_index, output_inverse, gate_index, gate_inverse)
        ctx.save_for_forward(index)
        ctx.layout = layout

    @staticmethod
    def backward(ctx, gradient):
        output_index, output_inverse, gate_index, gate_inverse = ctx.saved_tensors
        sizes, dimensions, output_identity, gate_identity = ctx.layout
        if not output_identity:
            gradient = _Permute.apply(gradient, output_inverse, output_index)
        widths = tuple(dim * mul for dim, mul in zip(dimensions, sizes))
        values = [
            value.view(*gradient.shape[:-1], dim, mul).sum(-2)
            for value, dim, mul in zip(
                gradient.split(widths, dim=-1), dimensions, sizes
            )
        ]
        gradient = values[0] if len(values) == 1 else torch.cat(values, dim=-1)
        if not gate_identity:
            gradient = _Permute.apply(gradient, gate_inverse, gate_index)
        return gradient, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx, tangent, *unused):
        (index,) = ctx.saved_tensors
        return tangent.index_select(-1, index)


class Gate(torch.nn.Module):
    """Apply scalar activations and scalar gates to O(2) features.

    Parameters
    ----------
    irreps_scalars : Irreps, str, or sequence
        Scalar entries transformed directly by ``act_scalars``.
    act_scalars : sequence of callable or None
        One activation per scalar entry, normalized to unit second moment.
        ``None`` leaves the entry unchanged.
    irreps_gates : Irreps, str, or sequence
        Order-zero entries used as gates. Their total multiplicity must equal
        the total multiplicity in ``irreps_gated``. Output irreps follow the
        tensor product of each activated gate and its gated entry.
    act_gates : sequence of callable or None
        One activation per gate entry, normalized to unit second moment.
        ``None`` leaves the entry unchanged.
    irreps_gated : Irreps, str, or sequence
        Entries multiplied channel-wise by the activated gates.
    """

    def __init__(
        self,
        irreps_scalars: IrrepsLike,
        act_scalars: Sequence[Optional[Callable]],
        irreps_gates: IrrepsLike,
        act_gates: Sequence[Optional[Callable]],
        irreps_gated: IrrepsLike,
    ) -> None:
        super().__init__()
        irreps_scalars = Irreps(irreps_scalars)
        irreps_gates = Irreps(irreps_gates)
        irreps_gated = Irreps(irreps_gated)
        if any(ir.m != 0 for ir, _ in irreps_scalars):
            raise ValueError("Gate scalars must be scalar O(2) irreps.")
        if any(ir.m != 0 for ir, _ in irreps_gates):
            raise ValueError("Gate inputs must be scalar O(2) irreps.")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} gated irreps, but "
                f"{irreps_gates.num_irreps} gate scalars."
            )

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        self.act_gates = Activation(irreps_gates, act_gates)
        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated

        self.irreps_in, p, _ = (irreps_scalars + irreps_gates + irreps_gated).sort()
        scalar_end = len(irreps_scalars)
        gate_end = scalar_end + len(irreps_gates)
        self._scalar_locations = p[:scalar_end]
        self._gate_locations = p[scalar_end:gate_end]
        self._gated_locations = p[gate_end:]
        self._input_slices = self.irreps_in.slices()
        self._gate_slices = self.act_gates.irreps_out.slices()

        paths = []
        output_irrep_list = []
        i_gate = i_gated = 0
        gate_start = gated_start = 0
        while i_gate < len(self.act_gates.irreps_out):
            ir_gate, mul_gate = self.act_gates.irreps_out[i_gate]
            ir_gated, mul_gated = irreps_gated[i_gated]
            count = min(mul_gate - gate_start, mul_gated - gated_start)
            (ir_out,) = ir_gated * ir_gate
            paths.append(
                _GatePath(
                    i_gate,
                    gate_start,
                    i_gated,
                    gated_start,
                    count,
                    ir_gate,
                    ir_gated,
                    ir_out,
                )
            )
            output_irrep_list.append((ir_out, count))
            gate_start += count
            gated_start += count
            if gate_start == mul_gate:
                i_gate += 1
                gate_start = 0
            if gated_start == mul_gated:
                i_gated += 1
                gated_start = 0
        self._paths = tuple(paths)
        self.irreps_out = self.act_scalars.irreps_out + Irreps(output_irrep_list)
        # Select adjacent entries with a single view.
        selections, spans = [], []
        for locations in (
            self._scalar_locations,
            self._gate_locations,
            self._gated_locations,
        ):
            runs = []
            for index in locations:
                if runs and index == runs[-1][-1] + 1:
                    runs[-1].append(index)
                else:
                    runs.append([index])
            selections.append(tuple(run[0] for run in runs))
            spans.extend(
                (run[0], sum(self.irreps_in[i].dim for i in run)) for run in runs
            )
        spans.sort()
        positions = {index: i for i, (index, _) in enumerate(spans)}
        self._input_sizes = tuple(size for _, size in spans)
        self._selections = tuple(
            tuple(positions[index] for index in selected) for selected in selections
        )
        self._dimensions = tuple(sorted({path.ir_gated.dim for path in paths}))
        self._gate_sizes = tuple(
            sum(path.mul for path in paths if path.ir_gated.dim == dim)
            for dim in self._dimensions
        )
        gated_indices, gate_indices, signs = [], [], []
        output_indices = [0] * irreps_gated.dim
        gated_slices = irreps_gated.slices()
        output_slices = Irreps(output_irrep_list).slices()
        offset = 0
        for dim, mul in zip(self._dimensions, self._gate_sizes):
            selected = [
                (i, path) for i, path in enumerate(paths) if path.ir_gated.dim == dim
            ]
            for _, path in selected:
                start = self._gate_slices[path.i_gate].start + path.gate_start
                gate_indices.extend(range(start, start + path.mul))
            for component in range(dim):
                channel = 0
                for i, path in selected:
                    swapped = path.ir_gate.is_odd_scalar() and dim == 2
                    row = 1 - component if swapped else component
                    start = (
                        gated_slices[path.i_gated].start
                        + row * irreps_gated[path.i_gated].mul
                        + path.gated_start
                    )
                    gated_indices.extend(range(start, start + path.mul))
                    signs.extend(
                        [-1.0 if swapped and component == 0 else 1.0] * path.mul
                    )
                    start = output_slices[i].start + component * path.mul
                    output_indices[start : start + path.mul] = range(
                        offset + component * mul + channel,
                        offset + component * mul + channel + path.mul,
                    )
                    channel += path.mul
            offset += dim * mul
        expanded_indices = []
        offset = 0
        for dim, mul in zip(self._dimensions, self._gate_sizes):
            expanded_indices.extend(gate_indices[offset : offset + mul] * dim)
            offset += mul
        expanded_indices = [expanded_indices[i] for i in output_indices]
        gated_indices = [gated_indices[i] for i in output_indices]
        signs = [signs[i] for i in output_indices]
        for name, indices in (
            ("gated", gated_indices),
            ("gate", gate_indices),
            ("output", output_indices),
        ):
            setattr(self, f"_{name}_identity", indices == list(range(len(indices))))
            self.register_buffer(
                f"_{name}_indices",
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_{name}_inverse",
                torch.tensor(
                    sorted(range(len(indices)), key=indices.__getitem__),
                    dtype=torch.long,
                ),
                persistent=False,
            )
        self.register_buffer("_gated_signs", torch.tensor(signs), persistent=False)
        self.register_buffer(
            "_expanded_indices",
            torch.tensor(expanded_indices, dtype=torch.long),
            persistent=False,
        )
        self._has_signs = -1.0 in signs
        self._gate_layout = (
            self._gate_sizes,
            self._dimensions,
            self._output_identity,
            self._gate_identity,
        )

    def _select_entries(
        self,
        inputs: tuple[torch.Tensor, ...],
        locations: tuple[int, ...],
    ) -> torch.Tensor:
        if not locations:
            return inputs[0][..., :0]
        if len(locations) == 1:
            return inputs[locations[0]]
        return torch.cat(
            [inputs[index] for index in locations],
            dim=-1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Activate scalar entries and apply the gates.

        Parameters
        ----------
        features : torch.Tensor
            Input with shape ``(..., irreps_in.dim)`` in flattened ``ir_mul`` order.
            ``irreps_in`` sorts the scalar, gate, and gated entries together.

        Returns
        -------
        torch.Tensor
            Activated scalars followed by gated entries, with shape
            ``(..., irreps_out.dim)`` in flattened ``ir_mul`` order.
        """
        if features.ndim < 1 or features.size(-1) != self.irreps_in.dim:
            raise ValueError(
                "Gate feature trailing dimension must be "
                f"{self.irreps_in.dim}, got {tuple(features.shape)}."
            )
        if not self.irreps_in:
            return features
        inputs = features.split(self._input_sizes, dim=-1)
        scalars = self.act_scalars(self._select_entries(inputs, self._selections[0]))
        if not self._paths:
            return scalars
        gates = self.act_gates(self._select_entries(inputs, self._selections[1]))
        values = self._select_entries(inputs, self._selections[2])
        if not self._gated_identity:
            values = _Permute.apply(values, self._gated_indices, self._gated_inverse)
        if self._has_signs:
            values = values * self._gated_signs
        gates = _GateExpand.apply(
            gates,
            self._expanded_indices,
            self._output_indices,
            self._output_inverse,
            self._gate_indices,
            self._gate_inverse,
            self._gate_layout,
        )
        output = values * gates
        return torch.cat((scalars, output), dim=-1) if scalars.size(-1) else output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"
