################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import NamedTuple, Sequence

import torch

from .irreps import Irrep, Irreps, IrrepsLike
from .tensor_product import _cg_product


class _Path(NamedTuple):
    leaves: tuple[int, ...]
    intermediates: tuple[Irrep, ...]
    output_index: int


class AsymmetricContraction(torch.nn.Module):
    """Contract independent O(2) inputs into many-body features.

    Parameters
    ----------
    irreps_in : Irreps, str, or sequence
        Representation of every independent input. All entries must have one
        common multiplicity, which is interpreted as the channel count.
    irreps_out : Irreps, str, or sequence
        Requested output types. Their multiplicities must equal the common
        input multiplicity.
    correlation : int
        Highest correlation order to enumerate. The module consumes one
        independent input tensor for every order up to this value.
    algorithm : {"recursive", "dense"}
        Evaluation strategy. ``"recursive"`` recursively contracts individual
        paths; ``"dense"`` evaluates precomputed generalized coupling tensors.
    path_mode : {"sum", "expand"}, optional
        ``"sum"`` accumulates equivalent paths into each requested output
        type with variance normalization. ``"expand"`` preserves every path
        as a separate output multiplicity.

    Notes
    -----
    Inputs and outputs use flattened ``ir_mul`` layout. Weights are always
    supplied externally and are applied directly to the enumerated paths.
    ``weight_numel`` gives the required trailing weight dimension. Every path
    multiplies the time parities of its leaves; external weights are invariant.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        irreps_out: IrrepsLike,
        correlation: int,
        *,
        algorithm: str,
        path_mode: str = "sum",
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        muls = {mul for _, mul in self.irreps_in}
        if len(muls) != 1:
            raise ValueError("Input irreps must have one common multiplicity.")
        self.num_channels = muls.pop()
        self._input_irreps = tuple(ir for ir, _ in self.irreps_in)
        requested_irreps_out = Irreps(irreps_out)
        if any(mul != self.num_channels for _, mul in requested_irreps_out):
            raise ValueError(
                "AsymmetricContraction input and requested output entries must "
                "use the same multiplicity."
            )
        if not isinstance(correlation, int):
            raise TypeError("correlation must be an integer.")
        if correlation < 1:
            raise ValueError("correlation must be positive.")
        if algorithm not in ("recursive", "dense"):
            raise ValueError("algorithm must be 'recursive' or 'dense'.")
        if path_mode not in ("sum", "expand"):
            raise ValueError("path_mode must be 'sum' or 'expand'.")
        self.correlation = correlation
        self.algorithm = algorithm
        self.path_mode = path_mode

        output_types = []
        for ir, _ in requested_irreps_out:
            if ir not in output_types:
                output_types.append(ir)
        self.irreps_out_types = Irreps([(ir, self.num_channels) for ir in output_types])
        allowed_outputs = set(output_types)
        filtered_states = tuple(
            tuple(
                (leaves, intermediates)
                for leaves, intermediates in states
                if intermediates[-1] in allowed_outputs
            )
            for states in self._enumerate_states()
        )

        if path_mode == "sum":
            base_irreps_out = Irreps(output_types)
            output_indices = {ir: index for index, ir in enumerate(output_types)}
            paths_by_order = [
                tuple(
                    _Path(leaves, intermediates, output_indices[intermediates[-1]])
                    for leaves, intermediates in states
                )
                for states in filtered_states
            ]
        else:
            output_counts = {ir: 0 for ir in output_types}
            for states in filtered_states:
                for _, intermediates in states:
                    output_counts[intermediates[-1]] += 1
            base_irreps_out = Irreps(
                [(ir, output_counts[ir]) for ir in output_types if output_counts[ir]]
            )
            next_output_index = {}
            offset = 0
            for ir, mul in base_irreps_out:
                next_output_index[ir] = offset
                offset += mul
            paths_by_order = []
            for states in filtered_states:
                paths = []
                for leaves, intermediates in states:
                    ir_out = intermediates[-1]
                    paths.append(
                        _Path(leaves, intermediates, next_output_index[ir_out])
                    )
                    next_output_index[ir_out] += 1
                paths_by_order.append(tuple(paths))

        self._base_irreps_out = base_irreps_out
        self.irreps_out = Irreps(
            [(ir, mul * self.num_channels) for ir, mul in base_irreps_out]
        )
        self._paths_by_order = tuple(paths_by_order)
        self.order_num_paths = tuple(len(paths) for paths in self._paths_by_order)
        self.num_paths = sum(self.order_num_paths)
        self.weight_numel = self.num_paths * self.num_channels
        self.weight_shape = (self.weight_numel,)

        if path_mode == "sum":
            output_path_counts = [0] * base_irreps_out.num_irreps
            for paths in self._paths_by_order:
                for path in paths:
                    output_path_counts[path.output_index] += 1
            self._path_scales = tuple(
                tuple(output_path_counts[path.output_index] ** -0.5 for path in paths)
                for paths in self._paths_by_order
            )
        else:
            self._path_scales = tuple(
                tuple(1.0 for _ in paths) for paths in self._paths_by_order
            )

        order_weight_slices = []
        offset = 0
        for num_paths in self.order_num_paths:
            width = num_paths * self.num_channels
            order_weight_slices.append(slice(offset, offset + width))
            offset += width
        self._order_weight_slices = tuple(order_weight_slices)
        self._base_input_slices = Irreps(self._input_irreps).slices()
        self._base_input_dim = sum(ir.dim for ir in self._input_irreps)
        self._base_output_slices = Irreps(base_irreps_out.expanded()).slices()
        paths_by_output = [[] for _ in range(base_irreps_out.num_irreps)]
        for order_index, paths in enumerate(self._paths_by_order):
            for path_index, path in enumerate(paths):
                paths_by_output[path.output_index].append(
                    (order_index, path_index, path)
                )
        self._paths_by_output = tuple(tuple(paths) for paths in paths_by_output)
        if algorithm == "dense":
            self._register_coupling_tensors()

    def _enumerate_states(self):
        previous = tuple(
            (((index,), (ir,))) for index, ir in enumerate(self._input_irreps)
        )
        states_by_order = [previous]
        for _ in range(2, self.correlation + 1):
            current = []
            for leaves, intermediates in previous:
                for input_index, ir in enumerate(self._input_irreps):
                    for ir_out in intermediates[-1] * ir:
                        current.append(
                            (leaves + (input_index,), intermediates + (ir_out,))
                        )
            previous = tuple(current)
            states_by_order.append(previous)
        return tuple(states_by_order)

    def _coupling_tensor(self, path: _Path) -> torch.Tensor:
        first_ir = self._input_irreps[path.leaves[0]]
        coefficient = torch.eye(first_ir.dim, dtype=torch.float64)
        for order_index in range(1, len(path.leaves)):
            ir1 = path.intermediates[order_index - 1]
            ir2 = self._input_irreps[path.leaves[order_index]]
            pair = (
                _cg_product(
                    torch.eye(ir1.dim, dtype=torch.float64),
                    ir1,
                    torch.eye(ir2.dim, dtype=torch.float64),
                    ir2,
                    path.intermediates[order_index],
                )
                .permute(1, 2, 0)
                .contiguous()
            )
            coefficient = torch.tensordot(
                coefficient,
                pair,
                dims=([-1], [0]),
            ).contiguous()
        return coefficient

    def _register_coupling_tensors(self) -> None:
        input_dim = self._base_input_dim
        output_dim = self._base_irreps_out.dim
        for order_index, paths in enumerate(self._paths_by_order):
            order = order_index + 1
            coefficient = torch.zeros(
                (output_dim,) + (input_dim,) * order + (len(paths),),
                dtype=torch.float64,
            )
            for path_index, (path, scale) in enumerate(
                zip(paths, self._path_scales[order_index])
            ):
                compact = self._coupling_tensor(path)
                compact = compact.permute(
                    compact.ndim - 1,
                    *range(compact.ndim - 1),
                )
                coefficient[
                    (
                        self._base_output_slices[path.output_index],
                        *(self._base_input_slices[index] for index in path.leaves),
                        path_index,
                    )
                ] = compact * scale
            self.register_buffer(
                f"generalized_cg_{order}",
                coefficient,
                persistent=False,
            )

    def _flatten_output(self, features: torch.Tensor) -> torch.Tensor:
        outputs = []
        offset = 0
        for ir, mul in self._base_irreps_out:
            width = ir.dim * mul
            values = features[..., offset : offset + width, :].reshape(
                *features.shape[:-2], mul, ir.dim, self.num_channels
            )
            outputs.append(
                values.transpose(-3, -2).reshape(
                    *features.shape[:-2], ir.dim * mul * self.num_channels
                )
            )
            offset += width
        if outputs:
            return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        return features.flatten(-2)

    def _forward_recursive(self, inputs, order_weights):
        leading_shape = inputs[0].shape[:-2]
        zero = sum(features[..., :0].sum() for features in inputs)
        zero = zero + sum(weight[..., :0].sum() for weight in order_weights)
        outputs = []
        for output_index, ir_out in enumerate(self._base_irreps_out.expanded()):
            output = (
                inputs[0].new_zeros((*leading_shape, ir_out.dim, self.num_channels))
                + zero
            )
            for order_index, path_index, path in self._paths_by_output[output_index]:
                value = inputs[0][..., self._base_input_slices[path.leaves[0]], :]
                for i in range(1, len(path.leaves)):
                    i_in = path.leaves[i]
                    value = _cg_product(
                        value,
                        path.intermediates[i - 1],
                        inputs[i][..., self._base_input_slices[i_in], :],
                        self._input_irreps[i_in],
                        path.intermediates[i],
                        elementwise=True,
                    )
                output = output + (
                    value
                    * order_weights[order_index][..., path_index, :].unsqueeze(-2)
                    * self._path_scales[order_index][path_index]
                )
            outputs.append(output)
        output = (
            torch.cat(outputs, dim=-2)
            if outputs
            else inputs[0].new_empty((*leading_shape, 0, self.num_channels)) + zero
        )
        return self._flatten_output(output)

    def _forward_dense(self, inputs, order_weights):
        letters = "abdefghijklmnqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.correlation > len(letters):
            raise ValueError("correlation exceeds the number of einsum indices.")
        output = None
        for order_index in range(self.correlation):
            order = order_index + 1
            indices = letters[:order]
            equation = (
                ",".join(
                    [f"o{indices}p"] + [f"...{index}c" for index in indices] + ["...pc"]
                )
                + "->...oc"
            )
            coefficient = getattr(self, f"generalized_cg_{order}").to(inputs[0])
            contribution = torch.einsum(
                equation, coefficient, *inputs[:order], order_weights[order_index]
            )
            output = contribution if output is None else output + contribution
        return self._flatten_output(output)

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate all correlation orders.

        Parameters
        ----------
        inputs : sequence of torch.Tensor
            Exactly ``correlation`` independent tensors with shape
            ``(..., irreps_in.dim)``. Leading dimensions must broadcast.
        weight : torch.Tensor
            External path weights with shape ``(..., weight_numel)``.
            Leading dimensions broadcast with all inputs.

        Returns
        -------
        torch.Tensor
            Contracted features with shape ``(..., irreps_out.dim)``.
        """
        if len(inputs) != self.correlation:
            raise ValueError(
                f"Expected {self.correlation} independent inputs, got {len(inputs)}."
            )
        for features in inputs:
            if features.ndim < 1 or features.size(-1) != self.irreps_in.dim:
                raise ValueError(
                    f"Input trailing dimension must be {self.irreps_in.dim}."
                )
            if features.is_complex():
                raise TypeError("AsymmetricContraction requires real inputs.")
        if weight.ndim < 1 or weight.size(-1) != self.weight_numel:
            raise ValueError(f"Weight trailing dimension must be {self.weight_numel}.")
        if weight.is_complex():
            raise TypeError("AsymmetricContraction requires real weights.")
        leading_shape = torch.broadcast_shapes(
            *(features.shape[:-1] for features in inputs), weight.shape[:-1]
        )
        inputs = tuple(
            features.expand(*leading_shape, self.irreps_in.dim).reshape(
                *leading_shape, self._base_input_dim, self.num_channels
            )
            for features in inputs
        )
        weight = weight.expand(*leading_shape, self.weight_numel)
        order_weights = tuple(
            weight[..., weight_slice].reshape(
                *leading_shape, num_paths, self.num_channels
            )
            for weight_slice, num_paths in zip(
                self._order_weight_slices, self.order_num_paths
            )
        )
        if self.algorithm == "recursive":
            return self._forward_recursive(inputs, order_weights)
        return self._forward_dense(inputs, order_weights)

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, irreps_out={self.irreps_out}, "
            f"correlation={self.correlation}, algorithm={self.algorithm!r}, "
            f"path_mode={self.path_mode!r}, num_paths={self.num_paths}"
        )
