################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from collections import Counter
from typing import Optional, Sequence, Tuple

import torch

from .irreps import Irrep, IrrepsLike, check_o2_irreps


class Linear(torch.nn.Module):
    """A complete real O(2)-equivariant linear layer.

    Inputs use shape ``(..., irreps_in.dim, channels_in)`` and outputs use
    ``(..., irreps_out.dim, channels_out)``. Paths connect only identical
    irreps. In particular, ``0e`` and ``0o`` never mix, and both real
    components of every positive-order ``m`` block share one channel matrix.

    Args:
        irreps_in: Input :class:`Irreps`.
        irreps_out: Output :class:`Irreps`.
        channels_in: Number of input channels.
        channels_out: Number of output channels. Defaults to ``channels_in``.
        path_mode: ``"uv"`` uses one full input-output channel matrix per
            path. ``"uu"`` requires equal input and output channel counts and
            uses one channel-wise weight per path.
        internal_weights: Store trainable weights in the module. If ``False``,
            weights must be passed to :meth:`forward`.
        bias: Add a trainable bias to every output ``0e`` copy. Complete O(2)
            symmetry forbids bias on ``0o`` and positive-order blocks.
        path_norm: Divide paths entering each output irrep copy by the square
            root of their count.
        path: Optional ``(output_index, input_index)`` entries indexing the
            expanded irrep-copy layouts. By default, all identical-irrep paths
            are included.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        irreps_out: IrrepsLike,
        channels_in: int,
        channels_out: Optional[int] = None,
        *,
        path_mode: str = "uv",
        internal_weights: bool = True,
        bias: bool = True,
        path_norm: bool = True,
        path: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()

        self.irreps_in = check_o2_irreps(irreps_in)
        self.irreps_out = check_o2_irreps(irreps_out)
        if not isinstance(channels_in, int) or isinstance(channels_in, bool):
            raise TypeError("channels_in must be an integer.")
        if channels_out is None:
            channels_out = channels_in
        if not isinstance(channels_out, int) or isinstance(channels_out, bool):
            raise TypeError("channels_out must be an integer.")
        if channels_in < 1 or channels_out < 1:
            raise ValueError("Linear channel counts must be positive.")
        if path_mode not in ("uv", "uu"):
            raise ValueError("path_mode must be 'uv' or 'uu'.")
        if path_mode == "uu" and channels_in != channels_out:
            raise ValueError("path_mode='uu' requires channels_in == channels_out.")
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.path_mode = path_mode
        self.internal_weights = bool(internal_weights)
        self.path_norm = bool(path_norm)

        inputs = self.irreps_in.expanded()
        outputs = self.irreps_out.expanded()
        if path is None:
            paths = tuple(
                (output_index, input_index)
                for output_index, output in enumerate(outputs)
                for input_index, input_irrep in enumerate(inputs)
                if output == input_irrep
            )
        else:
            paths = tuple(path)
        for item in paths:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Each Linear path must be (output_index, input_index).")
            output_index, input_index = item
            if not isinstance(output_index, int) or isinstance(output_index, bool):
                raise TypeError("Linear path indices must be integers.")
            if not isinstance(input_index, int) or isinstance(input_index, bool):
                raise TypeError("Linear path indices must be integers.")
            if not 0 <= output_index < len(outputs):
                raise ValueError(f"Invalid Linear output path index: {output_index}.")
            if not 0 <= input_index < len(inputs):
                raise ValueError(f"Invalid Linear input path index: {input_index}.")
            if outputs[output_index] != inputs[input_index]:
                raise ValueError(
                    "A complete O(2) Linear path must connect identical irreps; "
                    f"got {inputs[input_index]} -> {outputs[output_index]}."
                )
        self.path = paths

        path_counts = Counter(output_index for output_index, _ in paths)
        scales = [
            path_counts[output_index] ** -0.5 if self.path_norm else 1.0
            for output_index, _ in paths
        ]
        self.register_buffer(
            "path_scales",
            torch.tensor(scales, dtype=torch.get_default_dtype()),
            persistent=False,
        )

        paths_by_output = [[] for _ in outputs]
        for path_index, (output_index, input_index) in enumerate(paths):
            paths_by_output[output_index].append((path_index, input_index))
        self._paths_by_output = tuple(tuple(group) for group in paths_by_output)
        self._input_slices = self.irreps_in.expanded_slices()

        input_group_offsets = {}
        input_offset = 0
        for multiplicity, irrep in self.irreps_in:
            if irrep in input_group_offsets:
                input_group_offsets[irrep] = None
            else:
                input_group_offsets[irrep] = (input_offset, multiplicity)
            input_offset += multiplicity

        output_offset = 0
        grouped_paths = []
        path_lookup = {
            (output_index, input_index): path_index
            for path_index, (output_index, input_index) in enumerate(paths)
        }
        for output_multiplicity, output_irrep in self.irreps_out:
            input_group = input_group_offsets.get(output_irrep, False)
            if input_group is False:
                grouped_paths.append((0, 0, 0, output_multiplicity))
                output_offset += output_multiplicity
                continue
            if input_group is None:
                grouped_paths.append(None)
            else:
                input_start, input_multiplicity = input_group
                path_indices = tuple(
                    path_lookup.get((output_index, input_index), -1)
                    for output_index in range(
                        output_offset,
                        output_offset + output_multiplicity,
                    )
                    for input_index in range(
                        input_start,
                        input_start + input_multiplicity,
                    )
                )
                if any(path_index < 0 for path_index in path_indices):
                    grouped_paths.append(None)
                else:
                    first = path_indices[0]
                    if path_indices != tuple(range(first, first + len(path_indices))):
                        grouped_paths.append(None)
                    else:
                        grouped_paths.append(
                            (
                                first,
                                len(path_indices),
                                input_multiplicity,
                                output_multiplicity,
                            )
                        )
            output_offset += output_multiplicity
        self._grouped_paths = tuple(grouped_paths)

        if self.path_mode == "uv":
            self.weight_shape = (len(paths), channels_in, channels_out)
        else:
            self.weight_shape = (len(paths), channels_in)
        self.weight_numel = math.prod(self.weight_shape)
        if self.internal_weights:
            self.weight = torch.nn.Parameter(torch.empty(self.weight_shape))
        else:
            self.register_parameter("weight", None)

        bias_row_by_output = []
        num_biases = 0
        for output in outputs:
            if output.m == 0 and output.p == 1:
                bias_row_by_output.append(num_biases)
                num_biases += 1
            else:
                bias_row_by_output.append(-1)
        self._bias_row_by_output = tuple(bias_row_by_output)
        self.bias_numel = num_biases * channels_out
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(num_biases, channels_out))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None and self.weight.numel() > 0:
            bound = 1.0 / math.sqrt(self.channels_in) if self.path_mode == "uv" else 1.0
            torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _resolve_weight(
        self,
        weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.internal_weights:
            if weight is not None:
                raise ValueError(
                    "Do not pass weight when Linear uses internal weights."
                )
            weight = self.weight
        elif weight is None:
            raise ValueError("Linear requires external weight.")
        if weight is None:
            raise RuntimeError("Linear weight resolution failed.")
        if weight.is_complex():
            raise TypeError("Complete O(2) Linear supports real weights only.")
        weight_ndim = len(self.weight_shape)
        if (
            weight.ndim < weight_ndim
            or tuple(weight.shape[-weight_ndim:]) != self.weight_shape
        ):
            raise ValueError(
                "Linear weight trailing shape must be "
                f"{self.weight_shape}, got {tuple(weight.shape)}."
            )
        return weight

    def forward(
        self,
        input: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input.is_complex():
            raise TypeError("Complete O(2) Linear supports real inputs only.")
        expected_input_shape = (self.irreps_in.dim, self.channels_in)
        if input.ndim < 2 or tuple(input.shape[-2:]) != expected_input_shape:
            raise ValueError(
                "Linear input trailing shape must be "
                f"{expected_input_shape}, got {tuple(input.shape)}."
            )
        weight = self._resolve_weight(weight)
        weight_ndim = len(self.weight_shape)
        try:
            leading_shape = torch.broadcast_shapes(
                input.shape[:-2],
                weight.shape[:-weight_ndim],
            )
        except RuntimeError as error:
            raise ValueError(
                "Linear input and external weight batch dimensions do not broadcast."
            ) from error

        output_blocks = []
        zero_dependency = input.sum() * 0 + weight.sum() * 0
        outputs = self.irreps_out.expanded()
        for output_index, output in enumerate(outputs):
            contributions = []
            for path_index, input_index in self._paths_by_output[output_index]:
                input_block = input[..., self._input_slices[input_index], :]
                if self.path_mode == "uv":
                    contribution = torch.matmul(
                        input_block,
                        weight[..., path_index, :, :],
                    )
                else:
                    contribution = input_block * weight[..., path_index, :].unsqueeze(
                        -2
                    )
                contributions.append(contribution * self.path_scales[path_index])
            if contributions:
                output_block = sum(contributions[1:], contributions[0])
            else:
                output_block = input.new_zeros(
                    leading_shape + (output.dim, self.channels_out)
                )
                output_block = output_block + zero_dependency

            bias_row = self._bias_row_by_output[output_index]
            if self.bias is not None and bias_row >= 0:
                output_block = output_block + self.bias[bias_row]
            output_blocks.append(output_block)

        if output_blocks:
            return torch.cat(output_blocks, dim=-2)
        output = input.new_empty(leading_shape + (0, self.channels_out))
        return output + zero_dependency

    def forward_grouped(
        self,
        input_blocks: Sequence[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Apply one dense UV contraction per contiguous O(2) irrep group.

        Each input block has shape ``(..., irrep.dim, mul * channels_in)``.
        The multiplicity and channel axes are already contiguous, so every
        ``0e``, ``0o``, or positive-order block is evaluated by one large
        matrix multiplication instead of one multiplication per path.
        """
        if len(input_blocks) != len(self.irreps_in):
            raise ValueError("Expected one input block per O(2) irrep group.")
        if self.path_mode != "uv":
            raise ValueError("Grouped Linear is only available in path_mode='uv'.")
        if any(specification is None for specification in self._grouped_paths):
            raise ValueError(
                "Grouped Linear requires regrouped irreps and fully connected paths."
            )

        weight = self._resolve_weight(weight)
        input_by_irrep = {
            irrep: (multiplicity, block)
            for (multiplicity, irrep), block in zip(self.irreps_in, input_blocks)
        }
        output_blocks = []
        for (
            (output_multiplicity, output_irrep),
            specification,
        ) in zip(self.irreps_out, self._grouped_paths):
            if specification is None:
                raise RuntimeError("Grouped Linear path resolution failed.")
            first_path, num_paths, input_multiplicity, _ = specification
            if num_paths == 0:
                reference = input_blocks[0]
                output_block = reference.new_zeros(
                    *reference.shape[:-2],
                    output_irrep.dim,
                    output_multiplicity * self.channels_out,
                )
                output_block = output_block + reference.sum() * 0 + weight.sum() * 0
                if self.bias is not None and output_irrep == Irrep("0e"):
                    output_block = output_block + self.bias.reshape(
                        1,
                        output_multiplicity * self.channels_out,
                    )
                output_blocks.append(output_block)
                continue
            input_group = input_by_irrep.get(output_irrep)
            if input_group is None:
                raise RuntimeError("Grouped Linear input resolution failed.")
            observed_multiplicity, input_block = input_group
            if observed_multiplicity != input_multiplicity:
                raise RuntimeError("Grouped Linear multiplicity resolution failed.")
            expected_shape = (
                output_irrep.dim,
                input_multiplicity * self.channels_in,
            )
            if tuple(input_block.shape[-2:]) != expected_shape:
                raise ValueError(
                    "Grouped Linear input block trailing shape must be "
                    f"{expected_shape}, got {tuple(input_block.shape)}."
                )

            path_weight = weight.narrow(-3, first_path, num_paths)
            path_weight = path_weight.reshape(
                *path_weight.shape[:-3],
                output_multiplicity,
                input_multiplicity,
                self.channels_in,
                self.channels_out,
            )
            scales = self.path_scales.narrow(0, first_path, num_paths).reshape(
                output_multiplicity,
                input_multiplicity,
                1,
                1,
            )
            path_weight = path_weight * scales
            matrix = path_weight.permute(
                *range(path_weight.ndim - 4),
                -3,
                -2,
                -4,
                -1,
            ).reshape(
                *path_weight.shape[:-4],
                input_multiplicity * self.channels_in,
                output_multiplicity * self.channels_out,
            )
            if matrix.ndim == 2:
                output_block = torch.mm(
                    input_block.reshape(-1, matrix.size(0)),
                    matrix,
                ).reshape(
                    *input_block.shape[:-2],
                    output_irrep.dim,
                    output_multiplicity * self.channels_out,
                )
            else:
                output_block = torch.matmul(input_block, matrix)

            if self.bias is not None and output_irrep == Irrep("0e"):
                output_block = output_block + self.bias.reshape(
                    1,
                    output_multiplicity * self.channels_out,
                )
            output_blocks.append(output_block)
        return tuple(output_blocks)

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, irreps_out={self.irreps_out}, "
            f"channels_in={self.channels_in}, channels_out={self.channels_out}, "
            f"path_mode={self.path_mode}, "
            f"internal_weights={self.internal_weights}, "
            f"bias={self.bias is not None}, num_paths={len(self.path)}"
        )
