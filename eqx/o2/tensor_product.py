import math
from typing import Iterator, Optional, Sequence

import torch

from ..backends.torch.tensor_product import TensorProduct as GroupedTensorProduct
from ..backends.torch.tensor_product import instruction_groups
from ..utils.tensor_product import tensor_product_instructions
from .clebsch_gordan import channel_product as couple_channels
from .clebsch_gordan import clebsch_gordan_products, coupling_entries, coupling_type
from .irreps import Irreps, IrrepsLike


class TensorProduct(torch.nn.Module):
    """Tensor product with parametrized O(2) coupling paths.

    Parameters
    ----------
    irreps_in1 : Irreps, str, or sequence
        Representation of the first input.
    irreps_in2 : Irreps, str, or sequence
        Representation of the second input.
    irreps_out : Irreps, str, or sequence
        Requested output representation.
    instructions : sequence of tuple
        Coupling paths written as ``(i_in1, i_in2, i_out, mode, train)`` or
        ``(i_in1, i_in2, i_out, mode, train, path_weight)``. ``"u1u"`` couples
        one second-input channel to matching first-input/output channels;
        ``"uuu"`` couples matching channels; ``"uvw"`` mixes all channels.
        ``train`` selects whether the path has a weight.
    in1_var : sequence of float, optional
        Expected variance for each first-input entry. Used only to calculate
        path normalization. Defaults to one.
    in2_var : sequence of float, optional
        Expected variance for each second-input entry. Defaults to one.
    out_var : sequence of float, optional
        Requested variance for each output entry. Defaults to one.
    irrep_normalization : {"component", "norm", "none"}, optional
        ``"component"`` gives unit output variance for independent unit-variance
        input components. ``"norm"`` rescales by
        ``sqrt(dim_in1 * dim_in2 / dim_out)``. ``"none"`` uses coupling tensors
        with unit squared Frobenius norm.
    path_normalization : {"element", "path", "none"}, optional
        Normalization across paths contributing to the same output entry.
    internal_weights : bool, optional
        If ``True``, trainable weights are stored by the module. The default
        is inferred from ``shared_weights`` and the weighted instructions.
    shared_weights : bool, optional
        Whether one weight vector is shared over all leading dimensions.
    backend : {"torch", "cuda"}, optional
        CUDA evaluates ``uuu`` and ``u1u`` paths with float32 or float64
        CUDA tensors. Unsupported modes raise an error.

    Notes
    -----
    CUDA supports repeated differentiation. Shared-weight gradients use atomic
    reductions and are not bitwise deterministic.

    Examples
    --------
    Couple two order-one inputs into an even scalar, an odd scalar, and an
    order-two output, with independent weights for each path.

    >>> from eqx import o2
    >>> tp = o2.TensorProduct(
    ...     "2x1m", "2x1m", "2x0e + 2x0o + 2x2m",
    ...     [(0, 0, i, "uuu", True) for i in range(3)],
    ...     shared_weights=False,
    ... )
    >>> x = tp.irreps_in1.randn(5, -1)
    >>> y = tp.irreps_in2.randn(5, -1)
    >>> weight = torch.randn(5, tp.weight_numel)
    >>> tp(x, y, weight).shape
    torch.Size([5, 8])
    """

    def __init__(
        self,
        irreps_in1: IrrepsLike,
        irreps_in2: IrrepsLike,
        irreps_out: IrrepsLike,
        instructions: Sequence[tuple],
        in1_var: Optional[Sequence[float]] = None,
        in2_var: Optional[Sequence[float]] = None,
        out_var: Optional[Sequence[float]] = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be 'torch' or 'cuda'.")
        self.backend = backend
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization

        self.instructions = tensor_product_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            in1_var,
            in2_var,
            out_var,
            irrep_normalization,
            path_normalization,
        )
        self.path_order, self.groups = instruction_groups(
            self.instructions,
            [
                coupling_type(
                    self.irreps_in1[p.i_in1].ir,
                    self.irreps_in2[p.i_in2].ir,
                    self.irreps_out[p.i_out].ir,
                )
                for p in self.instructions
            ],
        )
        self.register_buffer(
            "path_scales",
            torch.tensor(
                [self.instructions[i].path_weight for i in self.path_order],
                dtype=torch.float64,
            ),
            persistent=False,
        )
        self.contraction = GroupedTensorProduct(
            [
                tuple((entry.mul, entry.ir.dim) for entry in irreps)
                for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
            ],
            self.groups,
            couple_channels,
            layout="ir_mul",
        )
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = bool(shared_weights) and any(
                instruction.has_weight for instruction in self.instructions
            )
        if internal_weights and not shared_weights:
            raise ValueError("Internal weights require shared_weights=True.")
        self.internal_weights = bool(internal_weights)
        self.shared_weights = bool(shared_weights)
        self.weight_numel = sum(
            math.prod(instruction.path_shape)
            for instruction in self.instructions
            if instruction.has_weight
        )
        self.weight_shape = (self.weight_numel,)
        if self.internal_weights and self.weight_numel > 0:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0))

        offsets = []
        offset = 0
        for instruction in self.instructions:
            if instruction.has_weight:
                size = math.prod(instruction.path_shape)
                offsets.append((offset, size))
                offset += size
            else:
                offsets.append(None)
        self._weight_offsets = tuple(offsets)
        self._weight_sizes = tuple(0 if item is None else item[1] for item in offsets)
        couplings = {}
        for index, ins in enumerate(self.instructions):
            key = ins.i_in1, ins.i_in2, ins.connection_mode != "uvw"
            irreps, paths = couplings.setdefault(key, ([], []))
            ir = self.irreps_out[ins.i_out].ir
            if ir not in irreps:
                irreps.append(ir)
            paths.append((index, irreps.index(ir)))
        self.couplings = tuple(
            (*key, tuple(irreps), tuple(paths))
            for key, (irreps, paths) in couplings.items()
        )
        output_mask = []
        for i_out, ir_mul in enumerate(self.irreps_out):
            connected = any(
                instruction.i_out == i_out and instruction.path_weight != 0
                for instruction in self.instructions
            )
            output_mask.append(
                torch.ones(ir_mul.dim) if connected else torch.zeros(ir_mul.dim)
            )
        self.register_buffer(
            "output_mask",
            torch.cat(output_mask) if output_mask else torch.ones(0),
            persistent=False,
        )
        self.kernel_metadata = None
        multiplicities = {
            mul
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
            for _, mul in irreps
        }
        if backend == "cuda" and all(
            ins.connection_mode != "uvw" for ins in self.instructions
        ):
            channels = next(iter(multiplicities)) if len(multiplicities) == 1 else 0
            slices1 = self.irreps_in1.slices()
            slices2 = self.irreps_in2.slices()
            slices_out = self.irreps_out.slices()
            rows = []
            for ins, weight_slice in zip(self.instructions, self._weight_offsets):
                ir1 = self.irreps_in1[ins.i_in1].ir
                ir2 = self.irreps_in2[ins.i_in2].ir
                ir_out = self.irreps_out[ins.i_out].ir
                for i, j, k, value in coupling_entries(ir1, ir2, ir_out):
                    c1 = self.irreps_in1[ins.i_in1].mul
                    c2 = self.irreps_in2[ins.i_in2].mul
                    co = self.irreps_out[ins.i_out].mul
                    indices = (
                        slices1[ins.i_in1].start + i * c1,
                        slices2[ins.i_in2].start + j * c2,
                        weight_slice[0] if weight_slice else -1,
                        slices_out[ins.i_out].start + k * co,
                    )
                    coefficient = value * ins.path_weight
                    if channels:
                        rows.append(
                            (
                                tuple(v // channels if v >= 0 else -1 for v in indices),
                                coefficient,
                            )
                        )
                    else:
                        rows.append(
                            (
                                indices,
                                coefficient,
                                co,
                                (1, int(c2 != 1 or co == 1), 1, 1),
                            )
                        )
            self.kernel_metadata = repr(
                (
                    channels,
                    (
                        self.irreps_in1.dim // (channels or 1),
                        self.irreps_in2.dim // (channels or 1),
                        self.weight_numel // (channels or 1),
                        self.irreps_out.dim // (channels or 1),
                    ),
                    tuple(rows),
                )
            )

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when internal_weights=False."
                )
            weight = self.weight
        if weight.is_complex():
            raise TypeError("O(2) TensorProduct supports real weights only.")
        if weight.ndim < 1 or weight.size(-1) != self.weight_numel:
            raise ValueError(
                "TensorProduct weight trailing dimension must be "
                f"{self.weight_numel}, got {tuple(weight.shape)}."
            )
        return weight

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate all tensor-product instructions.

        Parameters
        ----------
        input1 : torch.Tensor
            First input with shape ``(..., irreps_in1.dim)`` in flattened
            ``ir_mul`` order.
        input2 : torch.Tensor
            Second input with shape ``(..., irreps_in2.dim)`` in the same layout.
        weight : torch.Tensor, optional
            External flattened weights with trailing shape ``weight_shape``.
            Leading dimensions broadcast with both inputs. Omit when internal
            weights are enabled.

        Returns
        -------
        torch.Tensor
            Coupled features with shape ``(..., irreps_out.dim)`` over the
            broadcast leading shape.
        """
        if input1.is_complex() or input2.is_complex():
            raise TypeError("O(2) TensorProduct supports real inputs only.")
        if input1.ndim < 1 or input1.size(-1) != self.irreps_in1.dim:
            raise ValueError(
                "TensorProduct input1 trailing dimension must be "
                f"{self.irreps_in1.dim}, got {tuple(input1.shape)}."
            )
        if input2.ndim < 1 or input2.size(-1) != self.irreps_in2.dim:
            raise ValueError(
                "TensorProduct input2 trailing dimension must be "
                f"{self.irreps_in2.dim}, got {tuple(input2.shape)}."
            )
        weight = self._get_weights(weight)
        try:
            torch.broadcast_shapes(
                input1.shape[:-1],
                input2.shape[:-1],
                weight.shape[:-1],
            )
        except RuntimeError as error:
            raise ValueError(
                "TensorProduct input and weight batch dimensions do not broadcast."
            ) from error

        if self.backend == "cuda":
            from ..ops.channel_product import tensor_product as cuda_product

            return cuda_product(
                self.kernel_metadata, input1, input2, weight, self.irreps_out.dim
            )

        if self.contraction.indexed:
            return self.contraction(input1, input2, weight, self.path_scales.to(input1))
        return self.forward_paths(input1, input2, weight)

    def forward_paths(self, input1, input2, weight):
        """Reuse angular products when instruction batches are too small to pack."""
        shape = torch.broadcast_shapes(
            input1.shape[:-1], input2.shape[:-1], weight.shape[:-1]
        )
        inputs = [
            [
                block.reshape(*value.shape[:-1], ir.dim, mul)
                for block, (ir, mul) in zip(
                    value.split([entry.dim for entry in irreps], -1), irreps
                )
            ]
            for value, irreps in ((input1, self.irreps_in1), (input2, self.irreps_in2))
        ]
        weights = weight.split(self._weight_sizes, -1)
        outputs = [None] * len(self.irreps_out)
        for i, j, elementwise, irreps, paths in self.couplings:
            values = clebsch_gordan_products(
                inputs[0][i],
                self.irreps_in1[i].ir,
                inputs[1][j],
                self.irreps_in2[j].ir,
                irreps,
                elementwise=elementwise,
            )
            for index, angular in paths:
                ins = self.instructions[index]
                value = values[angular]
                if ins.has_weight:
                    w = weights[index] * ins.path_weight
                    if ins.connection_mode == "uvw":
                        value = torch.einsum(
                            "...duv,...uvw->...dw",
                            value,
                            w.reshape(*weight.shape[:-1], *ins.path_shape),
                        )
                    else:
                        value = value * w.unsqueeze(-2)
                else:
                    value = value * ins.path_weight
                previous = outputs[ins.i_out]
                outputs[ins.i_out] = value if previous is None else previous + value
        zero = (
            input1[..., :0].sum() + input2[..., :0].sum() + weight[..., :0].sum()
            if not self.contraction.complete or not outputs
            else None
        )
        if not outputs:
            return input1.new_zeros((*shape, 0)) + zero
        result = torch.cat(
            [
                (
                    input1.new_zeros((*shape, ir.dim, mul))
                    if value is None
                    else value.expand(*shape, ir.dim, mul)
                ).flatten(-2)
                for value, (ir, mul) in zip(outputs, self.irreps_out)
            ],
            -1,
        )
        return result if zero is None else result + zero

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.path_scales = torch.tensor(
            [self.instructions[i].path_weight for i in self.path_order],
            dtype=torch.float64,
            device="cpu",
        ).to(self.path_scales)
        return self

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return an instruction-shaped view of the weight storage.

        Parameters
        ----------
        instruction : int
            Index into ``instructions``.
        weight : torch.Tensor, optional
            Weight storage to view. The module weight is used when omitted.

        Returns
        -------
        torch.Tensor
            View whose trailing dimensions equal the instruction path shape.
        """
        specification = self.instructions[instruction]
        if not specification.has_weight:
            raise ValueError("The selected instruction has no weights.")
        weight = self._get_weights(weight)
        offset, size = self._weight_offsets[instruction]
        return weight.narrow(-1, offset, size).view(
            *weight.shape[:-1],
            *specification.path_shape,
        )

    def weight_views(
        self,
        weight: Optional[torch.Tensor] = None,
        yield_instruction: bool = False,
    ) -> Iterator:
        """Iterate over the weighted instruction views.

        Parameters
        ----------
        weight : torch.Tensor, optional
            Weight storage to view. The module weight is used when omitted.
        yield_instruction : bool, optional
            If ``True``, also yield the instruction index and metadata.

        Yields
        ------
        torch.Tensor or tuple
            One weight view, optionally with its instruction metadata.
        """
        for index, instruction in enumerate(self.instructions):
            if not instruction.has_weight:
                continue
            view = self.weight_view_for_instruction(index, weight)
            yield (index, instruction, view) if yield_instruction else view

    def __repr__(self) -> str:
        num_paths = sum(math.prod(item.path_shape) for item in self.instructions)
        return (
            f"{self.__class__.__name__}({self.irreps_in1.simplify()} x "
            f"{self.irreps_in2.simplify()} -> {self.irreps_out.simplify()} | "
            f"{num_paths} paths | {self.weight_numel} weights)"
        )
