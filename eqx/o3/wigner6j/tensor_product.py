"""Three-input tensor products evaluated by Racah recoupling."""

import math
from functools import lru_cache

import torch
from e3nn import o3


@lru_cache(maxsize=None)
def recoupling_coefficient(l1, l2, l3, l12, l13, l_out):
    """Return the coefficient between two products of unit-norm Wigner-3j tensors."""
    from sympy.physics.wigner import wigner_6j

    return (
        (-1) ** (l2 + l3 + l12 + l13)
        * (2 * l13 + 1)
        * float(wigner_6j(l2, l1, l12, l3, l_out, l13))
    )


def layout_indices(irreps):
    """Return the permutation from mul_ir to ir_mul storage."""
    indices = [
        torch.arange(section.start, section.stop, device="cpu")
        .reshape(mul, ir.dim)
        .T.flatten()
        for (mul, ir), section in zip(irreps, irreps.slices())
    ]
    return torch.cat(indices) if indices else torch.empty(0, dtype=torch.long)


class Wigner6jTensorProduct(torch.nn.Module):
    r"""Evaluate ``(x1 x x2) x x3`` in the order ``(x1 x x3) x x2``.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_in3 : e3nn.o3.Irreps or str
        Input representations in ``mul_ir`` layout.
    irreps_mid, irreps_out : e3nn.o3.Irreps or str
        Intermediate and output representations of the standard coupling tree.
    instructions1, instructions2 : list of tuple
        e3nn instructions for the first and second standard tensor products.
        Connection mode ``uvu`` supports arbitrary input multiplicities.
    irrep_normalization : {"component", "norm", "none"}, optional
        Normalization of both standard tensor products. Defaults to "component".
    path_normalization : {"element", "path", "none"}, optional
        Path normalization of both standard tensor products. Defaults to "element".
    internal_weights, shared_weights : bool, optional
        Store trainable weights and share them over leading dimensions.
        Both default to true. Internal weights require shared weights.
    backend : {"cuda", "torch"}, optional
        CUDA contractions or PyTorch operations with automatic differentiation.
        Defaults to "cuda"; CPU inputs use PyTorch.

    Notes
    -----
    Each weight vector retains the ordering and normalization of its standard
    tensor product. Repeated irreps and paths retain independent channels.

    Examples
    --------
    >>> from eqx import o3
    >>> tp = o3.Wigner6jTensorProduct(
    ...     "2x1o", "1x1o", "1x1o", "2x2e", "2x1o",
    ...     [(0, 0, 0, "uvu", True)], [(0, 0, 0, "uvu", True)],
    ...     backend="torch",
    ... )
    >>> x1, x2, x3 = (torch.randn(5, dim) for dim in (6, 3, 3))
    >>> torch.testing.assert_close(tp(x1, x2, x3), tp.forward_reference(x1, x2, x3))
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_in3,
        irreps_mid,
        irreps_out,
        instructions1,
        instructions2,
        *,
        irrep_normalization="component",
        path_normalization="element",
        internal_weights=None,
        shared_weights=None,
        backend="cuda",
    ):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be cuda or torch.")
        self.backend = backend
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_in3 = o3.Irreps(irreps_in3)
        self.irreps_mid = o3.Irreps(irreps_mid)
        self.irreps_out = o3.Irreps(irreps_out)
        self.input_dims = (
            self.irreps_in1.dim,
            self.irreps_in2.dim,
            self.irreps_in3.dim,
        )
        self.output_dim = self.irreps_out.dim
        self.shared_weights = True if shared_weights is None else shared_weights
        self.internal_weights = (
            self.shared_weights if internal_weights is None else internal_weights
        )
        if self.internal_weights and not self.shared_weights:
            raise ValueError("Internal weights must be shared.")
        options = dict(
            internal_weights=False,
            shared_weights=self.shared_weights,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
        )
        self.tp1 = o3.TensorProduct(
            self.irreps_in1, self.irreps_in2, self.irreps_mid, instructions1, **options
        )
        self.tp2 = o3.TensorProduct(
            self.irreps_mid, self.irreps_in3, self.irreps_out, instructions2, **options
        )
        if any(
            ins.connection_mode != "uvu"
            for tp in (self.tp1, self.tp2)
            for ins in tp.instructions
        ):
            raise NotImplementedError(
                "Wigner6jTensorProduct supports uvu instructions."
            )
        self.instructions1 = self.tp1.instructions
        self.instructions2 = self.tp2.instructions
        self.weight_numel1 = self.tp1.weight_numel
        self.weight_numel2 = self.tp2.weight_numel
        for number, size in ((1, self.weight_numel1), (2, self.weight_numel2)):
            if self.internal_weights:
                self.register_parameter(
                    f"weight{number}", torch.nn.Parameter(torch.randn(size))
                )
            else:
                self.register_buffer(
                    f"weight{number}", torch.empty(0), persistent=False
                )

        offsets = []
        for tp in (self.tp1, self.tp2):
            start, entries = 0, []
            for ins in tp.instructions:
                entries.append(start if ins.has_weight else tp.weight_numel)
                if ins.has_weight:
                    start += math.prod(ins.path_shape)
            offsets.append(entries)

        split3, split_offsets = [], []
        for mul, ir in self.irreps_in3:
            split_offsets.append(len(split3))
            split3.extend([(1, ir)] * mul)
        self.irreps_split3 = o3.Irreps(split3)
        intermediates, instructions13, instructions132 = [], [], []
        lookup, routes = {}, []
        indices1, indices2 = [], []
        for j, ins2 in enumerate(self.tp2.instructions):
            mul3, ir3 = self.irreps_in3[ins2.i_in2]
            ir_out = self.irreps_out[ins2.i_out].ir
            for i, ins1 in enumerate(self.tp1.instructions):
                if ins1.i_out != ins2.i_in1:
                    continue
                mul1, ir1 = self.irreps_in1[ins1.i_in1]
                mul2, ir2 = self.irreps_in2[ins1.i_in2]
                ir12 = self.irreps_mid[ins1.i_out].ir
                if (
                    not mul1
                    or not mul2
                    or not mul3
                    or not ins1.path_weight
                    or not ins2.path_weight
                ):
                    continue
                for ir13 in ir1 * ir3:
                    if ir_out not in ir13 * ir2:
                        continue
                    coefficient = recoupling_coefficient(
                        ir1.l, ir2.l, ir3.l, ir12.l, ir13.l, ir_out.l
                    )
                    if coefficient == 0.0:
                        continue
                    for channel in range(mul3):
                        key = (ins1.i_in1, ins2.i_in2, channel, ir13)
                        if key not in lookup:
                            lookup[key] = len(intermediates)
                            intermediates.append((mul1, ir13))
                            instructions13.append(
                                (
                                    ins1.i_in1,
                                    split_offsets[ins2.i_in2] + channel,
                                    lookup[key],
                                    "uvu",
                                    False,
                                )
                            )
                        instructions132.append(
                            (lookup[key], ins1.i_in2, ins2.i_out, "uvu", True)
                        )
                        routes.append(
                            (
                                lookup[key],
                                coefficient * ins1.path_weight * ins2.path_weight,
                            )
                        )
                        for u in range(mul1):
                            for v in range(mul2):
                                indices1.append(
                                    offsets[0][i] + u * mul2 + v
                                    if ins1.has_weight
                                    else self.weight_numel1
                                )
                                indices2.append(
                                    offsets[1][j] + u * mul3 + channel
                                    if ins2.has_weight
                                    else self.weight_numel2
                                )

        self.irreps_recoupled = o3.Irreps(intermediates)
        self.tp13 = o3.TensorProduct(
            self.irreps_in1,
            self.irreps_split3,
            self.irreps_recoupled,
            instructions13,
            internal_weights=False,
            shared_weights=True,
            path_normalization="none",
        )
        self.tp132 = o3.TensorProduct(
            self.irreps_recoupled,
            self.irreps_in2,
            self.irreps_out,
            instructions132,
            internal_weights=False,
            shared_weights=False,
            path_normalization="none",
        )
        coefficients = []
        for ins, (index, coefficient) in zip(self.tp132.instructions, routes):
            factor = coefficient / (
                self.tp13.instructions[index].path_weight * ins.path_weight
            )
            coefficients.extend([factor] * math.prod(ins.path_shape))
        self.coefficient_values = tuple(coefficients)
        self.register_buffer(
            "coefficients",
            torch.tensor(coefficients, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "weight_index1", torch.tensor(indices1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "weight_index2", torch.tensor(indices2, dtype=torch.long), persistent=False
        )
        for name, irreps in (
            ("input1_index", self.irreps_in1),
            ("input2_index", self.irreps_in2),
            ("input3_index", self.irreps_split3),
            ("output_index", self.irreps_out),
        ):
            index = layout_indices(irreps)
            if name == "output_index":
                index = index.argsort()
            self.register_buffer(name, index, persistent=False)
        from ..cgtp.convolution import O3TensorProductConv

        self.conv13 = O3TensorProductConv(self.tp13, batch_paths=True)
        self.conv132 = O3TensorProductConv(self.tp132, batch_paths=True)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.coefficients = self.coefficients.new_tensor(self.coefficient_values)
        return self

    def get_weights(self, weight1, weight2):
        """Resolve the two standard tensor-product weight vectors."""
        weights = []
        for number, weight, size in (
            (1, weight1, self.weight_numel1),
            (2, weight2, self.weight_numel2),
        ):
            if weight is None:
                if size and not self.internal_weights:
                    raise ValueError(f"External weight{number} is required.")
                weight = getattr(self, f"weight{number}")
            if weight.ndim < 1 or weight.shape[-1] != size:
                raise ValueError(f"weight{number} must end with weight_numel{number}.")
            if self.shared_weights and weight.ndim != 1:
                raise ValueError("Shared weights must be one-dimensional.")
            weights.append(weight)
        return tuple(weights)

    def combine_weights(self, weight1, weight2):
        """Apply fixed recoupling coefficients to products of standard weights."""
        weight1 = torch.cat((weight1, weight1.new_ones(weight1.shape[:-1] + (1,))), -1)
        weight2 = torch.cat((weight2, weight2.new_ones(weight2.shape[:-1] + (1,))), -1)
        return (
            weight1.index_select(-1, self.weight_index1)
            * weight2.index_select(-1, self.weight_index2)
            * self.coefficients.to(weight1)
        )

    def couple_inputs(self, x1, x3, backend=None):
        """Couple node inputs in ``ir_mul`` for CUDA or ``mul_ir`` for PyTorch."""
        backend = self.backend if backend is None else backend
        if backend != "cuda" or not x1.is_cuda:
            return self.tp13(x1, x3)
        return self.conv13(
            x1.index_select(-1, self.input1_index),
            x3.index_select(-1, self.input3_index),
            x1.new_empty((1, 0)),
            x1.new_empty((0, 0)),
            None,
        )

    def forward(self, x1, x2, x3, weight1=None, weight2=None):
        """Evaluate the recoupled tensor product.

        Parameters
        ----------
        x1, x2, x3 : torch.Tensor
            Features with trailing sizes ``irreps_in1.dim``, ``irreps_in2.dim``,
            and ``irreps_in3.dim`` in ``mul_ir`` layout. Leading dimensions
            broadcast.
        weight1, weight2 : torch.Tensor, optional
            Weights of the two standard products, with trailing sizes
            ``weight_numel1`` and ``weight_numel2``. Shared weights are
            one-dimensional; otherwise their leading dimensions broadcast
            with the inputs. Defaults to the internal parameters.

        Returns
        -------
        torch.Tensor
            Features of shape ``(..., irreps_out.dim)`` in ``mul_ir`` layout.
        """
        weight1, weight2 = self.get_weights(weight1, weight2)
        for value, width in zip((x1, x2, x3), self.input_dims):
            if value.ndim < 1 or value.shape[-1] != width:
                raise ValueError("Input widths must match their irreps.")
        shape = torch.broadcast_shapes(
            *(v.shape[:-1] for v in (x1, x2, x3, weight1, weight2))
        )
        values = [
            v.expand(shape + (v.shape[-1],)).reshape(math.prod(shape), v.shape[-1])
            for v in (x1, x2, x3)
        ]
        x1, x2, x3 = values
        weight = (
            self.combine_weights(weight1, weight2)
            .expand(shape + (self.tp132.weight_numel,))
            .reshape(math.prod(shape), self.tp132.weight_numel)
        )
        intermediate = self.couple_inputs(x1, x3)
        if self.backend == "cuda" and x1.is_cuda:
            output = self.conv132(
                intermediate,
                x2.index_select(-1, self.input2_index),
                weight,
                weight.new_empty((0, self.tp132.weight_numel)),
                None,
            ).index_select(-1, self.output_index)
        else:
            output = self.tp132(intermediate, x2, weight)
        return output.reshape(shape + (self.output_dim,))

    def forward_reference(self, x1, x2, x3, weight1=None, weight2=None):
        """Evaluate the two standard tensor products without recoupling."""
        weight1, weight2 = self.get_weights(weight1, weight2)
        return self.tp2(self.tp1(x1, x2, weight1), x3, weight2)

    def extra_repr(self):
        return (
            f"({self.irreps_in1} x {self.irreps_in2}) x {self.irreps_in3} -> {self.irreps_out}, "
            f"{self.weight_numel1} + {self.weight_numel2} weights, backend={self.backend!r}"
        )
