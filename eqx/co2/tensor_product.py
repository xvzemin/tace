"""Cartesian O(2) products with independent channel paths."""

import math

import torch

from ..backends.torch.tensor_product import TensorProduct as GroupedTensorProduct
from ..backends.torch.tensor_product import instruction_groups
from ..utils.tensor_product import tensor_product_instructions
from .basis import Projector
from .irreps import Irrep, Irreps


def quarter_turn(tensor):
    """Apply the canonical quarter turn to one Cartesian tensor index."""
    value = tensor.reshape(*tensor.shape[:-1], 2, tensor.shape[-1] // 2)
    return torch.stack((-value[..., 1, :], value[..., 0, :]), -2).flatten(-2)


def cartesian_product(x, y, ir1, ir2, ir_out):
    """Return a component-normalized, unprojected Cartesian coupling.

    Parameters
    ----------
    x, y : torch.Tensor
        STF tensors with broadcastable leading dimensions and trailing
        Cartesian dimensions ir1.dim and ir2.dim.
    ir1, ir2, ir_out : Irrep
        An allowed input/output irrep triple.

    Returns
    -------
    torch.Tensor
        Raw Cartesian output. Sum-order outputs require STF projection.
    """
    a, b, c = ir1, ir2, ir_out
    if a.m == 0:
        return x * (quarter_turn(y) if a.p == -1 and b.m else y)
    if b.m == 0:
        return (quarter_turn(x) if b.p == -1 else x) * y
    if c.m == a.m + b.m:
        return (x.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2)
    if c.m == 0:
        if c.p == -1:
            y = quarter_turn(y)
        return (x * y).sum(-1, keepdim=True) / math.sqrt(2)
    if a.m > b.m:
        return (x.reshape(*x.shape[:-1], c.dim, b.dim) @ y.unsqueeze(-1)).squeeze(-1)
    return (x.unsqueeze(-2) @ y.reshape(*y.shape[:-1], a.dim, c.dim)).squeeze(-2)


def coupling_entries(a, b, c):
    """Return nonzero delta/epsilon entries before STF projection."""
    if not a.m or not b.m:
        tensor, scalar = (b, a) if not a.m else (a, b)
        half = tensor.dim // 2
        return tuple(
            (
                (0, (i + half) % tensor.dim, i, -1.0 if i < half else 1.0)
                if not a.m
                else ((i + half) % tensor.dim, 0, i, -1.0 if i < half else 1.0)
            )
            if scalar.p == -1 and tensor.m
            else ((0, i, i, 1.0) if not a.m else (i, 0, i, 1.0))
            for i in range(tensor.dim)
        )
    if c.m == a.m + b.m:
        return tuple(
            (i, j, i * b.dim + j, 1.0) for i in range(a.dim) for j in range(b.dim)
        )
    if not c.m:
        half = a.dim // 2
        return tuple(
            (
                i,
                (i + half) % a.dim if c.p == -1 else i,
                0,
                (-1.0 if c.p == -1 and i < half else 1.0) / math.sqrt(2),
            )
            for i in range(a.dim)
        )
    if a.m > b.m:
        return tuple(
            (i * b.dim + j, j, i, 1.0) for i in range(c.dim) for j in range(b.dim)
        )
    return tuple((i, i * c.dim + j, j, 1.0) for i in range(a.dim) for j in range(c.dim))


class TensorProduct(torch.nn.Module):
    """Couple two-dimensional Cartesian irreps.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        Cartesian representations in flattened ``mul_ir`` layout.
    instructions : sequence of tuple
        ``(i_in1, i_in2, i_out, mode, has_weight[, path_weight])`` entries.
        Modes are ``u1u``, ``uuu``, and ``uvw``.
    in1_var, in2_var, out_var : sequence of float, optional
        Variance of each irreducible entry. Defaults to one.
    irrep_normalization : {"component", "norm", "none"}, optional
        Normalization of the independent irreducible coordinates.
    path_normalization : {"element", "path", "none"}, optional
        Normalization across paths feeding the same output entry.
    internal_weights, shared_weights : bool, optional
        Store weights and share them over leading feature dimensions.
    project : bool, optional
        Apply STF projection. False defers it until after linear operations.
    backend : {"torch", "cuda"}, optional
        CUDA evaluates ``uuu`` and ``u1u`` paths and STF projection using
        float32 or float64 CUDA tensors. Unsupported modes raise an error.

    Notes
    -----
    Inputs must be symmetric traceless. Unprojected outputs must be projected
    before another tensor product or nonlinearity.

    Examples
    --------
    >>> from eqx import co2
    >>> tp = co2.TensorProduct(
    ...     "2x1m", "2x1m", "2x2m", [(0, 0, 0, "uuu", True)]
    ... )
    >>> x, y = torch.randn(5, 4), torch.randn(5, 4)
    >>> tp(x, y).shape
    torch.Size([5, 8])
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        in1_var=None,
        in2_var=None,
        out_var=None,
        irrep_normalization="component",
        path_normalization="element",
        internal_weights=None,
        shared_weights=None,
        *,
        project=True,
        backend="torch",
    ):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be 'torch' or 'cuda'.")
        self.backend = backend
        self.irreps_in1, self.irreps_in2, self.irreps_out = map(
            Irreps, (irreps_in1, irreps_in2, irreps_out)
        )
        self.input_dims = (self.irreps_in1.dim, self.irreps_in2.dim)
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
            dimensions=tuple(
                tuple(ir.circular_dim for _, ir in irreps)
                for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
            ),
        )
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.shared_weights = True if shared_weights is None else shared_weights
        self.internal_weights = (
            self.shared_weights and any(ins.has_weight for ins in self.instructions)
            if internal_weights is None
            else internal_weights
        )
        if self.internal_weights and not self.shared_weights:
            raise ValueError("Internal weights require shared_weights=True.")
        self.weight_numel = sum(
            math.prod(ins.path_shape) for ins in self.instructions if ins.has_weight
        )
        self.weight_shape = (self.weight_numel,)
        if self.internal_weights and self.weight_numel:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0))
        self.project = project
        self.paths = tuple(
            (
                self.irreps_in1[ins.i_in1].ir,
                self.irreps_in2[ins.i_in2].ir,
                self.irreps_out[ins.i_out].ir,
            )
            for ins in self.instructions
        )
        self.path_order, self.groups = instruction_groups(
            self.instructions,
            [tuple(Irrep(ir.m, ir.p) for ir in path) for path in self.paths],
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
            cartesian_product,
        )
        self.kernel_metadata = None
        if backend == "cuda" and all(
            p.connection_mode != "uvw" for p in self.instructions
        ):
            rows, offset = [], 0
            sections = tuple(
                irs.slices()
                for irs in (self.irreps_in1, self.irreps_in2, self.irreps_out)
            )
            for ins, (a, b, c) in zip(self.instructions, self.paths):
                width = self.irreps_out[ins.i_out].mul
                steps = (a.dim, b.dim if ins.connection_mode != "u1u" else 0, 1, c.dim)
                for i, j, k, coefficient in coupling_entries(a, b, c):
                    rows.append(
                        (
                            (
                                sections[0][ins.i_in1].start + i,
                                sections[1][ins.i_in2].start + j,
                                offset if ins.has_weight else -1,
                                sections[2][ins.i_out].start + k,
                            ),
                            coefficient * ins.path_weight,
                            width,
                            steps,
                        )
                    )
                if ins.has_weight:
                    offset += math.prod(ins.path_shape)
            self.kernel_metadata = repr(
                (
                    0,
                    (*self.input_dims, self.weight_numel, self.irreps_out.dim),
                    tuple(rows),
                )
            )
        self.projection = (
            Projector(self.irreps_out, backend=backend)
            if project
            else torch.nn.Identity()
        )
        connected = {ins.i_out for ins in self.instructions if ins.path_weight}
        self.register_buffer(
            "output_mask",
            (
                torch.cat(
                    [
                        torch.full((entry.dim,), float(i in connected))
                        for i, entry in enumerate(self.irreps_out)
                    ]
                )
                if self.irreps_out
                else torch.empty(0)
            ),
            persistent=False,
        )

    def forward(self, x, y, weight=None):
        """Evaluate the tensor product of symmetric traceless inputs.

        Parameters
        ----------
        x, y : torch.Tensor
            Features of shape ``(..., irreps_in1.dim)`` and
            ``(..., irreps_in2.dim)`` in ``mul_ir`` layout.
        weight : torch.Tensor, optional
            Flattened path weights of shape ``(..., weight_numel)``.
            Leading dimensions broadcast with both inputs. Defaults to
            the internal parameters.

        Returns
        -------
        torch.Tensor
            Features of shape ``(..., irreps_out.dim)`` in ``mul_ir`` layout.
        """
        if x.shape[-1] != self.input_dims[0] or y.shape[-1] != self.input_dims[1]:
            raise ValueError("Input feature dimensions do not match the irreps.")
        weight = self.weight if weight is None else weight
        if weight.ndim < 1 or weight.shape[-1] != self.weight_numel:
            raise ValueError("Weights must have trailing size weight_numel.")
        if x.is_complex() or y.is_complex() or weight.is_complex():
            raise TypeError("Cartesian O(2) products require real inputs and weights.")
        torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        if self.backend == "cuda":
            from ..ops.channel_product import tensor_product as cuda_product

            output = cuda_product(
                self.kernel_metadata, x, y, weight, self.irreps_out.dim
            )
            return self.projection(output)
        output = self.contraction(x, y, weight, self.path_scales.to(x))
        return self.projection(output)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.path_scales = torch.tensor(
            [self.instructions[i].path_weight for i in self.path_order],
            dtype=torch.float64,
            device="cpu",
        ).to(self.path_scales)
        return self

    def weight_view_for_instruction(self, instruction, weight=None):
        """Return a view of one instruction's weights."""
        ins = self.instructions[instruction]
        if not ins.has_weight:
            raise ValueError("The instruction has no weights.")
        weight = self.weight if weight is None else weight
        offset = sum(
            math.prod(p.path_shape)
            for p in self.instructions[:instruction]
            if p.has_weight
        )
        return weight[..., offset : offset + math.prod(ins.path_shape)].reshape(
            *weight.shape[:-1], *ins.path_shape
        )

    def weight_views(self, weight=None, yield_instruction=False):
        """Iterate over weights in instruction order."""
        for i, ins in enumerate(self.instructions):
            if ins.has_weight:
                value = self.weight_view_for_instruction(i, weight)
                yield (i, ins, value) if yield_instruction else value

    def extra_repr(self):
        return f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} | {self.weight_numel} weights, project={self.project}"


class FullyConnectedTensorProduct(TensorProduct):
    """Connect every allowed irrep triple with independent uvw weights.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        Input and output representations.
    **kwargs
        TensorProduct weight, normalization, and projection options.
    """

    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        a, b, c = map(Irreps, (irreps_in1, irreps_in2, irreps_out))
        instructions = [
            (i, j, k, "uvw", True)
            for i, (_, ir) in enumerate(a)
            for j, (_, jr) in enumerate(b)
            for k, (_, kr) in enumerate(c)
            if kr in ir * jr
        ]
        super().__init__(a, b, c, instructions, **kwargs)


class ElementwiseTensorProduct(TensorProduct):
    """Couple corresponding channels without trainable weights.

    Parameters
    ----------
    irreps_in1, irreps_in2 : Irreps or str
        Representations with the same total channel count.
    filter_ir_out : sequence of Irrep or str, optional
        Allowed output irrep types. Defaults to every allowed coupling.
    **kwargs
        TensorProduct normalization and projection options.
    """

    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, **kwargs):
        a, b = [list(Irreps(irreps).simplify()) for irreps in (irreps_in1, irreps_in2)]
        if sum(mul for mul, _ in a) != sum(mul for mul, _ in b):
            raise ValueError("Elementwise products require equal channel counts.")
        i = 0
        while i < len(a):
            u, ir = a[i]
            v, jr = b[i]
            if u < v:
                b[i : i + 1] = [(u, jr), (v - u, jr)]
            if v < u:
                a[i : i + 1] = [(v, ir), (u - v, ir)]
            i += 1
        keep = None if filter_ir_out is None else set(map(Irrep, filter_ir_out))
        output, instructions = [], []
        for i, ((mul, ir), (_, jr)) in enumerate(zip(a, b)):
            for kr in ir * jr:
                if keep is None or kr in keep:
                    instructions.append((i, i, len(output), "uuu", False))
                    output.append((mul, kr))
        super().__init__(a, b, output, instructions, **kwargs)
