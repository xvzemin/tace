"""Cartesian O(2) products with independent channel paths."""

import math

import torch

from .. import o2
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


class TensorProduct(torch.nn.Module):
    """Couple two-dimensional Cartesian irreps.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        Cartesian representations in flattened mul_ir layout.
    instructions : sequence of tuple
        (i_in1, i_in2, i_out, mode, has_weight[, path_weight]) entries.
        Modes are u1u, uuu, and uvw.
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
    ):
        super().__init__()
        self.irreps_in1, self.irreps_in2, self.irreps_out = map(
            Irreps, (irreps_in1, irreps_in2, irreps_out)
        )
        self.input_dims = (self.irreps_in1.dim, self.irreps_in2.dim)
        self.input_paths = tuple(
            tuple(
                (mul, ir.dim, section)
                for (mul, ir), section in zip(irreps, irreps.slices())
            )
            for irreps in (self.irreps_in1, self.irreps_in2)
        )
        self.output_shapes = tuple((mul, ir.dim) for mul, ir in self.irreps_out)
        metadata = o2.TensorProduct(
            self.irreps_in1.circular(),
            self.irreps_in2.circular(),
            self.irreps_out.circular(),
            instructions,
            in1_var,
            in2_var,
            out_var,
            irrep_normalization,
            path_normalization,
            internal_weights,
            shared_weights,
        )
        for name in (
            "instructions",
            "weight_numel",
            "weight_shape",
            "internal_weights",
            "shared_weights",
            "irrep_normalization",
            "path_normalization",
        ):
            setattr(self, name, getattr(metadata, name))
        if isinstance(metadata.weight, torch.nn.Parameter):
            self.weight = metadata.weight
        else:
            self.register_buffer("weight", metadata.weight)
        self.project = project
        self.paths = tuple(
            (
                self.irreps_in1[ins.i_in1].ir,
                self.irreps_in2[ins.i_in2].ir,
                self.irreps_out[ins.i_out].ir,
            )
            for ins in self.instructions
        )
        self.projection = Projector(self.irreps_out) if project else torch.nn.Identity()
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
        """Evaluate STF inputs and flattened weights over broadcast batch axes."""
        if x.shape[-1] != self.input_dims[0] or y.shape[-1] != self.input_dims[1]:
            raise ValueError("Input feature dimensions do not match the irreps.")
        weight = self.weight if weight is None else weight
        if weight.ndim < 1 or weight.shape[-1] != self.weight_numel:
            raise ValueError("Weights must have trailing size weight_numel.")
        if x.is_complex() or y.is_complex() or weight.is_complex():
            raise TypeError("Cartesian O(2) products require real inputs and weights.")
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        zero = x[..., :0].sum() + y[..., :0].sum() + weight[..., :0].sum()
        outputs = [
            x.new_zeros((*shape, mul, dim)) + zero for mul, dim in self.output_shapes
        ]
        inputs = [
            [
                value[..., section].reshape(*value.shape[:-1], mul, dim)
                for mul, dim, section in paths
            ]
            for value, paths in zip((x, y), self.input_paths)
        ]
        offset = 0
        for ins, (a, b, c) in zip(self.instructions, self.paths):
            left, right = inputs[0][ins.i_in1], inputs[1][ins.i_in2]
            if ins.connection_mode == "uvw":
                left, right = left.unsqueeze(-2), right.unsqueeze(-3)
            value = cartesian_product(left, right, a, b, c)
            if ins.has_weight:
                size = math.prod(ins.path_shape)
                w = weight[..., offset : offset + size].reshape(
                    *weight.shape[:-1], *ins.path_shape
                )
                offset += size
                value = (
                    torch.einsum("...uvd,...uvw->...wd", value, w)
                    if ins.connection_mode == "uvw"
                    else value * w.unsqueeze(-1)
                )
            outputs[ins.i_out] = outputs[ins.i_out] + value * ins.path_weight
        output = (
            torch.cat([value.flatten(-2) for value in outputs], -1)
            if outputs
            else x.new_zeros((*shape, 0)) + zero
        )
        return self.projection(output)

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
