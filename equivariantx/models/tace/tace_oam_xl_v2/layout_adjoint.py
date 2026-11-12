"""Partition and permutation adjoints without full-size zero scatter buffers.

These are exact layout maps. Their explicit transposes remain differentiable,
including mixed parameter/coordinate derivatives and forward-mode tangents.
"""
import torch


@torch.compiler.allow_in_graph
class ColumnSlices(torch.autograd.Function):
    """Partition columns with one concatenation for their joint transpose."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value, widths):
        return value.split(widths, dim=-1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        value, widths = inputs
        ctx.shape = value.shape
        ctx.widths = widths
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, *gradients):
        present = next((g for g in gradients if g is not None), None)
        if present is None:
            return None, None
        pieces = [
            g if g is not None else present.new_zeros((*ctx.shape[:-1], width))
            for g, width in zip(gradients, ctx.widths)
        ]
        return torch.cat(pieces, dim=-1), None

    @staticmethod
    def jvp(ctx, value_tangent, widths_tangent):
        if value_tangent is None:
            return (None,) * len(ctx.widths)
        return value_tangent.split(ctx.widths, dim=-1)


@torch.compiler.allow_in_graph
class PackedSlices(torch.autograd.Function):
    """Split a packed irrep tensor; concatenate disjoint adjoints once."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value, specification):
        nodes=value.shape[0]
        flat=value.reshape(-1)
        pieces=[]
        start=0
        for channels,degree in specification:
            size=channels*degree
            pieces.append(flat.narrow(0,nodes*start,nodes*size).reshape(nodes*degree,channels))
            start+=size
        if start!=value.shape[1]:
            raise ValueError('Packed slices must partition the complete feature axis')
        return tuple(pieces)

    @staticmethod
    def setup_context(ctx,inputs,output):
        value,specification=inputs
        ctx.shape=value.shape
        ctx.specification=specification
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx,*gradients):
        present=next((g for g in gradients if g is not None),None)
        if present is None:
            return None,None
        pieces=[g.reshape(-1) if g is not None else present.new_zeros(ctx.shape[0]*c*d)
                for g,(c,d) in zip(gradients,ctx.specification)]
        return torch.cat(pieces).reshape(ctx.shape),None

    @staticmethod
    def jvp(ctx,value_tangent,specification_tangent):
        if value_tangent is None:
            return (None,)*len(ctx.specification)
        return PackedSlices.forward(value_tangent,ctx.specification)


@torch.compiler.allow_in_graph
class Permutation(torch.autograd.Function):
    """Use the inverse permutation for the transpose of a complete gather."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value,index,inverse):
        return value.index_select(-1,index)

    @staticmethod
    def setup_context(ctx,inputs,output):
        _,index,inverse=inputs
        ctx.save_for_backward(index,inverse)
        ctx.save_for_forward(index,inverse)

    @staticmethod
    def backward(ctx,gradient):
        index,inverse=ctx.saved_tensors
        return Permutation.apply(gradient,inverse,index),None,None

    @staticmethod
    def jvp(ctx,value_tangent,index_tangent,inverse_tangent):
        index,_=ctx.saved_tensors
        return value_tangent.index_select(-1,index)


def packed_slices(value,irreps):
    """Return contiguous per-irrep matrix views with a single join in reverse."""
    return PackedSlices.apply(value,tuple((channels,ir.dim) for channels,ir in irreps))
