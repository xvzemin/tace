"""Shared feature permutations with inverse-permutation derivatives."""

import torch

from ..ops import config
from .autograd import nested_forward_ad


def channel_permutation(irreps):
    """Return the flat indices mapping blockwise mul_ir to ir_mul."""
    chunks = [
        torch.arange(sl.start, sl.stop, device="cpu")
        .reshape(block.mul, block.ir.dim)
        .T.flatten()
        for block, sl in zip(irreps, irreps.slices())
    ]
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)


class TransposeIrrepsLayout(torch.nn.Module):
    """Transpose multiplicities and representation components without changing basis.

    Parameters
    ----------
    irreps : Irreps or str
        Ordered representation blocks in this space.
    source, target : {"mul_ir", "ir_mul"}
        Component order in each input and output block.
    flat_in, flat_out : bool, optional
        Use one trailing feature dimension on input or output. Explicit axes
        require a common multiplicity across all blocks.
    backend : {None, "torch", "cuda"}, optional
        Execution backend.

    Examples
    --------
    >>> from equivariantx import o3
    >>> op = o3.TransposeIrrepsLayout("4x0e+4x1o", source="mul_ir", target="ir_mul", flat_out=False)
    >>> op(torch.randn(2, 16)).shape
    torch.Size([2, 4, 4])
    """

    def __init__(
        self,
        irreps,
        *,
        source="mul_ir",
        target="ir_mul",
        flat_in=True,
        flat_out=True,
        backend=None,
    ):
        from ..ops import resolve_backend

        super().__init__()
        self.irreps = self.irreps_type(irreps)
        if source not in ("ir_mul", "mul_ir") or target not in ("ir_mul", "mul_ir"):
            raise ValueError("source and target must be 'mul_ir' or 'ir_mul'.")
        self.source, self.target = source, target
        self.flat_in, self.flat_out = flat_in, flat_out
        self.backend = resolve_backend(backend)
        self.dim = self.irreps.dim
        self.num_components = sum(block.ir.dim for block in self.irreps)
        channels = {block.mul for block in self.irreps}
        self.num_channels = next(iter(channels)) if len(channels) == 1 else None
        if not (flat_in and flat_out) and self.num_channels is None:
            raise ValueError("Explicit axes require one common multiplicity.")
        self.single_block = len(self.irreps) == 1
        reorder = any(block.mul > 1 and block.ir.dim > 1 for block in self.irreps)
        self.widths = repr((self.dim,))
        needs_index = reorder and (
            source != target
            if flat_in and flat_out
            else not self.single_block
            and (flat_in and source == "mul_ir" or flat_out and target == "mul_ir")
        )
        index = channel_permutation(self.irreps) if needs_index else None
        self.register_buffer("index", index, persistent=False)
        self.register_buffer(
            "reverse", index.argsort() if needs_index else None, persistent=False
        )

    def reorder_flat(self, features, to_ir_mul):
        if self.index is None:
            return features
        index, reverse = (
            (self.index, self.reverse) if to_ir_mul else (self.reverse, self.index)
        )
        if self.backend == "cuda" and features.is_cuda:
            from ..ops.common.layout import grouped_permute

            return grouped_permute(
                features.reshape(-1, self.dim), index, reverse, self.widths
            ).reshape(features.shape)
        return permute(features, index, reverse)

    def transform(self, features, source, target, flat_in, flat_out):
        explicit = (self.num_components, self.num_channels)
        expected = (
            (self.dim,)
            if flat_in
            else explicit
            if source == "ir_mul"
            else explicit[::-1]
        )
        if features.shape[-len(expected) :] != expected:
            raise ValueError(
                f"Expected trailing shape {expected}, got {features.shape}."
            )
        if flat_in and flat_out:
            return (
                features
                if source == target
                else self.reorder_flat(features, target == "ir_mul")
            )
        if not flat_in and not flat_out:
            return features if source == target else features.transpose(-1, -2)
        if self.single_block:
            if flat_in:
                x = features.unflatten(
                    -1, explicit if source == "ir_mul" else explicit[::-1]
                )
                return x if source == target else x.transpose(-1, -2)
            x = features if source == target else features.transpose(-1, -2)
            return x.flatten(-2)
        if flat_in:
            x = self.reorder_flat(features, True) if source == "mul_ir" else features
            x = x.unflatten(-1, explicit)
            return x if target == "ir_mul" else x.transpose(-1, -2)
        x = features if source == "ir_mul" else features.transpose(-1, -2)
        x = x.flatten(-2)
        return self.reorder_flat(x, False) if target == "mul_ir" else x

    def forward(self, features):
        return self.transform(
            features, self.source, self.target, self.flat_in, self.flat_out
        )

    def inverse(self, features):
        """Transpose from target to source."""
        return self.transform(
            features, self.target, self.source, self.flat_out, self.flat_in
        )


@torch.compiler.allow_in_graph
def permute(features, index, inverse):
    """Permute features with an inverse adjoint and nested forward derivatives."""
    if (
        config.TORCH_REFERENCE
        or not torch.is_grad_enabled()
        or not features.requires_grad
        or nested_forward_ad()
    ):
        return features.index_select(-1, index)
    return Permute.apply(features, index, inverse)


@torch.compiler.allow_in_graph
class Permute(torch.autograd.Function):
    """Permute features and use the inverse permutation for the adjoint."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features, index, inverse):
        return features.index_select(-1, index)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, index, inverse = inputs
        ctx.save_for_backward(inverse, index)
        ctx.save_for_forward(index)

    @staticmethod
    def backward(ctx, gradient):
        inverse, index = ctx.saved_tensors
        return permute(gradient, inverse, index), None, None

    @staticmethod
    def jvp(ctx, tangent, index_tangent, inverse_tangent):
        (index,) = ctx.saved_tensors
        return tangent.index_select(-1, index)
