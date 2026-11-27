"""Channel-wise contractions of spherical O(3) tensors."""

import math
from itertools import groupby

import torch
from e3nn import o3

from ..ops import resolve_backend
from ..ops.precision import validate_uniform
from .cgtp import TensorProduct
from .paths import asymmetric_contraction_paths, symmetric_contraction_paths


class AsymmetricContraction(torch.nn.Module):
    """Couple independent factors with one external coefficient per path and channel.

    Parameters
    ----------
    irreps_in, irreps_out : Irreps or str
        Input and requested output blocks in flat mul_ir layout, with common multiplicity.
    correlation : int
        Highest product order. Order one is included.
    path_mode : {"sum", "expand"}, optional
        Sum paths of each type with inverse square-root path normalization, or retain each path.
    backend : {None, "torch", "cuda"}, optional
        Tensor product backend.

    Examples
    --------
    >>> from equivariantx import o3
    >>> op = o3.AsymmetricContraction("2x1o", "2x1e", 2)
    >>> x, y = torch.randn(3, 6), torch.randn(3, 6)
    >>> op((x, y), torch.ones(op.weight_numel)).shape
    torch.Size([3, 6])
    """

    irreps_type = o3.Irreps
    tensor_product = TensorProduct
    paths = staticmethod(asymmetric_contraction_paths)

    def __init__(
        self, irreps_in, irreps_out, correlation, *, path_mode="sum", backend=None
    ):
        super().__init__()
        self.backend = resolve_backend(backend)
        self.irreps_in = self.irreps_type(irreps_in)
        targets = self.irreps_type(irreps_out)
        channels = {block.mul for block in (*self.irreps_in, *targets)}
        if not self.irreps_in or not targets or len(channels) != 1 or not min(channels):
            raise ValueError(
                "Input and output irreps must have a common positive multiplicity."
            )
        if not isinstance(correlation, int) or correlation < 1:
            raise ValueError("correlation must be a positive integer.")
        if path_mode not in ("sum", "expand"):
            raise ValueError("path_mode must be 'sum' or 'expand'.")
        self.num_channels = channels.pop()
        self.correlation, self.path_mode = correlation, path_mode
        base = o3.Irreps([(b.mul, o3.Irrep(str(b.ir))) for b in self.irreps_in])
        requested = tuple(dict.fromkeys(o3.Irrep(str(b.ir)) for b in targets))
        reachable = [{b.ir for b in base}]
        for _ in range(1, correlation):
            reachable.append({c for a in reachable[-1] for _, b in base for c in a * b})
        needed = set(requested)
        for stage in range(correlation - 1, 0, -1):
            reachable[stage].intersection_update(needed)
            needed = set(requested) | {
                a
                for a in reachable[stage - 1]
                if any(c in reachable[stage] for _, b in base for c in a * b)
            }
        instructions = self.paths(
            base, [o3.Irreps(sorted(level)) for level in reachable[1:]]
        )
        self.products = torch.nn.ModuleList()
        levels = [self.irreps_in]
        for paths in instructions:
            out = self.irreps_type([(self.num_channels, str(p.ir_out)) for p in paths])
            self.products.append(
                self.tensor_product(
                    levels[-1],
                    self.irreps_in,
                    out,
                    [(p.i_in1, p.i_in2, k, "uuu", False) for k, p in enumerate(paths)],
                    internal_weights=False,
                    shared_weights=True,
                    backend=self.backend,
                )
                if out
                else torch.nn.Identity()
            )
            levels.append(out)
        self.live_levels = tuple(bool(level) for level in levels[1:])
        self.input_dim = self.irreps_in.dim
        self.groups = []
        offset = 0
        out = []
        for target in requested:
            pieces = []
            for degree, level in enumerate(levels):
                start = 0
                for ir, blocks in groupby(level, key=lambda b: b.ir):
                    count = sum(1 for _ in blocks)
                    width = count * self.num_channels * ir.dim
                    if str(ir) == str(target):
                        pieces.append((degree, start, width, count))
                    start += width
            paths = sum(block[3] for block in pieces)
            dim = self.irreps_type([(1, str(target))]).dim
            self.groups.append((tuple(pieces), offset, paths, dim))
            offset += paths * self.num_channels
            out.append(
                (
                    self.num_channels * (paths if path_mode == "expand" else 1),
                    str(target),
                )
            )
        self.irreps_out = self.irreps_type(out)
        self.weight_numel = offset
        self.num_paths = offset // self.num_channels

    def forward(self, inputs, weight):
        """Contract a sequence of factors with broadcastable leading dimensions."""
        if isinstance(inputs, torch.Tensor) or len(inputs) != self.correlation:
            raise ValueError(f"Expected a sequence of {self.correlation} factors.")
        if (
            any(x.shape[-1] != self.input_dim for x in inputs)
            or weight.shape[-1] != self.weight_numel
        ):
            raise ValueError(
                "Feature or weight dimensions do not match the contraction."
            )
        validate_uniform((*inputs, weight))
        leading = torch.broadcast_shapes(
            weight.shape[:-1], *(x.shape[:-1] for x in inputs)
        )
        levels = [inputs[0]]
        for k, product in enumerate(self.products):
            if not self.live_levels[k]:
                break
            levels.append(product(levels[-1], inputs[k + 1]))
        outputs = []
        for pieces, offset, paths, dim in self.groups:
            if not paths:
                width = 0 if self.path_mode == "expand" else self.num_channels * dim
                zero = weight[..., :0].sum(-1, keepdim=True)
                for x in inputs:
                    zero = zero + x[..., :0].sum(-1, keepdim=True)
                outputs.append(zero.expand(*leading, width))
                continue
            rows = [
                levels[k][..., start : start + size]
                .reshape(*levels[k].shape[:-1], count, self.num_channels, dim)
                .expand(*leading, count, self.num_channels, dim)
                for k, start, size, count in pieces
            ]
            values = torch.cat(rows, -3) if len(rows) > 1 else rows[0]
            scales = weight[..., offset : offset + paths * self.num_channels].unflatten(
                -1, (paths, self.num_channels)
            )
            value = values * scales.unsqueeze(-1)
            outputs.append(
                value.flatten(-3)
                if self.path_mode == "expand"
                else (value.sum(-3) * (1 / math.sqrt(paths))).flatten(-2)
            )
        return torch.cat(outputs, -1) if len(outputs) > 1 else outputs[0]


class SymmetricContraction(AsymmetricContraction):
    """Contract powers of one spherical feature. Parameters match AsymmetricContraction.

    Examples
    --------
    >>> from equivariantx import o3
    >>> op = o3.SymmetricContraction("2x1o", "2x0e", 2)
    >>> op(torch.randn(3, 6), torch.ones(op.weight_numel)).shape
    torch.Size([3, 2])
    """

    paths = staticmethod(symmetric_contraction_paths)

    def forward(self, features, weight):
        return super().forward((features,) * self.correlation, weight)
