"""Frozen OAM-L affine projection with bounded sparse CG contraction.

Keep the 128 real hidden coordinates (the checkpoint bias isn't a learned
feature). Emit the projection with addmm's bias epilogue, recompute in the
force VJP, and project path-weight cotangents to just those 128 coordinates.
A shared 65,536-edge scratch tile covers both sweeps. All learned and
geometry operands are read on this invocation; no projected weights survive
forward. Standard EQX supplies identical sparse CG factors and reductions.
"""

import torch

from equivariantx.ops.common.cuda.o3.convolution import contract_direct
from equivariantx.ops.common.o3 import contraction
from equivariantx.utils.metadata import compact_metadata

from .tangent_kernels import coordinate_adjoint

_TILE = 65536
_FORWARD = repr((((0, 1, 2, 3, 4), False, ((4, 0),)),))


class MessageTranspose:
    """Pull a node cotangent back to path-wise `ir_mul` convolution outputs.

    Parameters
    ----------
    linear : torch.nn.Module
        Frozen OAM-L linear-down map on merged `mul_ir` blocks.
    raw_irreps : e3nn.o3.Irreps
        Irreps of the convolution output before adjacent paths are merged.
    """

    def __init__(self, linear, raw_irreps):
        self.width = linear.irreps_in.dim
        self.blocks = []
        groups = []
        for mul, ir in raw_irreps:
            if not groups or groups[-1][0] != ir:
                groups.append((ir, []))
            groups[-1][1].append(mul)
        instructions = linear.linear.instructions
        if (
            raw_irreps.simplify() != linear.irreps_in
            or sorted(p.i_in for p in instructions) != list(range(len(groups)))
        ):
            raise ValueError("OAM-L needs one linear-down path per merged input.")
        inputs, outputs = linear.irreps_in.slices(), linear.irreps_out.slices()
        offset = 0
        for index, path in enumerate(instructions):
            a, b = path.i_in, path.i_out
            u, v = path.path_shape
            ir, widths = groups[a]
            channels = widths[0]
            if not all(w == channels for w in widths) or sum(widths) != u:
                raise ValueError("OAM-L needs equal channel widths within each irrep.")
            if isinstance(linear.weight, torch.nn.ParameterList):
                weight = linear.weight[index]
            else:
                weight = linear.weight[offset : offset + u * v].view(u, v)
            offset += u * v
            self.blocks.append(
                (inputs[a], outputs[b], ir.dim, v, len(widths), channels,
                 (weight.detach().T * path.path_weight).contiguous())
            )

    def __call__(self, gradient):
        n = len(gradient)
        out = gradient.new_empty(n, self.width)
        for dst, src, dim, v, paths, channels, weight in self.blocks:
            grad = gradient[:, src].reshape(n, v, dim).transpose(1, 2).reshape(n * dim, v)
            block = (grad @ weight).view(n, dim, paths, channels).transpose(1, 2)
            out[:, dst].view(n, paths, dim, channels).copy_(block)
        return out


@torch.library.custom_op("eqx_oam_l_conv::affine", mutates_args=())
def _affine(
    x: torch.Tensor,
    attrs: torch.Tensor,
    h: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    shape: torch.Tensor,
    metadata: str,
) -> torch.Tensor:
    z = x.new_zeros(shape.shape)
    edges = h.shape[0]
    scratch = h.new_empty((min(_TILE, edges), weight.shape[1]))
    empty = h.new_empty((0, weight.shape[1]))
    for start in range(0, edges, _TILE):
        sl = slice(start, min(start + _TILE, edges))
        rows = scratch[: h[sl].shape[0]]
        torch.addmm(bias, h[sl], weight, out=rows)
        args = (x, rows, empty, attrs[sl], z)
        contract_direct(metadata, source[sl], target[sl], [((4,), args, (z,), False)])
    return z


@_affine.register_fake
def _fake(x, attrs, h, weight, bias, source, target, shape, metadata):
    return x.new_empty(shape.shape)


@torch.library.custom_op("eqx_oam_l_conv::affine_vjp", mutates_args=())
def _affine_vjp(
    x: torch.Tensor,
    attrs: torch.Tensor,
    h: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    grad: torch.Tensor,
    metadata: str,
    node_grad: bool,
    attr_grad: bool,
    hidden_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.zeros_like(x) if node_grad else x.new_empty(0)
    da = torch.zeros_like(attrs) if attr_grad else x.new_empty(0)
    dh = torch.empty_like(h) if hidden_grad else x.new_empty(0)
    edges = h.shape[0]
    n = min(_TILE, edges)
    cols = weight.shape[1]
    scratch = h.new_empty((n, cols)) if node_grad or attr_grad else None
    dw = h.new_empty((n, cols)) if hidden_grad else None
    empty = h.new_empty((0, cols))
    wt = weight.T.contiguous() if hidden_grad else None
    # Independent, contiguous path columns assign every element of dw once.
    for start in range(0, edges, _TILE):
        sl = slice(start, min(start + _TILE, edges))
        count = h[sl].shape[0]
        if scratch is not None:
            rows = scratch[:count]
            torch.addmm(bias, h[sl], weight, out=rows)
        else:
            rows = h.new_empty(1).expand(count, cols)
        if attr_grad and hidden_grad:
            coordinate_adjoint(
                metadata,
                source[sl],
                target[sl],
                x,
                attrs[sl],
                rows,
                grad,
                dx,
                da[sl],
                dw[:count],
            )
        else:
            dest = {}
            if node_grad:
                dest[0] = dx
            if hidden_grad:
                dest[1] = dw[:count]
            if attr_grad:
                dest[3] = da[sl]
            args = (x, rows, empty, attrs[sl], grad)
            contract_direct(
                metadata,
                source[sl],
                target[sl],
                [(tuple(dest), args, tuple(dest.values()), False)],
                shared=(False, False),
                initialize=(dest[1],) if hidden_grad else (),
            )
        if hidden_grad:
            torch.mm(dw[:count], wt, out=dh[sl])
    return dx, da, dh


@_affine_vjp.register_fake
def _fake_vjp(
    x,
    attrs,
    h,
    weight,
    bias,
    source,
    target,
    grad,
    metadata,
    node_grad,
    attr_grad,
    hidden_grad,
):
    return tuple(
        torch.empty_like(v) if needed else x.new_empty(0)
        for v, needed in zip((x, attrs, h), (node_grad, attr_grad, hidden_grad))
    )


def _context(ctx, inputs, output):
    *args, ctx.cg_metadata = inputs
    ctx.save_for_backward(*args)


def _backward(ctx, grad):
    x, attrs, h, w, b, source, target, shape = ctx.saved_tensors
    mask = ctx.needs_input_grad[:3]
    if torch.is_grad_enabled():
        # Optional Hessians/force-loss gradients: delegate to EQX's recursive
        # analytic derivative rules; only MD first-VJP uses the fast kernel.
        args = [v.view_as(v) for v in (x, attrs, h)]
        weights = torch.addmm(b, args[2], w)
        (z,) = contraction(
            ctx.cg_metadata,
            _FORWARD,
            source,
            target,
            [
                args[0],
                weights,
                w.new_empty((0, w.shape[1])),
                args[1],
                shape,
            ],
        )
        grad_inputs = torch.autograd.grad(
            z, [v for v in args if v.requires_grad], grad, create_graph=True
        )
        active = iter(grad_inputs)
        dx, da, dh = (next(active) if m else None for m in mask)
    else:
        grads = _affine_vjp(
            x, attrs, h, w, b, source, target, grad.contiguous(), ctx.cg_metadata, *mask
        )
        dx, da, dh = (g if m else None for g, m in zip(grads, mask))
    return dx, da, dh, None, None, None, None, None, None


_affine.register_autograd(_backward, setup_context=_context)


def bias_convolution(conv, x, attrs, h, weight, bias, edge_index, num_nodes):
    """Apply a released CG interaction using an already-reshaped node input.

    Input/output node features have the same ir_mul layout as `conv`. For
    a learnable projection or unsupported device/type, use the live library
    implementation, including its checkpoint-gain gradients.
    """
    if (
        not x.is_cuda
        or torch.version.hip is not None
        or x.dtype != torch.float32
        or weight.requires_grad
        or bias.requires_grad
        or type(conv).__name__ != "O3TensorProductConv"
        or conv.backend != "cuda"
    ):
        affine = torch.nn.functional.pad(h, (0, 1), value=1.0)
        return conv(
            x,
            attrs,
            affine,
            torch.cat((weight, bias[None]), 0),
            edge_index,
            num_nodes=num_nodes,
        )
    width = weight.shape[0]
    channels = conv.paths[0][3]
    if (
        width not in (64, 128)
        or h.shape != (attrs.shape[0], width)
        or weight.shape != (width, conv.weight_numel)
        or bias.shape != (conv.weight_numel,)
        or x.shape[1] != conv.input_dim
        or attrs.shape[1] != conv.edge_dim
        or edge_index.shape != (2, h.shape[0])
        or channels not in (64, 128)
        or edge_index.dtype != torch.long
        or any(v.dtype != torch.float32 for v in (attrs, h, weight, bias))
        or len(conv.paths) * channels != conv.weight_numel
        or any(
            path[3:5] != (channels, 1) or path[8] != i * channels
            for i, path in enumerate(sorted(conv.paths, key=lambda path: path[8]))
        )
    ):
        raise ValueError("Mismatched OAM-L convolution operands")
    shape = x.new_empty(1).expand(num_nodes, conv.output_dim)
    source, target = edge_index
    return _affine(
        x.contiguous(),
        attrs.contiguous(),
        h.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        source.contiguous(),
        target.contiguous(),
        shape,
        compact_metadata(conv.kernel_metadata),
    )
