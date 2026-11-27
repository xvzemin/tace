"""Quadratic CG forward and pullback in native multiplicity-major storage."""

from collections import defaultdict
from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import compact_metadata, parse_metadata


def _layout(section):
    result = []
    start = 0
    for _, ir in section:
        result.extend((start, ir.dim, i) for i in range(ir.dim))
        start += ir.dim
    return tuple(result)


def square_metadata(product):
    """Return native-layout angular offsets and CG coefficients."""
    tp = product.aces[0].tp
    return compact_metadata(
        repr(
            (
                _layout(tp.irreps_in1),
                _layout(tp.irreps_out),
                parse_metadata(product.cg_metadata),
            )
        )
    )


def _scalar(v):
    return f"{v:.16e}f"


@lru_cache(maxsize=4)
def square_source(meta, reverse, mixed=False):
    """Generate one CUDA branch per output irrep and one writer per channel."""
    li, lo, (ch, dims, rows) = parse_metadata(meta)
    if ch != 64 or dims[0] != dims[1] or dims[0] != len(li) or dims[2] != len(lo):
        raise ValueError("Expected a 64-channel OAM-L self-product.")
    sums = defaultdict(float)
    for (i, j, k), w in rows:
        sums[min(i, j), max(i, j), k] += w
    contributions = defaultdict(list)
    for (i, j, k), w in sums.items():
        if not w:
            continue
        if reverse:
            contributions[i].append((j, k, 2 * w if i == j else w))
            if i != j:
                contributions[j].append((i, k, w))
        else:
            contributions[k].append((i, j, w))
    lw = li if reverse else lo
    sections = [(i, d) for i, d, t in lw if t == 0]
    cases = []
    for group, (start, dim) in enumerate(sections):
        code = []
        cache = {}

        def operand(a, table, key, name):
            base, d, t = table[a]
            slot = key, a
            if slot not in cache:
                var = f"{name}{a}"
                code.append(f"float {var}={key}[{base * ch}+c*{d}+{t}];")
                cache[slot] = var
            return cache[slot]

        for k in range(start, start + dim):
            products = []
            for a, b, w in contributions[k]:
                v = operand(a, li, "x", "h")
                u = operand(
                    b,
                    lo if reverse else li,
                    "g" if reverse else "x",
                    "b" if reverse else "h",
                )
                if mixed:
                    s = operand(b, li, "g", "d")
                    if a == b:
                        products.append(f"({_scalar(2 * w)}*{v})*{s}")
                    else:
                        t = operand(a, li, "g", "d")
                        products.append(f"{_scalar(w)}*fmaf({v},{s},{u}*{t})")
                else:
                    products.append(f"({_scalar(w)}*{v})*{u}")
            code.append(
                f"out[{start * ch}+c*{dim}+{k - start}]="
                + ("+".join(products) or "0.0f")
                + ";"
            )
        cases.append(f"case {group}: {{ {''.join(code)} break; }}")
    src = (
        """
    extern "C" __global__ void run(const float* __restrict__ x,const float* __restrict__ g,
                                   float* __restrict__ out,long long nodes) {
      long long n=(long long)blockIdx.x*2+threadIdx.x/64;
      int c=threadIdx.x%64;
      if(n>=nodes)return;
    """
        + f"x+=n*{dims[0] * ch}; g+=n*{(dims[2] if reverse else dims[0]) * ch}; out+=n*{len(lw) * ch};"
        + f"switch(blockIdx.y) {{ {''.join(cases)} }}"
        + "}"
    )
    return src, len(sections), len(lw) * ch


def _execute(meta, x, g, reverse, mixed=False):
    _, dims, _ = parse_metadata(meta)[2]
    if (
        x.dtype != torch.float32
        or g.dtype != x.dtype
        or g.device != x.device
        or x.ndim != 2
        or g.ndim != 2
        or x.shape != (g.shape[0], 64 * dims[0])
        or g.shape[1] != 64 * (dims[2] if reverse else dims[0])
    ):
        raise ValueError("Expected matching native FP32 OAM-L product rows.")
    x, g = x.contiguous(), g.contiguous()
    src, groups, width = square_source(meta, reverse, mixed)
    y = x.new_empty((x.shape[0], width))
    if x.shape[0]:
        kernel = kernels([src], x.device)[src]
        runtime().launch(
            [
                (
                    kernel,
                    [x.data_ptr(), g.data_ptr(), y.data_ptr(), x.shape[0]],
                    (x.shape[0] + 1) // 2,
                    groups,
                    128,
                    0,
                )
            ],
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    return y


@torch.library.custom_op(
    "eqx_oam_l_product::square", mutates_args=(), device_types="cuda"
)
def square(meta: str, x: torch.Tensor) -> torch.Tensor:
    """Apply all quadratic paths of an OAM-L product in native layout."""
    return _execute(meta, x, x, False)


@square.register_fake
def fake(meta, x):
    return x.new_empty((x.shape[0], parse_metadata(meta)[2][1][2] * 64))


@torch.library.custom_op(
    "eqx_oam_l_product::square_vjp", mutates_args=(), device_types="cuda"
)
def adjoint(meta: str, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Contract the cotangent with both operands of the self-product."""
    return _execute(meta, x, g, True)


@adjoint.register_fake
def fake_adjoint(meta, x, g):
    return torch.empty_like(x)


def context(ctx, inputs, output):
    ctx.meta = inputs[0]
    ctx.save_for_backward(inputs[1])


def pullback(ctx, g):
    (x,) = ctx.saved_tensors
    return None, adjoint(ctx.meta, x, g)


square.register_autograd(pullback, setup_context=context)


@torch.library.custom_op(
    "eqx_oam_l_product::mixed", mutates_args=(), device_types="cuda"
)
def mixed(meta: str, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Evaluate the Jacobian of a quadratic product on a tangent."""
    return _execute(meta, x, v, False, True)


@mixed.register_fake
def fake_mixed(meta, x, v):
    return fake(meta, x)


def pair_context(ctx, inputs, output):
    ctx.meta, x, v = inputs
    ctx.save_for_backward(
        x if ctx.needs_input_grad[2] else None,
        v if ctx.needs_input_grad[1] else None,
    )


def backward_adjoint(ctx, v):
    x, g = ctx.saved_tensors
    return (
        None,
        adjoint(ctx.meta, v, g) if g is not None else None,
        mixed(ctx.meta, x, v) if x is not None else None,
    )


def backward_mixed(ctx, g):
    x, v = ctx.saved_tensors
    return (
        None,
        adjoint(ctx.meta, v, g) if v is not None else None,
        adjoint(ctx.meta, x, g) if x is not None else None,
    )


adjoint.register_autograd(backward_adjoint, setup_context=pair_context)
mixed.register_autograd(backward_mixed, setup_context=pair_context)


@lru_cache(maxsize=4)
def _angular_order(entries):
    order = tuple(
        base * 64 + c * dim + m for base, dim, m in entries for c in range(64)
    )
    return order, tuple(sorted(range(len(order)), key=order.__getitem__))


def _transformed_product(meta, x, other, phase):
    """Evaluate transformed operands with differentiable PyTorch primitives."""
    from equivariantx.ops.common.torch.channel_product import channel_product

    from .product import ADJOINT, FORWARD

    inp, out, cg = parse_metadata(meta)
    ix, rx = _angular_order(inp)
    iy, ry = _angular_order(out)

    def reorder(value, order):
        return value.index_select(
            -1, torch.tensor(order, dtype=torch.long, device=value.device)
        )

    t = reorder(x, ix)
    shape = x.new_empty((0, len(iy)))
    if phase == "square":
        operands, program, restore = [t, shape], FORWARD, ry
    elif phase == "adjoint":
        operands, program, restore = [t, reorder(other, iy)], ADJOINT, rx
    else:
        operands = [t, reorder(other, ix), shape]
        program = "(((0, 1, 2), 2, 0, 0), ((1, 0, 2), 2, 0, 0))"
        restore = ry
    (value,) = channel_product(compact_metadata(repr(cg)), program, operands)
    return reorder(value, restore)


_transforms = torch.library.Library(
    "eqx_oam_l_product", "IMPL", "FuncTorchDynamicLayerFrontMode"
)
_transforms.impl(
    "square", lambda meta, x: _transformed_product(meta, x, None, "square")
)
_transforms.impl(
    "square_vjp", lambda meta, x, g: _transformed_product(meta, x, g, "adjoint")
)
_transforms.impl("mixed", lambda meta, x, v: _transformed_product(meta, x, v, "mixed"))
