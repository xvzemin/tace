"""Batched radial affine chains with operand-only VJPs and force adjoints."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ...utils.autograd import forward_ad_active, torch_gradient_mask
from ...utils.program import Program, activate
from .. import config
from .pointwise import silu_derivative, silu_evaluate, silu_product


def affine_shape(layer):
    if isinstance(layer, torch.nn.Linear):
        return layer.in_features, layer.out_features
    if (
        hasattr(layer, "get_weight")
        and hasattr(layer, "in_dim")
        and hasattr(layer, "out_dim")
    ):
        return layer.in_dim, layer.out_dim
    return None


def affine_weight(layer):
    return layer.weight.T if isinstance(layer, torch.nn.Linear) else layer.get_weight()


def pack_matrices(layers, width=None):
    """Pack a common affine scale once; unknown parameterizations remain independent."""
    terms = [
        layer.weight_and_scale()
        if hasattr(layer, "weight_and_scale")
        else (affine_weight(layer), 1.0)
        for layer in layers
    ]
    weights, scales = zip(*terms)
    if width is not None:
        weights = [
            w
            if w.shape[0] == width
            else torch.nn.functional.pad(w, (0, 0, 0, width - w.shape[0]))
            for w in weights
        ]
    if all(isinstance(s, (int, float)) for s in scales) and len(set(scales)) == 1:
        packed = torch.stack(tuple(weights))
        return packed if scales[0] == 1 else packed * scales[0]
    return torch.stack(
        [
            w if isinstance(s, (int, float)) and s == 1 else w * s
            for w, s in zip(weights, scales)
        ]
    )


def affine_chain(layers):
    """Record affine stages and their optional scaled SiLU activations."""
    steps = []
    for i, layer in enumerate(layers):
        if affine_shape(layer) is not None:
            steps.append((i, None))
            continue
        if (
            not steps
            or steps[-1][1] is not None
            or not isinstance(layer, torch.nn.Module)
            or tuple(layer.parameters())
            or tuple(layer.buffers())
        ):
            return ()
        program = Program()
        try:
            activate(program, program.input(0, "edge", 1), layer)
        except NotImplementedError:
            return ()
        ops = program.nodes
        if len(ops) == 1:
            continue
        if ops[1][0] != "silu" or not (
            len(ops) == 2
            or (len(ops) == 4 and ops[2][0] == "constant" and ops[3][0] == "mul")
        ):
            return ()
        steps[-1] = steps[-1][0], 1.0 if len(ops) == 2 else ops[2][3]
    return tuple(steps)


def apply_activation(x, scale, order, factors, native):
    if scale is None:
        if order == 0:
            return x
        if order == 1:
            return factors[0]
        return None  # The second derivative of an affine stage vanishes.
    op = (
        (silu_product if torch.is_grad_enabled() else silu_evaluate)
        if native
        else silu_derivative
    )
    return op(x, factors, scale, order)


def add_terms(x, y):
    return y if x is None else x if y is None else x + y


def states(plan, inputs, directions=None, *, save=True, cached=()):
    scales, _, native, _ = plan
    x = inputs[0]
    dx = directions[0] if directions is not None else None
    tape = []
    for i, scale in enumerate(scales):
        w, bias = inputs[1 + 2 * i : 3 + 2 * i]
        if cached:
            z = cached[2 * i]
        else:
            z = x.bmm(w)
            if bias.numel():
                z = z + bias
        dz = None if dx is None else dx.bmm(w)
        if directions is not None:
            dw, db = directions[1 + 2 * i : 3 + 2 * i]
            if dw is not None:
                dz = add_terms(dz, x.bmm(dw))
            if db is not None:
                dz = add_terms(dz, db)
        if save:
            tape.append((x, z, dx, dz))
        if (
            native
            and not cached
            and dz is not None
            and scale is not None
            and not torch.is_grad_enabled()
        ):
            from .cuda.pointwise import silu_dual

            x, dx = silu_dual(z, dz, scale)
        else:
            x = (
                cached[2 * i + 1]
                if cached
                else apply_activation(z, scale, 0, [], native)
            )
            dx = None if dz is None else apply_activation(z, scale, 1, [dz], native)
    return x, dx, tape


def linear_outputs(plan, x, inputs, dx=None, directions=None):
    scales, widths, _, _ = plan
    base = 1 + 2 * len(scales)
    w = inputs[base]
    if w.ndim == 3:
        if directions is None:
            out = x.bmm(w)
        else:
            out = add_terms(
                None if dx is None else dx.bmm(w),
                None if directions[base] is None else x.bmm(directions[base]),
            )
            if out is None:
                return x.new_zeros((x.shape[1], sum(widths)))
        return out.transpose(0, 1).reshape(x.shape[1], sum(widths))
    parts = []
    for i, width in enumerate(widths):
        if directions is None:
            part = x[i] @ inputs[base + i]
        else:
            part = None if dx is None else dx[i] @ inputs[base + i]
            if directions[base + i] is not None:
                part = add_terms(part, x[i] @ directions[base + i])
            if part is None:
                part = x.new_zeros((x.shape[1], width))
        parts.append(part)
    return parts[0] if len(parts) == 1 else torch.cat(parts, -1)


def reverse(
    plan, inputs, grad, active, directions=None, need_grad_output=False, cached=()
):
    """Compute a VJP or its directional derivative without edge outer products."""
    scales, widths, native, _ = plan
    base = 1 + 2 * len(scales)
    x, dx, tape = states(plan, inputs, directions, cached=cached)
    grads = [None] * len(inputs)
    output = (
        linear_outputs(plan, x, inputs, dx, directions) if need_grad_output else None
    )
    dgy = None
    h = x if directions is None else dx
    if inputs[base].ndim == 3:
        final = grad.reshape(grad.shape[0], len(widths), widths[0]).transpose(0, 1)
        gy = final.bmm(inputs[base].transpose(1, 2))
        if directions is not None and directions[base] is not None:
            dgy = final.bmm(directions[base].transpose(1, 2))
        if h is not None and active[base]:
            grads[base] = h.transpose(1, 2).bmm(final)
    else:
        final = grad.split(widths, -1)
        leaves = [g @ w.T for g, w in zip(final, inputs[base:])]
        gy = leaves[0].unsqueeze(0) if len(leaves) == 1 else torch.stack(leaves)
        del leaves
        if directions is not None and any(t is not None for t in directions[base:]):
            dgy = torch.stack(
                [
                    torch.zeros_like(gy[i]) if t is None else g @ t.T
                    for i, (g, t) in enumerate(zip(final, directions[base:]))
                ]
            )
        if h is not None:
            for i, g in enumerate(final):
                if active[base + i]:
                    grads[base + i] = h[i].T @ g
    del h, x, dx
    for i in range(len(scales) - 1, -1, -1):
        x, z, dx, dz = tape.pop()
        k, scale = 1 + 2 * i, scales[i]
        w = inputs[k]
        joint = native and not torch.is_grad_enabled() and scale is not None
        if joint:
            from .cuda.pointwise import silu_dual

            gz, dgz = silu_dual(z, dz, scale, gy, dgy)
        else:
            gz = apply_activation(z, scale, 1, [gy], native)
        if directions is None:
            if active[k]:
                grads[k] = x.transpose(1, 2).bmm(gz)
            if active[k + 1]:
                grads[k + 1] = gz.sum(1, keepdim=True)
        else:
            if not joint:
                dgz = (
                    None
                    if dgy is None
                    else apply_activation(z, scale, 1, [dgy], native)
                )
                if dz is not None:
                    dgz = add_terms(
                        dgz, apply_activation(z, scale, 2, [gy, dz], native)
                    )
            if active[k]:
                grads[k] = add_terms(
                    None if dx is None else dx.transpose(1, 2).bmm(gz),
                    None if dgz is None else x.transpose(1, 2).bmm(dgz),
                )
            if active[k + 1] and dgz is not None:
                grads[k + 1] = dgz.sum(1, keepdim=True)
            if i or active[0]:
                dw = directions[k]
                dgy = add_terms(
                    None if dgz is None else dgz.bmm(w.transpose(1, 2)),
                    None if dw is None else gz.bmm(dw.transpose(1, 2)),
                )
        if i or (active[0] and directions is None):
            gy = gz.bmm(w.transpose(1, 2))
        del gz, z, dz
        if directions is not None:
            del dgz
    if active[0]:
        grads[0] = gy if directions is None else dgy
    return output, grads


class RadialVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, active, grad, *inputs):
        ctx.plan, ctx.active = plan, active
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad, *inputs)
        n = len(active)
        _, result = reverse(plan, inputs[:n], grad, active, cached=inputs[n:])
        return tuple(g for need, g in zip(active, result) if need)

    @staticmethod
    def backward(ctx, *directions):
        grad, *saved = ctx.saved_tensors
        n = len(ctx.active)
        inputs, cached = saved[:n], saved[n:]
        it = iter(directions)
        directions = [next(it) if keep else None for keep in ctx.active]
        need_grad, *active = torch_gradient_mask(ctx, ctx.needs_input_grad[2:])
        # When differentiating this adjoint again, rebuild the primal dependence.
        dg, result = reverse(
            ctx.plan,
            inputs,
            grad,
            active[:n],
            directions,
            need_grad,
            () if torch.is_grad_enabled() else cached,
        )
        return None, None, dg, *result, *(None for _ in cached)


class RadialChain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, *inputs):
        ctx.plan = plan
        x, _, tape = states(plan, inputs, save=plan[3])
        cached = []
        for i, (_, z, _, _) in enumerate(tape):
            cached.extend((z, tape[i + 1][0] if i + 1 < len(tape) else x))
        ctx.save_for_backward(*inputs, *cached)
        return linear_outputs(plan, x, inputs)

    @staticmethod
    def backward(ctx, grad):
        active = torch_gradient_mask(ctx, ctx.needs_input_grad[1:])
        if not any(active):
            return (None,) * len(ctx.needs_input_grad)
        result = iter(RadialVJP.apply(ctx.plan, active, grad, *ctx.saved_tensors))
        return None, *(next(result) if need else None for need in active)


def eager_rows(value):
    return (
        value.is_cuda
        and not torch.compiler.is_compiling()
        and get_proxy_mode() is None
        and not forward_ad_active()
    )


def chain_tiles(x, params, scales, widths, native, mode, hidden):
    plan = scales, widths, native, mode == "last"
    tile = max(
        32,
        config.RADIAL_CHAIN_TILE_BYTES
        // (max(1, 4 * hidden) * x.element_size())
        // 32
        * 32,
    )

    def evaluate(v):
        if torch.is_grad_enabled():
            return RadialChain.apply(plan, v, *params)
        inputs = (v, *params)
        h, _, _ = states(plan, inputs, save=False)
        return linear_outputs(plan, h, inputs)

    if x.shape[1] <= tile:
        return evaluate(x)
    return torch.cat([evaluate(v) for v in x.split(tile, 1)], 0)
