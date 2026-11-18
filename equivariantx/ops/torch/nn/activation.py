"""Chebyshev evaluation with ordinary and bounded-tape analytic derivatives."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import nested_forward_ad
from ... import config


def scalar_blocks(sizes, activations):
    """Coalesce adjacent scalar entries with the same stateless activation."""
    from e3nn.math import normalize2mom

    kinds = {
        torch.nn.SiLU: 1,
        torch.nn.Sigmoid: 2,
        torch.nn.Tanh: 3,
        torch.nn.Identity: 0,
    }
    functions = {torch.nn.functional.silu: 1, torch.sigmoid: 2, torch.tanh: 3}
    groups = []
    for size, act in zip(sizes, activations):
        if not size:
            continue
        scale = 1.0
        if type(act) is normalize2mom:
            scale = 1.0 if act._is_id else act.cst
            act = act.f
        kind = 0 if act is None else kinds.get(type(act))
        if kind is None:
            kind = next((k for fn, k in functions.items() if act is fn), None)
        if kind is None or getattr(act, "inplace", False):
            return None
        if groups and groups[-1][1:] == (kind, scale):
            size += groups.pop()[0]
        groups.append((size, kind, scale))
    return tuple(groups)


def scalar_values(features, groups, gradient=None, order=0):
    """Evaluate contiguous scalar blocks and their first two derivatives."""
    from ...common.pointwise import activation_vjp

    values = []
    start = 0
    for size, kind, scale in groups:
        x = features if len(groups) == 1 else features[..., start : start + size]
        if order:
            g = gradient if len(groups) == 1 else gradient[..., start : start + size]
            if kind:
                value = activation_vjp(
                    x, g, ("", "silu", "sigmoid", "tanh")[kind], order
                )
            else:
                value = g if order == 1 else g * 0.0
        elif kind == 1:
            value = torch.nn.functional.silu(x)
        elif kind == 2:
            value = x.sigmoid()
        elif kind == 3:
            value = x.tanh()
        else:
            value = x
        values.append(value if scale == 1.0 else value * scale)
        start += size
    if not values:
        return features[..., :0]
    return values[0] if len(values) == 1 else torch.cat(values, -1)


def scalars(module, features, reference):
    """Dispatch normalized activations without crossing provider boundaries."""
    groups = module.scalar_blocks
    if groups is None or (config.TORCH_REFERENCE and module.backend == "torch"):
        return reference(module, features)
    if features.shape[-1] != module.irreps_in.dim:
        raise ValueError("The feature dimension does not match irreps_in.")
    if not groups or (len(groups) == 1 and groups[0][1:] == (0, 1.0)):
        return features
    if module.backend == "cuda" and features.is_cuda:
        from ....utils.autograd import forward_ad_active

        if not forward_ad_active():
            from ...cuda.nn.activation import block_activation

            sizes, kinds, scales = zip(*groups)
            return block_activation(features, None, sizes, kinds, scales)
    return scalar_values(features, groups)


def reference(features, coefficients):
    """Evaluate a Chebyshev series with the ordinary Clenshaw recurrence."""
    previous = torch.zeros_like(features)
    current = torch.zeros_like(features)
    for i in range(coefficients.shape[0] - 1, 0, -1):
        previous, current = current, coefficients[i] + 2 * features * current - previous
    return coefficients[0] + features * current - previous


def derivative(coefficients):
    """Return Chebyshev coefficients of the exact polynomial derivative."""
    degree = coefficients.shape[0] - 1
    if degree == 0:
        return coefficients * 0
    if degree >= 8 and not torch.compiler.is_compiling() and get_proxy_mode() is None:
        orders = torch.arange(
            1, degree + 1, device=coefficients.device, dtype=coefficients.dtype
        )
        weighted = coefficients[1:] * (2 * orders)
        even = weighted[::2].flip(0).cumsum(0).flip(0)
        odd = weighted[1::2].flip(0).cumsum(0).flip(0)
        if degree % 2:
            odd = torch.cat((odd, torch.zeros_like(even[:1])))
        values = torch.stack((even, odd), -1).flatten()[:degree]
        return torch.cat((values[:1] * 0.5, values[1:]))
    values = [torch.zeros_like(coefficients[0])] * (degree + 2)
    for i in range(degree, 0, -1):
        values[i - 1] = values[i + 1] + (2 * i) * coefficients[i]
    values[0] = values[0] * 0.5
    return torch.stack(values[:degree])


def coefficient_adjoint(features, gradient, size):
    """Reduce Chebyshev basis functions against the cotangent."""
    previous, current = torch.ones_like(features), features
    values = [gradient.sum()]
    if size > 1:
        values.append((gradient * current).sum())
    for _ in range(2, size):
        previous, current = current, 2 * features * current - previous
        values.append((gradient * current).sum())
    return torch.stack(values)


@torch.compiler.allow_in_graph
class Chebyshev(torch.autograd.Function):
    """Retain only operands and differentiate by another Chebyshev series."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features, coefficients):
        return reference(features, coefficients)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx, gradient):
        features, coefficients = ctx.saved_tensors
        dx = dc = None
        if ctx.needs_input_grad[0]:
            dx = gradient * Chebyshev.apply(features, derivative(coefficients))
        if ctx.needs_input_grad[1]:
            dc = coefficient_adjoint(features, gradient, coefficients.shape[0])
        return dx, dc

    @staticmethod
    def jvp(ctx, dx, dc):
        features, coefficients = ctx.saved_tensors
        result = features * 0
        if dx is not None:
            result = result + dx * Chebyshev.apply(features, derivative(coefficients))
        if dc is not None:
            result = result + Chebyshev.apply(features, dc)
        return result


@torch.compiler.allow_in_graph
def chebyshev(features, coefficients):
    """Expose the analytic Function to AOTAutograd without Dynamo tracing its JVP."""
    if nested_forward_ad():
        return reference(features, coefficients)
    return Chebyshev.apply(features, coefficients)


def evaluate(module, features):
    """Select ordinary or analytic AD without changing the polynomial."""
    features = features / module.bound
    if config.TORCH_REFERENCE or not config.TORCH_POLYNOMIAL_ADJOINT:
        return reference(features, module.coefficients)
    return chebyshev(features, module.coefficients)
