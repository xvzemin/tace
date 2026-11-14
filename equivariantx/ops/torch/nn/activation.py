"""Chebyshev evaluation with ordinary and bounded-tape analytic derivatives."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ....utils.autograd import nested_forward_ad


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
