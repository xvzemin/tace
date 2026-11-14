"""Sparse transverse couplings and an independent ordinary Torch reference."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ... import config
from ...common.torch import sparse_axis


def reference(module, features, vectors, method):
    """Apply the original dense generator and minimum-degree CG schedule."""
    direction = vectors / vectors.norm(dim=-1, keepdim=True)
    harmonic = module.harmonics(direction)

    def bridge(value):
        projected = (value @ module.cg.flatten(1)).unflatten(
            -1, (2 * module.delta + 1, 2 * module.l3 + 1)
        )
        return (projected * harmonic.unsqueeze(-1)).sum(-2)

    value = features if module.l1 < module.l3 else bridge(features)

    def apply_generator(value):
        projected = (value @ module.generator.flatten(0, 1).T).unflatten(
            -1, (3, 2 * module.degree + 1)
        )
        return (projected * direction.unsqueeze(-1)).sum(-2)

    if method == "recurrence":
        value = math.sqrt(2 * module.delta + 1) * value
        if module.odd:
            value = apply_generator(value) / module.recurrence[0]
        previous = None
        for k in range(module.odd, module.l2 - module.delta, 2):
            diagonal = module.recurrence[k] ** 2
            if k:
                diagonal += module.recurrence[k - 1] ** 2
            following = apply_generator(apply_generator(value)) + diagonal * value
            if k >= 2:
                following = following - (
                    module.recurrence[k - 1] * module.recurrence[k - 2] * previous
                )
            previous, value = (
                value,
                following / (module.recurrence[k] * module.recurrence[k + 1]),
            )
        if module.normalization == "norm":
            value = value / math.sqrt(2 * module.l2 + 1)
        elif module.normalization == "integral":
            value = value / math.sqrt(4 * math.pi)
        return bridge(value) if module.l1 < module.l3 else value

    if module.odd:
        value = apply_generator(value)
    previous = value
    result = module.coefficients[0] * value
    for k, coefficient in enumerate(module.coefficients[1:], start=1):
        following = (
            -2 * module.spectral_scale * apply_generator(apply_generator(value)) - value
        )
        if k > 1:
            following = 2 * following - previous
        previous, value = value, following
        result = result + coefficient * value
    return bridge(result) if module.l1 < module.l3 else result


def compiling():
    return torch.compiler.is_compiling() or get_proxy_mode() is not None


def sparse_product(module, name, features, direction, operation):
    indices = getattr(module, name + "_indices")
    columns = getattr(module, name + "_columns")
    coefficients = getattr(module, name + "_coefficients")
    weights = direction[..., columns] * coefficients
    return operation(features, indices, weights)


def evaluate(module, features, vectors, method, operation):
    """Keep generator actions sparse throughout the angular recurrence."""
    if not hasattr(module, "generator_indices"):
        return reference(module, features, vectors, method)
    direction = vectors / vectors.norm(dim=-1, keepdim=True)
    harmonic = module.harmonics(direction)

    def bridge(value):
        return sparse_product(module, "bridge", value, harmonic, operation)

    def generator(value):
        return sparse_product(module, "generator", value, direction, operation)

    value = features if module.l1 < module.l3 else bridge(features)
    if method == "recurrence":
        value = math.sqrt(2 * module.delta + 1) * value
        if module.odd:
            value = generator(value) / module.recurrence[0]
        previous = None
        for k in range(module.odd, module.l2 - module.delta, 2):
            diagonal = module.recurrence[k] ** 2
            if k:
                diagonal += module.recurrence[k - 1] ** 2
            following = generator(generator(value)) + diagonal * value
            if k >= 2:
                following = (
                    following
                    - module.recurrence[k - 1] * module.recurrence[k - 2] * previous
                )
            previous, value = (
                value,
                following / (module.recurrence[k] * module.recurrence[k + 1]),
            )
        if module.normalization == "norm":
            value = value / math.sqrt(2 * module.l2 + 1)
        elif module.normalization == "integral":
            value = value / math.sqrt(4 * math.pi)
        return bridge(value) if module.l1 < module.l3 else value
    if module.odd:
        value = generator(value)
    previous = value
    result = module.coefficients[0] * value
    for k, coefficient in enumerate(module.coefficients[1:], start=1):
        following = -2 * module.spectral_scale * generator(generator(value)) - value
        if k > 1:
            following = 2 * following - previous
        previous, value = value, following
        result = result + coefficient * value
    return bridge(result) if module.l1 < module.l3 else result


def shared_matrix(features, vectors):
    """Select a direction shared by sufficiently many feature channels."""
    minimum = config.SPHERICAL_MATRIX_MIN_CHANNELS
    return (
        minimum > 0
        and features.ndim >= 2
        and (vectors.ndim == 1 or statically_known_true(vectors.shape[-2] == 1))
        and statically_known_true(
            features.shape[-2] >= max(minimum, features.shape[-1])
        )
    )


def channel_matrix(module, features, vectors, method, operation=reference):
    """Compute the direction-dependent map once for every channel group."""
    identity = torch.eye(
        2 * module.l1 + 1, device=features.device, dtype=features.dtype
    )
    matrix = operation(module, identity, vectors, method)
    return features @ matrix


def forward(module, features, vectors, method):
    if config.TORCH_REFERENCE:
        return reference(module, features, vectors, method)
    if shared_matrix(features, vectors):
        return channel_matrix(module, features, vectors, method)
    if not config.TORCH_SPHERICAL_COUPLING:
        return reference(module, features, vectors, method)
    operation = (
        sparse_axis.evaluate if config.TORCH_SPHERICAL_ADJOINT else sparse_axis.primal
    )
    return evaluate(module, features, vectors, method, operation)
