"""Explicit real quaternion polynomials with arbitrary-order Torch derivatives."""

from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.quaternion import polynomial_data


@lru_cache(maxsize=64)
def polynomial_indices(lmax, device, dtype):
    """Create gather/reduction indices without runtime integer powers."""
    pointers, exponents, coefficients = polynomial_data(lmax)
    stride = 2 * lmax + 1
    return (
        tuple(
            torch.tensor(
                [axis * stride + powers[axis] for powers in exponents],
                device=device,
                dtype=torch.long,
            )
            for axis in range(4)
        ),
        torch.tensor(
            [
                i
                for i in range(len(pointers) - 1)
                for _ in range(pointers[i], pointers[i + 1])
            ],
            device=device,
            dtype=torch.long,
        ),
        torch.tensor(coefficients, device=device, dtype=dtype),
    )


def polynomial(q, lmax):
    """Evaluate quaternion monomials without singular derivatives at zero."""
    tracing = (
        torch.compiler.is_compiling()
        or is_fake(q)
        or get_proxy_mode() is not None
        or torch._C._are_functorch_transforms_active()
    )
    tables = polynomial_indices.__wrapped__ if tracing else polynomial_indices
    indices, destination, coefficient = tables(lmax, q.device, q.dtype)
    powers = [torch.ones_like(q)]
    for _ in range(2 * lmax):
        powers.append(powers[-1] * q)
    powers = torch.stack(powers, dim=-1).reshape(q.shape[0], 4 * (2 * lmax + 1))
    terms = coefficient[None]
    for index in indices:
        terms = terms * powers.index_select(1, index)
    width = sum((2 * degree + 1) ** 2 for degree in range(lmax + 1))
    return q.new_zeros((q.shape[0], width)).index_add(1, destination, terms)


def quaternion_wigner(vectors, lmax):
    """Construct the same chart and polynomial as the native CUDA provider."""
    from ....o2.rotation_matrix import alignment_quaternion, rotation_matrix_to_y_axis

    if lmax == 0:
        return vectors.new_ones((vectors.shape[0], 1))
    if lmax == 1:
        return torch.cat(
            (
                vectors.new_ones((vectors.shape[0], 1)),
                rotation_matrix_to_y_axis(vectors, backend="torch").flatten(1),
            ),
            dim=-1,
        )
    return polynomial(alignment_quaternion(vectors, backend="torch"), lmax)


def quaternion_polynomial(lmax, transpose, values):
    """Evaluate directional polynomial derivatives and their vector cotangent."""
    q, *directions = values
    cotangent = directions.pop() if transpose else None
    rank = len(directions)
    if rank == 0 and not transpose:
        return polynomial(q, lmax)
    tracing = (
        torch.compiler.is_compiling()
        or is_fake(q)
        or get_proxy_mode() is not None
        or torch._C._are_functorch_transforms_active()
    )
    tables = derivative_indices.__wrapped__ if tracing else derivative_indices
    powers_index, axes, destinations, coefficients, components = tables(
        lmax, rank, transpose, q.device, q.dtype
    )
    width = 4 if transpose else sum((2 * degree + 1) ** 2 for degree in range(lmax + 1))
    output = q.new_zeros((q.shape[0], width))
    zero = sum(value[..., :0].sum() for value in values)
    if not coefficients.numel():
        return output + zero
    powers = [torch.ones_like(q)]
    for _ in range(2 * lmax):
        powers.append(powers[-1] * q)
    powers = torch.stack(powers, dim=-1).flatten(1)
    terms = coefficients[None]
    for index in powers_index:
        terms = terms * powers.index_select(1, index)
    for direction, index in zip(directions, axes):
        terms = terms * direction.index_select(1, index)
    if transpose:
        terms = terms * cotangent.index_select(1, components)
    return output.index_add(1, destinations, terms) + zero


@lru_cache(maxsize=128)
def derivative_terms(lmax, rank, transpose):
    """Differentiate exact monomial coefficients without entering an AD transform."""
    if rank + int(transpose) > 2 * lmax:
        return ()
    pointers, exponents, coefficients = polynomial_data(lmax)
    terms = [
        (exponents[i], (), output, coefficients[i])
        for output in range(len(pointers) - 1)
        for i in range(pointers[output], pointers[output + 1])
    ]
    for _ in range(rank + int(transpose)):
        derivatives = []
        for powers, axes, output, coefficient in terms:
            for axis, power in enumerate(powers):
                if power:
                    reduced = list(powers)
                    reduced[axis] -= 1
                    derivatives.append(
                        (tuple(reduced), (*axes, axis), output, coefficient * power)
                    )
        terms = derivatives
    return tuple(terms)


@lru_cache(maxsize=64)
def derivative_indices(lmax, rank, transpose, device, dtype):
    """Place derivative gathers on the device without caching transformed tensors."""
    terms = derivative_terms(lmax, rank, transpose)
    stride = 2 * lmax + 1

    def indices(values):
        return torch.tensor(values, device=device, dtype=torch.long)

    return (
        tuple(
            indices([axis * stride + powers[axis] for powers, _, _, _ in terms])
            for axis in range(4)
        ),
        tuple(indices([axes[i] for _, axes, _, _ in terms]) for i in range(rank)),
        indices([axes[-1] if transpose else output for _, axes, output, _ in terms]),
        torch.tensor(
            [coefficient for _, _, _, coefficient in terms], device=device, dtype=dtype
        ),
        indices([output for _, _, output, _ in terms]),
    )
