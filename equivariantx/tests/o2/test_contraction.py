"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
from equivariantx.tests.o2.testing import (
    _asymmetric_contractions,
    _time_reverse,
    _transform,
)


@pytest.mark.parametrize("correlation", [2, 3])
@pytest.mark.parametrize("path_mode", ["sum", "expand"])
@pytest.mark.parametrize("batch_size", [0, 4])
def test_o2_asymmetric_contraction_algorithms_match(correlation, path_mode, batch_size):
    recursive, dense = _asymmetric_contractions(correlation, path_mode)
    inputs = [
        recursive.irreps_in.randn(
            batch_size, -1, dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        for _ in range(correlation)
    ]
    weights = torch.randn(
        batch_size,
        recursive.weight_numel,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    assert recursive.order_num_paths == dense.order_num_paths
    actual, expected = recursive(inputs, weights), dense(inputs, weights)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (*inputs, weights)),
        torch.autograd.grad(expected.square().sum(), (*inputs, weights)),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("algorithm", ["recursive", "dense"])
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_asymmetric_contraction_is_equivariant(algorithm, reflected):
    module = o2.AsymmetricContraction(
        "2x0e+2x0o+2x1m",
        "2x0e+2x0o+2x1m+2x2m",
        2,
        algorithm=algorithm,
        backend="torch" if algorithm == "dense" else None,
    ).to(DEVICE, DTYPE)
    inputs = [
        module.irreps_in.randn(4, -1, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]
    weights = torch.randn(4, module.weight_numel, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.31, dtype=DTYPE, device=DEVICE)
    output = module(inputs, weights)
    expected = _transform(output, module.irreps_out, angle, reflected)
    transformed = [
        _transform(features, module.irreps_in, angle, reflected) for features in inputs
    ]
    torch.testing.assert_close(module(transformed, weights), expected)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_o2_contraction_preserves_time_parity(backend, symmetric):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    irreps_in = o2.Irreps("2x0ee+2x0oo+2x1mo")
    contraction = (o2.SymmetricContraction if symmetric else o2.AsymmetricContraction)(
        irreps_in,
        "2x0ee+2x0eo+2x0oe+2x0oo+2x1me+2x1mo",
        correlation=3,
        algorithm="recursive",
        backend=backend,
    ).to(DEVICE, DTYPE)
    inputs = [irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE) for _ in range(3)]
    transformed = [_time_reverse(value, irreps_in) for value in inputs]
    inputs, transformed = (
        (inputs[0], transformed[0]) if symmetric else (inputs, transformed)
    )
    weights = torch.randn(5, contraction.weight_numel, dtype=DTYPE, device=DEVICE)
    output = contraction(inputs, weights)
    torch.testing.assert_close(
        contraction(transformed, weights),
        _time_reverse(output, contraction.irreps_out),
    )


@pytest.mark.parametrize("parity,correlation", [(False, 1), (False, 2), (True, 2)])
def test_symmetric_contraction_algorithms(parity, correlation, double_precision):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    irreps = o2.Irreps("2x0e+2x0o+2x1m+2x2m" if parity else "2x0e+2x1m+2x2m")
    module = o2.SymmetricContraction(irreps, irreps, correlation, backend="torch").to(device)
    x = torch.randn(4, irreps.dim, device=device, requires_grad=True)
    weight = torch.randn(4, module.weight_numel, device=device, requires_grad=True)
    expected = module(x, weight)
    module.set_algorithm("dense")
    actual = module(x, weight)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    for _ in range(2):
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    grad.flatten()
                    for grad in torch.autograd.grad(
                        (value * seed).sum(),
                        (x, weight),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    module.set_algorithm("recursive")
    assert not list(module.state_dict())


def test_symmetric_contraction_removes_antisymmetric_self_product(double_precision):
    module = o2.SymmetricContraction("2x1m", "2x0e+2x0o+2x2m", 2)
    assert module.order_num_paths == (0, 2)
    value = module(torch.randn(3, 4), torch.ones(module.weight_numel))
    torch.testing.assert_close(value[:, 2:4], torch.zeros(3, 2))


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("path_mode", ["sum", "expand"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_contraction_backends(symmetric, path_mode, dtype, backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cls = o2.SymmetricContraction if symmetric else o2.AsymmetricContraction
    args = ("2x0e+2x0o+2x1m", "2x0e+2x0o+2x1m+2x2m", 3)
    reference = cls(*args, algorithm="dense", path_mode=path_mode, backend="torch").to(DEVICE, dtype)
    module = cls(*args, algorithm="recursive", path_mode=path_mode, backend=backend).to(
        DEVICE, dtype
    )
    assert module.order_num_paths == reference.order_num_paths
    assert not module.state_dict()
    inputs = [
        torch.randn(
            3, module.irreps_in.dim, device=DEVICE, dtype=dtype, requires_grad=True
        )
        * 0.2
        for _ in range(1 if symmetric else 3)
    ]
    weight = (
        torch.randn(module.weight_numel, device=DEVICE, dtype=dtype, requires_grad=True)
        * 0.2
    )
    features = inputs[0] if symmetric else inputs
    actual, expected = module(features, weight), reference(features, weight)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        actual, expected = [
            torch.cat(
                [
                    x.flatten()
                    for x in torch.autograd.grad(
                        value.sin().sum(), (*inputs, weight), create_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]
    for batch in (0, 1):
        features = inputs[0][:batch] if symmetric else [x[:batch] for x in inputs]
        torch.testing.assert_close(
            module(features, weight),
            reference(features, weight),
            atol=tolerance,
            rtol=tolerance,
        )
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    features = inputs[0] if symmetric else inputs
    torch.testing.assert_close(
        compiled(features, weight),
        reference(features, weight),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("empty", [False, True])
def test_contraction_high_orders_and_broadcast(backend, empty):
    if backend == "cuda" and DEVICE.type != "cuda":
        pytest.skip("CUDA is unavailable")
    args = ("2x0o+2x8m", "2x31m" if empty else "2x0e+2x0o+2x8m+2x16m+2x24m+2x32m", 4)
    module = (
        o2.AsymmetricContraction(*args, algorithm="recursive", backend=backend)
        .to(DEVICE)
        .float()
        .double()
    )
    reference = o2.AsymmetricContraction(*args, algorithm="dense", backend="torch").to(
        DEVICE, torch.float64
    )
    inputs = [
        torch.randn(
            *shape,
            module.irreps_in.dim,
            device=DEVICE,
            dtype=torch.float64,
            requires_grad=True,
        )
        for shape in ((2, 1), (1, 3), (), (2, 3))
    ]
    weights = torch.randn(
        3, module.weight_numel, device=DEVICE, dtype=torch.float64, requires_grad=True
    )
    actual, expected = module(inputs, weights), reference(inputs, weights)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    gradients = [
        torch.autograd.grad(x.sin().sum(), (*inputs, weights), create_graph=True)
        for x in (actual, expected)
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("path_mode", ["sum", "expand"])
def test_staged_sparse_contraction(backend, symmetric, path_mode, monkeypatch):
    from equivariantx.ops import config

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_O2_CONTRACTION", True, raising=False)
    monkeypatch.setattr(config, "O2_CONTRACTION_STAGED", True, raising=False)
    monkeypatch.setattr(config, "O2_CONTRACTION_EAGER_STAGED", True)
    device = "cuda" if backend == "cuda" else "cpu"
    cls = o2.SymmetricContraction if symmetric else o2.AsymmetricContraction
    args = "2x0e+2x1m", "2x0e+2x0o+2x1m+2x2m", 3
    layer = (
        cls(*args, algorithm="recursive", path_mode=path_mode, backend=backend)
        .to(device)
        .float()
        .double()
    )
    ordinary = cls(*args, algorithm="dense", path_mode=path_mode, backend="torch").to(device).double()
    x = (
        torch.randn(2, 4, 6, dtype=torch.float64, device=device)[..., ::2] * 0.2
    ).requires_grad_()
    w = (
        torch.randn(1, layer.weight_numel, dtype=torch.float64, device=device) * 0.2
    ).requires_grad_()

    def call(a, b):
        return layer(a if symmetric else (a, a.sin(), a.cos()), b)

    def reference(a, b):
        return ordinary(a if symmetric else (a, a.sin(), a.cos()), b)

    # Noncontiguous feature batches and shared weights exercise all broadcasts.
    x = (
        x.transpose(-1, -2)
        .reshape(2, 12)[..., : layer.irreps_in.dim]
        .detach()
        .requires_grad_()
    )
    actual, expected = call(x, w), reference(x, w)
    for order in range(4):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        if order < 3:
            probe = torch.randn_like(actual) / actual.numel() ** 0.5
            actual, expected = [
                torch.cat(
                    [
                        g.flatten()
                        for g in torch.autograd.grad(
                            (value.sin() * probe).sum(), (x, w), create_graph=True
                        )
                    ]
                )
                for value in (actual, expected)
            ]
    torch.testing.assert_close(
        torch.func.jvp(call, (x, w), (x, w)), torch.func.jvp(reference, (x, w), (x, w))
    )
    torch.testing.assert_close(call(x[:0], w), reference(x[:0], w))
    assert not layer.state_dict()
