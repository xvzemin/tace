"""Tests for the corresponding EQX source module."""

import pytest
import torch

from .testing import (
    check_ictd_contraction,
    check_parent_aoti,
    check_parent_provider,
    check_spherical_compile,
)


@pytest.mark.parametrize("normalization", ["element", "path"])
def test_coefficient_basis_normalization(normalization, double_precision):
    from equivariantx import co3
    from equivariantx.mbe.ace.ictd_semi_coupled import cartesian_coefficient

    linear = co3.Linear(
        "2x0e+3x0e+2x2e+1x2e",
        "3x2e+2x0e",
        instructions=[(0, 1), (1, 1), (2, 0), (3, 0), (2, 0)],
        internal_weights=False,
        shared_weights=False,
        path_normalization=normalization,
        output_basis="spherical",
    )
    raw = cartesian_coefficient(linear)
    assert raw.instructions == linear.instructions
    assert linear.output_basis == "spherical" and raw.output_basis == "cartesian"
    assert cartesian_coefficient(raw) is raw
    transform = co3.ChangeOfBasis(raw.irreps_out, inverse=True)
    x = torch.randn(5, linear.irreps_in.dim, requires_grad=True)
    weight = torch.randn(5, linear.weight_numel, requires_grad=True)
    actual, expected = transform(raw(x, weight)), linear(x, weight)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        value.sin().sum(), (x, weight), create_graph=True
                    )
                ]
            )
            * 0.1
            for value in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_spherical_compile(backend, double_precision, monkeypatch):
    check_spherical_compile("semi", backend, monkeypatch)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_spherical_force_gradient_aoti(
    backend, tmp_path, double_precision, monkeypatch
):
    check_parent_aoti(
        "co3",
        "semi",
        tmp_path,
        monkeypatch,
        shared_weights=True,
        output_basis="spherical",
        backend=backend,
    )


@pytest.mark.parametrize(
    "device,nodes,backend,child_backend",
    [
        ("cpu", 0, "torch", "cuda"),
        ("cpu", 7, "torch", "torch"),
        ("cpu", 7, "cuda", "cuda"),
        ("cuda", 0, "cuda", "cuda"),
        ("cuda", 7, "torch", "cuda"),
        ("cuda", 7, "cuda", "torch"),
    ],
)
@pytest.mark.parametrize("shared_weights", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_spherical_output(
    device,
    nodes,
    backend,
    child_backend,
    shared_weights,
    reference,
    double_precision,
    monkeypatch,
):
    check_parent_provider(
        "co3",
        "semi",
        device,
        nodes,
        child_backend,
        reference,
        monkeypatch,
        shared_weights=shared_weights,
        output_basis="spherical",
        backend=backend,
    )


@pytest.mark.parametrize("invalid", ["basis", "bias", "internal"])
def test_coefficient_contract(invalid):
    from equivariantx import co3
    from equivariantx.mbe import ace

    irreps = "2x0e+2x2e"
    product = co3.TensorProduct(
        irreps,
        irreps,
        irreps,
        [(0, 0, 0, "uuu", False), (0, 1, 1, "uuu", False)],
        internal_weights=False,
        shared_weights=False,
    )
    linears = [
        co3.Linear(irreps, irreps, internal_weights=False, output_basis="spherical")
    ]
    linears.append(
        co3.Linear(
            irreps,
            irreps,
            output_basis="cartesian" if invalid == "basis" else "spherical",
            internal_weights=invalid == "internal",
            biases=invalid == "bias",
        )
    )
    with pytest.raises(ValueError, match="output basis|unbiased external"):
        ace.IctdSemiCoupledSymmetricContraction([product], linears)


@pytest.mark.parametrize("shared_weights", [False, True])
def test_force_gradient_aoti(shared_weights, tmp_path, double_precision, monkeypatch):
    check_parent_aoti(
        "co3", "semi", tmp_path, monkeypatch, shared_weights=shared_weights
    )


@pytest.mark.parametrize("device,nodes", [("cpu", 0), ("cpu", 7), ("cuda", 7)])
@pytest.mark.parametrize("reference", [False, True])
def test_shared_descriptors(device, nodes, reference, double_precision, monkeypatch):
    check_parent_provider(
        "co3",
        "semi",
        device,
        nodes,
        "torch",
        reference,
        monkeypatch,
        shared_weights=True,
    )


@pytest.mark.parametrize("device,nodes", [("cpu", 0), ("cpu", 7), ("cuda", 7)])
@pytest.mark.parametrize("child_backend", ["torch", "cuda"])
@pytest.mark.parametrize("reference", [False, True])
def test_parent_provider(
    device, nodes, child_backend, reference, double_precision, monkeypatch
):
    check_parent_provider(
        "co3", "semi", device, nodes, child_backend, reference, monkeypatch
    )


@pytest.mark.parametrize("kind", ["ace", "ace_split"])
def test_ictd_contractions(kind, double_precision, monkeypatch):
    check_ictd_contraction(kind, monkeypatch)
