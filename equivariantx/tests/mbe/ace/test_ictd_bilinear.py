"""Tests for the corresponding EQX source module."""

import pytest

from .testing import (
    check_ictd_contraction,
    check_parent_aoti,
    check_parent_provider,
    check_spherical_compile,
)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_spherical_compile(backend, double_precision, monkeypatch):
    check_spherical_compile("bilinear", backend, monkeypatch)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_spherical_force_gradient_aoti(
    backend, tmp_path, double_precision, monkeypatch
):
    check_parent_aoti(
        "co3",
        "bilinear",
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
        "bilinear",
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
        [(0, 0, 0, "uuu", True), (0, 1, 1, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    linear = co3.Linear(
        irreps,
        irreps,
        output_basis="spherical",
        internal_weights=invalid == "internal",
        biases=invalid == "bias",
    )
    shared = (
        co3.Linear(irreps, irreps, internal_weights=False)
        if invalid == "basis"
        else None
    )
    with pytest.raises(ValueError, match="output basis|unbiased external"):
        ace.IctdBilinearContraction(product, linear, shared_linear=shared)


@pytest.mark.parametrize("shared_weights", [False, True])
def test_force_gradient_aoti(shared_weights, tmp_path, double_precision, monkeypatch):
    check_parent_aoti(
        "co3", "bilinear", tmp_path, monkeypatch, shared_weights=shared_weights
    )


@pytest.mark.parametrize("device,nodes", [("cpu", 0), ("cpu", 7), ("cuda", 7)])
@pytest.mark.parametrize("reference", [False, True])
def test_shared_descriptors(device, nodes, reference, double_precision, monkeypatch):
    check_parent_provider(
        "co3",
        "bilinear",
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
        "co3", "bilinear", device, nodes, child_backend, reference, monkeypatch
    )


@pytest.mark.parametrize(
    "kind", ["bilinear", "bilinear_split", "bilinear_wide", "moe", "moe_split"]
)
def test_ictd_contractions(kind, double_precision, monkeypatch):
    check_ictd_contraction(kind, monkeypatch)
