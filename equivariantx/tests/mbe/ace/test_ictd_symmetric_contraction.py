"""ICTD uuu fully connected symmetric contractions."""

import pytest
import torch

from .testing import (
    check_ictd_contraction,
    check_parent_aoti,
    check_parent_provider,
    check_spherical_compile,
)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 6])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_projection_backend(rank, backend, double_precision):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx import co3
    from equivariantx.ops.torch.mbe.ace import ictd_symmetric_contraction as kernel

    if backend == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        from equivariantx.ops.cuda.mbe.ace import ictd_symmetric_contraction as kernel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    child_backend = "cuda" if backend == "torch" else "torch"
    irreps = co3.Irreps([(2, (rank, (-1) ** rank))])
    projection = co3.Projector(irreps, backend=child_backend).to(device)
    transform = co3.ChangeOfBasis(irreps, inverse=True, backend=child_backend).to(
        device
    )
    matrix = getattr(transform, f"basis_{rank}")
    features = torch.randn(3, irreps.dim, device=device, requires_grad=True)

    def evaluate(features):
        return kernel.project(projection, features), kernel.change_basis(
            transform, features
        )

    graph = make_fx(evaluate)(features)
    if backend == "torch":
        assert not any(
            str(node.target).startswith("eqx.") for node in graph.graph.nodes
        )
    spherical = features.reshape(3, 2, 3**rank) @ matrix
    expected = ((spherical @ matrix.T).flatten(1), spherical.flatten(1))
    actual = graph(features)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        actual, expected = [
            torch.autograd.grad(
                sum(value.sin().sum() for value in outputs), features, create_graph=True
            )
            for outputs in (actual, expected)
        ]


@pytest.mark.parametrize("normalization", ["element", "path"])
def test_coefficient_basis_normalization(normalization, double_precision):
    from equivariantx import co3
    from equivariantx.mbe.ace.ictd_symmetric_contraction import cartesian_coefficient

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
    check_spherical_compile("uuu_fully_connected", backend, monkeypatch)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_spherical_force_gradient_aoti(
    backend, tmp_path, double_precision, monkeypatch
):
    check_parent_aoti(
        "co3",
        "uuu_fully_connected",
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
        "uuu_fully_connected",
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
        ace.IctdUuuFullyConnectedSymmetricContraction([product], linears)


@pytest.mark.parametrize("shared_weights", [False, True])
def test_force_gradient_aoti(shared_weights, tmp_path, double_precision, monkeypatch):
    check_parent_aoti(
        "co3",
        "uuu_fully_connected",
        tmp_path,
        monkeypatch,
        shared_weights=shared_weights,
    )


@pytest.mark.parametrize("device,nodes", [("cpu", 0), ("cpu", 7), ("cuda", 7)])
@pytest.mark.parametrize("reference", [False, True])
def test_shared_descriptors(device, nodes, reference, double_precision, monkeypatch):
    check_parent_provider(
        "co3",
        "uuu_fully_connected",
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
        "co3",
        "uuu_fully_connected",
        device,
        nodes,
        child_backend,
        reference,
        monkeypatch,
    )


@pytest.mark.parametrize("kind", ["ace", "ace_split"])
def test_ictd_contractions(kind, double_precision, monkeypatch):
    check_ictd_contraction(kind, monkeypatch)
