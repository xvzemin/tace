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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wide_mixed_projection_live_map_and_adjoints(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA mixed Cartesian projector")
    from equivariantx import co3
    from equivariantx.ops.cuda.co3.basis import project_mixed_layout
    from equivariantx.ops.cuda.mbe.ace import ictd_symmetric_contraction as kernel
    from equivariantx.ops.torch.co3.basis import rank_two

    module = co3.Projector("80x0e+80x2e+80x3o+80x1o+80x2e+80x3o").to("cuda", dtype)
    sections = list(kernel._mixed_sections(module.irreps_in, module.groups))
    x = torch.randn(
        5, module.irreps_in.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    b = torch.randn(27, 27, device="cuda", dtype=dtype, requires_grad=True) / 8

    def reference(a, matrix):
        parts = []
        for (mul, ir), section in zip(module.irreps_in, module.slices, strict=True):
            value = a[:, section].reshape(a.size(0), mul, ir.dim)
            if ir.l == 2:
                value = rank_two(value)
            elif ir.l == 3:
                value = value @ matrix.T
            parts.append(value.flatten(1))
        return torch.cat(parts, -1)

    actual = project_mixed_layout(x, b, sections)
    expected = reference(x, b)
    tol = 6e-5 if dtype == torch.float32 else 2e-11
    direction = torch.randn_like(x)
    matrix_direction = torch.randn_like(b) / 20

    def derivatives(y):
        dx, db = torch.autograd.grad(y.sin().mean(), (x, b), create_graph=True)
        hx, hb = torch.autograd.grad(
            (dx * direction).sum() + (db * matrix_direction).sum(),
            (x, b),
            create_graph=True,
        )
        return (
            y,
            dx,
            db,
            hx,
            hb,
            torch.autograd.grad(
                (hx * direction).sum() + (hb * matrix_direction).sum(), (x, b)
            ),
        )

    from torch.utils._pytree import tree_leaves

    values, refs = tree_leaves(derivatives(actual)), tree_leaves(derivatives(expected))
    for value, ref in zip(values, refs, strict=True):
        torch.testing.assert_close(value, ref, rtol=tol, atol=tol)
        assert (value - ref).norm() / ref.norm() < tol
    # Mutated or differentiable module tables are never replaced by a
    # precomputed isotropic formula. Reuse the independent dense reference.
    module.basis_3.add_(0.017)
    torch.testing.assert_close(
        kernel.project(module, x),
        reference(x, module.basis_3 @ module.basis_3.T),
        rtol=tol,
        atol=tol,
    )
    if dtype == torch.float32:
        from torch.fx.experimental.proxy_tensor import make_fx

        captured = make_fx(lambda values: kernel.project(module, values))(x.detach())
        assert all(
            "stf_mixed_layout" not in str(n.target) for n in captured.graph.nodes
        )
    module.basis_3.requires_grad_(True)
    y = kernel.project(module, x)
    expected = reference(x, module.basis_3 @ module.basis_3.T)
    (g,) = torch.autograd.grad(y.square().mean(), (module.basis_3,))
    (target,) = torch.autograd.grad(expected.square().mean(), (module.basis_3,))
    torch.testing.assert_close(g, target, rtol=tol, atol=tol)
