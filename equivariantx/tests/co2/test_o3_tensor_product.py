"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import co2, co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    from equivariantx.tests.utils.testing import check_method_precision

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = co2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    inputs = (
        co3.Irreps("2x1o").randn(3, -1, device=device),
        torch.randn(3, 3, device=device),
        torch.randn(3, layer.weight_numel, device=device),
    )
    check_method_precision(layer, [("contract", inputs, {})])


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_o3_equivalence(double_precision, mode, normalization, path_normalization):
    a, b = o3.Irreps("2x1o+3x2e"), o3.Irreps.spherical_harmonics(2)
    output, instructions = [], []
    for i, (mul, ir) in enumerate(a):
        for j, (_, jr) in enumerate(b):
            for kr in ir * jr:
                instructions.append((i, j, len(output), mode, True, 0.8))
                output.append((mul if mode == "uvu" else 4, kr))
    c = o3.Irreps(output)
    instructions.append((*instructions[0][:5], 1.3))
    options = dict(
        internal_weights=False,
        shared_weights=False,
        path_normalization=path_normalization,
        irrep_normalization="norm",
        in1_var=[0.8, 1.3],
        in2_var=[0.7, 1.2, 1.1],
    )
    ref = o3.TensorProduct(a, b, c, instructions, **options).to(DEVICE)
    module = co2.O3TensorProduct(
        a,
        b,
        c,
        instructions,
        normalization=normalization,
        input_basis="spherical",
        output_basis="spherical",
        **options,
    ).to(DEVICE)
    cartesian = co3.TensorProduct(a, b, c, instructions, **options).to(DEVICE)
    x = torch.randn(3, a.dim, device=DEVICE, requires_grad=True)
    vectors = torch.randn(3, 3, device=DEVICE, requires_grad=True)
    weight = torch.randn(3, ref.weight_numel, device=DEVICE, requires_grad=True)
    harmonics = o3.spherical_harmonics(b, vectors, True, normalization)
    expected = ref(x, harmonics, weight)
    actual = module(x, vectors, weight)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    transforms = [
        co3.ChangeOfBasis(irreps, inverse).to(DEVICE)
        for irreps, inverse in ((a, False), (b, False), (c, True))
    ]
    torch.testing.assert_close(
        actual,
        transforms[2](cartesian(transforms[0](x), transforms[1](harmonics), weight)),
        atol=3e-12,
        rtol=3e-12,
    )
    for _ in range(2):
        probe = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (value * probe).sum(),
                        (x, vectors, weight),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
    assert module(x[:0], vectors[:0], weight[:0]).shape == (0, c.dim)


def test_scatter_compile_and_high_degree(double_precision):
    a, b, c = "2x5e", "1x3o", "2x4o+2x5o"
    ins = [(0, 0, 0, "uvu", True), (0, 0, 1, "uvu", True)]
    module = co2.O3TensorProduct(
        a,
        b,
        c,
        ins,
        shared_weights=False,
        internal_weights=False,
        input_basis="spherical",
        output_basis="spherical",
    ).to(DEVICE)
    ref = o3.TensorProduct(
        a, b, c, ins, shared_weights=False, internal_weights=False
    ).to(DEVICE)
    x = torch.randn(4, o3.Irreps(a).dim, device=DEVICE)
    edge_index = torch.tensor([[0, 1, 2, 3, 2], [1, 0, 1, 1, 0]], device=DEVICE)
    vectors = torch.randn(5, 3, device=DEVICE)
    weight = torch.randn(5, ref.weight_numel, device=DEVICE)
    sh = o3.spherical_harmonics(o3.Irreps(b), vectors, True, normalization="component")
    expected = ref(x[edge_index[0]], sh, weight)
    actual = module.forward_scatter(x, vectors, edge_index, weight)
    torch.testing.assert_close(
        actual,
        expected.new_zeros(4, expected.shape[-1]).index_add(0, edge_index[1], expected),
        atol=3e-12,
        rtol=3e-12,
    )
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(x[edge_index[0]], vectors, weight), expected, atol=3e-12, rtol=3e-12
    )


def test_third_derivative(double_precision):
    a, b, c = "1x3o", "1x2e", "1x3o"
    options = dict(internal_weights=False, shared_weights=False)
    ins = [(0, 0, 0, "uvu", True)]
    module = co2.O3TensorProduct(
        a, b, c, ins, input_basis="spherical", output_basis="spherical", **options
    ).to(DEVICE)
    reference = o3.TensorProduct(a, b, c, ins, **options).to(DEVICE)
    x = torch.randn(2, 7, device=DEVICE, requires_grad=True)
    r = torch.randn(2, 3, device=DEVICE, requires_grad=True)
    w = torch.randn(2, 1, device=DEVICE, requires_grad=True)
    actual = module(x, r, w)
    expected = reference(
        x, o3.spherical_harmonics(o3.Irreps(b), r, True, "component"), w
    )
    for _ in range(3):
        probe = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (v * probe).sum(),
                        (x, r, w),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for v in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)


def test_single_precision_and_poles():
    torch.set_default_dtype(torch.float32)
    a, b, c = "2x4e", "1x3o", "2x3o+2x4o+2x5o"
    ins = [(0, 0, i, "uvu", True) for i in range(3)]
    ref = o3.TensorProduct(a, b, c, ins).to(DEVICE)
    module = co2.O3TensorProduct.from_tensor_product(
        ref, input_basis="spherical", output_basis="spherical"
    )
    vectors = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1e-8, 1.0, 1e-8],
        ],
        device=DEVICE,
        requires_grad=True,
    )
    x = torch.randn(5, o3.Irreps(a).dim, device=DEVICE)
    actual = module(x, vectors)
    expected = ref(x, o3.spherical_harmonics(o3.Irreps(b), vectors, True, "component"))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual, expected = [
        torch.autograd.grad(v.square().sum(), vectors, retain_graph=True)[0]
        for v in (actual, expected)
    ]
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


def test_unweighted_paths_and_validation(double_precision):
    a, b, c = "1x0e+2x1o", "1x0e+1x1o", "1x0e+2x1o+2x0e"
    instructions = [
        (0, 0, 0, "uvu", False),
        (1, 0, 1, "uvu", True),
        (1, 1, 2, "uvu", False),
    ]
    reference = o3.TensorProduct(a, b, c, instructions).to(DEVICE)
    module = co2.O3TensorProduct.from_tensor_product(
        reference, input_basis="spherical", output_basis="spherical"
    )
    x = reference.irreps_in1.randn(3, -1, device=DEVICE)
    r = torch.randn(3, 3, device=DEVICE)
    sh = o3.spherical_harmonics(reference.irreps_in2, r, True, "component")
    torch.testing.assert_close(module(x, r), reference(x, sh), atol=2e-12, rtol=2e-12)
    for mode, weighted in (("uvu", True), ("uvw", False)):
        with pytest.raises(ValueError):
            co2.O3TensorProduct("1x1o", "1x1o", "2x0e", [(0, 0, 0, mode, weighted)])


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("input_basis", ["cartesian", "spherical"])
@pytest.mark.parametrize("output_basis", ["cartesian", "spherical"])
def test_compact_backend(
    backend, input_basis, output_basis, double_precision, monkeypatch
):
    import copy

    from equivariantx.ops import config
    from equivariantx.ops.torch.co2.o3_tensor_product import reference

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    a, b, c = "2x2e", "3o", "2x2o+2x3o"
    layer = (
        co2.O3TensorProduct(
            a,
            b,
            c,
            [(0, 0, 0, "uvu", True), (0, 0, 1, "uvu", True)],
            backend=backend,
            input_basis=input_basis,
            output_basis=output_basis,
            shared_weights=False,
            internal_weights=False,
        )
        .double()
        .to(device)
    )
    source = (torch.randn(2, 10, device=device) * 0.1).requires_grad_()
    vectors = (torch.randn(2, 3, device=device) + 0.5).requires_grad_()
    weight = (torch.randn(1, layer.weight_numel, device=device) * 0.1).requires_grad_()
    embedding = co3.ChangeOfBasis(a).double().to(device)
    features = embedding(source) if input_basis == "cartesian" else source
    actual = layer(features, vectors, weight)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    ordinary = copy.deepcopy(layer)
    ordinary.backend = "torch"
    expected = ordinary(features, vectors, weight)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        probe = torch.randn_like(actual) / actual.numel() ** 0.5
        actual, expected = [
            torch.cat(
                [
                    v.flatten()
                    for v in torch.autograd.grad(
                        (value.sin() * probe).sum(),
                        (source, vectors, weight),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(
        layer.contract(embedding(source), vectors, weight),
        reference(layer, embedding(source), vectors, weight),
        atol=3e-12,
        rtol=3e-12,
    )
    # Old serialized layers without compact metadata keep the original path.
    del layer.couplings
    del layer.feature_dim
    torch.testing.assert_close(
        layer(features, vectors, weight),
        layer.projection(reference(layer, layer.embedding(features), vectors, weight)),
        atol=3e-12,
        rtol=3e-12,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_backend_routing(device, monkeypatch, double_precision):
    import copy
    from unittest.mock import patch

    from equivariantx.ops import config
    from equivariantx.ops.cuda.co2 import o3_tensor_product as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = co2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        backend="cuda",
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    reference = copy.deepcopy(module)
    reference.backend = "torch"
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        values = (
            torch.randn(count, 6, device=device),
            torch.randn(count, 3, device=device),
            torch.randn(count, module.weight_numel, device=device),
        )
        for name in ("forward", "contract"):
            with patch.object(kernel, name, wraps=getattr(kernel, name)) as selected:
                actual = getattr(module, name)(*values)
            assert selected.call_count == (device == "cuda")
            torch.testing.assert_close(actual, getattr(reference, name)(*values))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("spherical", [False, True])
def test_reference_bypasses_children(device, spherical, monkeypatch, double_precision):
    import copy

    from equivariantx.ops import config
    from equivariantx.ops.torch.co2 import o3_tensor_product as kernel
    from equivariantx.tests.conv.testing import assert_native_autograd

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    basis = "spherical" if spherical else "cartesian"
    module = co2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        input_basis=basis,
        output_basis=basis,
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    reference = copy.deepcopy(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("An ordinary product dispatched its child backend.")

    for child in module.modules():
        if child is not module and hasattr(child, "backend"):
            child.backend = "cuda"
            monkeypatch.setattr(child, "forward", forbidden)
    backends = [
        (name, child.backend)
        for name, child in module.named_modules()
        if hasattr(child, "backend")
    ]
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        values = tuple(
            value.requires_grad_()
            for value in (
                torch.randn(count, 6, device=device) * 0.1,
                torch.randn(count, 3, device=device) + 1,
                torch.randn(count, module.weight_numel, device=device) * 0.1,
            )
        )
        actual, expected = module(*values), reference(*values)
        for _ in range(4):
            torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
            assert_native_autograd(actual)
            actual, expected = [
                sum(
                    g.sin().sum()
                    for g in torch.autograd.grad(
                        v.sin().sum(), values, create_graph=True
                    )
                )
                for v in (actual, expected)
            ]
        tangent = tuple(torch.randn_like(value) for value in values)
        torch.testing.assert_close(
            torch.func.jvp(module, values, tangent),
            torch.func.jvp(reference, values, tangent),
        )
    assert backends == [
        (name, child.backend)
        for name, child in module.named_modules()
        if hasattr(child, "backend")
    ]
    assert config.TORCH_REFERENCE is True
    # A direct oracle remains ordinary for CUDA-declared modules without the switch.
    module.backend = "cuda"
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    values = (
        torch.randn(2, 6, device=device, requires_grad=True),
        (torch.randn(2, 3, device=device) + 1).requires_grad_(),
        torch.randn(2, module.weight_numel, device=device, requires_grad=True),
    )
    actual = kernel.reference(module, *values)
    expected = kernel.reference(reference, *values)
    torch.testing.assert_close(actual, expected)
    assert_native_autograd(actual)


def test_unprojected_contract(double_precision):
    layer = co2.O3TensorProduct(
        "2x2e", "2e", "2x2e", [(0, 0, 0, "uvu", True)], project=False
    )
    x = co3.ChangeOfBasis("2x2e")(torch.randn(3, 10))
    r = torch.randn(3, 3)
    torch.testing.assert_close(layer(x, r), layer.contract(x, r))


def test_reference_mixed_child_aoti(tmp_path, monkeypatch, double_precision):
    import copy
    import os
    import subprocess
    import sys

    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    module = co2.O3TensorProduct(
        "1o",
        "1o",
        "0e+1e+2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        internal_weights=False,
        shared_weights=True,
    ).double()
    reference = copy.deepcopy(module)
    for child in module.modules():
        if child is not module and hasattr(child, "backend"):
            child.backend = "cuda"
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)

    def joint(layer, x, vectors, weight):
        energy = layer(x, vectors, weight).sin().sum()
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        loss = energy.square() * 0.01 + (force - 0.1).square().sum()
        gradient = torch.autograd.grad(loss, weight)[0]
        return energy, force, loss, gradient

    def inputs(count):
        return (
            (torch.randn(count, 3) * 0.1).requires_grad_(),
            (torch.randn(count, 3) + 1).requires_grad_(),
        )

    weight = (torch.randn(module.weight_numel) * 0.1).requires_grad_()
    example = (*inputs(7), weight)
    graph = make_fx(
        lambda *values: joint(module, *values),
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )(*example)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    rows = torch.export.Dim("rows", min=2, max=16)
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in example),
        dynamic_shapes=({0: rows}, {0: rows}, None),
    )
    package = tmp_path / "ordinary-product.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    initial = weight.detach().clone()
    records = []
    for count in (3, 7, 5):
        values = inputs(count)
        for _ in range(3):
            expected = joint(reference, *values, weight)
            assert torch.count_nonzero(expected[1]) and torch.count_nonzero(
                expected[-1]
            )
            with torch.no_grad():
                weight.add_(expected[-1], alpha=-1e-3)
            records.append(
                {
                    "inputs": tuple(value.detach().clone() for value in values),
                    "outputs": tuple(value.detach().clone() for value in expected),
                    "weight": weight.detach().clone(),
                }
            )
    data = tmp_path / "records.pt"
    torch.save({"initial": initial, "records": records}, data)
    script = """
import sys
import torch
runner = torch._inductor.aoti_load_package(sys.argv[1])
data = torch.load(sys.argv[2], weights_only=True)
weight = data['initial'].clone()
for case in data['records']:
    with torch.no_grad():
        actual = runner(*case['inputs'], weight)
        torch.testing.assert_close(tuple(actual), case['outputs'], atol=2e-10, rtol=2e-9)
        weight.add_(actual[-1], alpha=-1e-3)
    torch.testing.assert_close(weight, case['weight'], atol=2e-10, rtol=2e-9)
assert not torch.equal(weight, data['initial'])
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        env=dict(os.environ, AOTI_RUNTIME_CHECK_INPUTS="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert config.TORCH_REFERENCE is True
    assert all(
        child.backend == "cuda"
        for child in module.modules()
        if child is not module and hasattr(child, "backend")
    )
