"""CPU polynomial prefix schedules through the public ACE contraction."""

import subprocess
import sys
import zipfile

import pytest
import torch
from e3nn import o3
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.dynamic_spec import (
    ParamsSpec,
    ShapesSpec,
    ShapeVar,
    TensorSpec,
)
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.mbe.ace import SymmetricContraction
from equivariantx.ops import config
from equivariantx.ops.common.symmetric import symmetric_inputs
from equivariantx.ops.torch.mbe.ace import symmetric as kernel
from equivariantx.tests.utils.testing import check_forward_ad
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.module import default_dtype


def contraction(correlation=3, channels=2):
    entries = [((0,), 0, 0, 0.7)]
    path = 1
    for degree in range(2, correlation + 1):
        entries.append(((0,) * degree, path, 0, 0.3))
        for components in (range(1, 4), range(4, 9)):
            path += 1
            entries.extend(
                ((0,) * (degree - 2) + (i, i), path, 0, 1 / len(components) ** 0.5)
                for i in components
            )
        path += 1
    return SymmetricContraction.from_entries(
        f"{channels}x0e+{channels}x1o+{channels}x2e",
        f"{channels}x0e",
        entries,
        path,
    )


def selects_prefix(layer, features):
    program, _ = symmetric_inputs(
        features, features, layer.correlation, layer.output_dim
    )
    return kernel.prefix_enabled(layer, features, program)


@pytest.fixture(scope="module")
def nonscalar_contraction():
    irreps = o3.Irreps.spherical_harmonics(2)
    tensors = []
    with default_dtype(torch.float64):
        for degree, formula in enumerate(("i", "ij=ji", "ijk=jik=ikj"), 1):
            product = o3.ReducedTensorProducts(
                formula, i=irreps, filter_ir_out=[ir for _, ir in irreps]
            )
            for (mul, ir), section in zip(
                product.irreps_out, product.irreps_out.slices()
            ):
                tensor = product.change_of_basis[section].reshape(
                    mul, ir.dim, *([irreps.dim] * degree)
                )
                tensors.append((ir.l, degree, tensor.movedim(0, -1)))
    return SymmetricContraction(
        [(64, ir) for _, ir in irreps], [(64, ir) for _, ir in irreps], tensors
    )


@pytest.mark.parametrize("correlation", [2, 3, 4])
def test_prefix_values_and_four_derivatives(correlation, monkeypatch):
    monkeypatch.setattr(kernel, "prefix_enabled", lambda *_: True)
    layer = contraction(correlation)
    x = (torch.randn(3, 2, 9, dtype=torch.float64) * 0.2).requires_grad_()
    w = (torch.randn(2, layer.num_paths, 2, dtype=torch.float64) * 0.2).requires_grad_()
    types = torch.tensor([0, 1, 0])
    actual = layer(x, w, types)
    with monkeypatch.context() as flags:
        flags.setattr(config, "TORCH_REFERENCE", True)
        expected = layer(x, w, types)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
        actual, expected = [
            torch.cat(
                [
                    value.flatten()
                    for value in torch.autograd.grad(
                        output.sin().sum(), (x, w), create_graph=True
                    )
                ]
            )
            for output in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
    assert torch.autograd.gradcheck(
        lambda x, w: layer(x, w, types), (x, w), fast_mode=True
    )
    assert torch.autograd.gradgradcheck(
        lambda x, w: layer(x, w, types), (x, w), fast_mode=True
    )


def test_prefix_forward_ad_and_vmap(monkeypatch):
    monkeypatch.setattr(kernel, "prefix_enabled", lambda *_: True)
    layer = contraction(4)
    x = torch.randn(3, 2, 9, dtype=torch.float64) * 0.2
    weight = torch.randn(2, layer.num_paths, 2, dtype=torch.float64) * 0.2
    types = torch.tensor([0, 1, 0])
    directions = torch.randn_like(x), torch.randn_like(weight)

    def function(x, w):
        return layer(x, w, types)

    results = []
    for reference in (False, True):
        with monkeypatch.context() as flags:
            flags.setattr(config, "TORCH_REFERENCE", reference)
            functional = torch.func.jvp(function, (x, weight), directions)
            with torch.autograd.forward_ad.dual_level():
                duals = tuple(
                    torch.autograd.forward_ad.make_dual(value, tangent)
                    for value, tangent in zip((x, weight), directions)
                )
                ordinary = torch.autograd.forward_ad.unpack_dual(function(*duals))
            batched = torch.vmap(function, in_dims=(0, None))(
                torch.stack((x, x * 0.5)), weight
            )
            results.append((functional, ordinary, batched))
    torch.testing.assert_close(*results, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("nodes", [0, 7])
def test_prefix_noncontiguous_and_empty(nodes, monkeypatch):
    monkeypatch.setattr(kernel, "prefix_enabled", lambda *_: True)
    layer = contraction(4)
    x = torch.randn(nodes, 4, 18, dtype=torch.float64)[:, ::2, ::2].requires_grad_()
    w = torch.randn(3, layer.num_paths * 2, 4, dtype=torch.float64)[
        :, ::2, ::2
    ].requires_grad_()
    types = torch.arange(nodes, dtype=torch.int32) % 3
    values = []
    for reference in (False, True):
        with monkeypatch.context() as flags:
            flags.setattr(config, "TORCH_REFERENCE", reference)
            y = layer(x, w, types)
            values.append((y, *torch.autograd.grad(y.square().sum(), (x, w))))
    torch.testing.assert_close(*values, atol=2e-10, rtol=2e-9)


def test_prefix_selection_limits(monkeypatch):
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_PREFIX", True)
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", False)
    assert selects_prefix(contraction(4, 64), torch.empty(512, 64, 9))
    assert not selects_prefix(contraction(4, 64), torch.empty(7, 64, 9))
    assert not selects_prefix(contraction(2, 1024), torch.empty(512, 1024, 9))
    assert not selects_prefix(contraction(5, 64), torch.empty(512, 64, 9))
    many = SymmetricContraction.from_entries(
        "64x0e",
        "64x0e",
        [((0, 0, 0), path, 0, 0.1) for path in range(257)],
        257,
    )
    assert not selects_prefix(many, torch.empty(512, 64, 1))
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", True)
    assert not selects_prefix(contraction(4, 64), torch.empty(512, 64, 9))
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", False)
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_PREFIX", False)
    assert not selects_prefix(contraction(4, 64), torch.empty(512, 64, 9))


def test_reference_does_not_use_prefix(monkeypatch):
    def reject(*args):
        raise AssertionError("The independent reference entered the prefix evaluator")

    monkeypatch.setattr(kernel, "evaluate_prefix", reject)
    monkeypatch.setattr(kernel, "prefix_enabled", lambda *_: True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    layer = contraction()
    result = layer(
        torch.randn(3, 2, 9), torch.randn(2, layer.num_paths, 2), torch.arange(3) % 2
    )
    assert result.shape == (3, 2)


def test_default_prefix_dynamic_force_loss(monkeypatch):
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_PREFIX", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", False)
    flags_before = (
        config.TORCH_SYMMETRIC_PREFIX,
        config.TORCH_REFERENCE,
        config.TORCH_RECOMPUTE,
    )
    layer = contraction(3, 64)
    weight = (
        torch.randn(3, layer.num_paths, 64, dtype=torch.float64) * 0.2
    ).requires_grad_()

    def joint(x, w, types):
        energy = layer(x, w, types).sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() / x.shape[0] + force.square().mean()
        return energy, force, torch.autograd.grad(loss, w)[0]

    count = ShapeVar("nodes", min=2, optimization_hint=7)
    x = (torch.randn(7, 64, 9, dtype=torch.float64) * 0.2).requires_grad_()
    types = torch.arange(7) % 3
    graph = make_fx(
        joint,
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec([count, None, None]),
                    "w": TensorSpec([None, None, None]),
                    "types": TensorSpec([count]),
                }
            )
        ),
    )(x, weight, types)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled_graphs = []

    def backend(module, inputs):
        from torch._inductor.compile_fx import compile_fx

        compiled_graphs.append(module)
        return compile_fx(module, inputs)

    run = torch.compile(graph, fullgraph=True, dynamic=True, backend=backend)
    for nodes in (7, 11, 9):
        x = (torch.randn(nodes, 64, 9, dtype=torch.float64) * 0.2).requires_grad_()
        types = torch.arange(nodes) % 3
        arguments = x.detach(), weight.detach(), types
        # Seven nodes must stay independent of the seven fixed weight paths.
        torch._dynamo.mark_dynamic(arguments[0], 0, min=2)
        torch._dynamo.mark_static(arguments[0], 1)
        torch._dynamo.mark_static(arguments[0], 2)
        torch._dynamo.mark_static(arguments[1])
        torch._dynamo.mark_dynamic(types, 0, min=2)
        actual = run(*arguments)
        with monkeypatch.context() as flags:
            flags.setattr(config, "TORCH_REFERENCE", True)
            expected = joint(x, weight, types)
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    assert len(compiled_graphs) == 1
    assert flags_before == (
        config.TORCH_SYMMETRIC_PREFIX,
        config.TORCH_REFERENCE,
        config.TORCH_RECOMPUTE,
    )


def test_compiled_prefix_dag_budget(nonscalar_contraction, monkeypatch):
    layer = nonscalar_contraction
    x = torch.empty(512, 64, 9)
    program, _ = symmetric_inputs(x, x, layer.correlation, layer.output_dim)
    terms = len(parse_metadata(layer.metadata)[-1])
    assert terms == 317
    nodes = len(kernel.prefix_plan(layer.metadata, program)[0])
    assert terms < nodes <= 1024
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_PREFIX", True)
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", False)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_TERMS", 256)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_NODES", 1024)
    assert not selects_prefix(layer, x)
    monkeypatch.setattr(kernel, "get_proxy_mode", lambda: object())
    assert selects_prefix(layer, x)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_NODES", nodes)
    assert selects_prefix(layer, x)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_NODES", nodes - 1)
    assert not selects_prefix(layer, x)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_NODES", 0)
    assert not selects_prefix(layer, x)
    assert selects_prefix(contraction(3, 64), x)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_NODES", 1024)
    for attribute, value in (("num_channels", 32), ("correlation", 5)):
        with monkeypatch.context() as limits:
            limits.setattr(layer, attribute, value)
            assert not selects_prefix(layer, x)
    for attribute, value in (
        ("TORCH_RECOMPUTE", True),
        ("TORCH_SYMMETRIC_PREFIX", False),
    ):
        with monkeypatch.context() as limits:
            limits.setattr(config, attribute, value)
            assert not selects_prefix(layer, x)
    if torch.cuda.is_available():
        with FakeTensorMode():
            assert not selects_prefix(layer, torch.empty(7, 64, 9, device="cuda"))

    def reject(*args):
        raise AssertionError("An oversized polynomial built a prefix plan")

    monkeypatch.setattr(kernel, "prefix_plan", reject)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_NODES", terms - 1)
    assert not selects_prefix(layer, x)


@pytest.mark.parametrize("nodes", [0, 3])
def test_nonscalar_prefix_derivatives(nodes, nonscalar_contraction, monkeypatch):
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_PREFIX", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", False)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_NODES", 1024)
    layer = nonscalar_contraction
    x = (torch.randn(nodes, 128, 18, dtype=torch.float64) * 0.1)[
        :, ::2, ::2
    ].requires_grad_()
    w = (torch.randn(3, layer.num_paths * 2, 128, dtype=torch.float64) * 0.1)[
        :, ::2, ::2
    ].requires_grad_()
    types = torch.arange(nodes, dtype=torch.int32) % 3
    selected = []
    original = kernel.evaluate_prefix

    def prefix(*args):
        selected.append(True)
        return original(*args)

    monkeypatch.setattr(kernel, "evaluate_prefix", prefix)
    graph = make_fx(layer)(x, w, types)
    assert selected == [True]
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    actual, expected = graph(x, w, types), layer(x, w, types)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
        actual, expected = [
            torch.cat(
                [
                    value.flatten()
                    for value in torch.autograd.grad(
                        output.sin().sum(), (x, w), create_graph=True
                    )
                ]
            )
            * 0.1
            for output in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
    check_forward_ad(
        lambda x, w: graph(x, w, types),
        lambda x, w: layer(x, w, types),
        (x, w),
    )


def test_nonscalar_dynamic_force_loss_aoti(
    nonscalar_contraction, monkeypatch, tmp_path
):
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_PREFIX", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", False)
    monkeypatch.setattr(config, "SYMMETRIC_PREFIX_MAX_NODES", 1024)
    layer = nonscalar_contraction
    frequencies = torch.linspace(0.1, 2, 64, dtype=torch.float64)[None]

    def joint(positions, weight, types):
        radius = (positions.square().sum(-1, keepdim=True) + 0.5).sqrt()
        radial = (radius * frequencies).sin() / radius
        angular = o3.spherical_harmonics(
            [0, 1, 2], positions, normalize=False, normalization="component"
        )
        output = layer(radial[:, :, None] * angular[:, None, :] * 0.1, weight, types)
        energy = output.square().sum() / 64
        force = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        energy = energy / positions.shape[0]
        loss = (energy - 0.2).square() + 0.1 * (force + 0.01).square().mean()
        return energy, force, loss, torch.autograd.grad(loss, weight)[0]

    def inputs(nodes):
        x = torch.randn(3, nodes, dtype=torch.float64).T.requires_grad_()
        return x, torch.arange(nodes, dtype=torch.int32) % 3

    x, types = inputs(7)
    weight = (
        torch.randn(3, layer.num_paths, 64, dtype=torch.float64) * 0.2
    ).requires_grad_()
    nodes = ShapeVar("nodes", min=2, optimization_hint=7)
    shapes = ShapesSpec(
        ParamsSpec(
            {
                "positions": TensorSpec([nodes, None]),
                "weight": TensorSpec([None, None, None]),
                "types": TensorSpec([nodes]),
            }
        )
    )
    graph = make_fx(
        joint,
        tracing_mode="fake",
        dynamic_shapes=shapes,
        _allow_non_fake_inputs=True,
    )(x, weight, types)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert not any("eqx." in str(node.target) for node in graph.graph.nodes)
    nodes = torch.export.Dim("nodes", min=2, max=8192)
    exported = torch.export.export(
        graph,
        (x.detach(), weight.detach(), types),
        strict=True,
        dynamic_shapes=({0: nodes}, None, {0: nodes}),
    )
    package = tmp_path / "nonscalar.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        assert any(name.endswith(".so") for name in names)
        assert not any(name.endswith((".cubin", ".ptx")) for name in names)
        source = "\n".join(
            archive.read(name).decode() for name in names if name.endswith(".cpp")
        )
        assert source and "eqx::" not in source
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    oracle = make_fx(
        joint,
        tracing_mode="fake",
        dynamic_shapes=shapes,
        _allow_non_fake_inputs=True,
    )(x, weight, types)
    torch.save(oracle, tmp_path / "oracle.pt")
    cases = [tuple(value.detach() for value in inputs(n)) for n in (7, 11, 9)]
    torch.save((weight.detach(), cases), tmp_path / "inputs.pt")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys, torch
torch.set_num_threads(2)
run = torch._inductor.aoti_load_package(sys.argv[1])
oracle = torch.load(sys.argv[2], weights_only=False)
weight, cases = torch.load(sys.argv[3], weights_only=True)
expected_weight = weight.clone()
for positions, types in cases:
    for _ in range(3):
        actual = run(positions, weight, types)
        expected = oracle(positions, expected_weight, types)
        assert all(value.device.type == 'cpu' for value in actual)
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        weight = weight - 0.001 * actual[-1]
        expected_weight = expected_weight - 0.001 * expected[-1]
        torch.testing.assert_close(weight, expected_weight, atol=2e-9, rtol=2e-9)
assert 'equivariantx' not in sys.modules and 'e3nn' not in sys.modules
""",
            str(package),
            str(tmp_path / "oracle.pt"),
            str(tmp_path / "inputs.pt"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
