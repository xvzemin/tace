"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx.co3.symmetric import SymmetricBasis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def force_loss(function, positions, features, weight):
    """Differentiate a position-dependent energy and its force loss."""
    import math

    matrix = torch.eye(3, device=positions.device, dtype=positions.dtype)
    matrix = matrix + 0.15 * positions
    output = function(features * weight, matrix.unsqueeze(-3))
    energy = output.sin().sum((-2, -1)) / math.sqrt(weight.numel())
    force = -torch.autograd.grad(energy.sum(), positions, create_graph=True)[0]
    loss = (energy - 0.2).square().sum() + 10 * (force - 0.1).square().sum()
    return energy, force, loss, torch.autograd.grad(loss, weight)[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA shared matrix")
@pytest.mark.parametrize("rank,channels", [(4, 128), (6, 1024)])
def test_native_shared_matrix_force_loss(rank, channels, monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config
    from equivariantx.ops.torch.co3.symmetric import transform_reference

    monkeypatch.setattr(config, "SYMMETRIC_MATRIX_NATIVE", True)
    module = SymmetricBasis(rank, backend="cuda").cuda().float()
    reference = SymmetricBasis(rank).double()
    positions = torch.randn(3, 3, 3, device="cuda", requires_grad=True) * 0.1
    features = torch.randn(3, channels, module.dim, device="cuda") * 0.1
    weight = (torch.randn(channels, module.dim, device="cuda") * 0.1).requires_grad_()

    graph = make_fx(lambda *args: force_loss(module, *args), tracing_mode="fake")(
        positions, features, weight
    )
    graph.graph.eliminate_dead_code()
    graph.recompile()
    targets = [str(node.target) for node in graph.graph.nodes]
    assert targets.count("eqx.symmetric_compiled.default") == 3
    assert targets.count("eqx.ieee_bmm.default") == 5
    assert not any(
        target in targets for target in ("aten.mm.default", "aten.bmm.default")
    )
    actual = graph(positions.detach(), features, weight.detach())
    expected = force_loss(
        lambda x, m: transform_reference(reference, x, m),
        positions.detach().cpu().double().requires_grad_(),
        features.cpu().double(),
        weight.detach().cpu().double().requires_grad_(),
    )
    torch.testing.assert_close(
        tuple(value.detach().cpu().double() for value in actual),
        expected,
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
@pytest.mark.parametrize("rank,channels", [(4, 128), (6, 1024)])
def test_native_shared_matrix_dynamic_aoti(rank, channels, monkeypatch, tmp_path):
    import os
    import subprocess
    import sys

    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config
    from equivariantx.ops.torch.co3.symmetric import transform_reference

    monkeypatch.setattr(config, "SYMMETRIC_MATRIX_NATIVE", True)
    module = SymmetricBasis(rank, backend="cuda").cuda().float()
    reference = SymmetricBasis(rank).double()
    initial = torch.randn(channels, module.dim, dtype=torch.float32) * 0.1
    weight, cases = initial.clone(), []
    for nodes in (3, 7, 2, 11, 3):
        positions = torch.randn(nodes, 3, 3, dtype=torch.float32) * 0.1
        features = torch.randn(nodes, channels, module.dim, dtype=torch.float32) * 0.1
        expected = force_loss(
            lambda x, m: transform_reference(reference, x, m),
            positions.double().requires_grad_(),
            features.double(),
            weight.double().requires_grad_(),
        )
        expected = tuple(value.detach() for value in expected)
        weight = weight - 0.01 * expected[-1].float()
        cases.append(
            dict(
                positions=positions, features=features, expected=expected, weight=weight
            )
        )

    inputs = (
        cases[0]["positions"].cuda().requires_grad_(),
        cases[0]["features"].cuda(),
        initial.cuda().requires_grad_(),
    )
    nodes = ShapeVar("nodes", min=2, max=32, optimization_hint=3)
    specs = [
        TensorSpec((nodes, None, None)),
        TensorSpec((nodes, None, None)),
        TensorSpec((None, None)),
    ]
    graph = make_fx(
        lambda *args: force_loss(module, *args),
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(ParamsSpec({"*args": specs})),
    )(*inputs)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    targets = [str(node.target) for node in graph.graph.nodes]
    assert targets.count("eqx.symmetric_compiled.default") == 3
    assert targets.count("eqx.ieee_bmm.default") == 5
    dim = torch.export.Dim("nodes", min=2, max=32)
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in inputs),
        dynamic_shapes=({0: dim}, {0: dim}, {}),
        strict=True,
    )
    package = tmp_path / "shared_matrix.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = tmp_path / "records.pt"
    torch.save(dict(initial=initial, cases=cases), records)
    script = """
import sys
import torch

torch.backends.cuda.matmul.allow_tf32 = True
run = torch._inductor.aoti_load_package(sys.argv[1])
records = torch.load(sys.argv[2], weights_only=True)
weight = records['initial'].cuda()
for case in records['cases']:
    positions, features = case['positions'].cuda(), case['features'].cuda()
    before = weight.clone()
    actual = tuple(run(positions, features, weight))
    torch.testing.assert_close(tuple(v.cpu().double() for v in actual), case['expected'], atol=3e-6, rtol=3e-5)
    assert actual[1].abs().max() > 0 and actual[-1].abs().max() > 0
    assert len({v.untyped_storage().data_ptr() for v in (*actual, positions, features, weight)}) == 7
    predictions = tuple(v.clone() for v in actual[:-1])
    actual[-1].mul_(0.01)
    torch.testing.assert_close(actual[:-1], predictions, atol=0, rtol=0)
    torch.testing.assert_close(weight, before, atol=0, rtol=0)
    weight.sub_(actual[-1])
    assert not torch.equal(weight, before)
    torch.testing.assert_close(weight.cpu(), case['weight'], atol=3e-6, rtol=3e-5)
for count in (0, 1, 33):
    try:
        run(positions.new_zeros(count, 3, 3), features.new_zeros(count, *features.shape[1:]), weight)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f'Accepted out-of-contract batch {count}')
try:
    run(positions, features[:2], weight)
except RuntimeError:
    pass
else:
    raise AssertionError('Accepted inconsistent dynamic batch dimensions')
assert torch.backends.cuda.matmul.allow_tf32
assert not any(name.split('.')[0] in ('equivariantx', 'e3nn', 'tace') for name in sys.modules)
print('Five dynamic force-loss parameter updates passed with Torch only.')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(package), str(records)],
        cwd=tmp_path,
        env={**os.environ, "AOTI_RUNTIME_CHECK_INPUTS": "1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "loader.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    from equivariantx.tests.utils.testing import check_method_precision

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = SymmetricBasis(2).to(device)
    tensor = torch.randn(3, 9, device=device)
    packed = layer.pack(tensor)
    check_method_precision(
        layer,
        [
            ("pack", (tensor,), {}),
            ("unpack", (packed,), {}),
            ("contract", (packed, torch.randn(3, 3, device=device)), {}),
            ("trace", (packed,), {}),
            ("multiply", (torch.randn(3, 1, device=device), packed), {}),
        ],
    )


@pytest.mark.parametrize("rank", [2, 4, 8])
def test_symmetric_basis_adjoint(double_precision, rank):
    basis = SymmetricBasis(rank).to(DEVICE)
    tensor = torch.randn(2, 3**rank, device=DEVICE, requires_grad=True)
    compact = torch.randn(2, basis.dim, device=DEVICE, requires_grad=True)
    projected = basis.pack(tensor)
    loss = (projected * compact).sum()
    expected = (tensor * basis.unpack(compact)).sum()
    torch.testing.assert_close(loss, expected, atol=3e-13, rtol=3e-13)
    grad = torch.autograd.grad(loss, tensor, create_graph=True)[0]
    torch.testing.assert_close(grad, basis.unpack(compact), atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("rank", [0, 1, 2, 4, 6])
def test_reference_switch(rank, double_precision, monkeypatch):
    from equivariantx.ops import config

    layer = SymmetricBasis(rank).to(DEVICE)
    value = (torch.randn(2, layer.dim, device=DEVICE) * 0.1).requires_grad_()
    matrix = (torch.randn(2, 3, 3, device=DEVICE) * 0.2).requires_grad_()
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    actual = layer(value, matrix)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = layer(value, matrix)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        actual, expected = [
            torch.autograd.grad(output.sin().sum(), value, create_graph=True)[0]
            for output in (actual, expected)
        ]
    full = torch.randn(2, 3**rank, device=DEVICE)
    expected = layer.pack(full)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(layer.pack(full), expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("backend", ["torch", "analytic", "cuda"])
@pytest.mark.parametrize("rank", [2, 4, 6])
@pytest.mark.parametrize(
    "method", ["pack", "unpack", "contract", "trace", "multiply", "forward"]
)
def test_backend_derivatives(backend, rank, method, double_precision, monkeypatch):
    from equivariantx.ops import config

    monkeypatch.setattr(config, "TORCH_SYMMETRIC_KERNELS", True)
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_ADJOINT", backend == "analytic")
    module = SymmetricBasis(rank, backend="cuda" if backend == "cuda" else "torch").to(
        DEVICE
    )
    reference = SymmetricBasis(rank).to(DEVICE)
    size = (
        3**rank
        if method == "pack"
        else (rank - 1) * rank // 2
        if method == "multiply"
        else module.dim
    )
    x = (torch.randn(2, 2, size, device=DEVICE) * 0.1).requires_grad_()
    values = (x,)
    if method in ("contract", "multiply", "forward"):
        shape = (
            (3, 3) if method == "forward" else (6,) if method == "multiply" else (3,)
        )
        values += (torch.randn(2, 1, *shape, device=DEVICE) * 0.1,)
        values[-1].requires_grad_()
    actual = getattr(module, method)(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = getattr(reference, method)(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    losses = [v.sin().sum() for v in (actual, expected)]
    for _ in range(5 if rank == 2 else 3 if rank == 4 else 2):
        gradients = [
            torch.autograd.grad(loss, values, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-9, rtol=3e-9)
        tangents = tuple(torch.randn_like(v) * 0.1 for v in values)
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
    tangents = tuple(torch.randn_like(v) * 0.1 for v in values)
    actual = torch.func.jvp(getattr(module, method), values, tangents)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = torch.func.jvp(getattr(reference, method), values, tangents)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(
        torch.vmap(getattr(module, method))(*values),
        getattr(module, method)(*values),
        atol=3e-11,
        rtol=3e-11,
    )
    assert getattr(module, method)(*(v[:0] for v in values)).shape[0] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.ops import config
    from equivariantx.ops.cuda.co3 import symmetric as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module, reference = (
        SymmetricBasis(3, backend="cuda").to(device),
        SymmetricBasis(3).to(device),
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        x = torch.randn(count, 27, device=device)
        with patch.object(kernel, "pack", wraps=kernel.pack) as selected:
            actual = module.pack(x)
        assert selected.call_count == (device == "cuda")
        torch.testing.assert_close(actual, reference.pack(x))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


def test_legacy_sparse_metadata(double_precision, monkeypatch):
    from equivariantx.ops import config

    module = SymmetricBasis(3).to(DEVICE)
    x = torch.randn(2, 27, device=DEVICE)
    compact = torch.randn(2, module.dim, device=DEVICE)
    matrix = torch.randn(2, 3, 3, device=DEVICE)
    expected = module.pack(x), module(compact, matrix)
    for name in list(module._buffers):
        if name.startswith("output_") and name.endswith(("_indices", "_weights")):
            delattr(module, name)
    del module.backend
    del module.permutation
    del module.offsets
    module.float().double()
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_KERNELS", True)
    torch.testing.assert_close((module.pack(x), module(compact, matrix)), expected)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_normalization_derivatives(expand, backend, double_precision):
    from equivariantx.ops.torch.co3.symmetric import map_reference, symmetry_map

    module = SymmetricBasis(2).to(DEVICE)
    x = (torch.randn(2, 6 if expand else 9, device=DEVICE) * 0.1).requires_grad_()
    norm = (module.norm.detach() + 0.2).requires_grad_()

    def reference(x, norm):
        return map_reference(x, module.indices, norm, expand)

    if backend == "cuda" and DEVICE.type == "cuda":
        from equivariantx.ops.cuda.co3.symmetric import map_operator

        def actual(x, norm):
            return map_operator(
                x, module.indices, norm, module.permutation, module.offsets, 2, expand
            )
    else:

        def actual(x, norm):
            return symmetry_map(x, module.indices, norm, expand)

    torch.testing.assert_close(
        actual(x, norm), reference(x, norm), atol=3e-12, rtol=3e-12
    )
    assert torch.autograd.gradcheck(actual, (x, norm), fast_mode=True)
    assert torch.autograd.gradgradcheck(actual, (x, norm), fast_mode=True)
    if backend == "torch":
        tangents = torch.randn_like(x), torch.randn_like(norm)

        def grad(f):
            return torch.func.grad(lambda x, n: f(x, n).sin().sum(), argnums=(0, 1))

        torch.testing.assert_close(
            torch.func.jvp(grad(actual), (x, norm), tangents),
            torch.func.jvp(grad(reference), (x, norm), tangents),
            atol=3e-10,
            rtol=3e-10,
        )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("rank", [8, 10])
def test_high_rank_single_precision(backend, rank, monkeypatch):
    from equivariantx.ops import config

    monkeypatch.setattr(config, "TORCH_SYMMETRIC_KERNELS", True)
    module = SymmetricBasis(rank, backend=backend).float().to(DEVICE)
    reference = SymmetricBasis(rank).double().to(DEVICE)
    compact = torch.linspace(0.1, 1, reference.dim, device=DEVICE, dtype=torch.float64)
    full = (
        (compact / reference.norm)
        .index_select(-1, reference.indices)
        .expand(3, -1)
        .float()
    )
    expected = reference.pack(full.double())
    actual = module.pack(full)
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-6)
