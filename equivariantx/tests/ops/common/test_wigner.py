"""Tests for the corresponding EQX source module."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import o2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Wigner transforms")
@pytest.mark.parametrize("method", ["quaternion", "recursive"])
@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_transforms(method, degree, compiler, double_precision):
    frame = o2.WignerD(degree, degree, backend="cuda", method=method).cuda()
    reference = o2.WignerD(degree, degree, backend="torch", method=method).cuda()
    vectors = torch.randn(4, 3, device="cuda") + torch.tensor(
        [0.4, 0.7, 1.1], device="cuda"
    )
    direction = torch.randn_like(vectors)

    def transformed(function, vectors):
        gradient = torch.func.grad(lambda x: function(x).sin().sum())
        return (
            torch.func.jvp(function, (vectors,), (direction,)),
            torch.func.jvp(gradient, (vectors,), (direction,)),
        )

    expected = transformed(reference.forward_packed, vectors)
    torch.compiler.reset()

    def run(x):
        return transformed(frame.forward_packed, x)

    if compiler:
        run = torch.compile(run, backend=compiler, fullgraph=True)
    torch.testing.assert_close(run(vectors), expected, atol=2e-8, rtol=2e-8)

    def nested(function, vectors):
        return torch.func.jvp(
            lambda x: torch.func.jvp(function, (x,), (direction,))[1],
            (vectors,),
            (direction,),
        )

    expected = nested(reference.forward_packed, vectors)

    def nested_run(x):
        return nested(frame.forward_packed, x)

    torch.testing.assert_close(nested_run(vectors), expected, atol=2e-8, rtol=2e-8)
    if compiler:
        graph = make_fx(nested_run)(vectors)
        graph.graph.eliminate_dead_code()
        graph.recompile()
        torch.testing.assert_close(
            torch.compile(graph, backend=compiler, fullgraph=True)(vectors),
            expected,
            atol=2e-8,
            rtol=2e-8,
        )
    batch = torch.stack((vectors, vectors + 0.3))
    torch.testing.assert_close(
        torch.vmap(frame.forward_packed)(batch),
        torch.vmap(reference.forward_packed)(batch),
        atol=2e-10,
        rtol=2e-10,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
@pytest.mark.parametrize("method", ["recursive", "quaternion"])
def test_force_derivatives_aoti(method, double_precision, tmp_path):
    frame = o2.WignerD(2, 2, method=method, backend="cuda").cuda()
    reference = o2.WignerD(2, 2, method=method, backend="torch").cuda()
    vectors = torch.randn(4, 3, device="cuda", requires_grad=True)

    def joint(module, vectors):
        energy = module.forward_packed(vectors).sin().mean()
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        curvature = torch.autograd.grad((force - 0.1).square().sum(), vectors)[0]
        return energy, force, curvature

    graph = make_fx(
        lambda vectors: joint(frame, vectors),
    )(vectors)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, (vectors,), strict=True)
    assert all(
        not str(node.target).startswith("eqx.")
        for node in exported.graph_module.graph.nodes
    )
    package = tmp_path / "wigner.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for offset in (0.0, 0.3):
        changed = vectors.detach().clone().add_(offset).requires_grad_()
        cases.append(
            {
                "input": changed.detach().cpu(),
                "expected": tuple(
                    value.detach().cpu() for value in joint(reference, changed)
                ),
            }
        )
    data = tmp_path / "cases.pt"
    torch.save(cases, data)
    script = """
import sys
import torch
runner = torch._inductor.aoti_load_package(sys.argv[1])
for case in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = runner(case['input'].cuda())
    for value, expected in zip(actual, case['expected']):
        torch.testing.assert_close(value.cpu(), expected, atol=1e-9, rtol=1e-9)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Wigner transforms")
@pytest.mark.parametrize("method", ["auto", "quaternion", "recursive"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_ordinary_forward_ad(method, requires_grad, double_precision):
    frame = o2.WignerD(2, 2, backend="cuda", method=method).cuda()
    reference = o2.WignerD(2, 2, backend="torch", method=method).cuda()
    vectors = torch.randn(4, 3, device="cuda", requires_grad=requires_grad)
    tangent = torch.randn_like(vectors)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(vectors, tangent)
        result = torch.autograd.forward_ad.unpack_dual(frame.forward_packed(dual))
    expected = torch.autograd.functional.jvp(
        reference.forward_packed, (vectors,), (tangent,)
    )
    assert result.tangent is not None
    torch.testing.assert_close(tuple(result), expected, atol=2e-10, rtol=2e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
@pytest.mark.parametrize("method", ["auto", "recursive"])
def test_default_aoti_parameter_updates(method, tmp_path, double_precision):
    frame = o2.WignerD(2, 2, method=method, backend="cuda").cuda()
    reference = o2.WignerD(2, 2, method=method, backend="torch").cuda()
    vectors = torch.randn(3, 3, device="cuda", requires_grad=True)
    weight = (torch.randn(35, device="cuda") * 0.1).requires_grad_()

    def joint(module, vectors, weight):
        energy = (module.forward_packed(vectors) @ weight).sin().mean() + 1
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    graph = make_fx(lambda *values: joint(frame, *values))(vectors, weight)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    exported = torch.export.export(graph, (vectors.detach(), weight.detach()))
    package = tmp_path / "wigner_force.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    rows = []
    for _ in range(3):
        expected = joint(reference, vectors, weight)
        rows.append(
            dict(
                inputs=(vectors.detach().cpu(), weight.detach().cpu()),
                expected=tuple(value.detach().cpu() for value in expected),
            )
        )
        weight = (weight - 0.001 * expected[-1]).detach().requires_grad_()
    data = tmp_path / "cases.pt"
    torch.save(rows, data)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    actual=run(*(value.cuda() for value in row['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual),row['expected'],atol=2e-9,rtol=2e-9)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("lmax", [0, 1, 3, 5])
def test_quaternion_wigner_frames(lmax):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.wigner import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        axes = torch.eye(3, device="cuda")
        vectors = torch.cat((axes, -axes, axes + 1e-8, -axes + 1e-8))
        # Non-contiguous input, including directions on the chart boundaries.
        vectors = torch.stack((vectors, vectors), dim=-1)[..., 0].requires_grad_()
        frame = o2.WignerD(min(1, lmax), lmax, backend="cuda").cuda()
        reference = o2.WignerD(min(1, lmax), lmax, method="recursive").cuda()
        for actual, expected in zip(frame(vectors), reference(vectors)):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(
            frame.forward_packed(vectors),
            wigner_D(frame, vectors, backend="cuda"),
            atol=0,
            rtol=0,
        )
        for actual, expected in zip(
            frame.matrix_blocks(vectors), reference.matrix_blocks(vectors)
        ):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        for actual, expected in zip(frame(vectors[:0]), reference(vectors[:0])):
            torch.testing.assert_close(actual, expected)
        if lmax:
            for method in ("quaternion", "recursive"):
                values = frame.forward_packed(vectors, method=method)
                gradient = torch.autograd.grad(
                    values.sin().sum(), vectors, create_graph=True
                )[0]
                assert torch.isfinite(gradient).all()
                assert torch.isfinite(
                    torch.autograd.grad(gradient.square().sum(), vectors)[0]
                ).all()
    finally:
        torch.set_default_dtype(previous)


def test_quaternion_wigner_compile_and_capture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    frame = o2.WignerD(3, 3, backend="cuda").cuda()
    function = torch.compile(
        frame.forward_packed, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for count in (3, 7, 0):
        vectors = torch.randn(count, 3, device="cuda", requires_grad=True)
        actual, expected = function(vectors), frame.forward_packed(vectors)
        torch.testing.assert_close(actual, expected)
        # The squared norm of an orthogonal Wigner matrix is constant; use a
        # directional readout so this checks derivatives rather than roundoff.
        probe = torch.randn_like(actual)
        actual_grad = torch.autograd.grad((actual.sin() * probe).sum(), vectors)[0]
        expected_grad = torch.autograd.grad((expected.sin() * probe).sum(), vectors)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    vectors = torch.randn(11, 3, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            frame.forward_packed(vectors)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = frame.forward_packed(vectors)
    vectors.normal_()
    graph.replay()
    torch.testing.assert_close(output, frame.forward_packed(vectors))


@pytest.mark.parametrize("degree", [0, 1, 3, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["recursive", "quaternion"])
def test_fused_wigner_derivatives(degree, dtype, method):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.wigner import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(27)
        frame = o2.WignerD(degree, degree, backend="cuda").cuda()
        vectors = torch.randn(5, 3, device="cuda", requires_grad=True)
        actual = wigner_D(frame, vectors, method=method, backend="cuda")
        expected = frame.forward_packed(vectors, method="recursive")
        tolerance = 2e-5 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(
            wigner_D(frame, vectors.detach(), method=method, backend="cuda"),
            expected.detach(),
            atol=tolerance,
            rtol=tolerance,
        )
        if degree:
            actual, expected = actual.sin(), expected.sin()
            for _ in range(4 if degree == 1 and dtype == torch.float64 else 3):
                seed = torch.randn_like(actual) / actual.numel() ** 0.5
                actual = torch.autograd.grad(
                    (actual * seed).sum(), vectors, create_graph=True, retain_graph=True
                )[0]
                expected = torch.autograd.grad(
                    (expected * seed).sum(),
                    vectors,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                torch.testing.assert_close(
                    actual, expected, atol=10 * tolerance, rtol=10 * tolerance
                )
        empty = vectors[:0]
        torch.testing.assert_close(
            wigner_D(frame, empty, method=method, backend="cuda"),
            frame.forward_packed(empty, method="recursive"),
        )
        torch.testing.assert_close(
            wigner_D(frame, empty.detach(), method=method, backend="cuda"),
            frame.forward_packed(empty, method="recursive").detach(),
        )
    finally:
        torch.set_default_dtype(previous)
