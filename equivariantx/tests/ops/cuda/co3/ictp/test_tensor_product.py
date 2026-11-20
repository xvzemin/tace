"""Native angular kernels in independently loaded joint training artifacts."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops import config
from equivariantx.ops.compiler import compile_and_package
from equivariantx.tests.ops.torch.co3.ictp.test_tensor_product import training_problem
from equivariantx.tests.utils.testing import check_forward_ad, package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


@pytest.mark.parametrize("degrees", [(3, 3, 6), (4, 3, 6), (4, 4, 6)])
def test_symmetric_free_indices(degrees):
    from equivariantx.co3.ictp.tensor_product import CartesianProduct
    from equivariantx.ops.torch.co3.ictp.tensor_product import reference

    modules = [
        CartesianProduct(*degrees, backend=b).cuda().double() for b in ("torch", "cuda")
    ]
    x, y = [
        (torch.randn(*shape, 3**degree * 2, device="cuda", dtype=torch.float64) * 0.1)[
            ..., ::2
        ].requires_grad_()
        for shape, degree in zip(((2, 1), (1, 3)), degrees)
    ]
    expected = reference(modules[0], x, y)
    for module in modules:
        actual, oracle = module(x, y), expected
        for _ in range(3):
            torch.testing.assert_close(actual, oracle, rtol=2e-10, atol=2e-11)
            seed = torch.randn_like(oracle)
            actual, oracle = [
                torch.cat(
                    [
                        a.flatten()
                        for a in torch.autograd.grad(
                            v.sin(), (x, y), seed, create_graph=True
                        )
                    ]
                )
                for v in (actual, oracle)
            ]
        torch.testing.assert_close(actual, oracle, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("route", ["ictd", "ictp"])
@pytest.mark.parametrize("mode", ["uvu", "uvv", "uuu"])
def test_independent_weighted_paths(backend, route, mode, monkeypatch):
    from equivariantx import co3

    c = 4
    paths = [(0, 0, 0, mode, True, 0.5), (1, 0, 0, mode, True, 1.5)]
    cls = getattr(co3, route).TensorProduct

    def layer(b):
        return (
            cls(
                f"{c}x1o+{c}x1o",
                f"{c}x1o",
                f"{c}x2e",
                paths,
                internal_weights=False,
                shared_weights=False,
                backend=b,
            )
            .cuda()
            .double()
        )

    op, ref = layer(backend), layer("torch")
    values = [
        torch.randn(*s, device="cuda", dtype=torch.float64).requires_grad_()
        for s in ((2, 1, 6 * c), (1, 3, 3 * c), (2, 1, op.weight_numel))
    ]
    actual = op(*values)
    with monkeypatch.context() as change:
        change.setattr(config, "TORCH_REFERENCE", True)
        expected = ref(*values)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    a.flatten()
                    for a in torch.autograd.grad(
                        z.sin(), values, seed, create_graph=True
                    )
                ]
            )
            for z in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("degrees", [(4, 3, 6), (4, 4, 6), (7, 7, 0)])
def test_indexed_cartesian_product(degrees):
    from equivariantx.ops import get_kernel

    def operation(backend, x, y):
        return get_kernel("cartesian_product", backend)(x, y, *degrees)

    x, y = [
        (torch.randn(*shape, 2 * 3**l, device="cuda", dtype=torch.float64) * 0.1)[
            ..., ::2
        ].requires_grad_()
        for l, shape in zip(degrees[:2], ((2, 1), (1, 3)))
    ]
    actual, expected = [operation(backend, x, y) for backend in ("cuda", "torch")]
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(4):
        seed = torch.randn_like(expected)
        derivative = [
            torch.autograd.grad(value.sin(), (x, y), seed, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*derivative, atol=2e-10, rtol=2e-10)
        actual, expected = [
            torch.cat([v.flatten() for v in pair]) for pair in derivative
        ]
    check_forward_ad(
        lambda a, b: operation("cuda", a, b),
        lambda a, b: operation("torch", a, b),
        (x, y),
    )
    empty = operation("cuda", x[:0], y)
    assert empty.shape == (0, 3, 3 ** degrees[2])


@pytest.mark.parametrize(
    "degrees,inline",
    [
        ((2, 2, 2), False),
        ((4, 3, 6), False),
        ((4, 3, 6), True),
        ((5, 6, 2), False),
        ((6, 2, 5), False),
    ],
)
def test_joint_aoti(degrees, inline, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "ICTP_INLINE", False)
    monkeypatch.setattr(config, "CARTESIAN_REDUCTION_INLINE", inline)
    monkeypatch.setattr(config, "SPHERICAL_INLINE", False)
    operation, values = training_problem("cuda", "cuda", degrees)
    reference, _ = training_problem("cuda", "torch", degrees)
    shapes = None
    if degrees != (2, 2, 2):
        from torch.fx.experimental.dynamic_spec import (
            ParamsSpec,
            ShapesSpec,
            ShapeVar,
            TensorSpec,
        )

        rows = ShapeVar("rows", min=2, optimization_hint=3)
        specs = {"x": TensorSpec((rows, None)), "y": TensorSpec((rows, None))}
        graph = make_fx(
            operation,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
            dynamic_shapes=ShapesSpec(ParamsSpec(specs)),
        )(*values)
        rows = torch.export.Dim("rows", min=2, max=32)
        shapes = ({0: rows}, {0: rows}, None)
    else:
        graph = make_fx(operation)(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    if degrees == (4, 3, 6):
        assert ("eqx.cartesian_contract" in graph.code) != inline
    if degrees == (5, 6, 2):
        assert "eqx.ictp_symmetric_epsilon" in graph.code
    if degrees == (6, 2, 5):
        assert "eqx.ictp_orbit_product" in graph.code
    exported = torch.export.export(
        graph, tuple(v.detach() for v in values), strict=True, dynamic_shapes=shapes
    )
    package = tmp_path / "module.pt2"
    compile_and_package(exported, package_path=str(package))
    records = []
    x, y, w = values
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for rows in (3, 7, 9) if shapes else (3, 3, 3):
        if shapes:
            x, y = [
                (v.new_empty(rows, v.shape[-1]).normal_() * 0.1).requires_grad_()
                for v in (x, y)
            ]
        expected = reference(x, y, w)
        records.append(
            {
                "inputs": tuple(v.detach().cpu() for v in (x, y, w)),
                "outputs": tuple(v.detach().cpu() for v in expected),
            }
        )
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script("""
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad():actual=run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.cpu() for v in actual),row['outputs'],atol=3e-9,rtol=3e-9)
assert 'equivariantx' in sys.modules
"""),
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
