"""Tests for the corresponding EQX source module."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from e3nn import o3

from equivariantx import conv as eqx_conv
from equivariantx import o2
from equivariantx.ops.precision import (
    validate_module_precision,
    validate_precision,
    validate_uniform,
)


def test_public_harmonics_reject_autocast():
    from equivariantx.co3 import CartesianHarmonics

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast"):
            CartesianHarmonics(2, normalize=False)(torch.ones(3, 3))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_public_harmonics_reject_reduced_precision(dtype):
    from equivariantx.co3 import CartesianHarmonics

    module = CartesianHarmonics(2, normalize=False)
    with pytest.raises(TypeError, match="float32 or float64"):
        module(torch.ones(3, 3, dtype=dtype))


def test_module_state_checked():
    module = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="float32 or float64"):
        validate_module_precision(module, [torch.ones(2, 3)])


def test_validation_is_traceable():
    def function(value):
        validate_precision([value])
        return value.square()

    compiled = torch.compile(function, backend="eager", fullgraph=True)
    value = torch.randn(3, requires_grad=True)
    torch.testing.assert_close(compiled(value), function(value))


def run_precision_process(code, tmp_path):
    """Keep policy mutations outside the pytest process."""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join((root, env.get("PYTHONPATH", "")))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_matrix_precision_policy(tmp_path):
    if not hasattr(torch._C, "_get_fp32_precision_getter"):
        pytest.skip("Per-backend matrix precision is unavailable")
    run_precision_process(
        """
import pytest
import torch
from equivariantx import o2
from equivariantx.ops.precision import validate_precision

torch.set_num_threads(2)
get = torch._C._get_fp32_precision_getter
set_precision = torch._C._set_fp32_precision_setter
x, w = torch.randn(7, 4), torch.randn(2, 12)
indices = torch.arange(7) % 2
linear = o2.Linear('4x0ee', '3x0ee', internal_weights=False, shared_weights=False)
module = o2.ElementLinear(linear, backend='torch', element_mode='single')
for setting in ('medium', 'explicit', 'inherited'):
    set_precision('generic', 'all', 'none')
    set_precision('mkldnn', 'all', 'none')
    set_precision('mkldnn', 'matmul', 'none')
    if setting == 'medium':
        torch.set_float32_matmul_precision('medium')
    elif setting == 'explicit':
        torch.backends.mkldnn.matmul.fp32_precision = 'bf16'
    else:
        torch.backends.fp32_precision = 'bf16'
    before = tuple(get(b, o) for b, o in (
        ('generic', 'all'), ('mkldnn', 'all'), ('mkldnn', 'matmul')))
    with pytest.raises(TypeError, match='MKLDNN BF16'):
        validate_precision([{'x': [x], 'weight': w}])
    with pytest.raises(TypeError, match='MKLDNN BF16'):
        module(x, w, indices)
    assert before == tuple(get(b, o) for b, o in (
        ('generic', 'all'), ('mkldnn', 'all'), ('mkldnn', 'matmul')))
    validate_precision([x.double(), w.double()])
    module(x.double(), w.double(), indices)
    torch.backends.mkldnn.enabled = False
    module(x, w, indices)
    torch.backends.mkldnn.enabled = True
    for allowed in ('ieee', 'tf32'):
        torch.backends.mkldnn.matmul.fp32_precision = allowed
        validate_precision([x, w])
        module(x, w, indices)
torch.set_float32_matmul_precision('high')
validate_precision([x, w])
module(x, w, indices)
""",
        tmp_path,
    )


def test_cpu_policy_checked_once(monkeypatch):
    from equivariantx.ops import precision

    if not hasattr(torch._C, "_get_fp32_precision_getter"):
        pytest.skip("Per-backend matrix precision is unavailable")
    calls = []

    def policy():
        calls.append(True)
        return True

    monkeypatch.setattr(precision, "cpu_matmul_allowed", policy)
    precision.validate_precision([torch.ones(2), {"weight": [torch.ones(3)]}])
    assert calls == [True]
    calls.clear()
    precision.validate_precision([torch.ones(3, dtype=torch.float64)])
    assert not calls


def test_cpu_policy_compiler_guard(tmp_path):
    if not hasattr(torch._C, "_get_fp32_precision_getter"):
        pytest.skip("Per-backend matrix precision is unavailable")
    run_precision_process(
        """
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from equivariantx import o2
from equivariantx.ops.precision import validate_precision

torch.set_num_threads(2)
graphs = []
def backend(graph, inputs):
    graphs.append(graph)
    return graph.forward
def function(x, w):
    validate_precision([x, w])
    return (x @ w).sin()
x, w = torch.randn(7, 4), torch.randn(4, 3)
compiled = torch.compile(function, backend=backend, fullgraph=True, dynamic=True)
expected = function(x, w)
torch.testing.assert_close(compiled(x, w), expected)
assert len(graphs) == 1
linear = o2.Linear('4x0ee', '3x0ee', internal_weights=False, shared_weights=False)
module = o2.ElementLinear(linear, backend='torch', element_mode='single')
module_run = torch.compile(module, backend='eager', fullgraph=True, dynamic=True)
indices = torch.zeros(7, dtype=torch.long)
torch.testing.assert_close(module_run(x, w.reshape(1, -1), indices), module(x, w.reshape(1, -1), indices))
torch.backends.mkldnn.matmul.fp32_precision = 'tf32'
torch.testing.assert_close(compiled(x, w), function(x, w))
assert len(graphs) == 1
torch.backends.mkldnn.matmul.fp32_precision = 'bf16'
with pytest.raises(torch._dynamo.exc.Unsupported, match='MKLDNN BF16'):
    compiled(x, w)
with pytest.raises(torch._dynamo.exc.Unsupported, match='MKLDNN BF16'):
    module_run(x, w.reshape(1, -1), indices)
with pytest.raises(TypeError, match='MKLDNN BF16'):
    make_fx(function)(x, w)
assert torch.backends.mkldnn.matmul.fp32_precision == 'bf16'
assert len(graphs) == 1
torch.testing.assert_close(compiled(x.double(), w.double()), function(x.double(), w.double()))
torch.backends.mkldnn.enabled = False
torch.testing.assert_close(compiled(x, w), expected)
torch.backends.mkldnn.enabled = True
with pytest.raises(torch._dynamo.exc.Unsupported, match='MKLDNN BF16'):
    compiled(x, w)
torch.backends.mkldnn.matmul.fp32_precision = 'none'
torch.backends.fp32_precision = 'bf16'
with pytest.raises(torch._dynamo.exc.Unsupported, match='MKLDNN BF16'):
    compiled(x, w)
torch.backends.mkldnn.matmul.fp32_precision = 'ieee'
torch.testing.assert_close(compiled(x, w), expected)
assert torch.backends.fp32_precision == 'bf16'
assert torch.backends.mkldnn.matmul.fp32_precision == 'ieee'
""",
        tmp_path,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matrix_precision_policy(tmp_path):
    run_precision_process(
        """
import torch
from equivariantx.ops.precision import validate_precision

torch.set_num_threads(2)
x, w = torch.randn(7, 4, device='cuda'), torch.randn(4, 3, device='cuda')
def function(x, w):
    validate_precision([x, w])
    return x @ w
compiled = torch.compile(function, backend='eager', fullgraph=True)
for policy in ('high', 'medium'):
    torch.set_float32_matmul_precision(policy)
    torch.testing.assert_close(compiled(x, w), function(x, w))
    assert torch.get_float32_matmul_precision() == policy
""",
        tmp_path,
    )


def test_cpu_dynamic_joint_aoti(tmp_path):
    pytest.importorskip("torch.fx.experimental.dynamic_spec")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    linear = o2.Linear("4x0ee", "3x0ee", internal_weights=False, shared_weights=False)
    module = o2.ElementLinear(linear, backend="torch", element_mode="single")

    def operation(x, weight, indices):
        energy = module(x, weight, indices).sin().square().mean()
        force = torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return loss, force, *torch.autograd.grad(loss, weight)

    def inputs(rows, weight):
        return torch.randn(rows, 4).requires_grad_(), weight, torch.arange(rows) % 2

    weight = (torch.randn(2, linear.weight_numel) * 0.1).requires_grad_()
    values = inputs(7, weight)
    rows = ShapeVar("rows", min=2)
    shapes = ShapesSpec(
        ParamsSpec({"x": TensorSpec((rows, None)), "indices": TensorSpec((rows,))})
    )
    graph = make_fx(
        operation,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=shapes,
    )(*values)
    rows = torch.export.Dim("rows", min=2)
    exported = torch.export.export(
        graph,
        tuple(v.detach() for v in values),
        strict=True,
        dynamic_shapes=({0: rows}, None, {0: rows}),
    )
    package = tmp_path / "precision.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for count in (7, 11, 5):
        values = inputs(count, weight)
        expected = operation(*values)
        records.append(
            {
                "inputs": tuple(v.detach() for v in values),
                "outputs": tuple(v.detach() for v in expected),
            }
        )
        weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
    data = tmp_path / "states.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
torch.set_num_threads(2)
run = torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = run(*row['inputs'])
    torch.testing.assert_close(tuple(actual), row['outputs'], atol=3e-6, rtol=3e-5)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_autocast_rejected():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast"):
            validate_precision([torch.ones(2)])


def test_mixed_geometry_precision():
    values = [torch.ones(2, dtype=torch.float32), torch.ones(3, dtype=torch.float64)]
    validate_precision(values)
    with pytest.raises(ValueError, match="share a dtype"):
        validate_uniform(values)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.complex64, torch.complex128]
)
def test_reduced_storage_rejected(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        validate_precision([{"features": [torch.ones(3, dtype=dtype)]}])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supported_storage(dtype):
    validate_precision([torch.ones(3, dtype=dtype), {"index": torch.arange(3)}])


@pytest.mark.parametrize("kind", ["o3", "o2", "co3", "uu"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_convolution_geometry_precision(kind, backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    args = (
        "2x0e+2x1o",
        "0e+1o",
        "2x0e+2x1o",
        [(0, 0, 0, "uvu", True), (1, 1, 0, "uvu", True), (0, 1, 1, "uvu", True)],
    )
    from equivariantx import co3

    tp_cls = {
        "o3": o3.TensorProduct,
        "o2": o2.O3TensorProduct,
        "co3": co3.TensorProduct,
        "uu": o2.O3TensorProduct,
    }[kind]
    tp = (
        tp_cls(
            *args,
            internal_weights=False,
            shared_weights=False,
        )
        .float()
        .to(device)
    )
    if kind == "uu":
        local_frame = o2.LocalFrame(args[0]).to(device)
        tp = o2.UuLinear(local_frame.irreps_out, local_frame.irreps_out, num_channel=2)
        module = (
            eqx_conv.UuO2TensorProductConv(
                local_frame,
                tp,
                local_frame,
                backend=backend,
            )
            .float()
            .to(device)
        )
    else:
        module = {
            "o3": eqx_conv.O3TensorProductConv,
            "o2": eqx_conv.O2O3TensorProductConv,
            "co3": eqx_conv.CartesianTensorProductConv,
        }[kind](
            tp,
            backend=backend,
        ).to(device)
    frame = o2.WignerD(1, 1).to(device)
    x = torch.randn(3, 8, dtype=torch.float32, device=device, requires_grad=True)
    vectors = torch.randn(5, 3, dtype=torch.float64, device=device, requires_grad=True)
    radial = torch.randn(5, 3, dtype=torch.float64, device=device, requires_grad=True)
    projection = torch.randn(
        3,
        tp.weight_numel,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 0]], device=device)
    amplitudes = torch.ones(
        1, 1 if kind == "uu" else 2, dtype=torch.float64, device=device
    )
    angular = o3.spherical_harmonics(
        [0, 1],
        vectors,
        normalize=True,
        normalization="component",
    )
    if kind in ("o3", "co3"):
        actual = module(x, angular, radial, projection, index, vectors=vectors)
        expected = module(x, angular.float(), radial.float(), projection, index)
    else:
        actual = module(
            x,
            radial,
            projection,
            None,
            amplitudes,
            index,
            3,
            vectors=vectors,
        )
        expected = module(
            x,
            radial.float(),
            projection,
            frame.forward_packed(vectors).float(),
            amplitudes.float(),
            index,
            3,
        )
    assert actual.dtype == torch.float32
    variables = (x, vectors, radial, projection)
    for _ in range(2):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        results = [
            torch.autograd.grad(value.square().sum(), variables, create_graph=True)
            for value in (actual, expected)
        ]
        for gradients in results:
            assert tuple(g.dtype for g in gradients) == tuple(
                v.dtype for v in variables
            )
            assert all(torch.isfinite(g).all() for g in gradients)
        actual, expected = (
            torch.cat([g.flatten() for g in grads]) for grads in results
        )


def test_cuda_operand_precision_validation():
    from equivariantx.ops.registry import validate_operands

    validate_operands([torch.zeros(2), torch.ones(3), torch.zeros(2, dtype=torch.long)])
    with pytest.raises(ValueError, match="share a dtype"):
        validate_operands([torch.zeros(2), [torch.ones(2, dtype=torch.float64)]])
    with pytest.raises(TypeError, match="float32 or float64"):
        validate_operands([torch.zeros(2, dtype=torch.bfloat16)])


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_uv_convolution_geometry_precision(backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    frame = o2.LocalFrame("2x0e+2x1o").to(device)
    scalars = frame.irreps_out.filter(keep=lambda entry: entry.ir.m == 0)
    tensors = frame.irreps_out.filter(keep=lambda entry: entry.ir.m > 0)
    gate = (
        o2.Gate(
            scalars,
            [torch.nn.SiLU() for _ in scalars],
            o2.Irreps([(o2.Irrep("0e"), tensors.num_irreps)]),
            [torch.nn.Sigmoid()],
            tensors,
        )
        .float()
        .to(device)
    )
    up = o2.Linear((frame.irreps_out * 2).regroup(), gate.irreps_in).float().to(device)
    down = o2.Linear(gate.irreps_out, frame.irreps_out).float().to(device)
    layer = (
        eqx_conv.UvO2TensorProductConv(
            frame,
            up,
            gate,
            down,
            frame,
            backend=backend,
        )
        .float()
        .to(device)
    )
    features = torch.randn(3, frame.input_dim, device=device, requires_grad=True)
    vectors = torch.randn(4, 3, dtype=torch.float64, device=device, requires_grad=True)
    weights = torch.randn(
        4,
        layer.weight_numel,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    cutoff = torch.rand(4, 1, dtype=torch.float64, device=device, requires_grad=True)
    index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)
    parameters = up.weight, up.bias, down.weight, down.bias
    actual = layer(
        features,
        None,
        weights,
        index,
        None,
        None,
        cutoff,
        parameters,
        vectors=vectors,
    )
    wigner = layer.wigner.forward_packed(vectors).float()
    expected = layer(
        features,
        None,
        weights.float(),
        index,
        wigner,
        None,
        cutoff.float(),
        parameters,
    )
    variables = features, vectors, weights, cutoff, up.weight, down.weight
    for _ in range(2):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        gradients = [
            torch.autograd.grad(value.square().sum(), variables, create_graph=True)
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([g.flatten() for g in result]) for result in gradients
        ]
