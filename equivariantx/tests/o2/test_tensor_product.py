"""Tests for the corresponding EQX source module."""

from copy import deepcopy
import pytest
import torch
from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
from equivariantx.tests.o2.testing import _transform
from equivariantx.tests.o2.testing import _tensor_product_case
from equivariantx.tests.o2.testing import _time_reverse


@pytest.mark.parametrize("mode", ["u1u", "uuu", "uvw"])
@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_o2_tensor_product_is_equivariant(mode, reflected, backend):
    if backend == "cuda" and DEVICE.type != "cuda":
        pytest.skip("CUDA requires a CUDA device")
    irreps_in1, irreps_in2, irreps_out = _tensor_product_case(mode)
    module = o2.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        [(0, 0, 0, mode, True)],
        internal_weights=False,
        shared_weights=False,
        backend=backend,
    ).to(DEVICE, DTYPE)
    input1 = module.irreps_in1.randn(5, -1, dtype=DTYPE, device=DEVICE)
    input2 = module.irreps_in2.randn(5, -1, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(5, module.weight_numel, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    output = module(input1, input2, weights)
    expected = _transform(output, module.irreps_out, angle, reflected)
    actual = module(
        _transform(input1, module.irreps_in1, angle, reflected),
        _transform(input2, module.irreps_in2, angle, reflected),
        weights,
    )
    torch.testing.assert_close(actual, expected)


def test_o2_tensor_product_zero_pads_missing_outputs():
    module = o2.TensorProduct(
        "2x1m",
        "0e",
        "2x1m+0o",
        [(0, 0, 0, "u1u", False)],
    )
    output = module(torch.randn(3, 4), torch.randn(3, 1))
    torch.testing.assert_close(output[:, -1], torch.zeros_like(output[:, -1]))


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("shared", [False, True])
def test_mixed_channel_modes(backend, shared, monkeypatch, double_precision):
    from equivariantx.ops import config

    if backend == "cuda" and DEVICE.type != "cuda":
        pytest.skip("CUDA unavailable")
    module = o2.TensorProduct(
        "2x1m+3x0o",
        "3x1m+0e",
        "4x0e+2x1m+3x0o+0e",
        [(0, 0, 0, "uvw", True), (0, 1, 1, "u1u", False), (1, 1, 2, "u1u", True)],
        backend=backend,
        internal_weights=False,
        shared_weights=shared,
    ).to(DEVICE)
    reference = deepcopy(module)
    reference.backend = "torch"
    inputs = tuple(
        torch.randn(*shape, device=DEVICE, requires_grad=True)
        for shape in (
            (2, 1, 7),
            (1, 3, 7),
            (module.weight_numel,) if shared else (2, 3, module.weight_numel),
        )
    )
    actual = module(*inputs)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    a, b = actual.sin().sum(), expected.sin().sum()
    for _ in range(4):
        ga, gb = [
            torch.autograd.grad(value, inputs, create_graph=True) for value in (a, b)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-10, rtol=2e-10)
        a, b = [sum(g.sin().sum() for g in gradients) / 10 for gradients in (ga, gb)]
    tangents = tuple(torch.randn_like(value) for value in inputs)
    expected_jvp = torch.func.jvp(reference, inputs, tangents)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(
        torch.func.jvp(module, inputs, tangents), expected_jvp, atol=2e-10, rtol=2e-10
    )
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(*inputs), expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_o2_tensor_product_mixed_channels(shared, dtype):
    if DEVICE.type != "cuda":
        pytest.skip("CUDA is unavailable")
    options = dict(internal_weights=False, shared_weights=shared)
    instructions = [
        (0, 0, 0, "uuu", True),
        (1, 1, 1, "uuu", True),
        (2, 2, 2, "u1u", True),
        (2, 2, 2, "u1u", False),
        (1, 1, 3, "uuu", False),
    ]
    args = (
        "2x1m+3x2m+4x1m",
        "2x1m+3x2m+1x0o",
        "2x0e+3x0o+4x1m+3x4m+1x0e",
        instructions,
    )
    reference = o2.TensorProduct(*args, backend="torch", **options).to(DEVICE, dtype)
    module = o2.TensorProduct(*args, backend="cuda", **options).to(DEVICE, dtype)
    assert module.kernel_metadata is not None
    inputs = [
        torch.randn(5, ir.dim, device=DEVICE, dtype=dtype, requires_grad=True)
        for ir in (module.irreps_in1, module.irreps_in2)
    ]
    inputs.append(
        torch.randn(
            (module.weight_numel,) if shared else (5, module.weight_numel),
            device=DEVICE,
            dtype=dtype,
            requires_grad=True,
        )
    )
    actual, expected = module(*inputs), reference(*inputs)
    tol = 2e-5 if dtype == torch.float32 else 2e-12
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        probe = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    grad.flatten()
                    for grad in torch.autograd.grad(
                        (value * probe).sum(),
                        inputs,
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("irreps_out", ["", "0o"])
def test_o2_disconnected_operators_have_zero_gradients(double_precision, irreps_out):
    linear = o2.Linear("0e", irreps_out, instructions=[])
    tp = o2.TensorProduct("0e", "0e", irreps_out, [])
    for batch_size in (3, 0):
        x = torch.randn(batch_size, 1, requires_grad=True)
        y = torch.randn(batch_size, 1, requires_grad=True)
        for output, inputs in ((linear(x), (x,)), (tp(x, y), (x, y))):
            assert output.shape == (batch_size, linear.irreps_out.dim)
            torch.testing.assert_close(output, torch.zeros_like(output))
            for grad in torch.autograd.grad(output.sum(), inputs):
                torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.parametrize("normalization", ["component", "norm", "none"])
@pytest.mark.parametrize(
    ("ir1", "ir2", "ir_out"),
    [
        ("0o", "1m", "1m"),
        ("1m", "1m", "0e"),
        ("1m", "1m", "0o"),
        ("1m", "2m", "1m"),
        ("1m", "2m", "3m"),
    ],
)
def test_o2_tensor_product_coupling_normalization(
    double_precision, normalization, ir1, ir2, ir_out
):
    ir1, ir2, ir_out = o2.Irrep(ir1), o2.Irrep(ir2), o2.Irrep(ir_out)
    module = o2.TensorProduct(
        ir1,
        ir2,
        ir_out,
        [(0, 0, 0, "uuu", False)],
        irrep_normalization=normalization,
        path_normalization="none",
    )
    coefficients = module(torch.eye(ir1.dim)[:, None], torch.eye(ir2.dim)[None, :])
    squared_norm = {"component": ir_out.dim, "norm": ir1.dim * ir2.dim, "none": 1}
    torch.testing.assert_close(
        coefficients.square().sum(), torch.tensor(float(squared_norm[normalization]))
    )


@pytest.mark.parametrize("batch_size", [0, 4])
@pytest.mark.parametrize(
    ("ir1", "ir2"), [("0oo", "1mo"), ("1me", "1mo"), ("1mo", "2me")]
)
def test_o2_uuu_matches_diagonal_uvw(double_precision, batch_size, ir1, ir2):
    channels = 16
    ir1, ir2 = o2.Irrep(ir1), o2.Irrep(ir2)
    irreps_out = o2.Irreps([(ir, channels) for ir in ir1 * ir2])
    kwargs = dict(
        internal_weights=False, shared_weights=False, path_normalization="none"
    )
    module = o2.TensorProduct(
        channels * ir1,
        channels * ir2,
        irreps_out,
        [(0, 0, i, "uuu", True) for i in range(len(irreps_out))],
        **kwargs,
    )
    reference = o2.TensorProduct(
        channels * ir1,
        channels * ir2,
        irreps_out,
        [(0, 0, i, "uvw", True) for i in range(len(irreps_out))],
        **kwargs,
    )
    x = torch.randn(batch_size, ir1.dim * channels, requires_grad=True)
    y = torch.randn(batch_size, ir2.dim * channels, requires_grad=True)
    weight = torch.randn(batch_size, module.weight_numel, requires_grad=True)
    reference_weight = torch.einsum(
        "bpu,uv,uw->bpuvw",
        weight.reshape(batch_size, len(irreps_out), channels),
        torch.eye(channels),
        torch.eye(channels),
    ).flatten(-4)
    actual, expected = module(x, y, weight), reference(x, y, reference_weight)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, y, weight)),
        torch.autograd.grad(expected.square().sum(), (x, y, weight)),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_o2_tensor_product_preserves_time_parity(backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    irreps_in = o2.Irreps("2x0ee+2x0oo+2x1me+2x1mo")
    irreps_out = o2.Irreps("2x0ee+2x0eo+2x0oe+2x0oo+2x1me+2x1mo+2x2me+2x2mo")
    tensor_product = o2.TensorProduct(
        irreps_in,
        irreps_in,
        irreps_out,
        [
            (i, j, k, "uuu", True)
            for i, (a, _) in enumerate(irreps_in)
            for j, (b, _) in enumerate(irreps_in)
            for k, (c, _) in enumerate(irreps_out)
            if c in a * b
        ],
        backend=backend,
    ).to(DEVICE, DTYPE)
    first = irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    second = irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    output = tensor_product(first, second)
    torch.testing.assert_close(
        tensor_product(
            _time_reverse(first, irreps_in),
            _time_reverse(second, irreps_in),
        ),
        _time_reverse(output, irreps_out),
    )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_o2_tensor_product_cuda_derivatives(shared, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    irreps = o2.Irreps("3x0e+3x0o+3x1m+3x2m")
    instructions = [
        (i, j, k, "uuu", (i + j + k) % 3 != 0)
        for i, (a, _) in enumerate(irreps)
        for j, (b, _) in enumerate(irreps)
        for k, (c, _) in enumerate(irreps)
        if c in a * b
    ]
    reference = o2.TensorProduct(
        irreps, irreps, irreps, instructions, internal_weights=False
    ).to("cuda", dtype)
    module = o2.TensorProduct(
        irreps, irreps, irreps, instructions, internal_weights=False, backend="cuda"
    ).to("cuda", dtype)
    x, y = [
        torch.randn(37, 2 * irreps.dim, device="cuda", dtype=dtype)[
            :, ::2
        ].requires_grad_()
        for _ in range(2)
    ]
    weight = torch.randn(
        1 if shared else 37,
        module.weight_numel,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    values = [f(x, y, weight) for f in (reference, module)]
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(*values, atol=tolerance, rtol=tolerance)
    # Nonlinear losses keep mixed derivatives of all three operands live.
    for _ in range(3):
        gradients = [
            torch.autograd.grad(
                value.square().sum() / value.numel(), (x, y, weight), create_graph=True
            )
            for value in values
        ]
        for expected, actual in zip(*gradients):
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        values = [torch.cat([g.flatten() for g in gs]) for gs in gradients]


def test_o2_tensor_product_cuda_compile_and_empty():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = (
        o2.TensorProduct(
            "5x1m",
            "5x1m",
            "5x0e+5x0o+5x2m+5x3m",
            [(0, 0, 0, "uuu", True), (0, 0, 1, "uuu", False), (0, 0, 2, "uuu", True)],
            backend="cuda",
        )
        .cuda()
        .double()
    )
    reference = deepcopy(module)
    reference.backend = "torch"
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for nodes in (9, 1, 0):
        x = torch.randn(
            nodes, 10, device="cuda", dtype=torch.float64, requires_grad=True
        )
        for function in (module, compiled):
            actual, expected = function(x, x), reference(x, x)
            torch.testing.assert_close(actual, expected)
            gradients = [
                torch.autograd.grad(y.square().sum(), (x, f.weight))
                for y, f in ((actual, module), (expected, reference))
            ]
            for a, b in zip(*gradients):
                torch.testing.assert_close(a, b)
