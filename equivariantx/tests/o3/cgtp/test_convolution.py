"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import conv as eqx_conv
from equivariantx import o2
from equivariantx.ops import config
from equivariantx.tests.conv.o2_o3.test_convolution import (
    test_streaming_projected_tiles as check_streaming_projected_tiles,
)
from equivariantx.tests.conv.testing import layout
from equivariantx.tests.conv.testing import reference
from equivariantx.tests.conv.testing import assert_native_autograd
from equivariantx.tests.conv.testing import tensor_product


@pytest.mark.parametrize("rows", [0, 1, 9])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("copies", [2, 4])
def test_o3_rowwise_tensor_product(rows, shared, copies, device, double_precision):
    tp = o3.TensorProduct(
        "+".join(["3x1o"] * copies + ["2x0e"]),
        "2x1o+0e",
        "3x0e+3x1e+3x2e+2x1o",
        [(i, 0, k, "uvu", k != 1) for i in range(copies) for k in range(3)]
        + [(copies, 0, 3, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    conv = eqx_conv.O3TensorProductConv(tp, batch_paths=True, backend="cuda").to(device)
    inputs = [
        torch.randn(*shape, device=device, requires_grad=True)
        for shape in (
            (rows, tp.irreps_in1.dim),
            (1 if shared else rows, tp.irreps_in2.dim),
            (1 if shared else rows, tp.weight_numel),
        )
    ]
    projection = inputs[0].new_empty(0, tp.weight_numel)
    actual = conv(*inputs, projection, None)
    expected = conv.reference(*inputs, projection, None, rows)
    reference = layout(
        tp(
            layout(inputs[0], tp.irreps_in1, inverse=True),
            layout(inputs[1], tp.irreps_in2, inverse=True),
            inputs[2],
        ),
        tp.irreps_out,
    )
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for _ in range(3):
        seed = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad((value.sin() * seed).sum(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        actual, expected = [torch.cat([g.flatten() for g in gs]) for gs in gradients]


@pytest.mark.parametrize("implementation", ["o3", "o2"])
def test_partitioned_paths(monkeypatch, double_precision, implementation):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if implementation == "o3":
        from equivariantx.ops.common.cuda.o3 import convolution
    else:
        from equivariantx.ops.common.cuda.o2_o3 import convolution

    monkeypatch.setattr(convolution, "SOURCE_SIZE_LIMIT", 0)
    convolution.execution_plan.cache_clear()
    try:
        if implementation == "o3":
            test_o3_rowwise_tensor_product(9, False, 4, "cuda", double_precision)
        else:
            check_streaming_projected_tiles(monkeypatch, 1, 35)
    finally:
        convolution.execution_plan.cache_clear()


@pytest.mark.parametrize("edges", [0, 4])
@pytest.mark.parametrize(
    "algorithm",
    [
        "o3",
        "cartesian",
        "generator",
        "recurrence",
        "cg",
        "wigner",
        "uu_wigner",
        "uu_transverse",
    ],
)
@pytest.mark.parametrize("ordinary", [False, True])
def test_torch_scalar_direction_gradient(
    algorithm, edges, ordinary, double_precision, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3

    monkeypatch.setattr(config, "TORCH_REFERENCE", ordinary)

    args = ("2x0e", "0e", "2x0e", [(0, 0, 0, "uvu", True)])
    options = dict(internal_weights=False, shared_weights=False)
    if algorithm.startswith("uu"):
        frame = o2.LocalFrame("2x0e")
        linear = o2.UuLinear(frame.irreps_out, frame.irreps_out, 2)
        module = eqx_conv.UuO2TensorProductConv(frame, linear, frame, backend="torch")
    elif algorithm == "o3":
        module = eqx_conv.O3TensorProductConv(
            o3.TensorProduct(*args, **options), backend="torch"
        )
    elif algorithm == "cartesian":
        module = eqx_conv.CartesianTensorProductConv(
            co3.TensorProduct(*args, **options), backend="torch"
        )
    else:
        module = eqx_conv.O2O3TensorProductConv(
            o2.O3TensorProduct(*args, **options), backend="torch", method=algorithm
        )
    module = module.cuda()
    x = torch.randn(3, 2, device="cuda", requires_grad=True)
    vectors = torch.randn(edges, 3, device="cuda", requires_grad=True)
    radial = torch.randn(edges, 3, device="cuda", requires_grad=True)
    projection = torch.randn(3, module.weight_numel, device="cuda", requires_grad=True)
    amplitude = torch.randn(edges, 1, device="cuda", requires_grad=True)
    index = torch.randint(3, (2, edges), device="cuda")
    if algorithm in ("o3", "cartesian"):
        actual = module(
            x, None, radial, projection, index, vectors=vectors, amplitudes=amplitude
        )
    else:
        evaluate = module.forward_wigner if algorithm == "uu_wigner" else module
        actual = evaluate(
            x, radial, projection, None, amplitude, index, 3, vectors=vectors
        )
    if ordinary:
        assert_native_autograd(actual)
    grads = torch.autograd.grad(
        actual.square().sum(), (x, radial, projection, vectors, amplitude)
    )
    torch.testing.assert_close(grads[3], torch.zeros_like(vectors))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("shared,direct", [(False, False), (True, True)])
def test_o3_cartesian_derivatives(
    device, normalization, shared, direct, double_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = o3.TensorProduct(
        "3x1o+3x2e",
        "2x1o+2e",
        "3x1o+3x2e+3x3o",
        [
            (0, 0, 1, "uvu", True),
            (1, 0, 0, "uvu", True),
            (1, 0, 2, "uvu", True),
            (1, 1, 1, "uvu", False),
        ],
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    conv = eqx_conv.O3TensorProductConv(
        tp, normalization=normalization, backend="cuda"
    ).to(device)
    reference = eqx_conv.O3TensorProductConv(
        tp, normalization=normalization, backend="torch"
    ).to(device)
    edges = torch.randint(4, (2, 9), device=device)
    x = torch.randn(4, tp.irreps_in1.dim, device=device, requires_grad=True)
    vectors = torch.randn(9, 3, device=device, requires_grad=True)
    amplitudes = torch.randn(1 if shared else 9, 3, device=device, requires_grad=True)
    radial = torch.randn(
        1 if shared else 9,
        tp.weight_numel if direct else 4,
        device=device,
        requires_grad=True,
    )
    projection = torch.randn(
        0 if direct else 4, tp.weight_numel, device=device, requires_grad=True
    )
    inputs = x, vectors, amplitudes, radial, projection
    actual = conv(
        x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes
    )
    expected = reference(
        x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(3):
        seed = torch.randn_like(actual) / max(1, actual.numel()) ** 0.5
        a = torch.autograd.grad((actual.sin() * seed).sum(), inputs, create_graph=True)
        b = torch.autograd.grad(
            (expected.sin() * seed).sum(), inputs, create_graph=True
        )
        for value, target in zip(a, b):
            torch.testing.assert_close(value, target, atol=3e-9, rtol=3e-9)
        actual, expected = (
            torch.cat([value.flatten() for value in values]) for values in (a, b)
        )


@pytest.mark.parametrize("implementation", ["o3", "o2"])
@pytest.mark.parametrize("channels", [3, 64])
def test_reverse_edge_convolution(implementation, channels, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(105)
    args = (
        f"{channels}x1o",
        "1o+3o",
        f"{channels}x0e+{channels}x2e+{channels}x4e",
        [
            (0, 0, 0, "uvu", True),
            (0, 1, 1, "uvu", True),
            (0, 1, 2, "uvu", True),
        ],
    )
    options = dict(internal_weights=False, shared_weights=False)
    tp = o3.TensorProduct(*args, **options).cuda()
    reference = eqx_conv.O3TensorProductConv(tp, backend="torch").cuda()
    module = (
        eqx_conv.O3TensorProductConv(tp, backend="cuda")
        if implementation == "o3"
        else eqx_conv.O2O3TensorProductConv(
            o2.O3TensorProduct(*args, **options), backend="cuda"
        )
    ).cuda()
    # Repeated endpoints with distinct images, a self-image, duplicates and
    # one unmatched edge. Every directed edge has independent radial weights.
    pairs = torch.tensor([[0, 0, 1, 2], [1, 1, 2, 2]], device="cuda")
    edges = torch.cat(
        (pairs, pairs.flip(0), pairs[:, :1], pairs.flip(0)[:, :1], pairs[:, 2:3]),
        dim=1,
    )
    r = torch.randn(4, 3, device="cuda")
    vectors = torch.cat((r, -r, r[:1], -r[:1], r[2:3] * 2)).requires_grad_()
    x = torch.randn(3, tp.irreps_in1.dim, device="cuda", requires_grad=True)
    amplitudes = torch.randn(11, 2, device="cuda", requires_grad=True)
    radial = torch.randn(11, 4, device="cuda", requires_grad=True)
    projection = torch.randn(4, tp.weight_numel, device="cuda", requires_grad=True)
    inputs = x, vectors, amplitudes, radial, projection
    expected = reference(
        x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes
    )
    if implementation == "o3":
        actual = module(
            x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes
        )
    else:
        frame = o2.WignerD(4, 4, method="recursive").cuda()
        actual = module(
            x,
            radial,
            projection,
            frame.forward_packed(vectors.detach()),
            amplitudes,
            edges,
            3,
            vectors=vectors,
        )
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        seed = torch.randn_like(actual) / max(1, actual.numel()) ** 0.5
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (value.sin() * seed).sum(), inputs, create_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize(
    "implementation,symmetric_inputs",
    [("o3", False), ("co3", False), ("co3", True), ("compact", True)],
)
def test_o3_cartesian_compile_and_empty_graph(
    double_precision, implementation, symmetric_inputs
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3

    torch._dynamo.reset()

    cartesian = implementation != "o3"
    tp_cls = co3.TensorProduct if cartesian else o3.TensorProduct
    conv_cls = (
        eqx_conv.CartesianTensorProductConv
        if cartesian
        else eqx_conv.O3TensorProductConv
    )
    tp = tp_cls(
        "2x2e" if cartesian else "2x1o",
        "2e" if cartesian else "1o",
        "2x0e+2x2e",
        [(0, 0, 0, "uvu", True), (0, 0, 1, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
        **({"project": False} if implementation == "compact" else {}),
    ).cuda()
    kwargs = {"symmetric_inputs": symmetric_inputs} if cartesian else {}
    if implementation == "compact":
        kwargs.update(input_basis="spherical", compact_output=True)
    conv = conv_cls(tp, normalize=False, **kwargs).cuda()
    reference = conv_cls(tp, normalize=False, backend="torch", **kwargs).cuda()
    if implementation == "compact":
        from equivariantx.conv.co3.linear import Linear

        linear = Linear(tp.irreps_out, tp.irreps_out, conv.harmonic_output_index).cuda()
    else:
        linear = torch.nn.Identity()

    def operation(x, radial, projection, edges, vectors, amplitudes):
        return linear(
            conv(
                x,
                None,
                radial,
                projection,
                edges,
                vectors=vectors,
                amplitudes=amplitudes,
            )
        )

    compiled = torch.compile(
        operation, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for nodes, edges_count in ((3, 1), (3, 7), (3, 0), (0, 0)):
        edges = torch.randint(max(1, nodes), (2, edges_count), device="cuda")
        input_dim = (
            tp.irreps_in1.spherical().dim
            if implementation == "compact"
            else tp.irreps_in1.dim
        )
        x = torch.randn(nodes, input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(edges_count, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edges_count, 2, device="cuda", requires_grad=True)
        projection = torch.randn(2, tp.weight_numel, device="cuda", requires_grad=True)
        amplitudes = torch.randn(edges_count, 1, device="cuda", requires_grad=True)
        inputs = x, vectors, radial, projection, amplitudes
        actual = compiled(x, radial, projection, edges, vectors, amplitudes)
        expected = linear(
            reference(
                x,
                None,
                radial,
                projection,
                edges,
                vectors=vectors,
                amplitudes=amplitudes,
            )
        )
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        a = torch.autograd.grad(actual.square().sum(), inputs)
        b = torch.autograd.grad(expected.square().sum(), inputs)
        for value, target in zip(a, b):
            torch.testing.assert_close(value, target, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize(
    "device,backend,ordinary",
    [
        ("cpu", "torch", False),
        ("cpu", "torch", True),
        ("cuda", "torch", False),
        ("cuda", "torch", True),
        ("cuda", "cuda", False),
    ],
)
@pytest.mark.parametrize(
    "shared,direct,merge,unweighted",
    [
        (False, False, False, False),
        (True, False, True, False),
        (False, True, True, True),
    ],
)
def test_values_and_recursive_derivatives(
    device,
    backend,
    ordinary,
    shared,
    direct,
    merge,
    unweighted,
    double_precision,
    monkeypatch,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_REFERENCE", ordinary)
    tp = tensor_product(merge=merge, unweighted=unweighted).to(device)
    conv = eqx_conv.O3TensorProductConv(tp, backend=backend).to(device)
    edges = torch.randint(5, (2, 19), device=device)
    x = torch.randn(5, tp.irreps_in1.dim, device=device, requires_grad=True)
    attrs = torch.randn(
        1 if shared else 19, tp.irreps_in2.dim, device=device, requires_grad=True
    )
    radial = torch.randn(
        1 if shared else 19,
        tp.weight_numel if direct else 4,
        device=device,
        requires_grad=True,
    )
    projection = torch.randn(
        0 if direct else 4, tp.weight_numel, device=device, requires_grad=True
    )
    inputs = (x, attrs, radial) if direct else (x, attrs, radial, projection)
    actual = conv(x, attrs, radial, projection, edges)
    expected = reference(tp, x, attrs, radial, projection, edges)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    if ordinary:
        assert_native_autograd(actual)
    for _ in range(3):
        seed = torch.randn_like(actual) / actual.numel() ** 0.5
        a = torch.autograd.grad((actual.sin() * seed).sum(), inputs, create_graph=True)
        b = torch.autograd.grad(
            (expected.sin() * seed).sum(), inputs, create_graph=True
        )
        for value, target in zip(a, b):
            torch.testing.assert_close(value, target, atol=2e-10, rtol=2e-10)
        actual, expected = (
            torch.cat([value.flatten() for value in values]) for values in (a, b)
        )
        if ordinary:
            assert_native_autograd(actual)


@pytest.mark.parametrize("channels", [1, 32, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_degrees_and_channels(channels, dtype, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.set_default_dtype(dtype)
    tp = o3.TensorProduct(
        f"{channels}x3o",
        "2e",
        f"{channels}x3o+{channels}x5o",
        [(0, 0, 0, "uvu", True), (0, 0, 1, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    conv = eqx_conv.O3TensorProductConv(tp, backend="cuda").cuda()
    edges = torch.randint(7, (2, 33), device="cuda")
    # Noncontiguous operands are accepted without changing their derivatives.
    x = torch.randn(tp.irreps_in1.dim, 7, device="cuda").T.requires_grad_()
    attrs = torch.randn(tp.irreps_in2.dim, 33, device="cuda").T.requires_grad_()
    radial = torch.randn(8, 33, device="cuda").T.requires_grad_()
    projection = torch.randn(tp.weight_numel, 8, device="cuda").T.requires_grad_()
    args = x, attrs, radial, projection
    actual, expected = conv(*args, edges), reference(tp, *args, edges)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    a = torch.autograd.grad(actual.square().mean(), args)
    b = torch.autograd.grad(expected.square().mean(), args)
    for value, target in zip(a, b):
        torch.testing.assert_close(value, target, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("edge_count", [35, 1031])
def test_cuda_unequal_channels(edge_count, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(17)
    irreps = o3.Irreps("3x0e + 33x1o + 65x2e")
    harmonics = o3.Irreps("0e + 1o + 2e")
    outputs, instructions = [], []
    for i, (mul, ir1) in enumerate(irreps):
        for j, (_, ir2) in enumerate(harmonics):
            for ir in ir1 * ir2:
                if ir.l <= 2:
                    instructions.append((i, j, len(outputs), "uvu", True))
                    outputs.append((mul, ir))
    tp = o3.TensorProduct(
        irreps,
        harmonics,
        outputs,
        instructions,
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    module = eqx_conv.O3TensorProductConv(tp, backend="cuda").cuda()
    edges = torch.randint(7, (2, edge_count), device="cuda")
    inputs = tuple(
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in (
            (7, irreps.dim),
            (edge_count, harmonics.dim),
            (edge_count, 3),
            (3, tp.weight_numel),
        )
    )
    actual, expected = module(*inputs, edges), reference(tp, *inputs, edges)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        seed = torch.randn_like(actual) / actual.numel() ** 0.5
        actual, expected = (
            torch.cat(
                [
                    value.flatten()
                    for value in torch.autograd.grad(
                        (value.sin() * seed).sum(), inputs, create_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        )
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


def test_cuda_requested_gradients(monkeypatch, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.cuda.runtime import runtime

    tp = tensor_product().cuda()
    module = eqx_conv.O3TensorProductConv(tp, backend="cuda").cuda()
    edges = torch.randint(4, (2, 13), device="cuda")
    inputs = tuple(
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in (
            (4, tp.irreps_in1.dim),
            (13, tp.irreps_in2.dim),
            (13, 3),
            (3, tp.weight_numel),
        )
    )
    masks = []
    query = runtime().gradient_mask

    def record(offset, count):
        mask = query(offset, count)
        masks.append(mask)
        return mask

    monkeypatch.setattr(runtime(), "gradient_mask", record)
    actual, expected = module(*inputs, edges), reference(tp, *inputs, edges)
    actual = torch.autograd.grad(actual.sin().sum(), inputs[0], create_graph=True)[0]
    expected = torch.autograd.grad(expected.sin().sum(), inputs[0], create_graph=True)[
        0
    ]
    assert masks[0] == [True, False, False, False, False]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    # Pruning an unrequested first adjoint must preserve mixed derivatives.
    actual = torch.autograd.grad(actual.square().sum(), inputs)
    expected = torch.autograd.grad(expected.square().sum(), inputs)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, atol=2e-9, rtol=2e-9)


def test_empty_and_unsupported(double_precision):
    tp = tensor_product()
    conv = eqx_conv.O3TensorProductConv(tp, backend="cuda")
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        inputs = [
            torch.randn(shape, device=device, requires_grad=True)
            for shape in (
                (0, tp.irreps_in1.dim),
                (0, tp.irreps_in2.dim),
                (0, 3),
                (3, tp.weight_numel),
            )
        ]
        result = conv.to(device)(
            *inputs, torch.empty(2, 0, dtype=torch.long, device=device)
        )
        assert result.shape == (0, tp.irreps_out.dim)
        for value in torch.autograd.grad(result.sum(), inputs):
            assert not value.count_nonzero()
    with pytest.raises(ValueError, match="backend"):
        eqx_conv.O3TensorProductConv(tp, backend="triton")
    with pytest.raises(NotImplementedError, match="uvu"):
        eqx_conv.O3TensorProductConv(
            o3.FullyConnectedTensorProduct("0e", "0e", "0e"), backend="cuda"
        )


def test_compile_and_cuda_graph(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = tensor_product().cuda()
    conv = eqx_conv.O3TensorProductConv(tp, backend="cuda").cuda()
    compiled = torch.compile(conv, backend="aot_eager", fullgraph=True, dynamic=True)
    for count in (17, 23, 0):
        args = [
            torch.randn(shape, device="cuda", requires_grad=True)
            for shape in (
                (5, tp.irreps_in1.dim),
                (count, tp.irreps_in2.dim),
                (count, 4),
                (4, tp.weight_numel),
            )
        ]
        edges = torch.randint(5, (2, count), device="cuda")
        actual, expected = compiled(*args, edges), conv(*args, edges)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), args),
            torch.autograd.grad(expected.square().sum(), args),
        ):
            torch.testing.assert_close(a, b)

    args = [
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in (
            (5, tp.irreps_in1.dim),
            (17, tp.irreps_in2.dim),
            (17, 4),
            (4, tp.weight_numel),
        )
    ]
    edges = torch.randint(5, (2, 17), device="cuda")

    def evaluate():
        output = conv(*args, edges)
        gradient = torch.autograd.grad(output.square().sum(), args, create_graph=True)
        second = torch.autograd.grad(
            sum(value.square().sum() for value in gradient), args
        )
        return output, *gradient, *second

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = evaluate()
    with torch.no_grad():
        for value in args:
            value.normal_()
        edges.random_(5)
    graph.replay()
    for a, b in zip(actual, evaluate()):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-10)


@pytest.mark.parametrize("channels", [3, 65])
@pytest.mark.parametrize("radial_channels", [5, 1024])
@pytest.mark.parametrize(
    "shared_attrs,shared_radial",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_o3_chunked_shared_gradients(
    monkeypatch,
    channels,
    shared_attrs,
    shared_radial,
    radial_channels,
    double_precision,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.common.cuda.o3.convolution as cuda

    # Two grouped chunks and a one-edge tail exercise private and broadcast
    # gradients without mistaking the tail for a shared operand.
    monkeypatch.setattr(cuda, "CHUNK_SIZE", 1025)
    tp = tensor_product(channels, merge=True, unweighted=True).cuda()
    conv = eqx_conv.O3TensorProductConv(tp, backend="cuda").cuda()
    edges = torch.randint(257, (2, 2051), device="cuda")
    inputs = [
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in (
            (257, tp.irreps_in1.dim),
            (1 if shared_attrs else 2051, tp.irreps_in2.dim),
            (1 if shared_radial else 2051, radial_channels),
            (radial_channels, tp.weight_numel),
        )
    ]
    with torch.no_grad():
        inputs[-1].div_(radial_channels**0.5)
    actual = conv(*inputs, edges)
    expected = reference(tp, *inputs, edges)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-11)
    for _ in range(2):
        seed = torch.randn_like(actual) / actual.numel() ** 0.5
        gradients = [
            torch.autograd.grad((value.sin() * seed).sum(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-10)
        actual, expected = (
            torch.cat([g.flatten() for g in values]) for values in gradients
        )


def test_o3_weight_adjoint_reuses_angular_contractions(double_precision):
    from equivariantx.ops.common.cuda.o3.codegen import angular_source

    tp = tensor_product()
    conv = eqx_conv.O3TensorProductConv(tp, backend="cuda")
    for path in conv.paths:
        values = angular_source(
            path,
            tuple(range(5)),
            (
                tp.irreps_in1.dim,
                tp.weight_numel,
                0,
                tp.irreps_in2.dim,
                tp.irreps_out.dim,
            ),
            (False,) * 5,
            {0, 1, 3},
            {},
            [],
        )
        assert all(role != 4 for _, role, _ in values)
        assert all((v, 1, 0) in values for v in range(path[4]))


def test_o3_mixed_node_adjoint_destination(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.o3 import contraction

    tp = o3.TensorProduct(
        "3x0e",
        "0e",
        "3x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    conv = eqx_conv.O3TensorProductConv(tp, backend="cuda")
    target = torch.arange(257, device="cuda").repeat_interleave(8)
    source = torch.randint(257, target.shape, device="cuda")
    edges = torch.stack((source, target))
    x = torch.randn(257, 3, device="cuda", requires_grad=True)
    attrs = torch.randn(target.numel(), 1, device="cuda")
    weights = torch.randn(target.numel(), 3, device="cuda")
    projection = x.new_empty(0, 3)
    cotangent = torch.randn_like(x)
    expected = reference(tp, x, attrs, weights, projection, edges)
    expected = expected + torch.autograd.grad((expected * cotangent).sum(), x)[0]
    program = repr((((0, 1, 2, 3, 4), False, ((0, 0), (4, 0))),))
    for _ in range(3):
        actual = contraction(
            conv.kernel_metadata,
            program,
            source,
            target,
            [x, weights, projection, attrs, cotangent],
        )[0]
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("background_loading", [False, True])
@pytest.mark.parametrize("dense", [False, True])
def test_row_product_cuda_graph(
    monkeypatch, double_precision, background_loading, dense
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.cuda import graph
    from equivariantx.ops.common.cuda.edge import product_graph_layout
    from equivariantx.ops.common.torch.edge import evaluate_torch
    from equivariantx.ops.common.edge import evaluate
    from equivariantx.utils.program import Program

    worker = None
    pinned = []
    if background_loading:
        from concurrent.futures import ThreadPoolExecutor

        worker = ThreadPoolExecutor(max_workers=1)
        worker.submit(torch.empty, 1, pin_memory=True).result(timeout=15)
        capture_begin = torch.cuda.CUDAGraph.capture_begin

        def begin_with_loading(self, *args, **kwargs):
            capture_begin(self, *args, **kwargs)
            pinned.append(
                worker.submit(
                    torch.empty, 1 << 20, dtype=torch.uint8, pin_memory=True
                ).result(timeout=15)
            )

        monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", begin_with_loading)

    monkeypatch.delenv("EQX_USE_CUDA_GRAPH", raising=False)
    monkeypatch.setattr(graph, "MIN_ROWS", 1)
    monkeypatch.setattr(graph, "ROW_BUCKET", 8)
    program = Program()
    x = program.input(0, "edge", 2)
    w = program.input(1, "shared", 4)
    bias = program.input(2, "shared", 2)
    projected = (
        program.matmul(x, w, 1, 2, 2)
        if dense
        else program.binary("mul", x, program.slice(w, 0, 2))
    )
    hidden = program.unary("silu", program.binary("add", projected, bias))
    value = program.product(
        (hidden, hidden, program.constant(2, 0)),
        (1, (2, 2, 2), (((0, 0, 0), 1.0), ((1, 1, 1), 0.7))),
        2,
    )
    metadata = repr((tuple(program.nodes), ((value, 0, "edge"),)))
    # A sum of biased primals is not safe to pad; adjoints with zero seeds are.
    unsafe = repr((tuple(program.nodes), ((value, 0, "shared"),)))
    assert product_graph_layout(unsafe) is None
    assert product_graph_layout(metadata) is not None
    from equivariantx.ops.common.cuda.execution import uses_matrix_products

    assert uses_matrix_products(metadata)
    assert graph.enabled("product") and not graph.enabled("o3")
    graph._GRAPHS.clear()
    try:
        previous = None
        for rows in (5, 7, 5, 9, 5, 0, 129, 137, 129):
            inputs = [
                (torch.randn(shape, device="cuda") * 0.2).requires_grad_()
                for shape in ((rows, 2), (1, 4), (1, 2))
            ]
            index = torch.arange(rows, device="cuda")
            actual = evaluate(metadata, inputs, index, index, rows)[0]
            expected = evaluate_torch(metadata, inputs, index, index, rows)[0]
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
            if rows:
                for order in range(3):
                    # Exercise nonconsecutive output slots on the first adjoint.
                    active = (inputs[0], inputs[2]) if order == 0 else inputs
                    gradients = [
                        torch.autograd.grad(y.sin().sum(), active, create_graph=True)
                        for y in (actual, expected)
                    ]
                    actual, expected = [
                        torch.cat([g.flatten() for g in grads]) for grads in gradients
                    ]
                    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
                captured = {key: entry[0] for key, entry in graph._GRAPHS.items()}
                assert captured
                if rows in (5, 7, 137) and previous is not None:
                    assert captured == previous
                previous = captured
        # Explicit opt-out keeps the ordinary CUDA implementation available.
        monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "0")
        assert not graph.enabled("product")
    finally:
        graph._GRAPHS.clear()
        if worker is not None:
            worker.shutdown()
