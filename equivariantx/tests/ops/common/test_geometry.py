"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import conv as eqx_conv
from equivariantx import o2


@pytest.mark.parametrize("source_degree", [0, 5])
def test_sparse_direction_rotations(source_degree, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    harmonic_degree = 5 if source_degree == 0 else 1
    tp = o2.O3TensorProduct(
        [(3, (source_degree, (-1) ** source_degree))],
        [(1, (harmonic_degree, (-1) ** harmonic_degree))],
        [(3, (5, (-1) ** (source_degree + harmonic_degree)))],
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    conv = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").cuda()
    reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").cuda()
    frame = o2.WignerD(5, 5).cuda()
    axes = torch.eye(3, device="cuda")
    vectors = torch.cat(
        (axes, -axes, axes + 1e-8, -axes + 1e-8, torch.randn(3, 3, device="cuda"))
    ).requires_grad_()
    edges = torch.randint(4, (2, vectors.size(0)), device="cuda")
    x = torch.randn(4, tp.input_dim, device="cuda", requires_grad=True)
    weights = torch.randn(
        vectors.size(0), tp.weight_numel, device="cuda", requires_grad=True
    )
    amplitude = torch.randn(vectors.size(0), 1, device="cuda", requires_grad=True)
    args = (
        x,
        weights,
        weights.new_empty(0, tp.weight_numel),
        frame.forward_packed(vectors),
        amplitude,
        edges,
        4,
    )
    actual, expected = conv(*args, vectors=vectors), reference(*args)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    for _ in range(2):
        seed = torch.randn_like(actual) / actual.numel() ** 0.5
        derivatives = [
            torch.autograd.grad(
                (value.sin() * seed).sum(),
                (x, vectors, weights, amplitude),
                create_graph=True,
                retain_graph=True,
            )
            for value in (actual, expected)
        ]
        actual, expected = (
            torch.cat([g.flatten() for g in gradients]) for gradients in derivatives
        )
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("normalization", ["norm", "component", "integral"])
def test_direction_coefficients_at_pole(degree, normalization):
    from equivariantx.utils.geometry import angular_coefficients

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = o2.O3TensorProduct(
            "2x3e",
            o3.Irreps([(1, (degree, (-1) ** degree))]),
            o3.Irreps([(2, (3, (-1) ** degree))]),
            [(0, 0, 0, "uvu", True)],
            normalization=normalization,
        )
        conv = eqx_conv.O2O3TensorProductConv(tp, backend="cuda")
        pole = torch.tensor([0.0, 1.0, 0.0])

        def harmonics(vectors):
            return o3.spherical_harmonics(
                degree, vectors, normalize=True, normalization=normalization
            )

        def derivative(vectors):
            jacobian = torch.autograd.functional.jacobian(
                harmonics, vectors, create_graph=True
            )
            return torch.linalg.cross(vectors.expand_as(jacobian), jacobian)

        jacobian = torch.autograd.functional.jacobian(derivative, pole)
        values = (
            harmonics(pole),
            derivative(pole),
            torch.linalg.cross(pole.expand_as(jacobian), jacobian),
        )
        cg = o3.wigner_3j(3, degree, 3) * tp.instructions[0].path_weight
        actual = []
        for rank, value in enumerate(values):
            actual.append(angular_coefficients(conv.direction_metadata, rank)[0])
            expected = torch.einsum("aqb,q...->ab...", cg, value)
            torch.testing.assert_close(actual[-1], expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(
            actual[2][..., 0, 0] + actual[2][..., 2, 2],
            -degree * (degree + 1) * actual[0],
            atol=2e-12,
            rtol=2e-12,
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    "device,backend,mode,channels",
    [
        ("cpu", "torch", "uvu", 2),
        ("cpu", "torch", "uvw", 2),
        ("cuda", "torch", "uvu", 2),
        ("cuda", "cuda", "uvu", 2),
        ("cuda", "cuda", "uvu", 65),
        ("cuda", "cuda", "uvu", 257),
    ],
)
@pytest.mark.parametrize(
    "projected,shared", [(False, False), (True, False), (True, True)]
)
def test_direction_derivatives(
    monkeypatch, device, backend, mode, channels, projected, shared
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.common.cuda.o2_o3.convolution as cuda

    monkeypatch.setattr(cuda, "CHUNK_SIZE", 3)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(76)
        tp = o2.O3TensorProduct(
            f"{channels}x1o+{channels}x2e",
            "0e+1o+2e",
            "+".join(f"{channels}x{ir}" for ir in ("1o", "2e", "1e", "2o")),
            [
                (0, 0, 0, mode, True),
                (1, 0, 1, mode, True),
                (0, 1, 2, mode, True),
                (1, 1, 3, mode, True),
                (1, 2, 1, "uvu", False),
            ],
            internal_weights=False,
            shared_weights=False,
        ).to(device)
        plan = eqx_conv.O2O3TensorProductConv(tp, backend=backend).to(device)
        reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").to(device)
        frame = o2.WignerD(2, 2).to(device)
        edges = torch.tensor([[0, 1, 2, 0], [2, 2, 1, 1]], device=device)

        def rand(*shape):
            return torch.randn(*shape, device=device, requires_grad=True)

        rows = 1 if shared else 4
        x, vectors = rand(3, tp.input_dim), rand(rows, 3)
        radial = rand(rows, 3 if projected else tp.weight_numel)
        projection = (
            rand(3, tp.weight_numel)
            if projected
            else radial.new_empty(0, tp.weight_numel)
        )
        amplitude = rand(rows, 3)
        inputs = (
            (x, vectors, radial, amplitude, projection)
            if projected
            else (x, vectors, radial, amplitude)
        )
        packed = frame.forward_packed(vectors)
        args = x, radial, projection, packed, amplitude, edges, 3
        actual = plan(*args, vectors=vectors).sin()
        expected = reference(*args).sin()
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        for _ in range(3):
            cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
            derivatives = [
                torch.autograd.grad(
                    (value * cotangent).sum(),
                    inputs,
                    create_graph=True,
                    retain_graph=True,
                )
                for value in (actual, expected)
            ]
            actual, expected = [
                torch.cat([g.flatten() for g in grads]) for grads in derivatives
            ]
            torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("projected", [False, True])
def test_direction_zero_order_and_empty_edges(device, projected):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = (
        o2.O3TensorProduct("2x2e", "0e", "2x2e", [(0, 0, 0, "uvu", True)])
        .to(device)
        .double()
    )
    plan = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").to(device)
    frame = o2.WignerD(2, 2).to(device).double()
    for size in (0, 4):
        x = torch.randn(3, 10, device=device, dtype=torch.float64, requires_grad=True)
        vectors = torch.randn(
            size, 3, device=device, dtype=torch.float64, requires_grad=True
        )
        weights = torch.randn(
            size,
            3 if projected else 2,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        projection = torch.randn(
            3 if projected else 0,
            2,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        edges = torch.randint(3, (2, size), device=device)
        output = plan(
            x,
            weights,
            projection,
            frame.forward_packed(vectors),
            weights.new_ones(size, 1),
            edges,
            3,
            vectors=vectors,
        )
        gradient = torch.autograd.grad(
            output.square().sum(), vectors, create_graph=True
        )[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
        # A vanishing angular derivative must write zero even when the
        # projected weight workspace has not been initialized.
        second = torch.autograd.grad(gradient.sum(), (vectors, weights, projection))
        for value in second:
            torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)


@pytest.mark.parametrize("channels", [3, 65])
def test_direction_node_reductions(monkeypatch, channels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.common.cuda.o2_o3.convolution as cuda

    monkeypatch.setattr(cuda, "ROW_SIZE", 8)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(82)
        tp = o2.O3TensorProduct(
            f"{channels}x1o+{channels}x2e",
            "0e+1o+2e",
            f"{channels}x1o+{channels}x2e+{channels}x1e",
            [(0, 0, 0, "uvu", True), (1, 2, 1, "uvu", True), (0, 1, 2, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        module, reference = (
            eqx_conv.O2O3TensorProductConv(tp, backend="cuda"),
            eqx_conv.O2O3TensorProductConv(tp, backend="torch"),
        )
        frame = o2.WignerD(2, 2).cuda()
        edges = torch.randint(37, (2, 1031), device="cuda")
        # Include isolated nodes, complete rows, and split high-degree rows.
        edges[:, :257] = 0
        edges[:, 257:] = edges[:, 257:] % 29 + 1
        edges[:, -6:] = torch.tensor(
            [[31, 31, 32, 33, 33, 34], [33, 34, 31, 31, 32, 33]], device="cuda"
        )
        x = torch.randn(37, tp.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(1031, 3, device="cuda", requires_grad=True)
        radial = torch.randn(1031, 3, device="cuda", requires_grad=True)
        projection = torch.randn(3, tp.weight_numel, device="cuda", requires_grad=True)
        amplitude = torch.randn(1031, 3, device="cuda", requires_grad=True)
        inputs = x, vectors, radial, projection, amplitude
        args = (
            x,
            radial,
            projection,
            frame.forward_packed(vectors),
            amplitude,
            edges,
            37,
        )
        actual = module(*args, vectors=vectors).sin()
        expected = reference(*args).sin()
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        for _ in range(3):
            seed = torch.randn_like(actual) / actual.numel() ** 0.5
            actual, expected = [
                torch.cat(
                    [
                        grad.flatten()
                        for grad in torch.autograd.grad(
                            (value * seed).sum(),
                            inputs,
                            create_graph=True,
                            retain_graph=True,
                        )
                    ]
                )
                for value in (actual, expected)
            ]
            torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-9)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("edge_count", [35, 1031])
def test_direction_compile_and_capture(edge_count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.wigner import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = o2.O3TensorProduct(
            "2x1o", "1o", "2x0e+2x1e", [(0, 0, i, "uvu", True) for i in range(2)]
        ).cuda()
        plan = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").cuda()
        reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").cuda()
        frame = o2.WignerD(1, 1).cuda()
        x = torch.randn(3, 6, device="cuda", requires_grad=True)
        vectors = torch.randn(edge_count, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edge_count, 3, device="cuda", requires_grad=True)
        projection = torch.randn(3, tp.weight_numel, device="cuda", requires_grad=True)
        edges = torch.randint(3, (2, edge_count), device="cuda")

        def evaluate(x, vectors, radial, projection, edges):
            packed = wigner_D(frame, vectors.detach())
            return plan(
                x,
                radial,
                projection,
                packed,
                radial.new_ones(radial.size(0), 1),
                edges,
                3,
                vectors=vectors,
            )

        for _ in range(2):
            torch.autograd.grad(
                evaluate(x, vectors, radial, projection, edges).square().sum(),
                (x, vectors, radial, projection),
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = evaluate(x, vectors, radial, projection, edges)
            actual_grads = torch.autograd.grad(
                actual.square().sum(), (x, vectors, radial, projection)
            )
        edges[0].add_(1).remainder_(3)
        edges[1].add_(2).remainder_(3)
        graph.replay()
        expected = evaluate(x, vectors, radial, projection, edges)
        torch.testing.assert_close(actual, expected)
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (x, vectors, radial, projection)
        )
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        compiled = torch.compile(evaluate, fullgraph=True, dynamic=True)
        for size in (edge_count, edge_count // 2):
            actual = compiled(
                x, vectors[:size], radial[:size], projection, edges[:, :size]
            )
            expected = reference(
                x,
                radial[:size],
                projection,
                frame.forward_packed(vectors[:size]),
                radial.new_ones(size, 1),
                edges[:, :size],
                3,
            )
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
            actual_grads = torch.autograd.grad(
                actual.square().sum(), (x, vectors, radial, projection)
            )
            expected_grads = torch.autograd.grad(
                expected.square().sum(), (x, vectors, radial, projection)
            )
            for a, b in zip(actual_grads, expected_grads):
                torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_direction_derivatives_on_axes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from e3nn import o3

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(42)
        instructions = [(0, 0, i, "uvu", True) for i in range(3)]
        tp = o2.O3TensorProduct(
            "1o",
            "1o",
            "0e+1e+2e",
            instructions,
            internal_weights=False,
            shared_weights=False,
        ).to(device)
        reference = o3.TensorProduct(
            "1o",
            "1o",
            "0e+1e+2e",
            instructions,
            internal_weights=False,
            shared_weights=False,
        ).to(device)
        plan, frame = (
            eqx_conv.O2O3TensorProductConv(tp, backend="cuda").to(device),
            o2.WignerD(2, 2).to(device),
        )
        vectors = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1e-8, 1.0, -1e-8],
                [-1e-8, -1.0, 1e-8],
            ],
            device=device,
            requires_grad=True,
        )
        edges = torch.stack(
            (torch.arange(8, device=device), torch.arange(8, device=device))
        )
        x = torch.randn(8, 3, device=device, requires_grad=True)
        weights = torch.randn(8, tp.weight_numel, device=device, requires_grad=True)
        # The unused empty projection remains a valid differentiable operand.
        projection = torch.empty(0, tp.weight_numel, device=device, requires_grad=True)
        actual = plan(
            x,
            weights,
            projection,
            frame.forward_packed(vectors),
            weights.new_ones(8, 1),
            edges,
            8,
            vectors=vectors,
        )
        expected = reference(
            x, o3.spherical_harmonics([1], vectors, True, "component"), weights
        )
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        for _ in range(3):
            cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
            actual = torch.autograd.grad(
                (actual.sin() * cotangent).sum(),
                vectors,
                create_graph=True,
                retain_graph=True,
            )[0]
            expected = torch.autograd.grad(
                (expected.sin() * cotangent).sum(),
                vectors,
                create_graph=True,
                retain_graph=True,
            )[0]
            torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        (gradient,) = torch.autograd.grad(actual.sum(), projection)
        assert gradient.shape == projection.shape
    finally:
        torch.set_default_dtype(previous)
