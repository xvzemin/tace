"""O(3) and aligned-frame convolutions, including training and higher derivatives."""

from copy import deepcopy

import pytest
import torch
from e3nn import o3

from eqx.conv import O2O3TensorProductConv, O3TensorProductConv
from eqx.o2 import O3TensorProduct, WignerD


def test_convolution_backends_and_modes():
    from eqx.kernels import wigner_D

    # Exported graphs can resolve the operator before any frame is evaluated.
    assert torch.ops.eqx.quaternion_polynomial.default is not None
    tp = O3TensorProduct("2x0e", "0e", "3x0e", [(0, 0, 0, "uvw", True)])
    assert tp.convolution.instructions[0].connection_mode == "uvw"
    with pytest.raises(NotImplementedError, match="only 'uvu'"):
        O2O3TensorProductConv(tp, backend="cuda")
    with pytest.raises(ValueError, match="backend must be torch or cuda"):
        O2O3TensorProductConv(tp, backend="triton")
    with pytest.raises(ValueError, match="backend must be torch or cuda"):
        wigner_D(WignerD(0, 0), torch.randn(2, 3), backend="triton")
    with pytest.raises(ValueError, match="method must be"):
        WignerD(0, 0, method="unknown")
    with pytest.raises(ValueError, match="CUDA float32 or float64"):
        WignerD(1, 1, method="quaternion")(torch.randn(2, 3))


def test_quaternion_coefficients():
    from e3nn import o3

    from eqx.kernels.quaternion import polynomial_coefficients
    from eqx.o2.rotation_matrix import _quaternion_to_matrix

    q = torch.randn(
        6, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(13)
    )
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.cat((q, -q))
    pointers, exponents, coefficients = polynomial_coefficients(6)
    terms = (q[:, None] ** exponents).prod(-1) * coefficients
    indices = torch.repeat_interleave(
        torch.arange(pointers.numel() - 1), pointers.diff()
    )
    actual = q.new_zeros(q.size(0), pointers.numel() - 1).index_add(1, indices, terms)
    rotation = _quaternion_to_matrix(q)
    blocks = [q.new_ones(q.size(0), 1, 1), rotation]
    for l in range(2, 7):
        cg = o3.wigner_3j(1, l - 1, l, dtype=torch.float64)
        blocks.append(
            torch.einsum("abm,eac,ebd,cdn->emn", cg, rotation, blocks[-1], cg)
            * (2 * l + 1)
        )
    expected = torch.cat([block.flatten(1) for block in blocks], dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("lmax", [0, 1, 3, 5])
def test_quaternion_wigner_frames(lmax):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        axes = torch.eye(3, device="cuda")
        vectors = torch.cat((axes, -axes, axes + 1e-8, -axes + 1e-8))
        # Non-contiguous input, including directions on the chart boundaries.
        vectors = torch.stack((vectors, vectors), dim=-1)[..., 0].requires_grad_()
        frame = WignerD(min(1, lmax), lmax).cuda()
        reference = WignerD(min(1, lmax), lmax, method="recursive").cuda()
        for actual, expected in zip(frame(vectors), reference(vectors)):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(
            frame.forward_packed(vectors), wigner_D(frame, vectors), atol=0, rtol=0
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
    frame = WignerD(3, 3).cuda()
    function = torch.compile(
        frame.forward_packed, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for count in (3, 7, 0):
        vectors = torch.randn(count, 3, device="cuda", requires_grad=True)
        actual, expected = function(vectors), frame.forward_packed(vectors)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.square().sum(), vectors)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), vectors)[0]
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


@pytest.mark.parametrize("channels,edge_count", [(3, 5), (64, 5), (129, 5), (3, 1031)])
@pytest.mark.parametrize("merge_paths", [False, True])
def test_same_degree_output_rotations(channels, edge_count, merge_paths):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(37)
        tp = O3TensorProduct(
            f"{channels}x0e+{channels}x1o+{channels}x2e",
            "0e+1o+2e",
            (
                f"{channels}x1o+{channels}x2e"
                if merge_paths
                else "+".join([f"{channels}x1o"] * 4 + [f"{channels}x2e"] * 2)
            ),
            [
                (a, b, int(i >= 4) if merge_paths else i, "uvu", True)
                for i, (a, b) in enumerate(
                    [(0, 1), (1, 0), (1, 2), (2, 1), (1, 1), (2, 0)]
                )
            ],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        plan = O2O3TensorProductConv(tp).cuda()
        reference = O2O3TensorProductConv(tp, backend="torch").cuda()
        frame = WignerD(2, 2).cuda()
        edges = torch.randint(4, (2, edge_count), device="cuda")
        x = torch.randn(4, tp.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(edge_count, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edge_count, 4, device="cuda", requires_grad=True)
        projection = torch.randn(4, tp.weight_numel, device="cuda", requires_grad=True)
        amplitude = torch.randn(edge_count, 3, device="cuda", requires_grad=True)
        inputs = x, vectors, radial, projection, amplitude
        arguments = (
            x,
            radial,
            projection,
            frame.forward_packed(vectors),
            amplitude,
            edges,
            4,
        )
        actual, expected = plan(*arguments), reference(*arguments)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        # Force training requires double backward. Exercise one further
        # derivative with the default CUDA backend.
        for _ in range(3):
            seed = torch.randn_like(actual) / actual.numel() ** 0.5
            derivatives = []
            for value in (actual, expected):
                grads = torch.autograd.grad(
                    (value * seed).sum(), inputs, create_graph=True, retain_graph=True
                )
                derivatives.append(torch.cat([grad.flatten() for grad in grads]))
            actual, expected = derivatives
            torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-9)
    finally:
        torch.set_default_dtype(previous)


def test_shared_path_rotation_schedule():
    from eqx.conv.o2_o3.schedule import rotation_groups

    # Three instructions sum into one output, while the fourth writes an
    # independent entry of the same degree. Do not split or merge those roles.
    paths = [
        ((0, output, 2, 2, 3, 17, 1, 680, 2 * i, i), ())
        for i, output in enumerate((0, 34, 0, 0))
    ]
    _, groups = rotation_groups(paths)
    assert groups == ((0, 2, 3), (1,))


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("contributions,inactive", [(1, False), (2, False), (1, True)])
def test_projected_weight_only_workspace(monkeypatch, shared, contributions, inactive):
    from types import SimpleNamespace

    from eqx.conv.o2_o3.schedule import project

    torch.manual_seed(46)
    rows = 1 if shared else 7
    radial = torch.randn(rows, 3, dtype=torch.float64)
    projection = torch.randn(3, 11, dtype=torch.float64)
    expected = torch.randn(7, 11, dtype=torch.float64)
    if inactive:
        expected[:, -1] = 0
    plan = SimpleNamespace(
        weight_numel=11,
        path_data=(("uvu", (0, 0, 10 if inactive else 11, 11, 1, 1, 0, 0, 0, 0)),),
    )
    gr, gp = torch.zeros_like(radial), torch.zeros_like(projection)
    edges = torch.arange(7)
    dummy = torch.ones(rows, 1, dtype=torch.float64)
    values = (dummy, radial, projection, dummy, dummy, dummy, dummy)
    products = []
    clears = []
    mm = torch.mm
    zero = torch.Tensor.zero_

    def record(a, b, **kwargs):
        products.append(a.shape)
        return mm(a, b, **kwargs)

    def record_zero(value):
        clears.append(value.shape)
        return zero(value)

    monkeypatch.setattr(torch, "mm", record)
    monkeypatch.setattr(torch.Tensor, "zero_", record_zero)

    def contract(plan, source, target, calls, layout, initialize):
        assert bool(initialize) == (not shared and contributions == 1 and not inactive)
        for outputs, operands, results, weighted in calls:
            assert outputs == (1,)
            assert operands[1].untyped_storage().nbytes() == radial.element_size()
            result = expected[source] / contributions
            if initialize:
                assert initialize[0] is results[0]
                results[0].copy_(result)
            else:
                results[0].add_(result.sum(0, keepdim=True) if shared else result)

    project(
        plan,
        edges,
        edges,
        [((1, 2), values, (gr, gp), False)] * contributions,
        contract,
        chunk_size=3,
    )
    assert not products
    assert len(clears) == (1 if shared else 3 if contributions > 1 or inactive else 0)
    reduced = expected.sum(0, keepdim=True) if shared else expected
    torch.testing.assert_close(gr, reduced @ projection.T)
    torch.testing.assert_close(gp, radial.T @ reduced)

    # Shared projections are evaluated once per call, not once per chunk and
    # not cached across parameter updates.
    def forward(plan, source, target, calls, layout, initialize):
        assert not initialize
        weights = calls[0][1][1]
        inputs = radial if shared else radial[source]
        torch.testing.assert_close(weights, mm(inputs, projection))

    for _ in range(2):
        products.clear()
        project(
            plan,
            edges,
            edges,
            [((6,), values, (dummy,), False)],
            forward,
            chunk_size=3,
        )
        assert len(products) == (1 if shared else 3)
        projection.add_(0.25)


def test_derivative_partition_reuses_rotations():
    from eqx.conv.o2_o3.schedule import split_program

    x, w, projection, din, dout, s = (object() for _ in range(6))
    cotangents = (object(), object())
    calls = [
        ((0,), (x, w, projection, din, dout, s, y), (object(),), False)
        for y in cotangents * 2
    ]
    groups = split_program(calls)
    assert sorted(id(call[2][0]) for group in groups for call in group) == sorted(
        id(call[2][0]) for call in calls
    )
    assert [len(group) for group in groups] == [2, 2]
    for group in groups:
        assert group[0][1][6] is group[1][1][6]


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
    from eqx.conv.o2_o3 import cuda

    monkeypatch.setattr(cuda, "CHUNK_SIZE", 3)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(76)
        tp = O3TensorProduct(
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
        plan = O2O3TensorProductConv(tp, backend=backend).to(device)
        reference = O2O3TensorProductConv(tp, backend="torch").to(device)
        frame = WignerD(2, 2).to(device)
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
        O3TensorProduct("2x2e", "0e", "2x2e", [(0, 0, 0, "uvu", True)])
        .to(device)
        .double()
    )
    plan = O2O3TensorProductConv(tp).to(device)
    frame = WignerD(2, 2).to(device).double()
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
    from eqx.conv.o2_o3 import cuda

    monkeypatch.setattr(cuda, "ROW_SIZE", 8)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(82)
        tp = O3TensorProduct(
            f"{channels}x1o+{channels}x2e",
            "0e+1o+2e",
            f"{channels}x1o+{channels}x2e+{channels}x1e",
            [(0, 0, 0, "uvu", True), (1, 2, 1, "uvu", True), (0, 1, 2, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        module, reference = (
            O2O3TensorProductConv(tp),
            O2O3TensorProductConv(tp, backend="torch"),
        )
        frame = WignerD(2, 2).cuda()
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
    from eqx.kernels import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = O3TensorProduct(
            "2x1o", "1o", "2x0e+2x1e", [(0, 0, i, "uvu", True) for i in range(2)]
        ).cuda()
        plan = O2O3TensorProductConv(tp).cuda()
        reference = O2O3TensorProductConv(tp, backend="torch").cuda()
        frame = WignerD(1, 1).cuda()
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
        tp = O3TensorProduct(
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
        plan, frame = O2O3TensorProductConv(tp).to(device), WignerD(2, 2).to(device)
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


@pytest.mark.parametrize("owner", [0, 1])
def test_convolution_graph_order(owner):
    from eqx.conv.graph import prepare_graph

    edges = torch.tensor([[0, 0, 1, 0, 3, 0, 2, 0, 0], [2, 1, 2, 0, 2, 2, 1, 2, 2]])
    order = prepare_graph(edges[0], edges[1], owner)
    assert prepare_graph(edges[0], edges[1], owner) is order
    assert prepare_graph(edges[0].detach(), edges[1].detach(), owner) is order
    torch.testing.assert_close(order, edges[owner].argsort(stable=True))
    edges[owner, 0] = 4
    updated = prepare_graph(edges[0].detach(), edges[1].detach(), owner)
    assert updated is not order
    torch.testing.assert_close(updated, edges[owner].argsort(stable=True))
    assert prepare_graph(edges[0, :0], edges[1, :0], owner).shape == (0,)


@pytest.mark.parametrize("degree", [0, 1, 3, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["recursive", "quaternion"])
def test_fused_wigner_derivatives(degree, dtype, method):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(27)
        frame = WignerD(degree, degree).cuda()
        vectors = torch.randn(5, 3, device="cuda", requires_grad=True)
        actual = wigner_D(frame, vectors, method=method)
        expected = frame.forward_packed(vectors, method="recursive")
        tolerance = 2e-5 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(
            wigner_D(frame, vectors.detach(), method=method),
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
            wigner_D(frame, empty, method=method),
            frame.forward_packed(empty, method="recursive"),
        )
        torch.testing.assert_close(
            wigner_D(frame, empty.detach(), method=method),
            frame.forward_packed(empty, method="recursive").detach(),
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("row_size", [3, 128])
def test_cuda_streams_capture_compile(monkeypatch, row_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o2_o3 import cuda

    monkeypatch.setattr(cuda, "ROW_SIZE", row_size)
    tp = O3TensorProduct("2x0e", "0e", "2x0e", [(0, 0, 0, "uvu", True)]).cuda().double()
    plan = O2O3TensorProductConv(tp).cuda().double()
    reference = O2O3TensorProductConv(tp, backend="torch").cuda().double()
    x = torch.randn(4, 2, device="cuda", dtype=torch.float64)
    weights = torch.randn(1031, 2, device="cuda", dtype=torch.float64)
    projection = weights.new_empty(0, 2)
    d = weights.new_ones(1031, 1)
    edges = torch.randint(4, (2, 1031), device="cuda")
    args = (x, weights, projection, d, d, edges, 4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(plan(*args), reference(*args), atol=2e-10, rtol=2e-10)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = plan(*args)
    for shift in (0, 1, 2):
        edges[1].add_(shift).remainder_(4)
        graph.replay()
        torch.testing.assert_close(actual, reference(*args), atol=2e-10, rtol=2e-10)
    x.requires_grad_()
    compiled = torch.compile(plan, fullgraph=True, dynamic=True)
    for size in (1031, 777):
        args = (x, weights[:size], projection, d[:size], d[:size], edges[:, :size], 4)
        actual, expected = compiled(*args), reference(*args)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        (a,) = torch.autograd.grad(actual.square().sum(), x)
        (b,) = torch.autograd.grad(expected.square().sum(), x)
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-10)


@pytest.mark.parametrize(
    "device,backend,mode",
    [
        ("cpu", "torch", "uvu"),
        ("cpu", "torch", "uvw"),
        ("cuda", "torch", "uvu"),
        ("cuda", "torch", "uvw"),
        ("cuda", "cuda", "uvu"),
    ],
)
@pytest.mark.parametrize(
    "radial_channels,dtype",
    [
        (0, torch.float64),
        (3, torch.float64),
        (128, torch.float64),
        (129, torch.float32),
        (33, torch.float32),
    ],
)
def test_streaming_derivatives(
    monkeypatch, device, backend, mode, radial_channels, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if backend == "cuda":
        from eqx.conv.o2_o3 import cuda

        monkeypatch.setattr(cuda, "CHUNK_SIZE", 3)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        channels = 17 if radial_channels == 33 else 2
        tp = O3TensorProduct(
            f"{channels}x1o",
            "0e+1o+2e",
            "+".join(f"{channels}x{ir}" for ir in ("0e", "1e", "2e")),
            [(0, 1, i, mode, True) for i in range(3)],
            shared_weights=False,
            internal_weights=False,
        ).to(device)
        tp.convolution = O2O3TensorProductConv(tp, backend=backend).to(device)
        frame = WignerD(2, 2).to(device)
        generator = torch.Generator(device=device).manual_seed(42)

        def rand(*shape):
            return torch.randn(
                *shape, generator=generator, device=device, requires_grad=True
            )

        edges = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 1]], device=device)
        shared = radial_channels == 129
        size = 1 if shared else 4
        x, r = rand(3, tp.input_dim), rand(size, 3)
        amplitude = rand(size, 3)
        if radial_channels:
            radial = rand(size, radial_channels)
            projection = rand(radial_channels, tp.weight_numel) / radial_channels**0.5
            weights = (radial @ projection).expand(4, -1)
            arguments = dict(radial_features=radial, radial_weight=projection)
            inputs = x, r, radial, projection, amplitude
        else:
            weights = rand(4, tp.weight_numel)
            arguments = dict(weight=weights)
            inputs = x, r, weights, amplitude
        d, di = frame(r.expand(4, -1))
        expected = x.new_zeros(3, tp.irreps_out.dim).index_add(
            0, edges[1], tp(x[edges[0]], d, di, weights, amplitude.expand(4, -1))
        )
        actual = tp.forward_scatter(
            x, edges, frame.forward_packed(r), harmonic_scale=amplitude, **arguments
        )
        tolerance = 3e-5 if dtype == torch.float32 else 1e-11
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        if radial_channels == 33:
            # A nonlinear loss also differentiates the output cotangent.
            actual, expected = actual.sin(), expected.sin()
        for _ in range(3):
            cotangent = torch.randn(actual.shape, generator=generator, device=device)
            if radial_channels == 33:
                cotangent = cotangent / actual.numel() ** 0.5
            actual_grads = torch.autograd.grad(
                (actual * cotangent).sum(), inputs, create_graph=True, retain_graph=True
            )
            expected_grads = torch.autograd.grad(
                (expected * cotangent).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
            )
            for grad, reference in zip(actual_grads, expected_grads):
                torch.testing.assert_close(
                    grad,
                    reference,
                    atol=5e-4 if dtype == torch.float32 else 3e-9,
                    rtol=5e-5 if dtype == torch.float32 else 3e-9,
                )
            actual = torch.cat([grad.flatten() for grad in actual_grads])
            expected = torch.cat([grad.flatten() for grad in expected_grads])
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize(
    "device,backend",
    [("cpu", "torch"), ("cuda", "torch"), ("cuda", "cuda")],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_streaming_mixed_paths_and_empty_edges(device, backend, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        module = O3TensorProduct(
            "3x0e+35x1o+2x1e",
            "0e+1o+2e",
            f"3x0e+35x0e+{2 if backend == 'cuda' else 19}x1e+35x2e+35x2e",
            [
                (0, 0, 0, "uvu", False),
                (1, 1, 1, "uvu", True),
                (2, 0, 2, "uvu" if backend == "cuda" else "uvw", True),
                (1, 1, 3, "uvu", True),
                (1, 1, 4, "uvu", True),
            ],
            internal_weights=True,
            shared_weights=True,
        ).to(device)
        module.convolution = O2O3TensorProductConv(module, backend=backend).to(device)
        frame = WignerD(3, 3).to(device)
        features = torch.randn(3, module.input_dim, device=device, requires_grad=True)
        vectors = torch.randn(1, 3, device=device, requires_grad=True)
        edges = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]], device=device)
        d, di = frame(vectors.expand(4, 3))
        reference = features.new_zeros(3, module.irreps_out.dim).index_add(
            0, edges[1], module(features[edges[0]], d, di)
        )
        actual = module.forward_scatter(features, edges, frame.forward_packed(vectors))
        tolerance = 3e-5 if dtype == torch.float32 else 1e-10
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
        inputs = features, vectors, module.weight
        a = torch.autograd.grad(actual.square().sum(), inputs, create_graph=True)
        b = torch.autograd.grad(reference.square().sum(), inputs, create_graph=True)
        for value, expected in zip(a, b):
            torch.testing.assert_close(
                value, expected, atol=tolerance * 100, rtol=tolerance * 10
            )

        empty = module.forward_scatter(
            features, edges[:, :0], frame.forward_packed(vectors[:0])
        )
        assert empty.shape == (3, module.irreps_out.dim)
        assert torch.count_nonzero(empty) == 0
        grads = torch.autograd.grad(empty.sum(), (features, module.weight))
        assert all(torch.count_nonzero(grad) == 0 for grad in grads)
        no_nodes = module.forward_scatter(
            features[:0], edges[:, :0], frame.forward_packed(vectors[:0])
        )
        assert no_nodes.shape == (0, module.irreps_out.dim)
        grads = torch.autograd.grad(no_nodes.sum(), (features, module.weight))
        assert all(torch.count_nonzero(grad) == 0 for grad in grads)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize(
    "device,dtype,channels,mode",
    [
        ("cpu", torch.float64, 2, "uvu"),
        ("cpu", torch.float64, 2, "uvw"),
        ("cuda", torch.float64, 2, "uvu"),
        ("cuda", torch.float32, 17, "uvu"),
    ],
)
@pytest.mark.parametrize(
    "projected,shared", [(False, False), (False, True), (True, False), (True, True)]
)
def test_mixed_path_higher_derivatives(
    device, dtype, channels, mode, projected, shared
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with torch.random.fork_rng():
        torch.manual_seed(4)
        tp = O3TensorProduct(
            f"{channels}x0e",
            "0e",
            f"{channels}x0e+{channels}x0e",
            [(0, 0, 0, "uvu", False), (0, 0, 1, mode, True)],
            internal_weights=False,
            shared_weights=False,
        ).to(device=device, dtype=dtype)
        plan = O2O3TensorProductConv(
            tp, backend="cuda" if device == "cuda" else "torch"
        ).to(device=device, dtype=dtype)
        edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=device)

        def rand(*shape):
            return torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)

        x = rand(3, channels)
        r = rand(1 if shared else 4, 3 if projected else tp.weight_numel)
        w = rand(3 if projected else 0, tp.weight_numel)
        d, s = rand(4, 1), rand(4, 1)
        weight = r @ w if projected else r
        expected = x.new_zeros(3, 2 * channels).index_add(
            0,
            edges[1],
            tp(x[edges[0]], d[:, :, None], d[:, :, None], weight.expand(4, -1), s),
        )
        actual = plan(x, r, w, d, s, edges, 3)
        inputs = (x, r, w, d, s) if projected else (x, r, d, s)
        torch.testing.assert_close(actual, expected)
        if not projected:
            # An absent projection has no dependence on the remaining factors.
            empty = torch.autograd.grad(
                actual.sum(), w, create_graph=True, retain_graph=True
            )[0]
            assert empty.numel() == 0
            gradients = torch.autograd.grad(
                empty.sum(), inputs, allow_unused=True, retain_graph=True
            )
            assert all(g is None or torch.count_nonzero(g) == 0 for g in gradients)
        # Starting with a weight derivative catches reintroduced unweighted paths.
        actual = torch.autograd.grad(actual.sum(), r, create_graph=True)[0]
        expected = torch.autograd.grad(expected.sum(), r, create_graph=True)[0]
        for _ in range(2):
            a = torch.autograd.grad(
                actual.sum(), inputs, create_graph=True, allow_unused=True
            )
            b = torch.autograd.grad(
                expected.sum(), inputs, create_graph=True, allow_unused=True
            )
            for value, reference, input in zip(a, b, inputs):
                value = torch.zeros_like(input) if value is None else value
                reference = torch.zeros_like(input) if reference is None else reference
                tolerance = 1e-10 if dtype == torch.float64 else 2e-5
                torch.testing.assert_close(
                    value, reference, atol=tolerance, rtol=tolerance
                )
            actual = sum(value.square().sum() for value in a if value is not None)
            expected = sum(value.square().sum() for value in b if value is not None)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "interaction,bias",
    [("cgtp", False), ("cgtp", True), ("o2_cgtp", True), (["cgtp", "o2"], False)],
)
def test_streaming_force_training(monkeypatch, device, interaction, bias):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.lightning import convert_cgtp, load_tace
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    for name in ("TACE_USE_EQX", "TACE_USE_EQT", "TACE_USE_OEQ", "TACE_USE_CUE"):
        monkeypatch.setenv(name, "0")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        config = deepcopy(DEFAULT_MODEL_CONFIG)
        config.update(
            cutoff=4.0,
            max_neighbors=None,
            num_layers=2,
            num_channel=3,
            Lmax=2,
            lmax=2,
            mmax=2,
            parity=True,
            statistics=[
                dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
            ],
            target_property=["energy", "forces", "stress", "virials"],
        )
        config["node_embedding"]["type"] = "linear"
        config["atomic_basis"]["type"] = interaction
        config["readout_emlp"]["use_one_body_magmoms"] = False
        config["radial_basis"]["hidden"] = [128 if bias else 4]
        config["radial_basis"]["bias"] = bias
        if isinstance(interaction, list):
            config["radial_basis"]["apply_cutoff"] = False
        config["readout_emlp"]["hidden"] = [2]
        config["scale_shift"]["enable"] = False
        reference = TensorModel(e3nnTACE(**deepcopy(config))).train().to(device)
        monkeypatch.setenv("TACE_USE_EQX", "1")
        model = convert_cgtp(reference, "o2")
        representation = model.readout_fn.representation
        mixed = isinstance(interaction, list)
        assert representation.use_packed_wigner == (not mixed)
        assert representation.use_o3_angular_basis == mixed

        def no_edge_message(*args, **kwargs):
            raise AssertionError("The streamed convolution must not call an edge TP")

        for layer in representation.interactions:
            if getattr(layer, "use_eqx", False):
                monkeypatch.setattr(layer.rejector.tp, "forward", no_edge_message)
                monkeypatch.setattr(layer.edge_info, "forward", no_edge_message)

        data = dict(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]], device=device
            ),
            node_attrs=torch.ones(3, 1, device=device),
            edge_index=torch.tensor(
                [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device=device
            ),
            edge_shifts=torch.zeros(6, 3, device=device),
            lattice=torch.eye(3, device=device).unsqueeze(0) * 8,
            batch=torch.zeros(3, dtype=torch.long, device=device),
            ptr=torch.tensor([0, 3], device=device),
            fidelity_idx=torch.zeros(1, dtype=torch.long, device=device),
        )
        outputs = []
        for network, enabled in ((reference, "0"), (model, "1")):
            monkeypatch.setenv("TACE_USE_EQX", enabled)
            output = network({key: value.clone() for key, value in data.items()})
            outputs.append(output)
            sum(
                output[key].square().sum() for key in ("energy", "forces", "stress")
            ).backward()
        for key in ("energy", "forces", "stress", "virials"):
            torch.testing.assert_close(
                outputs[0][key], outputs[1][key], atol=2e-9, rtol=2e-8
            )
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            expected = reference_parameters[name].grad
            if expected is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(
                    parameter.grad, expected, atol=2e-8, rtol=2e-7
                )
        if device == "cpu" and interaction == "cgtp" and not bias:
            next(reference.parameters()).requires_grad_(False)
            reference.retain_graph = True
            loaded = load_tace(reference, device="cpu")
            assert loaded is reference
            assert not loaded.readout_fn.representation.use_o2
            converted = convert_cgtp(reference)
            assert converted.readout_fn.representation.use_packed_wigner
            for enabled in ("0", "1", "0"):
                monkeypatch.setenv("TACE_USE_EQX", enabled)
                assert converted.readout_fn.representation.use_packed_wigner == (
                    enabled == "1"
                )
                output = converted({key: value.clone() for key, value in data.items()})
                for key in ("energy", "forces", "stress", "virials"):
                    torch.testing.assert_close(
                        output[key], outputs[0][key], atol=2e-9, rtol=2e-8
                    )
            monkeypatch.setenv("TACE_USE_EQX", "0")
            restored = convert_cgtp(converted)
            assert not next(restored.parameters()).requires_grad
            assert restored.retain_graph
            assert not restored.readout_fn.representation.use_packed_wigner
            assert not any(
                getattr(layer, "use_eqx", False)
                for layer in restored.readout_fn.representation.interactions
            )
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[name], value, atol=0, rtol=0
                )
    finally:
        torch.set_default_dtype(previous_dtype)


def test_streaming_finite_differences():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = O3TensorProduct(
            "1o",
            "1o",
            "0e+1e",
            [(0, 0, i, "uvu", True) for i in range(2)],
            internal_weights=False,
            shared_weights=False,
        )
        edges = torch.tensor([[0, 1], [1, 0]])
        frame = WignerD(1, 1)
        inputs = (
            torch.randn(2, 3, requires_grad=True),
            torch.randn(2, 3, requires_grad=True),
            torch.randn(2, 2, requires_grad=True),
            torch.randn(2, tp.weight_numel, requires_grad=True),
        )

        def function(x, r, radial, projection):
            return tp.forward_scatter(
                x,
                edges,
                frame.forward_packed(r),
                radial_features=radial,
                radial_weight=projection,
            )

        assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
        assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("degree", [4, 8])
def test_streaming_high_degree(degree):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = O3TensorProduct(
            f"3x{degree}e",
            "1o",
            f"3x{degree}o",
            [(0, 0, 0, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        tp.convolution = O2O3TensorProductConv(tp, backend="cuda").cuda()
        frame = WignerD(degree, degree).cuda()
        torch.manual_seed(71)
        x = torch.randn(3, tp.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(5, 3, device="cuda", requires_grad=True)
        weight = torch.randn(5, tp.weight_numel, device="cuda", requires_grad=True)
        edges = torch.tensor([[0, 1, 2, 0, 1], [2, 0, 1, 1, 2]], device="cuda")
        d, di = frame(vectors)
        expected = x.new_zeros(3, tp.irreps_out.dim).index_add(
            0, edges[1], tp(x[edges[0]], d, di, weight)
        )
        actual = tp.forward_scatter(x, edges, frame.forward_packed(vectors), weight)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        a = torch.autograd.grad(actual.square().sum(), (x, vectors, weight))
        b = torch.autograd.grad(expected.square().sum(), (x, vectors, weight))
        for value, reference in zip(a, b):
            torch.testing.assert_close(value, reference, atol=2e-9, rtol=2e-9)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("degree,edges_count", [(1, 35), (1, 1025), (5, 1025), (8, 35)])
def test_streaming_projected_tiles(monkeypatch, degree, edges_count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o2_o3 import cuda as convolution_kernel

    monkeypatch.setattr(convolution_kernel, "CHUNK_SIZE", 1024)
    previous_dtype = torch.get_default_dtype()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(23)
        tp = O3TensorProduct(
            f"17x{degree}e",
            "1o",
            "+".join(f"17x{l}o" for l in (degree - 1, degree, degree + 1)),
            [(0, 0, i, "uvu", True) for i in range(3)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        tp.convolution = O2O3TensorProductConv(tp, backend="cuda").cuda()
        frame = WignerD(degree + 1, degree + 1).cuda()
        x = torch.randn(7, tp.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(edges_count, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edges_count, 129, device="cuda", requires_grad=True)
        projection = (
            torch.randn(129, tp.weight_numel, device="cuda") / 129**0.5
        ).requires_grad_()
        # Noncontiguous indices and incomplete tiles exercise both masks.
        edges = torch.randint(7, (2, 2 * edges_count), device="cuda")[:, ::2]
        d, di = frame(vectors)
        packed = frame.forward_packed(vectors)
        expected = x.new_zeros(7, tp.irreps_out.dim).index_add(
            0, edges[1], tp(x[edges[0]], d, di, radial @ projection)
        )
        expected_output = expected
        inputs = x, vectors, radial, projection
        output_cotangent = torch.randn_like(expected) / edges_count**0.5
        cotangent = output_cotangent
        references = []
        for _ in range(3):
            grads = torch.autograd.grad(
                (expected * cotangent).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
            )
            references.append(grads)
            expected = torch.cat([grad.flatten() for grad in grads])
            cotangent = torch.ones_like(expected) / expected.numel() ** 0.5

        actual = tp.forward_scatter(
            x, edges, packed, radial_features=radial, radial_weight=projection
        )
        torch.testing.assert_close(actual, expected_output, atol=2e-4, rtol=2e-4)
        cotangent = output_cotangent
        for reference in references:
            grads = torch.autograd.grad(
                (actual * cotangent).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
            )
            for value, expected in zip(grads, reference):
                torch.testing.assert_close(value, expected, atol=2e-4, rtol=2e-4)
            actual = torch.cat([grad.flatten() for grad in grads])
            cotangent = torch.ones_like(actual) / actual.numel() ** 0.5
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_default_dtype(previous_dtype)


@pytest.fixture
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(12)
    yield
    torch.set_default_dtype(previous)


def layout(features, irreps, inverse=False):
    values = []
    for (mul, ir), section in zip(irreps, irreps.slices()):
        shape = (ir.dim, mul) if inverse else (mul, ir.dim)
        values.append(
            features[..., section]
            .reshape(*features.shape[:-1], *shape)
            .transpose(-1, -2)
            .flatten(-2)
        )
    return torch.cat(values, dim=-1)


def tensor_product(channels=3, merge=False, unweighted=False):
    return o3.TensorProduct(
        f"{channels}x0e+{channels}x1o+2x2e",
        "0e+2x1o+1x2e",
        f"{channels}x1o" + ("" if merge else f"+{channels}x1o") + "+2x2e",
        [
            (0, 1, 0, "uvu", True),
            (1, 0, 0 if merge else 1, "uvu", not unweighted),
            (2, 0, 1 if merge else 2, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )


def reference(tp, x, attrs, radial, projection, edges):
    weights = radial @ projection if projection.numel() else radial
    message = tp(
        layout(x, tp.irreps_in1, True)[edges[0]],
        layout(attrs, tp.irreps_in2, True),
        weights,
    )
    result = x.new_zeros(x.size(0), tp.irreps_out.dim).index_add(0, edges[1], message)
    return layout(result, tp.irreps_out)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "shared,direct,merge,unweighted",
    [
        (False, False, False, False),
        (True, False, True, False),
        (False, True, True, True),
    ],
)
def test_values_and_recursive_derivatives(
    device, shared, direct, merge, unweighted, double_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = tensor_product(merge=merge, unweighted=unweighted).to(device)
    conv = O3TensorProductConv(tp).to(device)
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
    conv = O3TensorProductConv(tp).cuda()
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


def test_empty_and_unsupported(double_precision):
    tp = tensor_product()
    conv = O3TensorProductConv(tp)
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
        O3TensorProductConv(tp, backend="triton")
    with pytest.raises(NotImplementedError, match="uvu"):
        O3TensorProductConv(o3.FullyConnectedTensorProduct("0e", "0e", "0e"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tace_radial_bias_cutoff_and_force_training(
    monkeypatch, device, double_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.fused import O3ScatterTensorProduct
    from tace.models.mlp import MLP

    monkeypatch.setenv("TACE_USE_OEQ", "0")
    monkeypatch.setenv("TACE_USE_CUE", "0")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    module = O3ScatterTensorProduct("3x0e+3x1o", "0e+1o", "3x0e+3x1o+3x2e").to(device)
    mlp = MLP([4, 5, module.weight_numel], bias=True).to(device)
    positions = torch.randn(5, 3, device=device, requires_grad=True)
    x = torch.randn(5, module.irreps_in1.dim, device=device, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 0]], device=device)
    vectors = positions[edges[1]] - positions[edges[0]]
    lengths = vectors.square().sum(-1, keepdim=True)
    radial = torch.cat([lengths**n for n in range(4)], -1)
    cutoff = torch.exp(-lengths)
    attrs = o3.spherical_harmonics(module.irreps_in2, vectors, normalize=True)
    expected = module(x, attrs, mlp(radial) * cutoff, edges)
    hidden = mlp.mlp[:-1](radial)
    last = mlp.mlp[-1]
    hidden = torch.cat((hidden, torch.ones_like(hidden[:, :1])), -1)
    weight = torch.cat((last.get_weight(), last.bias.unsqueeze(0)), 0)
    actual = module.forward_stream(x, attrs, hidden, weight, edges, cutoff)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    monkeypatch.setenv("TACE_USE_EQX", "1")
    torch.testing.assert_close(module(x, attrs, mlp(radial) * cutoff, edges), expected)
    forces = [
        torch.autograd.grad(value.square().sum(), positions, create_graph=True)[0]
        for value in (actual, expected)
    ]
    torch.testing.assert_close(*forces, atol=2e-9, rtol=2e-10)
    parameters = tuple(mlp.parameters()) + (x,)
    gradients = [
        torch.autograd.grad(force.square().sum(), parameters, retain_graph=True)
        for force in forces
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-9)


def test_compile_and_cuda_graph(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = tensor_product().cuda()
    conv = O3TensorProductConv(tp).cuda()
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
@pytest.mark.parametrize(
    "shared_attrs,shared_radial",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_o3_chunked_shared_gradients(
    monkeypatch, channels, shared_attrs, shared_radial, double_precision
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o3 import cuda

    # Two grouped chunks and a one-edge tail exercise private and broadcast
    # gradients without mistaking the tail for a shared operand.
    monkeypatch.setattr(cuda, "CHUNK_SIZE", 1025)
    tp = tensor_product(channels, merge=True, unweighted=True).cuda()
    conv = O3TensorProductConv(tp).cuda()
    edges = torch.randint(257, (2, 2051), device="cuda")
    inputs = [
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in (
            (257, tp.irreps_in1.dim),
            (1 if shared_attrs else 2051, tp.irreps_in2.dim),
            (1 if shared_radial else 2051, 5),
            (5, tp.weight_numel),
        )
    ]
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
    from eqx.conv.o3.codegen import angular_source

    tp = tensor_product()
    conv = O3TensorProductConv(tp)
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


def test_tace_model_force_training(monkeypatch, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    for name in ("EQX", "OEQ", "CUE", "EQT"):
        monkeypatch.setenv(f"TACE_USE_{name}", "0")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=3,
        Lmax=2,
        lmax=2,
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
        ],
        target_property=["energy", "forces", "stress", "virials"],
    )
    config["atomic_basis"]["type"] = "cgtp"
    config["node_embedding"]["type"] = "linear"
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["bias"] = True
    config["readout_emlp"]["hidden"] = [3]
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["scale_shift"]["enable"] = False
    reference_model = TensorModel(e3nnTACE(**config)).cuda().train()
    model = deepcopy(reference_model)
    assert model.state_dict().keys() == reference_model.state_dict().keys()
    data = dict(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]], device="cuda"
        ),
        node_attrs=torch.ones(3, 1, device="cuda"),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device="cuda"
        ),
        edge_shifts=torch.zeros(6, 3, device="cuda"),
        lattice=torch.eye(3, device="cuda").unsqueeze(0) * 8,
        batch=torch.zeros(3, dtype=torch.long, device="cuda"),
        ptr=torch.tensor([0, 3], device="cuda"),
        fidelity_idx=torch.zeros(1, dtype=torch.long, device="cuda"),
    )

    def no_edge_message(*args):
        raise AssertionError(
            "The fused interaction must not materialize edge messages or weights"
        )

    for layer in model.readout_fn.representation.interactions:
        monkeypatch.setattr(layer.rejector.tp, "forward", no_edge_message)
        monkeypatch.setattr(layer.edge_info, "forward", no_edge_message)
    results = []
    for network, enabled in ((reference_model, "0"), (model, "1")):
        monkeypatch.setenv("TACE_USE_EQX", enabled)
        output = network({key: value.clone() for key, value in data.items()})
        results.append(output)
        sum(
            output[key].square().sum() for key in ("energy", "forces", "stress")
        ).backward()
    for key in ("energy", "forces", "stress", "virials"):
        torch.testing.assert_close(
            results[0][key], results[1][key], atol=2e-9, rtol=2e-8
        )
    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in model.named_parameters():
        expected = reference_parameters[name].grad
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, atol=2e-8, rtol=2e-7)
