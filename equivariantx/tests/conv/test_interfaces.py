"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import conv as eqx_conv
from equivariantx import o2
from equivariantx import o3 as eqx_o3
from equivariantx.tests.conv.testing import layout


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"), reason="Requires time-reversal e3nn"
)
@pytest.mark.parametrize("kind", ["o3", "o2", "co3"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_convolution_time_reversal(kind, backend, device, double_precision):
    if backend == "cuda" and device != "cuda":
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3

    args = (
        "2x1ee+2x1eo",
        "1oe",
        "2x0oe+2x1oe+2x2oe+2x0oo+2x1oo+2x2oo",
        [(i, 0, 3 * i + k, "uvu", True) for i in range(2) for k in range(3)],
    )
    reference = o3.TensorProduct(
        *args, internal_weights=False, shared_weights=False
    ).to(device)
    x = torch.randn(3, reference.irreps_in1.dim, device=device, requires_grad=True)
    vectors = torch.randn(7, 3, device=device, requires_grad=True)
    weight = torch.randn(7, reference.weight_numel, device=device, requires_grad=True)
    index = torch.randint(3, (2, 7), device=device)
    sh = o3.spherical_harmonics(1, vectors, normalize=True, normalization="component")
    expected = x.new_zeros(3, reference.irreps_out.dim).index_add(
        0, index[1], reference(x[index[0]], sh, weight)
    )
    projection = x.new_empty(0, reference.weight_numel)
    if kind == "co3":
        tp = co3.TensorProduct(*args, internal_weights=False, shared_weights=False).to(
            device
        )
        module = eqx_conv.CartesianTensorProductConv(tp, backend=backend).to(device)
        embed = co3.ChangeOfBasis(tp.irreps_in1).to(device)
        inverse = co3.ChangeOfBasis(tp.irreps_out, inverse=True).to(device)
        actual = inverse(module(embed(x), sh, weight, projection, index))
    elif kind == "o3":
        module = eqx_conv.O3TensorProductConv(reference, backend=backend).to(device)
        actual = layout(
            module(layout(x, reference.irreps_in1), sh, weight, projection, index),
            reference.irreps_out,
            inverse=True,
        )
    else:
        tp = o2.O3TensorProduct(*args, internal_weights=False, shared_weights=False).to(
            device
        )
        module = eqx_conv.O2O3TensorProductConv(tp, backend=backend).to(device)
        actual = layout(
            module(
                layout(x, reference.irreps_in1),
                weight,
                projection,
                None,
                x.new_ones(1, 1),
                index,
                3,
                vectors=vectors,
            ),
            reference.irreps_out,
            inverse=True,
        )
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        probe = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad(
                (z.sin() * probe).sum(), (x, vectors, weight), create_graph=True
            )
            for z in (actual, expected)
        ]
        actual, expected = [torch.cat([g.flatten() for g in gs]) for gs in gradients]


@pytest.mark.parametrize("implementation", ["o3", "o2", "o2_direction"])
@pytest.mark.parametrize("shared", [False, True])
def test_native_cuda_graph_convolution(monkeypatch, implementation, shared):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.kernels.cuda.graph as cuda_graph

    monkeypatch.setattr(cuda_graph, "EDGE_BUCKET", 8)
    monkeypatch.setattr(cuda_graph, "MIN_EDGES", 1)
    cuda_graph._GRAPHS.clear()
    tp_args = (
        "2x0e+2x1o",
        "0e+1o",
        "2x0e+2x1o+2x0e",
        [(0, 0, 0, "uvu", True), (0, 1, 1, "uvu", True), (1, 1, 2, "uvu", False)],
    )
    tp_cls = o3.TensorProduct if implementation == "o3" else o2.O3TensorProduct
    tp = tp_cls(*tp_args, internal_weights=False, shared_weights=False)
    cls = (
        eqx_conv.O3TensorProductConv
        if implementation == "o3"
        else eqx_conv.O2O3TensorProductConv
    )
    module = cls(tp, backend="cuda").cuda().double()
    frame = o2.WignerD(1, 1).cuda().double()
    try:
        for edges in (9, 13, 9, 5, 1, 0):
            index = torch.randint(4, (2, edges), device="cuda")

            def rand(*shape):
                return (
                    torch.randn(*shape, device="cuda", dtype=torch.float64) * 0.2
                ).requires_grad_()

            x, vectors = rand(4, 8), rand(edges, 3)
            rows = 1 if shared else edges
            radial, projection, amplitudes = (
                rand(rows, 3),
                rand(3, tp.weight_numel),
                rand(rows, 2),
            )
            inputs = x, vectors, radial, projection, amplitudes

            def evaluate(enabled):
                monkeypatch.setenv("EQX_USE_CUDA_GRAPH", str(int(enabled)))
                if implementation == "o3":
                    value = module(
                        x,
                        None,
                        radial,
                        projection,
                        index,
                        vectors=vectors,
                        amplitudes=amplitudes,
                    )
                else:
                    value = module(
                        x,
                        radial,
                        projection,
                        frame.forward_packed(vectors),
                        amplitudes,
                        index,
                        4,
                        vectors=vectors if implementation == "o2_direction" else None,
                    )
                result = [value]
                if edges:
                    for _ in range(3):
                        grads = torch.autograd.grad(
                            value.sin().sum(), inputs, create_graph=True
                        )
                        result.extend(grads)
                        value = torch.cat([g.flatten() for g in grads]) / 10
                return result

            reference = evaluate(False)
            actual = evaluate(True)
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, atol=5e-9, rtol=5e-9)
            if edges:
                assert cuda_graph._GRAPHS
    finally:
        cuda_graph._GRAPHS.clear()


def test_native_cuda_graph_outer_capture(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.kernels.cuda.graph as cuda_graph

    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "1")
    monkeypatch.setattr(cuda_graph, "MIN_EDGES", 1)
    monkeypatch.setattr(cuda_graph, "EDGE_BUCKET", 8)
    cuda_graph._GRAPHS.clear()
    tp = o3.TensorProduct(
        "2x0e",
        "0e",
        "2x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    module = eqx_conv.O3TensorProductConv(tp, backend="cuda").cuda().double()
    x = torch.randn(4, 2, device="cuda", dtype=torch.float64, requires_grad=True)
    attrs = torch.randn(11, 1, device="cuda", dtype=torch.float64)
    weight = torch.randn(11, 2, device="cuda", dtype=torch.float64)
    projection = weight.new_empty(0, 2)
    index = torch.randint(4, (2, 11), device="cuda")

    def evaluate():
        value = module(x, attrs, weight, projection, index)
        return value, torch.autograd.grad(value.square().sum(), x)[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Warm the ordinary (unpadded) kernels that the outer graph will capture.
    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "0")
    with torch.cuda.stream(stream):
        evaluate()
        evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "1")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = evaluate()
    assert not cuda_graph._GRAPHS
    with torch.no_grad():
        x.mul_(0.7)
        weight.add_(0.3)
        index.copy_(torch.randint_like(index, 4))
    graph.replay()
    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "0")
    for a, b in zip(actual, evaluate()):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("stride", [1, 3])
def test_convolution_graph_order(owner, device, dtype, stride):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.kernels.cuda.reduction import prepare_graph

    edges = torch.tensor(
        [[0, 0, 1, 0, 3, 0, 2, 0, 0], [2, 1, 2, 0, 2, 2, 1, 2, 2]],
        device=device,
        dtype=dtype,
    ).repeat_interleave(stride, dim=1)[:, ::stride]
    order = prepare_graph(edges[0], edges[1], owner)
    assert prepare_graph(edges[0], edges[1], owner) is order
    assert prepare_graph(edges[0].detach(), edges[1].detach(), owner) is order
    indices = [edges[0], edges[1]]
    indices[1 - owner] = indices[1 - owner].clone()
    indices[1 - owner][0] = 5
    assert prepare_graph(*indices, owner) is order
    assert prepare_graph(indices[owner], indices[owner], 1 - owner) is order
    torch.testing.assert_close(order, edges[owner].argsort(stable=True))
    edges[owner, 0] = 4
    updated = prepare_graph(edges[0].detach(), edges[1].detach(), owner)
    assert updated is not order
    torch.testing.assert_close(updated, edges[owner].argsort(stable=True))
    assert prepare_graph(edges[0, :0], edges[1, :0], owner).shape == (0,)
    permutation, pointers = prepare_graph(edges[0], edges[1], owner, 7)
    assert prepare_graph(edges[0], edges[1], owner, 7)[1] is pointers
    expected = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            edges[owner].bincount(minlength=7).cumsum(0),
        )
    )
    torch.testing.assert_close(pointers, expected)
    torch.testing.assert_close(permutation, updated)
    edges[owner, 0] = 6
    assert prepare_graph(edges[0], edges[1], owner, 7)[1] is not pointers
    for nodes in (0, 7):
        permutation, pointers = prepare_graph(edges[0, :0], edges[1, :0], owner, nodes)
        assert permutation.numel() == 0
        torch.testing.assert_close(
            pointers, torch.zeros(nodes + 1, device=device, dtype=torch.int64)
        )
    if dtype == torch.int64:
        extremes = edges.new_tensor([2**63 - 1, -(2**63), 0, 2**31, 3, -1])
        extremes = extremes.repeat_interleave(2)[::2]
        permutation, pointers = prepare_graph(extremes, extremes, owner, 7)
        torch.testing.assert_close(permutation, extremes.argsort(stable=True))
        torch.testing.assert_close(
            pointers,
            torch.searchsorted(extremes[permutation], torch.arange(8, device=device)),
        )
    if device != "cuda" or owner != 0 or stride != 3:
        return
    index = torch.randint(19, (514,), device="cuda", dtype=dtype)[::2]
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        order, pointers = prepare_graph(index, index, 0, 23)
    with torch.cuda.stream(consumer):
        cached = prepare_graph(index, index, 0, 23)
        assert cached[0] is order and cached[1] is pointers
        copied = tuple(value.clone() for value in cached)
    consumer.synchronize()

    def check(actual):
        expected_order = index.argsort(stable=True)
        expected_ptr = torch.searchsorted(
            index[expected_order], torch.arange(24, device=index.device, dtype=dtype)
        )
        for value, expected in zip(actual, (expected_order, expected_ptr)):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    check(copied)
    producer.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=producer):
        captured = prepare_graph(index, index, 0, 23)
    graph.replay()
    check(captured)
    index.add_(3).remainder_(19)
    graph.replay()
    check(captured)
    assert prepare_graph(index, index, 0, 23)[1] is not pointers

    with torch.inference_mode():
        index = index.clone()
        before = prepare_graph(index, index, 0, 23)
        index.add_(1).remainder_(19)
        after = prepare_graph(index, index, 0, 23)
        assert before[1] is not after[1]
        check(after)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_product_and_convolution_compile(cls, device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = cls("0e+1o", "0e+1o", "0e+1o+2e", shared_weights=False)
    convolution = getattr(eqx_conv, cls.__name__ + "Conv")(tp).to(device)
    compiled = torch.compile(
        convolution, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for edges in (3, 6, 0):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(edges, 4, device=device, requires_grad=True)
        weight = torch.randn(edges, tp.weight_numel, device=device, requires_grad=True)
        index = torch.randint(4, (2, edges), device=device)
        actual, expected = (
            compiled(x, y, index, weight),
            convolution(x, y, index, weight),
        )
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), (x, y, weight)),
            torch.autograd.grad(expected.square().sum(), (x, y, weight)),
        ):
            torch.testing.assert_close(a, b)
