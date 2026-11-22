"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import conv as eqx_conv
from equivariantx import o2
from equivariantx import o3 as eqx_o3
from equivariantx.ops import config
from equivariantx.ops.common.radial import (
    convolve,
    materialize,
    radial_features,
    radial_parts,
    radial_reference,
)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("second_width", [2, 3])
def test_grouped_radial_program(backend, second_width, double_precision, monkeypatch):
    from equivariantx.nn import GroupedRadialMLP
    from equivariantx.ops.common.convolution import radial_input
    from equivariantx.ops.common.edge import evaluate
    from equivariantx.ops.common.torch.edge import evaluate_torch
    from equivariantx.utils.program import Program

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    branches = [
        torch.nn.Sequential(
            torch.nn.Linear(n, 4), torch.nn.SiLU(), torch.nn.Linear(4, m)
        )
        for n, m in [(3, 3), (4, second_width)]
    ]
    net = GroupedRadialMLP(
        [2, 3], branches, [(0, 1), (1, second_width), (0, 2)], context_features=1
    ).to(device)
    monkeypatch.setattr(config, "RADIAL_CHAIN_TILE_BYTES", 1024)
    x = torch.randn(35, 5, device=device, requires_grad=True)
    ctx = torch.randn(3, 1, device=device, requires_grad=True)
    index = torch.arange(len(x), device=device) % 3
    program, inputs = Program(), []
    root = radial_input(program, inputs, ((x, "edge"), (ctx, "target")), net)
    metadata = repr((tuple(program.nodes), ((root, 0, "edge"),)))
    evaluator = evaluate if backend == "cuda" else evaluate_torch
    actual = evaluator(metadata, inputs, index, index, 3)[0]
    a, b = [
        branch(torch.cat((part, ctx[index]), -1))
        for branch, part in zip(branches, x.split((2, 3), -1))
    ]
    expected = torch.cat((a[:, :1], b, a[:, 1:]), -1)
    torch.testing.assert_close(net(torch.cat((x, ctx[index]), -1)), expected)
    if backend == "cuda":
        # Unequal input widths, shared context, independently weighted
        # branches and a nontrivial output permutation. Probe mixed derivatives.
        inp = torch.cat((x, ctx[index]), -1)
        params = tuple(net.parameters())
        for mode in ("last", "full"):
            value, ref = net.stream_rows(inp, backend, mode), expected

            def force_grad(y):
                force = torch.autograd.grad(y.sin().sum(), x, create_graph=True)[0]
                return torch.autograd.grad(
                    force.square().sum(), params, retain_graph=True
                )

            torch.testing.assert_close(force_grad(value), force_grad(ref))
            for variables in ((x,), (x, *params), (x, *params)):
                torch.testing.assert_close(value, ref)
                probes = [torch.randn_like(p) for p in variables]

                def vjp(y):
                    grads = torch.autograd.grad(
                        y.sin().sum(), variables, create_graph=True
                    )
                    return sum((g * v).sum() for g, v in zip(grads, probes))

                value, ref = vjp(value), vjp(ref)
            torch.testing.assert_close(value, ref)
    active = (x, ctx, *net.parameters())
    for _ in range(2):
        torch.testing.assert_close(actual, expected)
        da, db = [
            torch.autograd.grad(y.sin().sum(), active, create_graph=True)
            for y in (actual, expected)
        ]
        torch.testing.assert_close(da, db)
        actual, expected = (sum(z.sum() for z in d) for d in (da, db))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_spline(dtype, device):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.nn import uniform_cubic_spline
    from equivariantx.ops.decompositions import torch_decompositions

    c = torch.randn(4, 4, 7, device=device, dtype=dtype)
    x = torch.tensor(
        [-0.1, 0.0, 0.48, 0.5, 1.87, 2.0],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    expected = uniform_cubic_spline(x, c, 0.5, backend="torch")

    def spline(value, table):
        return uniform_cubic_spline(value, table, 0.5)

    actual = spline(x, c)
    torch.testing.assert_close(spline(x[:0], c), expected[:0])
    graph = make_fx(spline, decomposition_table=torch_decompositions())(x, c)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    torch.testing.assert_close(graph(x, c), expected)
    for _ in range(3):
        torch.testing.assert_close(actual, expected)
        actual, expected = [
            torch.autograd.grad(y.sum(), x, create_graph=True)[0]
            for y in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fourier_radial(dtype, device):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.nn import fourier_radial
    from equivariantx.ops.decompositions import torch_decompositions

    x = torch.tensor(
        [-0.1, 0, 1.25, 9.99, 10.0], device=device, dtype=dtype, requires_grad=True
    )
    w = torch.linspace(0.1, 2.0, 400, device=device, dtype=dtype).reshape(2, 200)
    coef = (
        torch.arange(800, device=device, dtype=dtype).remainder(7).reshape(4, 200) / 700
    )

    def evaluate(value, w, coef):
        return fourier_radial(value, w, coef, 2, 10.0)

    expected = fourier_radial(x, w, coef, 2, 10.0, backend="torch")
    actual = evaluate(x, w, coef)
    graph = make_fx(evaluate, decomposition_table=torch_decompositions())(x, w, coef)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    torch.testing.assert_close(graph(x, w, coef), expected)
    torch.testing.assert_close(evaluate(x[:0], w, coef), expected[:0])
    assert actual.detach()[[0, -1]].count_nonzero() == 0
    for _ in range(3):
        torch.testing.assert_close(actual, expected)
        actual, expected = [
            torch.autograd.grad(y.sum(), x, create_graph=True)[0]
            for y in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected)


def embedded_case(kind, edges, shared, device):
    from equivariantx.tests.ops.common.torch.test_convolution import (
        convolution_case as aligned_case,
    )
    from equivariantx.tests.ops.common.torch.test_o3 import (
        convolution_case as spherical_case,
    )

    if kind == "o2":
        metadata, values, source, target = aligned_case(edges, shared, device=device)
        role, kernel = 6, "contraction"
    else:
        _, metadata, values, source, target = spherical_case(
            True, edges, shared, device=device
        )
        role, kernel = 4, "o3_contraction"
    values = tuple(value.detach().requires_grad_() for value in values)
    network = torch.nn.Sequential(
        torch.nn.Linear(4, 5), torch.nn.SiLU(), torch.nn.Linear(5, 4)
    ).to(device)
    program = repr(((tuple(range(len(values))), False, ((role, 0),)),))
    return (
        metadata,
        program,
        kernel,
        role,
        values,
        torch.stack((source, target)),
        network,
    )


@pytest.mark.parametrize("kind", ["o2", "o3"])
@pytest.mark.parametrize("edges,shared", [(0, False), (7, False), (7, True)])
@pytest.mark.parametrize("recompute", [False, True])
def test_embedded_convolution(
    kind, edges, shared, recompute, double_precision, monkeypatch
):
    from equivariantx.ops import get_kernel
    from equivariantx.ops.common import edge
    from equivariantx.ops.common.torch.edge import evaluate_recomputed, evaluate_torch

    metadata, program, kernel, role, values, index, network = embedded_case(
        kind, edges, shared, "cpu"
    )

    def evaluate(*args, inline):
        assert not inline
        return (evaluate_recomputed if recompute else evaluate_torch)(*args)

    monkeypatch.setattr(edge, "evaluate", evaluate)
    actual = convolve(kind, metadata, values, index, network)
    operands = [values[0], network(values[1]), *values[2:]]
    expected = get_kernel(kernel, "torch")(metadata, program, *index, operands)[0]
    active = (*values[:role], *values[role + 1 :], *network.parameters())
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        gradients = [
            torch.autograd.grad(output.sin().sum(), active, create_graph=True)
            for output in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [sum(g.sum() for g in gs) / 20 for gs in gradients]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA radial transforms")
@pytest.mark.parametrize("kind", ["o2", "o3"])
@pytest.mark.parametrize("compiler", [None, "inductor"])
def test_embedded_convolution_transforms(kind, compiler, double_precision):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import get_kernel
    from equivariantx.ops.decompositions import torch_decompositions
    from equivariantx.tests.ops.testing import assert_functional_derivatives

    metadata, program, kernel, role, values, index, network = embedded_case(
        kind, 7, False, "cuda"
    )

    def function(*inputs):
        return convolve(kind, metadata, inputs[:-1], inputs[-1], network)

    def reference(*inputs):
        values, index = inputs[:-1], inputs[-1]
        return get_kernel(kernel, "torch")(
            metadata, program, *index, [values[0], network(values[1]), *values[2:]]
        )[0]

    torch.testing.assert_close(
        function(*values, index), reference(*values, index), atol=1e-10, rtol=1e-10
    )
    graph = make_fx(function, decomposition_table=torch_decompositions())(
        *values, index
    )
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    torch.testing.assert_close(
        graph(*values, index), reference(*values, index), atol=1e-10, rtol=1e-10
    )
    assert_functional_derivatives(function, reference, (*values, index), compiler)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reference", [False, True])
def test_reject_low_precision_network(dtype, reference, monkeypatch):
    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    network = torch.nn.Sequential(torch.nn.Linear(2, 3)).to(dtype)
    with pytest.raises(TypeError, match="float32 or float64"):
        radial_features(network, torch.ones(3, 2), torch.zeros(2, 3, dtype=torch.long))


@pytest.mark.parametrize("normalization", [torch.nn.LayerNorm, torch.nn.RMSNorm])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_normalization_four_derivatives(normalization, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    network = torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.SiLU(),
        torch.nn.Sequential(normalization(8, eps=0.1), torch.nn.Linear(8, 4)),
        torch.nn.Tanh(),
    ).to(device=device, dtype=torch.float64)
    value = torch.randn(7, 3, dtype=torch.float64, device=device, requires_grad=True)
    index = torch.zeros(2, 7, dtype=torch.long, device=device)
    actual = radial_features(network, value, index)
    torch.testing.assert_close(actual, network(value))
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = radial_features(network, value, index)
    torch.testing.assert_close(actual, expected)
    variables = (value, *network.parameters())
    for _ in range(4):
        gradients = [
            torch.autograd.grad(output.square().mean(), variables, create_graph=True)
            for output in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-8, rtol=1e-8)
        actual, expected = [
            sum(gradient.square().mean() for gradient in values) for values in gradients
        ]
    tangent = torch.randn_like(value)
    reference_jvp = torch.func.jvp(
        lambda value: radial_reference(network, value), (value,), (tangent,)
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(
        torch.func.jvp(
            lambda value: radial_features(network, value, index), (value,), (tangent,)
        ),
        reference_jvp,
    )


@pytest.mark.parametrize(
    "parts",
    [
        (),
        ((torch.ones(2, 3), "shared"),),
        ((torch.ones(1, 3), "unknown"),),
        ((torch.ones(3), "edge"),),
    ],
)
def test_invalid_partitions(parts):
    with pytest.raises(ValueError):
        radial_parts(parts)
    with pytest.raises(ValueError):
        materialize(parts, torch.zeros(2, 2, dtype=torch.long))


@pytest.mark.parametrize("edges", [0, 7])
def test_materialize_partitions(edges):
    source = torch.randn(3, 2, requires_grad=True)
    target = torch.randn(5, 3, requires_grad=True)
    shared = torch.randn(1, 4, requires_grad=True)
    edge = torch.randn(edges, 2, requires_grad=True)
    index = torch.stack((torch.arange(edges) % 3, torch.arange(edges) % 5))
    parts = (
        (source, "source"),
        (target, "target"),
        (shared, "shared"),
        (shared, "edge"),
        (edge, "edge"),
    )
    actual = materialize(parts, index)
    expected = torch.cat(
        (
            source[index[0]],
            target[index[1]],
            shared.expand(edges, -1),
            shared.expand(edges, -1),
            edge,
        ),
        -1,
    )
    torch.testing.assert_close(actual, expected)
    values = source, target, shared, edge
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), values),
        torch.autograd.grad(expected.square().sum(), values),
    )


def test_local_gate_cutoff_factorization(double_precision):
    from equivariantx.conv.radial import MultiCutoffRadial

    r = torch.tensor([0.5, 2.0, 2.9, 3.0, 3.1, 5.99, 6.0, 6.1], requires_grad=True)
    inner = (1 - r / 3).clamp_min(0).pow(3)
    outer = (1 - r / 6).clamp_min(0).pow(3)
    envelopes = torch.stack((inner, outer), -1)
    relative = MultiCutoffRadial.relative_envelopes(envelopes)
    # The outer-only nonlinear message is exactly the single-cutoff message.
    torch.testing.assert_close(relative[:, 1], torch.ones_like(outer))
    actual = outer * (relative[:, 0] * 2 + relative[:, 1])
    expected = 2 * inner + outer
    for _ in range(3):
        torch.testing.assert_close(actual, expected)
        (actual,) = torch.autograd.grad(actual.sum(), r, create_graph=True)
        (expected,) = torch.autograd.grad(expected.sum(), r, create_graph=True)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("edges", [0, 7])
@pytest.mark.parametrize("paired", [False, True])
def test_multi_cutoff_convolution(backend, edges, paired, double_precision):
    from e3nn.nn import FullyConnectedNet

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    tp = o3.TensorProduct(
        "0e+1o",
        "0e+1o",
        "0e+1o",
        [(0, 0, 0, "uvu", True), (1, 1, 0, "uvu", True), (0, 1, 1, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    networks = [
        FullyConnectedNet([3, tp.weight_numel], variance_in=2, variance_out=3),
        torch.nn.Sequential(
            torch.nn.Linear(5, 7), torch.nn.SiLU(), torch.nn.Linear(7, tp.weight_numel)
        ),
    ]
    model = eqx_conv.MultiCutoffRadial(networks).to(device)
    assert not tuple(model.parameters(recurse=False))

    def rand(rows, width):
        return torch.randn(rows, width, device=device, requires_grad=True)

    index = torch.tensor([[0, 1, 0, 2, 3, 1, 2], [2, 0, 1, 0, 1, 3, 3]], device=device)[
        :, :edges
    ]
    pairs = eqx_conv.undirected_edge_pairs(index, num_nodes=4) if paired else None
    rows = edges if pairs is None else pairs.representatives.numel()
    x, attrs, source = rand(4, 4), rand(edges, 4), rand(4, 2)
    features, envelopes = rand(rows, 3), rand(edges, 2)
    shared = rand(1, 3)
    parts = ((features, "edge"), (source, "source"))
    input_index = index if pairs is None else index[:, pairs.representatives]
    dense = (shared.expand(rows, -1), materialize(parts, input_index))
    if pairs is not None:
        dense = tuple(pairs.expand(f) for f in dense)
    torch.testing.assert_close(
        model((shared, parts), envelopes, edge_index=index, edge_pairs=pairs),
        model(dense, envelopes),
    )
    expected = sum(
        x.new_zeros((4, 4)).index_add(
            0,
            index[1],
            tp(x[index[0]], attrs, net(f)) * envelopes[:, i : i + 1],
        )
        for i, (net, f) in enumerate(zip(model.networks, dense))
    )
    active = (x, attrs, source, features, shared, envelopes, *model.parameters())
    for mode in ("last", "full"):
        conv = eqx_conv.O3TensorProductConv(tp, radial_mode=mode, backend=backend).to(
            device
        )
        radial, projection, network = model.convolution_weights(
            (shared, parts),
            envelopes,
            radial_mode=mode,
            edge_index=index,
            edge_pairs=pairs,
            backend=backend,
        )
        actual = conv(x, attrs, radial, projection, index, radial_network=network)
        ref = expected
        for order in range(3):
            torch.testing.assert_close(actual, ref, atol=1e-9, rtol=1e-9)
            if order < 2:
                actual, ref = (
                    torch.cat(
                        [
                            g.reshape(-1)
                            for g in torch.autograd.grad(
                                y.sin().sum(), active, create_graph=True
                            )
                        ]
                    )
                    for y in (actual, ref)
                )


@pytest.mark.parametrize("kind", ["o3", "o2"])
@pytest.mark.parametrize("width", [3, 129])
def test_cuda_affine_projection_gradients(kind, width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    args = (
        "2x0e+2x1o",
        "0e+1o",
        "2x0e+2x1o",
        [(0, 0, 0, "uvu", True), (1, 1, 0, "uvu", True), (0, 1, 1, "uvu", True)],
    )
    tp_cls = o3.TensorProduct if kind == "o3" else o2.O3TensorProduct
    tp = tp_cls(*args, internal_weights=False, shared_weights=False).float().cuda()
    conv_cls = (
        eqx_conv.O3TensorProductConv if kind == "o3" else eqx_conv.O2O3TensorProductConv
    )
    module = conv_cls(tp, backend="cuda").cuda()
    x = torch.randn(3, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    r = torch.randn(5, width, device="cuda", dtype=torch.float32, requires_grad=True)
    p = (
        torch.randn(width, tp.weight_numel, device="cuda", dtype=torch.float32)
        / width**0.5
    ).requires_grad_()
    v = torch.randn(5, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    index = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 0]], device="cuda")
    amplitude = torch.ones(5, 2, device="cuda", dtype=torch.float32)

    def evaluate(backend):
        module.backend = backend
        if kind == "o3":
            return module(x, None, r, p, index, vectors=v, amplitudes=amplitude)
        return module(x, r, p, None, amplitude, index, 3, vectors=v)

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        actual, expected = evaluate("cuda"), evaluate("torch")
        for order in range(3):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            if order == 2:
                break
            actual, expected = (
                torch.cat(
                    [
                        g.flatten()
                        for g in torch.autograd.grad(
                            value.square().sum(),
                            (x, r, p, v),
                            create_graph=True,
                        )
                    ]
                )
                for value in (actual, expected)
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("kind", ["o3", "o2", "wigner", "uu", "cartesian"])
@pytest.mark.parametrize("width", [64, 128, 1024])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_radial_network(kind, width, shared, backend, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    args = (
        "2x0e+2x1o",
        "0e+1o",
        "2x0e+2x1o",
        [(0, 0, 0, "uvu", True), (1, 1, 0, "uvu", True), (0, 1, 1, "uvu", True)],
    )
    if kind == "uu":
        frame = o2.LocalFrame(args[0])
        linear = o2.UuLinear(frame.irreps_out, frame.irreps_out, 2)
        module = eqx_conv.UuO2TensorProductConv(
            frame, linear, frame, backend=backend, radial_mode="full"
        ).cuda()
    else:
        from equivariantx import co3

        tp = (
            co3.TensorProduct
            if kind == "cartesian"
            else o3.TensorProduct
            if kind == "o3"
            else o2.O3TensorProduct
        )(*args, internal_weights=False, shared_weights=False)
        module = (
            eqx_conv.CartesianTensorProductConv
            if kind == "cartesian"
            else eqx_conv.O3TensorProductConv
            if kind == "o3"
            else eqx_conv.O2O3TensorProductConv
        )(tp, backend=backend, radial_mode="full").cuda()
    network = torch.nn.Sequential(
        torch.nn.Linear(3, width),
        torch.nn.LayerNorm(width),
        torch.nn.SiLU(),
        torch.nn.Linear(width, width),
        torch.nn.RMSNorm(width),
        torch.nn.SiLU(),
    ).cuda()
    x = torch.randn(3, 8, device="cuda", requires_grad=True)
    radial = torch.randn(1 if shared else 7, 3, device="cuda", requires_grad=True)
    projection = (
        torch.randn(width, module.weight_numel, device="cuda", requires_grad=True)
        / width**0.5
    )
    vectors = torch.randn(7, 3, device="cuda", requires_grad=True)
    index = torch.randint(3, (2, 7), device="cuda")
    amplitude = torch.ones(
        1 if shared else 7, 1 if kind == "uu" else 2, device="cuda", requires_grad=True
    )
    if kind == "uu":
        with pytest.raises(ValueError, match="final linear weight"):
            module(
                x,
                radial,
                projection.new_empty((0, module.weight_numel)),
                None,
                amplitude,
                index,
                3,
                vectors=vectors,
                radial_network=network,
            )

    def evaluate(fused):
        r = radial
        if not fused:
            for layer in network:
                if isinstance(layer, torch.nn.LayerNorm):
                    # An explicit expression also supplies third derivatives;
                    # native LayerNorm saves detached variance statistics.
                    r = r - r.mean(-1, keepdim=True)
                    r = r * (r.square().mean(-1, keepdim=True) + layer.eps).rsqrt()
                    r = r * layer.weight + layer.bias
                else:
                    r = layer(r)
        kwargs = dict(vectors=vectors, radial_network=network if fused else None)
        if kind in ("o3", "cartesian"):
            return module(x, None, r, projection, index, amplitudes=amplitude, **kwargs)
        if kind == "uu":
            return module(x, r, projection, None, amplitude, index, 3, **kwargs)
        return module(
            x,
            r,
            projection,
            None,
            amplitude,
            index,
            3,
            method="wigner" if kind == "wigner" else "baseline",
            **kwargs,
        )

    actual, expected = evaluate(True), evaluate(False)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    inputs = (x, radial, projection, vectors, amplitude, *network.parameters())
    for order in range(3):
        seed = torch.randn_like(actual)
        ga = torch.autograd.grad(
            actual, inputs, seed, create_graph=True, allow_unused=True
        )
        gb = torch.autograd.grad(
            expected, inputs, seed, create_graph=True, allow_unused=True
        )
        for input_index, (a, b) in enumerate(zip(ga, gb)):
            if b is not None:
                torch.testing.assert_close(
                    a,
                    b,
                    atol=1e-8,
                    rtol=1e-8,
                    msg=lambda message: (
                        f"derivative order {order + 1}, input {input_index}: {message}"
                    ),
                )
        actual = sum(g.square().sum() for g in ga if g is not None) / 100
        expected = sum(g.square().sum() for g in gb if g is not None) / 100


@pytest.mark.parametrize("kind", ["o3", "o2"])
@pytest.mark.parametrize("edges", [0, 7])
def test_radial_network_compiled(kind, edges, monkeypatch, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.common.cuda.execution as execution

    launch = execution.launch
    monkeypatch.setattr(execution, "launch", lambda *args: launch(*args, tile_size=3))
    tp = (o3.TensorProduct if kind == "o3" else o2.O3TensorProduct)(
        "2x0e+2x1o",
        "0e+1o",
        "2x0e",
        [(0, 0, 0, "uvu", True), (1, 1, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    module = (
        eqx_conv.O3TensorProductConv if kind == "o3" else eqx_conv.O2O3TensorProductConv
    )(tp, backend="cuda", radial_mode="full").cuda()
    network = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.SiLU()).cuda()
    inputs = [
        torch.randn(*shape, device="cuda", requires_grad=True)
        for shape in (
            (3, 8),
            (edges, 2),
            (3, 2),
            (3, 2),
            (edges, 3),
            (9, tp.weight_numel),
        )
    ]
    index = torch.randint(3, (2, edges), device="cuda")
    amplitudes = torch.ones(edges, 2, device="cuda")

    def evaluate(x, radial, source, target, vectors, projection, stream=True):
        if stream:
            radial = ((radial, "edge"), (source, "source"), (target, "target"))
        else:
            radial = network(
                torch.cat((radial, source[index[0]], target[index[1]]), -1)
            )
            radial = torch.cat((radial, torch.ones_like(radial[:, :1])), -1)
        kwargs = dict(vectors=vectors, radial_network=network if stream else None)
        if kind == "o3":
            return module(
                x, None, radial, projection, index, amplitudes=amplitudes, **kwargs
            )
        return module(
            x,
            radial,
            projection,
            None,
            amplitudes,
            index,
            3,
            method="baseline",
            **kwargs,
        )

    compiled = torch.compile(
        evaluate, backend="aot_eager", fullgraph=True, dynamic=True
    )
    actual, expected = compiled(*inputs), evaluate(*inputs, stream=False)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    if edges:
        variables = (*inputs, *network.parameters())
        actual_grads = torch.autograd.grad(actual.sum(), variables)
        expected_grads = torch.autograd.grad(expected.sum(), variables)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)

        from torch.fx.experimental.proxy_tensor import make_fx

        def force(*values):
            energy = evaluate(*values)
            return torch.autograd.grad(
                energy.square().sum(), values[4], create_graph=True
            )[0]

        traced = make_fx(force, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
            *inputs
        )
        compiled = torch.compile(traced, backend="aot_eager", fullgraph=True)
        actual, expected = compiled(*inputs), force(*inputs)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        actual_grads = torch.autograd.grad(actual.square().sum(), variables)
        expected_grads = torch.autograd.grad(expected.square().sum(), variables)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("width", [64, 128, 1024])
@pytest.mark.parametrize("edges", [0, 7])
def test_edge_radial_projection(monkeypatch, width, edges, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.common.cuda.execution as execution
    import equivariantx.ops.common.edge as edge
    from equivariantx.ops.common.torch.edge import evaluate_torch
    from equivariantx.utils.metadata import parse_metadata
    from equivariantx.utils.program import Program

    program = Program()
    x = program.input(0, "source", 4)
    z = program.input(1, "edge", width)
    w = program.input(2, "shared", width * 4)
    b = program.input(3, "shared", 4)
    # Shared products keep one row across tiles, including a shorter final tile.
    b = program.product(
        (b, program.constant(4, 0)),
        (1, (4, 4), tuple(((i, i), 1.5) for i in range(4))),
        1,
    )
    value = program.binary("add", program.matmul(z, w, 1, width, 4), b)
    value = program.unary("silu", program.binary("mul", x, value))
    metadata = repr((tuple(program.nodes), ((value, 0, "target"),)))
    assert execution.uses_matrix_products(metadata)
    tiled = execution.launch

    def launch(*args):
        return tiled(*args, tile_size=3)

    monkeypatch.setattr(execution, "launch", launch)
    values = [
        (torch.randn(shape, device="cuda") * 0.1).requires_grad_()
        for shape in ((3, 4), (edges, width), (1, width * 4), (1, 4))
    ]
    index = torch.randint(3, (2, edges), device="cuda")
    actual = edge.evaluate(metadata, values, *index, 3)[0]
    expected = evaluate_torch(metadata, values, *index, 3)[0]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    adjoints = []
    evaluate = edge.edge_program

    def record(backend, metadata, *args, inline=True):
        adjoints.append({slot for _, slot, _ in parse_metadata(metadata)[1]})
        return evaluate(backend, metadata, *args, inline=inline)

    monkeypatch.setattr(edge, "edge_program", record)
    # A position-dependent radial input needs no parameter adjoints on the
    # first pass, but its mixed derivatives must still include all parameters.
    actual = torch.autograd.grad(actual.sin().sum(), values[1], create_graph=True)[0]
    expected = torch.autograd.grad(expected.sin().sum(), values[1], create_graph=True)[
        0
    ]
    assert adjoints[0] == {1}
    for _ in range(2):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        a = torch.autograd.grad(actual.sin().sum(), values, create_graph=True)
        b = torch.autograd.grad(expected.sin().sum(), values, create_graph=True)
        actual, expected = (torch.cat([v.flatten() for v in g]) for g in (a, b))
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
@pytest.mark.parametrize("num_edges", [0, 7])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_convolution_radial_network(cls, num_edges, device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = cls("2x0e+2x1o", "0e+1o", "2x0e+2x1o+2x2e", shared_weights=False)
    convolution = getattr(eqx_conv, cls.__name__ + "Conv")(tp).to(device)
    network = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.SiLU(), torch.nn.Linear(8, tp.weight_numel)
    ).to(device)
    features = torch.randn(4, tp.irreps_in1.dim, device=device, requires_grad=True)
    vectors = torch.randn(num_edges, 3, device=device, requires_grad=True)
    radial = torch.randn(num_edges, 3, device=device, requires_grad=True)
    index = torch.randint(4, (2, num_edges), device=device)
    attrs = o3.spherical_harmonics([0, 1], vectors, normalize=False)
    actual = convolution(
        features, attrs, index, num_nodes=6, radial=radial, radial_network=network
    )
    messages = tp(features[index[0]], attrs, network(radial))
    expected = messages.new_zeros(6, tp.irreps_out.dim).index_add(0, index[1], messages)
    inputs = (features, vectors, radial, *network.parameters())
    torch.testing.assert_close(actual, expected)
    for _ in range(2):
        gradients = [
            torch.autograd.grad(value.sin().sum(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([g.flatten() for g in values]) for values in gradients
        ]
        torch.testing.assert_close(actual, expected)
