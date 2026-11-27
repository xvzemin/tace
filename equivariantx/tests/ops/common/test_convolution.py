"""Radial policies, sparse paths and mixed parameter derivatives."""

import pytest
import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.nn import indexed_rows, indexed_weights
from equivariantx.o3 import O3TensorProductConv
from equivariantx.ops import config
from equivariantx.ops.common.convolution import convolve
from equivariantx.ops.compiler import compile_and_package
from equivariantx.tests.utils.testing import package_script
from equivariantx.utils import undirected_edge_pairs


def convolution_case(width=4, edges=7, shared=False, device="cpu"):
    tp = o3.TensorProduct(
        "2x0e+2x1o",
        "0e+2x1o",
        "2x0e+2x1o+2x1o",
        [
            (0, 0, 0, "uvu", True),
            (1, 1, 0, "uvu", True),
            (0, 1, 1, "uvu", True),
            (1, 0, 2, "uvu", False),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    module = O3TensorProductConv(tp).double().to(device)
    index = torch.randint(3, (2, edges), device=device)
    inputs = tuple(
        torch.randn(*shape, dtype=torch.float64, device=device).requires_grad_()
        for shape in (
            (3, tp.irreps_in1.dim),
            (1 if shared else edges, tp.irreps_in2.dim),
            (1 if shared else edges, width),
            (width, tp.weight_numel),
        )
    )
    return module, inputs, index


@pytest.mark.parametrize("mode", ["last", "full"])
@pytest.mark.parametrize("ordered", [False, True])
def test_pair_weight_derivatives(mode, ordered, device, double_precision, monkeypatch):
    # Include an unpaired edge, periodic images, duplicate arcs and self images.
    idx = torch.tensor(
        [[0, 1, 1, 0, 0, 0, 0, 2], [1, 0, 0, 1, 0, 0, 0, 1]], device=device
    )
    shifts = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, -2, 1],
            [0, 2, -1],
            [0, 2, -1],
            [0, 0, 0],
        ],
        device=device,
    )
    pairs = undirected_edge_pairs(idx, shifts, num_nodes=3)
    assert pairs.representatives.numel() == 4
    arc = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3], device=device)
    torch.testing.assert_close(
        pairs.inverse[:, None] == pairs.inverse, arc[:, None] == arc
    )
    general = undirected_edge_pairs(idx, shifts)
    torch.testing.assert_close(general.inverse, pairs.inverse)
    empty = undirected_edge_pairs(idx[:, :0], shifts[:0], num_nodes=3)
    assert empty.inverse.numel() == empty.representatives.numel() == 0
    module, (x, y, _, matrix), _ = convolution_case(edges=idx.size(1), device=device)
    module.radial_mode = mode
    r = x.new_empty(4, 4).normal_().requires_grad_()
    source = x.new_empty(3, module.weight_numel).normal_().requires_grad_()
    target = torch.randn_like(source, requires_grad=True)
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU()).to(x)
    variables = (x, y, r, matrix, *net.parameters())
    refweights = net(r[pairs.inverse]) @ matrix
    if ordered:
        refweights = refweights + source[idx[0]] + target[idx[1]]
        variables += source, target
        rad, p, network = indexed_weights(
            r,
            matrix,
            ((source, idx[0]), (target, idx[1])),
            radial_network=net,
            edge_pairs=pairs,
            radial_mode=mode,
        )
        actual = module(x, y, rad, p, idx, radial_network=network)
    else:
        actual = module(x, y, r, matrix, idx, radial_network=net, edge_pairs=pairs)
    expected = module.reference(
        x, y, refweights, matrix.new_empty(0, module.weight_numel), idx, len(x)
    )
    for _ in range(2):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
        probes = tuple(torch.randn_like(v) for v in variables)

        def derivative(value):
            gradients = torch.autograd.grad(
                value.sin().mean(), variables, create_graph=True
            )
            return sum((g * d).sum() for g, d in zip(gradients, probes))

        actual, expected = derivative(actual), derivative(expected)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    if ordered and mode == "last":
        bias = r.new_empty(module.weight_numel).normal_().requires_grad_()
        direct = (r @ matrix + bias)[pairs.inverse] + source[idx[0]] + target[idx[1]]
        packed = indexed_weights(
            r,
            torch.cat((matrix, bias[None])),
            ((source, idx[0]), (target, idx[1])),
            edge_pairs=pairs,
        )
        actual = module(
            x, y, packed.radial, packed.projection, idx, radial_network=packed.network
        )
        ref = module.reference(
            x, y, direct, matrix.new_empty(0, module.weight_numel), idx, len(x)
        )
        inputs = (x, y, r, matrix, bias, source, target)
        probes = [torch.randn_like(value) for value in inputs]
        torch.testing.assert_close(actual, ref)
        for i in range(2):
            a = torch.autograd.grad(actual.sin().sum(), inputs, create_graph=i == 0)
            b = torch.autograd.grad(ref.sin().sum(), inputs, create_graph=i == 0)
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
            if i == 0:
                actual = sum((v * q).sum() for v, q in zip(a, probes))
                ref = sum((v * q).sum() for v, q in zip(b, probes))
    if ordered:
        # Exactly enough room for pair outputs, not for last-mode hidden states.
        monkeypatch.setattr(
            config,
            "RADIAL_PROJECT_MAX_BYTES",
            len(r) * module.weight_numel * r.element_size(),
        )
        _, selected = network.bind_pairs(rad, module)
        assert (selected.pair is None) == (mode == "full")
        monkeypatch.setattr(
            config, "RADIAL_PROJECT_MAX_BYTES", r.numel() * r.element_size()
        )
        _, selected = network.bind_pairs(rad, module)
        assert selected.pair is not None and selected.network is None
        torch.testing.assert_close(selected.pair.features, net(r))
        hidden_only = module(x, y, rad, p, idx, radial_network=network)
        torch.testing.assert_close(
            hidden_only,
            module.reference(
                x, y, refweights, matrix.new_empty(0, module.weight_numel), idx, len(x)
            ),
        )
    # Large-table policy: keep the directed streaming kernel and narrow inputs.
    monkeypatch.setattr(config, "RADIAL_PROJECT_MAX_BYTES", 0)
    module.radial_mode = "full"
    if ordered:
        rad, p, network = indexed_weights(
            r,
            matrix,
            ((source, idx[0]), (target, idx[1])),
            radial_network=net,
            edge_pairs=pairs,
            radial_mode="full",
        )
        out = module(x, y, rad, p, idx, radial_network=network)
    else:
        out = module(x, y, r, matrix, idx, radial_network=net, edge_pairs=pairs)
    ref = module.reference(
        x, y, refweights, matrix.new_empty(0, module.weight_numel), idx, len(x)
    )
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(
        torch.autograd.grad(out.square().sum(), variables),
        torch.autograd.grad(ref.square().sum(), variables),
    )


@pytest.mark.parametrize("mode", ["last", "full"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_indexed_weight_derivatives(mode, backend, device, double_precision):
    if backend == "cuda" and device != "cuda":
        pytest.skip("CUDA is unavailable")
    module, (x, attrs, radial, matrix), index = convolution_case(device=device)
    module.radial_mode, module.backend = mode, backend
    # Differently sized tables, repeated rows and one table used twice.
    a, b, bias = (
        x.new_empty(shape).normal_().requires_grad_()
        for shape in (
            (2, module.weight_numel),
            (5, module.weight_numel),
            (module.weight_numel,),
        )
    )
    src, dst = index % 2
    ids = index[0] + 1
    terms = ((a, src), (a, dst), (b.square(), ids))
    r, p, net = indexed_weights(radial, matrix, terms, bias=bias)
    result = module(x, attrs, r, p, index, radial_network=net)
    weights = radial @ matrix + bias + a[src] + a[dst] + b[ids].square()
    expected = module.reference(x, attrs, weights, p, index, len(x))
    variables = (x, attrs, radial, matrix, a, b, bias)
    for _ in range(3):
        torch.testing.assert_close(result, expected)
        directions = tuple(torch.randn_like(v) for v in variables)

        def derivative(value):
            grads = torch.autograd.grad(value.sin().sum(), variables, create_graph=True)
            return sum((g * d).sum() for g, d in zip(grads, directions))

        result, expected = derivative(result), derivative(expected)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "categories,edges,mode",
    [(4, 0, "full"), (4, 9, "last"), (89, 9, "full"), (9, 75, "last")],
)
def test_reusable_weight_rows(
    categories, edges, mode, device, double_precision, monkeypatch
):
    module, (x, attrs, radial, matrix), index = convolution_case(
        edges=edges, device=device
    )
    module.radial_mode = "last" if mode == "full" else "full"
    monkeypatch.setattr(config, "RADIAL_PROJECT_MIN_WIDTH", 1)
    if categories == 89:
        monkeypatch.setattr(config, "INDEXED_PROJECTION_MAP_BYTES", 1)
    table = x.new_empty((categories, module.weight_numel)).normal_().requires_grad_()
    rows = indexed_rows(index, categories, dtype=x.dtype, reduce="mean")

    class QuadraticWeights(torch.nn.Module):
        def forward(self, v):
            return v + v.square()

        def to_edge_program(self, inputs, program, v):
            return program.binary("add", v, program.binary("mul", v, v))

    bias = torch.randn_like(matrix[0], requires_grad=True)
    r, p, net = indexed_weights(
        radial,
        matrix,
        [(table, rows)],
        radial_network=QuadraticWeights(),
        bias=bias,
        radial_mode=mode,
    )
    result = module(x, attrs, r, p, index, radial_network=net)
    ref = (
        (radial + radial.square()) @ matrix
        + (table[index[0]] + table[index[1]]) / 2
        + bias
    )
    expected = module.reference(x, attrs, ref, matrix[:0], index, len(x))
    torch.testing.assert_close(result, expected)
    if edges:
        for v in (matrix, table, bias):
            torch.testing.assert_close(
                torch.autograd.grad(result.sum(), v, retain_graph=True)[0],
                torch.autograd.grad(expected.sum(), v, retain_graph=True)[0],
            )


@pytest.mark.parametrize("kind", ["uv", "mtp", "gtp"])
@pytest.mark.parametrize("mode,rows", [("last", 4), ("full", 13)])
def test_indexed_network_interface(kind, mode, rows, device, double_precision):
    from equivariantx import o3 as eqx_o3
    from equivariantx.tests.o2.uv.test_convolution import RadialConvolution

    backend = "cuda" if device == "cuda" else "torch"
    edges = torch.randint(4, (2, 7), device=device)
    if kind == "uv":
        model = RadialConvolution(backend, mode).to(device)
        layer = model.conv
        vectors = torch.randn(7, 3, device=device)
        parameters = (
            model.up.weight,
            model.up.bias,
            model.down.weight,
            model.down.bias,
        )

        def run(x, w, net):
            return layer(
                x,
                None,
                w,
                edges,
                None,
                None,
                w.new_ones((7, 1)),
                parameters,
                vectors=vectors,
                radial_network=net,
            )
    else:
        name = "MatrixTensorProduct" if kind == "mtp" else "GauntTensorProduct"
        tp = getattr(eqx_o3, name)("0e+1o", "0e+1o", "0e+1o+2e", shared_weights=False)
        layer = getattr(eqx_o3, name + "Conv")(
            tp, backend=backend, radial_mode=mode
        ).to(device)
        attrs = torch.randn(7, 4, device=device)

        def run(x, w, net):
            return (
                layer(x, attrs, edges, w)
                if net is None
                else layer(x, attrs, edges, radial=w, radial_network=net)
            )

    x = torch.randn(4, layer.input_dim, device=device)
    r = x.new_empty((7, 3)).normal_().requires_grad_()
    p = x.new_empty((3, layer.weight_numel)).normal_().requires_grad_()
    table = x.new_empty((rows, layer.weight_numel)).normal_().requires_grad_()
    mapping = indexed_rows(edges, rows, dtype=x.dtype, reduce="sum")
    value, network = indexed_weights(r, p, [(table, mapping)]).as_network()
    actual, expected = (
        run(x, value, network),
        run(x, r @ p + table[edges[0]] + table[edges[1]], None),
    )
    torch.testing.assert_close(actual, expected)
    for lhs, rhs in zip(
        torch.autograd.grad(actual.square().sum(), (r, p, table)),
        torch.autograd.grad(expected.square().sum(), (r, p, table)),
    ):
        torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("edges", [0, 7])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_convolution_derivatives(edges, shared, recompute, double_precision):
    module, inputs, index = convolution_case(edges=edges, shared=shared)

    def optimized(*values):
        return convolve(
            module.kernel_metadata,
            *values,
            index,
            3,
            module.output_dim,
            recompute=recompute,
        )

    actual = optimized(*inputs)
    expected = module.reference(*inputs, index, 3)
    torch.testing.assert_close(actual, expected)
    actual, expected = actual.sin().mean(), expected.sin().mean()
    directions = tuple(
        torch.randn_like(value) / max(1, value.numel()) for value in inputs
    )
    for _ in range(4):
        gradients = [
            torch.autograd.grad(value, inputs, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-8, rtol=1e-8)
        actual, expected = [
            sum((g * direction).sum() for g, direction in zip(grads, directions))
            for grads in gradients
        ]
    tangents = tuple(torch.randn_like(value) for value in inputs)
    torch.testing.assert_close(
        torch.func.jvp(optimized, inputs, tangents),
        torch.func.jvp(
            lambda *values: module.reference(*values, index, 3), inputs, tangents
        ),
    )


@pytest.mark.parametrize("width", [64, 128, 1024])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_full_radial_force_training(
    width, shared, backend, monkeypatch, double_precision
):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    module, (x, attrs, radial, _), index = convolution_case(
        width=3,
        shared=shared,
        device=device,
    )
    module.backend = backend
    module.radial_mode = "full"
    network = torch.nn.Sequential(torch.nn.Linear(3, width), torch.nn.SiLU()).to(device)
    projection = torch.randn(width + 1, module.weight_numel, device=device) / width**0.5
    projection.requires_grad_()
    actual = module(x, attrs, radial, projection, index, radial_network=network)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module.backend = "torch"
    expected = module(x, attrs, radial, projection, index, radial_network=network)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    variables = (x, attrs, radial, projection, *network.parameters())
    actual, expected = actual.square().mean(), expected.square().mean()
    for _ in range(3):
        gradients = [
            torch.autograd.grad(value, variables, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-8, rtol=1e-8)
        actual, expected = [
            sum(g.square().mean() for g in grads) / len(grads) for grads in gradients
        ]


def test_partitioned_radial_and_compilation(double_precision):
    module, (x, attrs, radial, projection), index = convolution_case(width=4)
    module.radial_mode = "full"
    network = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.SiLU()).double()
    source = torch.randn(3, 2, requires_grad=True)
    target = torch.randn(3, 2, requires_grad=True)

    def function(x, attrs, radial, projection, source, target):
        partitions = ((radial, "edge"), (source, "source"), (target, "target"))
        return module(x, attrs, partitions, projection, index, radial_network=network)

    inputs = x, attrs, radial, projection, source, target
    expected = module.reference(
        x,
        attrs,
        network(torch.cat((radial, source[index[0]], target[index[1]]), -1)),
        projection,
        index,
        3,
    )
    actual = function(*inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), inputs, create_graph=True),
        torch.autograd.grad(expected.square().sum(), inputs, create_graph=True),
    )
    graph = make_fx(function)(*inputs)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    compiled = torch.compile(graph, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(*inputs), expected)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_bipartite_source_gradient(backend, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    module, inputs, index = convolution_case(device=device)
    module.backend, module.radial_mode = backend, "full"
    index[1] = torch.arange(index.shape[1], device=device) % 5
    actual = module(*inputs, index, 5, radial_network=torch.nn.Sequential())
    expected = module.reference(*inputs, index, 5)
    torch.testing.assert_close(actual, expected)
    for _ in range(3):
        gradients = [
            torch.autograd.grad(value.square().mean(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [
            sum(g.square().mean() for g in grads) / 100 for grads in gradients
        ]


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("radial_mode", ["last", "full"])
def test_dynamic_radial_program(backend, radial_mode, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.compiler.reset()
    device = "cuda" if backend == "cuda" else "cpu"
    module, _, _ = convolution_case(device=device)
    module.backend, module.radial_mode = backend, radial_mode
    network = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.SiLU()).to(device)

    def function(x, attrs, radial, projection, index):
        return module(x, attrs, radial, projection, index, radial_network=network)

    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(function, backend=compiler, dynamic=True, fullgraph=True)
    projection = torch.randn(8, module.weight_numel, device=device, requires_grad=True)
    for edges in (7, 11, 9):
        _, (x, attrs, radial, _), index = convolution_case(
            width=3, edges=edges, device=device
        )
        # Edge counts must stay independent of coincidentally equal model widths.
        for value, dimension in ((attrs, 0), (radial, 0), (index, 1)):
            torch._dynamo.mark_dynamic(value, dimension, min=2)
        actual = compiled(x, attrs, radial, projection, index)
        expected = function(x, attrs, radial, projection, index)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().mean(), projection)[0],
            torch.autograd.grad(expected.square().mean(), projection)[0],
        )
    assert len(graphs) == 1
    _, (empty_x, empty_attrs, empty_radial, _), empty_index = convolution_case(
        width=3, edges=0, device=device
    )
    empty_inputs = empty_x, empty_attrs, empty_radial, projection, empty_index
    actual = compiled(*empty_inputs)
    expected = function(*empty_inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().mean(), projection)[0],
        torch.autograd.grad(expected.square().mean(), projection)[0],
    )
    assert len(graphs) == 2
    torch.testing.assert_close(
        compiled(x, attrs, radial, projection, index),
        function(x, attrs, radial, projection, index),
    )
    assert len(graphs) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor package")
@pytest.mark.parametrize("radial_mode", ["last", "full"])
@pytest.mark.parametrize("width,bias", [(8, False), (1024, True)])
def test_force_gradient_aoti(
    radial_mode, width, bias, tmp_path, double_precision, monkeypatch
):
    import os
    import subprocess
    import sys

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, _, _ = convolution_case(device="cuda")
            self.conv.radial_mode = radial_mode
            self.network = torch.nn.Sequential(
                torch.nn.Linear(3, width), torch.nn.SiLU()
            ).cuda()

        def forward(self, x, attrs, radial, projection, index):
            return self.conv(
                x, attrs, radial, projection, index, radial_network=self.network
            )

    model = Model()
    _, (x, attrs, radial, _), index = convolution_case(width=3, device="cuda")
    projection = torch.randn(
        width + int(bias), model.conv.weight_numel, device="cuda", requires_grad=True
    )
    names, parameters = zip(*model.named_parameters())

    def joint(x, attrs, radial, projection, index, *parameters):
        prediction = torch.func.functional_call(
            model, dict(zip(names, parameters)), (x, attrs, radial, projection, index)
        )
        energy = prediction.square().mean()
        forces = -torch.autograd.grad(energy, radial, create_graph=True)[0]
        loss = (forces - 0.1).square().mean() + energy
        gradients = torch.autograd.grad(loss, (projection, *parameters))
        return loss, forces, *gradients

    inputs = (x, attrs, radial, projection, index, *parameters)
    graph = make_fx(joint)(*inputs)
    detached = tuple(value.detach() for value in inputs)
    exported = torch.export.export(graph, detached, strict=True)
    package = tmp_path / "convolution.pt2"
    compile_and_package(exported, package_path=str(package))
    # A packaged training graph must accept updated weights without recompiling.
    cases = []
    for scale in (1.0, 0.8):
        values = tuple(
            (
                value.detach() * scale if value.is_floating_point() else value.detach()
            ).requires_grad_(value.requires_grad)
            for value in inputs
        )
        with monkeypatch.context() as reference:
            reference.setattr(config, "TORCH_REFERENCE", True)
            reference.setattr(model.conv, "backend", "torch")
            expected = tuple(value.detach().cpu() for value in joint(*values))
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in values),
                "expected": expected,
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
        actual = runner(*(value.cuda() for value in case['inputs']))
    for value, expected in zip(actual, case['expected']):
        torch.testing.assert_close(value.cpu(), expected, atol=1e-8, rtol=1e-8)
assert 'equivariantx' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", package_script(script), str(package), str(data)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor package")
def test_dynamic_cuda_force_gradient_aoti(tmp_path, double_precision, monkeypatch):
    import subprocess
    import sys

    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, _, _ = convolution_case(device="cuda")
            self.conv.backend, self.conv.radial_mode = "cuda", "full"
            self.network = torch.nn.Sequential(
                torch.nn.Linear(3, 8), torch.nn.SiLU()
            ).cuda()

        def forward(self, x, attrs, radial, projection, index):
            return self.conv(
                x, attrs, radial, projection, index, radial_network=self.network
            )

    model = Model()
    names, parameters = zip(*model.named_parameters())

    def joint(*values):
        x, attrs, radial, projection, index, *parameters = values
        prediction = torch.func.functional_call(
            model, dict(zip(names, parameters)), (x, attrs, radial, projection, index)
        )
        energy = prediction.square().mean()
        force = -torch.autograd.grad(energy, radial, create_graph=True)[0]
        loss = (force - 0.1).square().mean() + energy
        return loss, force, *torch.autograd.grad(loss, (projection, *parameters))

    _, (x, attrs, radial, _), index = convolution_case(width=3, device="cuda")
    projection = torch.randn(
        8, model.conv.weight_numel, device="cuda", requires_grad=True
    )
    inputs = x, attrs, radial, projection, index, *parameters
    edges = ShapeVar("edges", min=2, optimization_hint=7)
    dynamic_axes = {1: 0, 2: 0, 4: 1}
    specs = []
    for i, value in enumerate(inputs):
        shape = [None] * value.ndim
        if i in dynamic_axes:
            shape[dynamic_axes[i]] = edges
        specs.append(TensorSpec(shape))
    graph = make_fx(
        joint,
        decomposition_table=torch._decomp.get_decompositions(
            [
                torch.ops.aten.constant_pad_nd.default,
                torch.ops.aten.silu_backward.default,
            ]
        ),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec({"*args": specs})),
    )(*inputs)
    (tmp_path / "graph.py").write_text(graph.code)
    count = torch.export.Dim("edges", min=2, max=1024)
    detached = tuple(value.detach() for value in inputs)
    exported = torch.export.export(
        graph,
        detached,
        strict=True,
        dynamic_shapes=tuple(
            {dynamic_axes[i]: count} if i in dynamic_axes else None
            for i in range(len(inputs))
        ),
    )
    package = tmp_path / "dynamic-convolution.pt2"
    compile_and_package(exported, package_path=str(package))
    cases = []
    state = projection, *parameters
    for edges in (7, 11, 9):
        _, (x, attrs, radial, _), index = convolution_case(
            width=3, edges=edges, device="cuda"
        )
        values = x, attrs, radial, state[0], index, *state[1:]
        with monkeypatch.context() as reference:
            reference.setattr(config, "TORCH_REFERENCE", True)
            reference.setattr(model.conv, "backend", "torch")
            expected = tuple(value.detach().cpu() for value in joint(*values))
        cases.append(
            dict(
                inputs=tuple(value.detach().cpu() for value in values),
                expected=expected,
            )
        )
        state = tuple(
            (value - 0.001 * gradient.to(value)).detach().requires_grad_()
            for value, gradient in zip(state, expected[2:])
        )
    data = tmp_path / "cases.pt"
    torch.save(cases, data)
    script = """
import sys
import torch
runner = torch._inductor.aoti_load_package(sys.argv[1])
for case in torch.load(sys.argv[2], weights_only=True):
    actual = runner(*(value.cuda() for value in case['inputs']))
    for value, expected in zip(actual, case['expected']):
        torch.testing.assert_close(value.cpu(), expected, atol=1e-8, rtol=1e-8)
assert 'equivariantx' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", package_script(script), str(package), str(data)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
