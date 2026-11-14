"""Shared independent checks for spherical and Cartesian ACE interfaces."""

import pytest
import torch

from equivariantx.mbe.ace import (
    IctdBilinearContraction,
    IctdUuuFullyConnectedSymmetricContraction,
)


def check_cpu_dynamic_coefficients(space, kind, monkeypatch, experts=1, dynamic=True):
    """Default wide coefficient maps, shared GEMMs and empty force-loss graphs."""
    from e3nn import o3 as spherical
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx import co3, o3
    from equivariantx.mbe import ace
    from equivariantx.ops import config

    namespace = spherical if space == "o3" else co3
    product_type = o3.TensorProduct if space == "o3" else co3.TensorProduct
    weighted = kind == "bilinear"
    channels = 64 * experts
    irreps = f"{channels}x0e+{channels}x1o"
    options = dict(internal_weights=False, shared_weights=False)
    product = product_type(
        irreps,
        irreps,
        irreps,
        [(0, 0, 0, "uuu", weighted), (0, 1, 1, "uuu", weighted)],
        **options,
    )
    linear = namespace.Linear("64x0e+64x1o", "64x0e+64x1o", **options)
    if weighted:
        shared = namespace.Linear(irreps, irreps, **options)
        cls = ace.BilinearContraction if space == "o3" else ace.IctdBilinearContraction
        module = cls(product, linear, experts, shared_linear=shared)
        sizes = (
            product.weight_numel,
            3 * experts * linear.weight_numel,
            shared.weight_numel,
        )
        weight_shapes = [(3, experts, linear.weight_numel), (shared.weight_numel,)]
    else:
        cls = (
            ace.UuuFullyConnectedSymmetricContraction
            if space == "o3"
            else ace.IctdUuuFullyConnectedSymmetricContraction
        )
        module = cls([product], [linear, linear])
        weight_shapes = [(3, linear.weight_numel), (3, linear.weight_numel)]
    module = module.double()
    weights = tuple(
        (torch.randn(shape, dtype=torch.float64) * 0.1).requires_grad_()
        for shape in weight_shapes
    )

    def data(nodes):
        x = (
            torch.randn(nodes, channels * 4, dtype=torch.float64) * 0.1
        ).requires_grad_()
        if weighted:
            gate = (
                torch.randn(nodes, sizes[0], dtype=torch.float64) * 0.1
            ).requires_grad_()
            return x, gate, *weights, torch.arange(nodes) % 3
        return x, *weights, torch.arange(nodes) % 3

    def joint(*args):
        x, *parameters, types = args
        if weighted:
            gate, weight, shared_weight = parameters
            output = module(x, x.cos(), gate, weight, types, shared_weight)
        else:
            output = module(x, parameters, types)
        energy = 1 + output.sin().sum() / (x.shape[0] + 1)
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force - 0.01).square().sum()
        return energy, force, *torch.autograd.grad(loss, parameters)

    count = ShapeVar("nodes", min=2, optimization_hint=7)
    values = data(7)
    specs = [TensorSpec([count, None])]
    if weighted:
        specs.append(TensorSpec([count, None]))
    specs.extend(TensorSpec([None] * len(shape)) for shape in weight_shapes)
    specs.append(TensorSpec([count]))
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    if dynamic:
        graph = make_fx(
            joint,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
            dynamic_shapes=ShapesSpec(ParamsSpec({"*args": specs})),
        )(*values)
        assert not any(
            node.target == torch.ops.aten.searchsorted.Tensor
            for node in graph.graph.nodes
        )
    else:
        graph = make_fx(joint)(*data(256))
        assert any(
            node.target == torch.ops.aten.searchsorted.Tensor
            for node in graph.graph.nodes
        )
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert not any(
        node.target == torch.ops.aten.sym_numel.default for node in graph.graph.nodes
    )
    compiled_graphs = []

    def backend(module, inputs):
        from torch._inductor.compile_fx import compile_fx

        compiled_graphs.append(module)
        return compile_fx(module, inputs)

    run = torch.compile(graph, fullgraph=True, dynamic=dynamic, backend=backend)
    for nodes in (7, 11, 5, 0) if dynamic else (256,):
        if not nodes:
            assert len(compiled_graphs) == 1
            empty_graph = make_fx(joint)(*data(0))
            run = torch.compile(empty_graph, fullgraph=True)
        for update in range(3):
            values = data(nodes)
            with torch.no_grad():
                for weight in weights:
                    weight.add_(0.001 * (update + 1))
            detached = tuple(value.detach() for value in values)
            if nodes and dynamic:
                for i, value in enumerate(detached):
                    if i == 0 or (weighted and i == 1) or i == len(detached) - 1:
                        torch._dynamo.mark_dynamic(value, 0, min=2)
                        for axis in range(1, value.ndim):
                            torch._dynamo.mark_static(value, axis)
                    else:
                        torch._dynamo.mark_static(value)
            actual = run(*detached)
            with monkeypatch.context() as flags:
                flags.setattr(config, "TORCH_REFERENCE", True)
                expected = joint(*values)
            torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


def check_parent_provider(
    space,
    kind,
    device,
    nodes,
    child_backend,
    reference,
    monkeypatch,
    *,
    shared_weights=False,
    output_basis="cartesian",
    backend="torch",
):
    from e3nn import o3 as spherical
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx import co3, o3
    from equivariantx.mbe import ace
    from equivariantx.ops import config

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(641)
    namespace = spherical if space == "o3" else co3
    product_type = o3.TensorProduct if space == "o3" else co3.TensorProduct
    weighted = kind == "bilinear"
    irreps = namespace.Irreps("4x0e+4x1o+4x2e")
    instructions = [
        (0, 0, 0, "uuu", weighted),
        (1, 1, 2, "uuu", weighted),
        (0, 1, 1, "uuu", weighted),
        (2, 0, 2, "uuu", weighted),
    ]
    options = dict(internal_weights=False, shared_weights=False)
    linear_options = (
        dict(options, output_basis=output_basis) if space == "co3" else options
    )
    product = product_type(
        irreps,
        irreps,
        irreps,
        instructions,
        backend=child_backend,
        **dict(options, shared_weights=shared_weights),
    )
    oracle_product = namespace.TensorProduct(
        irreps, irreps, irreps, instructions, **options
    ).to(device)
    transform = (
        torch.nn.Identity()
        if space == "o3"
        else co3.ChangeOfBasis(irreps, backend="torch").to(device)
    )
    source_dim = irreps.dim if space == "o3" else irreps.spherical().dim
    types = torch.arange(nodes, device=device) % 3

    def variable(*shape):
        return (torch.randn(*shape, device=device) * 0.2).requires_grad_()

    if weighted:
        linear = namespace.Linear(
            "2x0e+2x1o+2x2e",
            "3x0e+3x1o+3x2e",
            **dict(linear_options, shared_weights=shared_weights),
        )
        oracle_linear = namespace.Linear(
            "2x0e+2x1o+2x2e", "3x0e+3x1o+3x2e", **linear_options
        ).to(device)
        shared = namespace.Linear(irreps, "6x0e+6x1o+6x2e", **linear_options).to(device)
        cls = ace.BilinearContraction if space == "o3" else ace.IctdBilinearContraction
        module = cls(product, linear, 2, backend="cuda", shared_linear=shared).to(
            device
        )
        inputs = (
            variable(nodes, source_dim),
            variable(nodes, source_dim),
            variable(nodes, product.weight_numel),
            variable(3, 2, linear.weight_numel),
            variable(shared.weight_numel),
        )

        def run(x, y, gates, weight, shared_weight, indices):
            return module(
                transform(x), transform(y), gates, weight, indices, shared_weight
            )

        def oracle(x, y, gates, weight, shared_weight, indices):
            features = oracle_product(transform(x), transform(y), gates)
            parts = [
                features[:, section].reshape(nodes, 2, mul // 2, ir.dim)
                for (mul, ir), section in zip(irreps, irreps.slices())
            ]
            results = [
                oracle_linear(
                    torch.cat([part[:, expert].flatten(1) for part in parts], -1),
                    weight[indices, expert],
                )
                for expert in range(2)
            ]
            output = torch.cat(
                [
                    torch.stack([value[:, section] for value in results], 1).flatten(1)
                    for section in linear.irreps_out.slices()
                ],
                -1,
            )
            return output + shared(features, shared_weight.expand(nodes, -1))
    else:
        linears = [
            namespace.Linear(
                irreps,
                "3x0e+3x1o+3x2e",
                **dict(linear_options, shared_weights=shared_weights),
            ).to(device)
            for _ in range(3)
        ]
        oracle_linears = [
            namespace.Linear(irreps, "3x0e+3x1o+3x2e", **linear_options).to(device)
            for _ in range(3)
        ]
        cls = (
            ace.UuuFullyConnectedSymmetricContraction
            if space == "o3"
            else ace.IctdUuuFullyConnectedSymmetricContraction
        )
        module = cls([product, product], linears, backend="cuda").to(device)
        inputs = (
            variable(nodes, source_dim),
            *(variable(3, linear.weight_numel) for linear in linears),
        )

        def run(x, *args):
            return module(transform(x), args[:-1], args[-1])

        def oracle(x, *args):
            x = transform(x)
            weights, indices = args[:-1], args[-1]
            result = oracle_linears[0](x, weights[0][indices])
            value = x
            for linear, weight in zip(oracle_linears[1:], weights[1:]):
                value = oracle_product(value, x)
                result = result + linear(value, weight[indices])
            return result

    module.backend = backend
    with monkeypatch.context() as ordinary:
        ordinary.setattr(config, "TORCH_REFERENCE", True)
        expected = oracle(*inputs, types)
    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    actual = run(*inputs, types)
    assert actual.shape == (nodes, module.output_dim)
    assert not module.state_dict()
    if space == "co3":
        descriptors = (linear, shared) if weighted else linears
        assert all(linear.output_basis == output_basis for linear in descriptors)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        actual, expected = [
            torch.cat(
                [
                    value.flatten()
                    for value in torch.autograd.grad(
                        output.sin().sum(), inputs, create_graph=True
                    )
                ]
            )
            / 20
            for output in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    graph = make_fx(run)(*inputs, types)
    if backend == "torch" or device == "cpu":
        assert not any(
            str(node.target).startswith("eqx.") for node in graph.graph.nodes
        )
    changed = tuple(value.detach() * 0.8 for value in inputs)
    with monkeypatch.context() as ordinary:
        ordinary.setattr(config, "TORCH_REFERENCE", True)
        expected = oracle(*changed, types.roll(1))
    torch.testing.assert_close(
        graph(*changed, types.roll(1)), expected, atol=2e-10, rtol=2e-10
    )
    assert product.backend == child_backend
    assert product.shared_weights == shared_weights
    if (shared_weights or output_basis == "spherical") and nodes and not reference:
        tangents = tuple(torch.randn_like(value) * 0.1 for value in inputs)

        def transformed(function):
            def value(*args):
                return function(*args, types)

            def derivative(*args):
                return torch.func.jvp(value, args, tangents)[1]

            gradient = torch.func.grad(
                lambda *args: value(*args).sin().sum(),
                argnums=tuple(range(len(inputs))),
            )
            return (
                torch.func.jvp(derivative, inputs, tangents),
                torch.func.jvp(gradient, inputs, tangents),
                torch.vmap(value)(
                    *[torch.stack((argument, argument * 0.8)) for argument in inputs]
                ),
            )

        actual = transformed(run)
        with monkeypatch.context() as ordinary:
            ordinary.setattr(config, "TORCH_REFERENCE", True)
            expected = transformed(oracle)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    return run, oracle, inputs, types


def check_spherical_compile(kind, backend, monkeypatch):
    from equivariantx.ops import config

    torch.compiler.reset()
    run, oracle, inputs, types = check_parent_provider(
        "co3",
        kind,
        "cuda",
        7,
        "cuda",
        False,
        monkeypatch,
        shared_weights=True,
        output_basis="spherical",
        backend=backend,
    )
    compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
    for scale in (1.0, 0.8):
        changed = tuple((value.detach() * scale).requires_grad_() for value in inputs)
        indices = types if scale == 1 else types.roll(1)
        actual = compiled(*changed, indices)
        with monkeypatch.context() as ordinary:
            ordinary.setattr(config, "TORCH_REFERENCE", True)
            expected = oracle(*changed, indices)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        gradients = [
            torch.autograd.grad(value.sin().sum(), changed)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=2e-10, rtol=2e-10)


def check_parent_aoti(
    space,
    kind,
    tmp_path,
    monkeypatch,
    *,
    shared_weights=False,
    output_basis="cartesian",
    backend="torch",
):
    import subprocess
    import sys

    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config
    from equivariantx.ops.decompositions import torch_decompositions

    run, oracle, inputs, types = check_parent_provider(
        space,
        kind,
        "cuda",
        7,
        "cuda",
        False,
        monkeypatch,
        shared_weights=shared_weights,
        output_basis=output_basis,
        backend=backend,
    )

    def joint(function, *values):
        energy = function(*values).square().mean()
        force = -torch.autograd.grad(energy, values[0], create_graph=True)[0]
        loss = energy + (force - 0.01).square().mean()
        return loss, force, *torch.autograd.grad(loss, values[1:-1])

    graph = make_fx(
        lambda *values: joint(run, *values),
        decomposition_table=torch_decompositions(),
    )(*inputs, types)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    exported = torch.export.export(
        graph, (*[value.detach() for value in inputs], types), strict=True
    )
    package = tmp_path / "ace.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale in (1.0, 0.8):
        changed = tuple((value.detach() * scale).requires_grad_() for value in inputs)
        indices = types if scale == 1 else types.roll(1)
        with monkeypatch.context() as ordinary:
            ordinary.setattr(config, "TORCH_REFERENCE", True)
            expected = tuple(
                value.detach().cpu() for value in joint(oracle, *changed, indices)
            )
        cases.append(
            dict(
                inputs=(*[value.detach().cpu() for value in changed], indices.cpu()),
                expected=expected,
            )
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
    torch.testing.assert_close(tuple(value.cpu() for value in actual), case['expected'], atol=2e-9, rtol=2e-9)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def check_ictd_contraction(kind, monkeypatch):
    from equivariantx import co3

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if kind == "ace_split":
        from equivariantx.ops.common.cuda import ace

        batches = ace.schedule_batches
        monkeypatch.setattr(ace, "schedule_batches", lambda groups: batches(groups, 7))
        monkeypatch.setattr(ace, "weighted_source", lambda *args: None)
    elif kind.endswith("_split"):
        from equivariantx.ops.common.cuda import bilinear

        batches = bilinear.metadata_batches
        monkeypatch.setattr(
            bilinear, "metadata_batches", lambda value: batches(value, 1)
        )
    channels = 256 if kind == "bilinear_wide" else 4
    irreps = co3.Irreps(f"{channels}x0e+{channels}x1o+{channels}x2e")
    output = co3.Irreps(f"{channels}x0e+{channels}x1e+{channels}x2e+{channels}x3o")
    weighted = not kind.startswith("ace")
    tp = co3.TensorProduct(
        irreps,
        irreps,
        output,
        [
            (1, 1, 0, "uuu", weighted),
            (1, 1, 1, "uuu", weighted),
            (1, 1, 2, "uuu", weighted),
            (1, 2, 3, "uuu", weighted),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    experts = 2 if kind.startswith("moe") else 1
    coefficient = co3.Linear(
        [(mul // experts, ir) for mul, ir in output],
        [(mul // experts, ir) for mul, ir in output],
        internal_weights=False,
        shared_weights=False,
    )
    if weighted:
        modules = [
            IctdBilinearContraction(tp, coefficient, experts, backend=backend)
            for backend in ("torch", "cuda")
        ]
    else:
        first = co3.Linear(irreps, output, internal_weights=False, shared_weights=False)
        modules = [
            IctdUuuFullyConnectedSymmetricContraction(
                [tp], [first, coefficient], backend=backend
            )
            for backend in ("torch", "cuda")
        ]
    modules = [module.cuda() for module in modules]
    transform = co3.ChangeOfBasis(irreps).cuda()
    source = torch.randn(5, irreps.spherical().dim, device="cuda", requires_grad=True)
    features = transform(source)
    types = torch.tensor([0, 1, 0, 2, 1], device="cuda")
    if weighted:
        gates = torch.randn(5, tp.weight_numel, device="cuda", requires_grad=True)
        weight = torch.randn(
            3, experts, coefficient.weight_numel, device="cuda", requires_grad=True
        )
        inputs = (source, gates, weight)
        outputs = [
            module(features, features, gates, weight, types) for module in modules
        ]
    else:
        weights = [
            torch.randn(3, linear.weight_numel, device="cuda", requires_grad=True)
            for linear in (first, coefficient)
        ]
        inputs = (source, *weights)
        outputs = [module(features, weights, types) for module in modules]
    torch.testing.assert_close(*outputs, atol=1e-11, rtol=1e-11)
    losses = [output.sin().sum() for output in outputs]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for expected, actual in zip(*gradients):
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        tangents = [torch.randn_like(value) for value in inputs]
        losses = [
            sum((value * tangent).sum() for value, tangent in zip(grad, tangents))
            for grad in gradients
        ]
