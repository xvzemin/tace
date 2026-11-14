"""Uuu symmetric contractions with and without channel mixing."""

import pytest
import torch
from e3nn import o3

from equivariantx.mbe.ace import (
    UuuFullyConnectedSymmetricContraction,
    UuuSymmetricContraction,
)
from equivariantx.ops import config

from .testing import (
    check_cpu_dynamic_coefficients,
    check_parent_aoti,
    check_parent_provider,
)


def test_compiler_visible_derivatives(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    module, x, weight, types = contraction()
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", True)

    def joint(x, weight, types):
        energy = module(x, weight, types).square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        gradient = torch.autograd.grad(force.square().sum(), weight)[0]
        return energy, force, gradient

    graph = make_fx(joint)(x, weight, types)
    assert all(not str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    compiled = torch.compile(graph, fullgraph=True, backend="inductor")
    torch.testing.assert_close(
        compiled(x, weight, types), joint(x, weight, types), atol=1e-8, rtol=1e-8
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unsupported_precision(dtype):
    module, x, weight, types = contraction()
    with pytest.raises(TypeError, match="float32 or float64"):
        module.to(dtype)(x.to(dtype), weight.to(dtype), types)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("compiler", [None, "aot_eager"])
def test_gradcheck_and_transforms(monkeypatch, backend, compiler):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", True)
    module, x, weight, types = contraction("cuda" if backend == "cuda" else "cpu")
    module.backend = backend

    def function(x, weight):
        return module(x, weight, types)

    assert torch.autograd.gradcheck(function, (x, weight), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (x, weight), fast_mode=True)
    directions = (torch.randn_like(x), torch.randn_like(weight))

    def jvp(x, weight):
        return torch.func.jvp(function, (x, weight), directions)

    torch.compiler.reset()
    run = torch.compile(jvp, backend=compiler, fullgraph=True) if compiler else jvp
    actual = run(x, weight)
    batched = torch.vmap(function)(
        torch.stack((x, x * 0.3)), torch.stack((weight, weight * 0.7))
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module.backend = "torch"
    torch.testing.assert_close(
        actual, torch.func.jvp(function, (x, weight), directions)
    )
    torch.testing.assert_close(
        batched,
        torch.vmap(function)(
            torch.stack((x, x * 0.3)), torch.stack((weight, weight * 0.7))
        ),
    )


def test_functional_constants_do_not_escape():
    from equivariantx.ops.common.torch.symmetric import polynomial_indices

    polynomial_indices.cache_clear()
    module, x, weight, types = contraction()

    def gradient(x, weight):
        return torch.func.grad(
            lambda x, weight: module(x, weight, types).sin().sum(), argnums=(0, 1)
        )(x, weight)

    directions = torch.randn_like(x), torch.randn_like(weight)
    for scale in (1.0, 0.7, 1.0):
        values = x * scale, weight * scale
        actual, _ = torch.func.jvp(gradient, values, directions)
        expected = torch.autograd.grad(module(*values, types).sin().sum(), values)
        torch.testing.assert_close(actual, expected)
    polynomial_indices.cache_clear()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_and_four_derivatives(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module, x, weight, types = contraction(device)
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", True)
    results = []
    for reference in (False, True):
        monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
        results.append(module(x, weight, types))
    actual, expected = results
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        actual, expected = [
            torch.cat(
                [
                    value.flatten()
                    for value in torch.autograd.grad(
                        result.sin().sum(), (x, weight), create_graph=True
                    )
                ]
            )
            / 10
            for result in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


def contraction(device="cpu"):
    entries = [
        ((0,), 0, 0, 0.3),
        ((1, 1), 1, 0, 0.7),
        ((0, 1, 2), 2, 0, -0.2),
        ((2, 0, 1), 2, 0, 0.1),
        ((0, 0, 2, 2), 3, 0, 0.5),
    ]
    module = UuuSymmetricContraction.from_entries("2x1o", "2x0e", entries, 4).to(
        device, torch.float64
    )
    x = torch.randn(3, 2, 3, device=device, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(
        2, 4, 2, device=device, dtype=torch.float64, requires_grad=True
    )
    types = torch.tensor([1, 0, 1], device=device)
    return module, x, weight, types


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_uuu_symmetric_contraction(device, dtype, backend):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(9)
    coupling = torch.randn(3, 4, 4, 4, 2, dtype=torch.float64)
    first = torch.randn(1, 4, 1, dtype=torch.float64)
    module = UuuSymmetricContraction(
        "3x0e+3x1o",
        "3x0e+3x1o",
        [(0, 1, first), (1, 3, coupling)],
        backend=backend,
    ).to(device, dtype)
    x = torch.randn(4, 3, 4, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(
        2, module.num_paths, 3, device=device, dtype=dtype, requires_grad=True
    )
    assert module.num_paths == 3
    node_type = torch.tensor([0, 1, 1, 0], device=device)
    expected = torch.cat(
        (
            torch.einsum(
                "ak,nkc,nca->nc",
                first[0].to(device, dtype),
                weight[node_type, :1],
                x,
            ),
            torch.einsum(
                "mabck,nku,nua,nub,nuc->num",
                coupling.to(device, dtype),
                weight[node_type, 1:],
                x,
                x,
                x,
            ).flatten(1),
        ),
        dim=1,
    )
    actual = module(x, weight, node_type)
    atol, rtol = (1e-9, 1e-9) if dtype == torch.float64 else (3e-4, 1e-4)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for _ in range(3):
        vector = torch.randn_like(actual)
        actual = torch.cat(
            [
                g.flatten()
                for g in torch.autograd.grad(
                    (actual * vector).sum(), (x, weight), create_graph=True
                )
            ]
        )
        expected = torch.cat(
            [
                g.flatten()
                for g in torch.autograd.grad(
                    (expected * vector).sum(), (x, weight), create_graph=True
                )
            ]
        )
        torch.testing.assert_close(actual, expected, atol=atol * 10, rtol=rtol * 10)
    torch.compiler.reset()
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for nodes in (2, 0):
        a = x[:nodes].detach().requires_grad_()
        torch.testing.assert_close(
            compiled(a, weight, node_type[:nodes]),
            module(a, weight, node_type[:nodes]),
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nodes", [0, 5, 65])
def test_symmetric_shared_polynomials(dtype, nodes, wigner_3j):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cg = wigner_3j(2, 4, 4, dtype=torch.float64) * 5**0.5
    entries = []
    for output, left, right in cg.nonzero().tolist():
        value = cg[output, left, right].item()
        for path, scalar_factors, scale in (
            (0, 0, 1),
            (1, 1, 1),
            (2, 1, -0.75),
            (3, 2, 1),
        ):
            entries.append(
                (
                    (0,) * scalar_factors + (left + 1, right + 1),
                    path,
                    output,
                    scale * value,
                )
            )
    module = UuuSymmetricContraction.from_entries(
        "3x0e+3x4e", "3x2e", entries, 4, backend="cuda"
    ).to("cuda", dtype)
    x = (torch.randn(nodes, 3, 20, device="cuda", dtype=dtype) * 0.2)[
        ..., ::2
    ].requires_grad_()
    weights = torch.randn(3, 4, 3, device="cuda", dtype=dtype, requires_grad=True)
    species = torch.arange(nodes, device="cuda", dtype=torch.int32) % 3
    actual = module(x, weights, species)
    module.backend = "torch"
    expected = module(x, weights, species)
    tolerance = 2e-10 if dtype == torch.float64 else 5e-4
    for order in range(5):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        if order == 4 or (not nodes and order == 1):
            break
        direction = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    part.flatten()
                    for part in torch.autograd.grad(
                        (value * direction).sum(), (x, weights), create_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]


@pytest.mark.parametrize("space", ["o3", "co3"])
def test_cpu_dynamic_coefficients(space, monkeypatch):
    check_cpu_dynamic_coefficients(space, "uuu_fully_connected", monkeypatch)


def test_cpu_bounded_coefficients(monkeypatch):
    check_cpu_dynamic_coefficients(
        "co3", "uuu_fully_connected", monkeypatch, dynamic=False
    )


@pytest.mark.parametrize("shared_weights", [False, True])
def test_force_gradient_aoti(shared_weights, tmp_path, double_precision, monkeypatch):
    check_parent_aoti(
        "o3",
        "uuu_fully_connected",
        tmp_path,
        monkeypatch,
        shared_weights=shared_weights,
    )


@pytest.mark.parametrize("device,nodes", [("cpu", 0), ("cpu", 7), ("cuda", 7)])
@pytest.mark.parametrize("reference", [False, True])
def test_shared_descriptors(device, nodes, reference, double_precision, monkeypatch):
    check_parent_provider(
        "o3",
        "uuu_fully_connected",
        device,
        nodes,
        "torch",
        reference,
        monkeypatch,
        shared_weights=True,
    )


@pytest.mark.parametrize("device,nodes", [("cpu", 0), ("cpu", 7), ("cuda", 7)])
@pytest.mark.parametrize("child_backend", ["torch", "cuda"])
@pytest.mark.parametrize("reference", [False, True])
def test_parent_provider(
    device, nodes, child_backend, reference, double_precision, monkeypatch
):
    check_parent_provider(
        "o3",
        "uuu_fully_connected",
        device,
        nodes,
        child_backend,
        reference,
        monkeypatch,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Packed CUDA GEMMs")
@pytest.mark.parametrize(
    "space,project", [("o3", False), ("co3", False), ("co3", True)]
)
def test_packed_coefficient_maps(space, project, monkeypatch, double_precision):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx import co3
    from equivariantx.mbe.ace import IctdUuuFullyConnectedSymmetricContraction
    from equivariantx.ops import config

    namespace = o3 if space == "o3" else co3
    contraction = (
        UuuFullyConnectedSymmetricContraction
        if space == "o3"
        else IctdUuuFullyConnectedSymmetricContraction
    )
    irreps = namespace.Irreps("128x0e+128x2e")
    product = namespace.TensorProduct(
        irreps,
        irreps,
        irreps,
        [(0, 0, 0, "uuu", False), (1, 0, 1, "uuu", False)],
        internal_weights=False,
        shared_weights=False,
    )
    options = {} if space == "o3" else {"project": project}
    linears = [
        namespace.Linear(
            irreps, irreps, internal_weights=False, shared_weights=False, **options
        )
        for _ in range(2)
    ]
    module = contraction([product], linears).cuda()
    inputs = (
        (torch.randn(512, irreps.dim, device="cuda") / 3).requires_grad_(),
        *(
            torch.randn(4, linear.weight_numel, device="cuda", requires_grad=True)
            for linear in linears
        ),
    )
    types = torch.arange(512, device="cuda") % 4

    def joint(x, *weights):
        energy = module(x, weights, types).square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy + force.square().sum()
        return energy, force, *torch.autograd.grad(loss, weights)

    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = joint(*inputs)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    graph = make_fx(joint)(*inputs)
    torch.testing.assert_close(graph(*inputs), expected, atol=2e-10, rtol=2e-10)
    compiled = torch.compile(graph, fullgraph=True)
    torch.testing.assert_close(compiled(*inputs), expected, atol=2e-10, rtol=2e-10)
    assert not module.state_dict()


def test_native_cuda_graph_ace(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.common.cuda.graph as cuda_graph

    tp = o3.TensorProduct(
        "2x0e",
        "2x0e",
        "2x0e",
        [(0, 0, 0, "uuu", False)],
        internal_weights=False,
        shared_weights=False,
    )
    linears = [
        o3.Linear("2x0e", "3x0e", internal_weights=False, shared_weights=False)
        for _ in range(3)
    ]
    module = (
        UuuFullyConnectedSymmetricContraction([tp, tp], linears, backend="cuda")
        .cuda()
        .double()
    )
    cuda_graph._GRAPHS.clear()
    try:
        for seed in (2, 3):
            torch.manual_seed(seed)
            types = torch.randint(2, (7,), device="cuda")
            inputs = [
                torch.randn(
                    7, 2, device="cuda", dtype=torch.float64, requires_grad=True
                )
            ]
            inputs += [
                torch.randn(
                    2, 6, device="cuda", dtype=torch.float64, requires_grad=True
                )
                for _ in linears
            ]
            results = []
            for enabled in (False, True):
                monkeypatch.setenv("EQX_USE_CUDA_GRAPH", str(int(enabled)))
                value = module(inputs[0], inputs[1:], types)
                result = [value]
                for _ in range(3):
                    grads = torch.autograd.grad(
                        value.sin().sum(), inputs, create_graph=True
                    )
                    result.extend(grads)
                    value = torch.cat([g.flatten() for g in grads]) / 10
                results.append(result)
            for a, b in zip(*results):
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        assert cuda_graph._GRAPHS
    finally:
        cuda_graph._GRAPHS.clear()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_nodes", [0, 3, 9])
@pytest.mark.parametrize("channels_in,channels_out,degree", [(2, 2, 1), (33, 17, 5)])
def test_uuu_fully_connected_contraction(
    device, num_nodes, channels_in, channels_out, degree
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    irreps = o3.Irreps([(channels_in, (0, 1)), (channels_in, (degree, (-1) ** degree))])
    tp = o3.TensorProduct(
        irreps,
        irreps,
        irreps[:1] + irreps[:1] + irreps[1:] + irreps[1:],
        [
            (0, 0, 0, "uuu", False),
            (1, 1, 1, "uuu", False),
            (0, 1, 2, "uuu", False),
            (1, 0, 3, "uuu", False),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    irreps_out = o3.Irreps([(channels_out, ir) for _, ir in irreps])
    linears = [
        o3.Linear(inp, irreps_out, internal_weights=False, shared_weights=False)
        for inp in (irreps, tp.irreps_out.simplify())
    ]
    module = UuuFullyConnectedSymmetricContraction(
        [tp], linears, backend="cuda" if device == "cuda" else "torch"
    ).to(device=device, dtype=torch.float64)
    x = torch.randn(num_nodes, 2 * irreps.dim, device=device, dtype=torch.float64)[
        :, ::2
    ].requires_grad_()
    types = (torch.arange(num_nodes * 2, device=device) % 3)[::2]
    weights = [
        torch.randn(
            3,
            linear.weight_numel,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        for linear in linears
    ]
    actual = module(x, weights, types)
    expected = linears[0](x, weights[0][types]) + linears[1](
        tp(x, x), weights[1][types]
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    # Requesting only feature gradients must retain their parameter dependence.
    gradients = [
        torch.autograd.grad(value.sin().sum(), x, create_graph=True, retain_graph=True)[
            0
        ]
        for value in (actual, expected)
    ]
    torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-10)
    mixed = [
        torch.autograd.grad(value.square().sum(), weights, retain_graph=True)
        for value in gradients
    ]
    for a, b in zip(*mixed):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    losses = [y.square().sum() for y in (actual, expected)]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, (x, *weights), create_graph=True)
            for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-10)
        losses = [sum(g.square().sum() for g in values) / 100 for values in gradients]
    assert not module.state_dict()
    if device == "cuda" and num_nodes:
        compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x, weights, types), expected, atol=1e-12, rtol=1e-12
        )
