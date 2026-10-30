"""ACE coefficients, recursive derivatives, and CUDA graph replay."""

import pytest
import torch
from e3nn import o3

from eqx.ace.tace import TACE
from eqx.models.tace.tece_oam_rra.product import BilinearACE


@pytest.mark.parametrize(
    "kind",
    [
        "ace",
        "ace_split",
        "bilinear",
        "bilinear_split",
        "bilinear_wide",
        "moe",
        "moe_split",
    ],
)
def test_cartesian_products(kind, double_precision, monkeypatch):
    from eqx import co3
    from eqx.ace import CartesianBilinearACE, CartesianTACE

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if kind == "ace_split":
        from eqx.backends.cuda import ace

        batches = ace.schedule_batches
        monkeypatch.setattr(ace, "schedule_batches", lambda groups: batches(groups, 7))
        monkeypatch.setattr(ace, "weighted_source", lambda *args: None)
    elif kind.endswith("_split"):
        from eqx.backends.cuda.models.tace import bilinear

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
            CartesianBilinearACE(tp, coefficient, experts, backend=backend)
            for backend in ("torch", "cuda")
        ]
    else:
        first = co3.Linear(irreps, output, internal_weights=False, shared_weights=False)
        modules = [
            CartesianTACE([tp], [first, coefficient], backend=backend)
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


def test_native_cuda_graph_ace(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import eqx.backends.cuda.graph as cuda_graph

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
    module = TACE([tp, tp], linears).cuda().double()
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
def test_ace_external_coefficients(
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
    module = TACE([tp], linears).to(device=device, dtype=torch.float64)
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("experts", [1, 2, 4])
@pytest.mark.parametrize("nodes", [0, 3, 9])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("degree", [1, 3])
def test_bilinear_ace(device, experts, nodes, shared, degree):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = o3.TensorProduct(
        f"4x0e+4x{degree}o",
        f"4x0e+4x{degree}o+4x0e",
        f"4x0e+4x0e+4x{degree}o",
        [(0, 0, 0, "uuu", True), (1, 1, 1, "uuu", True), (1, 2, 2, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    linear = o3.Linear(
        [(mul // experts, ir) for mul, ir in tp.irreps_out.simplify()],
        f"{4 // experts}x0e+{4 // experts}x{degree}o",
        internal_weights=False,
        shared_weights=False,
    )
    shared_linear = (
        o3.Linear(
            tp.irreps_out.simplify(),
            f"4x0e+4x{degree}o",
            internal_weights=False,
            shared_weights=False,
        )
        if shared
        else None
    )
    module = BilinearACE(tp, linear, experts, shared_linear=shared_linear).to(
        device=device, dtype=torch.float64
    )
    x, y, gates = [
        torch.randn(nodes, 2 * size, dtype=torch.float64, device=device)[
            :, ::2
        ].requires_grad_()
        for size in (tp.irreps_in1.dim, tp.irreps_in2.dim, tp.weight_numel)
    ]
    weight = torch.randn(
        3,
        experts,
        linear.weight_numel,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    types = torch.arange(nodes, device=device) % 3
    shared_weight = (
        torch.randn(
            shared_linear.weight_numel,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        if shared
        else None
    )
    actual = module(x, y, gates, weight, types, shared_weight)
    # Compare against the existing tensor product and independent coefficient map.
    module.coefficients.backend = "torch"
    expected = module.coefficients(tp(x, y, gates), weight, types)
    inputs = (x, y, gates, weight)
    if shared:
        expected = expected + shared_linear(
            tp(x, y, gates), shared_weight.expand(nodes, -1)
        )
        inputs = (*inputs, shared_weight)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    gradients = [
        torch.autograd.grad(
            value.sin().sum(), (x, y), create_graph=True, retain_graph=True
        )
        for value in (actual, expected)
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    mixed = [
        torch.autograd.grad(
            sum(value.square().sum() for value in grad),
            inputs,
            retain_graph=True,
        )
        for grad in gradients
    ]
    for a, b in zip(*mixed):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-9)
    losses = [value.square().sum() for value in (actual, expected)]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(
                a, b, atol=1e-8, rtol=1e-9, msg=f"Derivative order {_ + 1}"
            )
        losses = [sum(value.square().sum() for value in grad) for grad in gradients]
    assert not module.state_dict()
    if device == "cuda" and nodes and experts == 2:
        compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x, y, gates, weight, types, shared_weight),
            expected,
            atol=1e-11,
            rtol=1e-11,
        )
