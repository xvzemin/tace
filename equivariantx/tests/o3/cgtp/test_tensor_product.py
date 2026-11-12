"""CG providers preserve e3nn paths, weights and mixed derivatives."""

import pytest
import torch
from e3nn import o3

from equivariantx import o3 as eqx_o3


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize(
    "mode,weighted",
    [
        ("uvw", True),
        ("uvu", True),
        ("uvu", False),
        ("uvv", True),
        ("uuw", True),
        ("uuw", False),
        ("uuu", True),
        ("uuu", False),
        ("uvuv", True),
        ("uvuv", False),
        ("uvu<v", True),
        ("uvu<v", False),
        ("u<vw", True),
    ],
)
def test_connection_modes(backend, mode, weighted, device, double_precision):
    if backend == "cuda" and device != "cuda":
        pytest.skip("CUDA unavailable")
    channels1 = 3
    channels2 = 3 if mode in ("uuu", "uuw", "uvu<v", "u<vw") else 2
    channels_out = {
        "uvw": 4,
        "uvu": 3,
        "uvv": 2,
        "uuu": 3,
        "uuw": 4 if weighted else 1,
        "uvuv": 6,
        "uvu<v": 3,
        "u<vw": 4,
    }[mode]
    args = (
        f"{channels1}x1o",
        f"{channels2}x1e",
        f"{channels_out}x2o",
        [(0, 0, 0, mode, weighted)],
    )
    options = dict(internal_weights=False, shared_weights=False)
    reference = o3.TensorProduct(*args, **options).to(device)
    module = eqx_o3.TensorProduct(*args, backend=backend, **options).to(device)
    inputs = tuple(
        torch.randn(*shape, device=device, requires_grad=True) / 3
        for shape in (
            (2, 1, reference.irreps_in1.dim),
            (1, 3, reference.irreps_in2.dim),
            (2, 3, reference.weight_numel),
        )
    )
    actual, expected = module(*inputs), reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        module.contraction(*inputs), expected, atol=2e-12, rtol=2e-12
    )
    a, b = actual.sin().sum(), expected.sin().sum()
    for _ in range(4):
        ga, gb = [
            torch.autograd.grad(value, inputs, create_graph=True, allow_unused=True)
            for value in (a, b)
        ]
        ga, gb = [
            tuple(
                torch.zeros_like(x) if g is None else g
                for x, g in zip(inputs, gradients)
            )
            for gradients in (ga, gb)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-10, rtol=2e-10)
        a, b = [sum(g.sin().sum() for g in gradients) / 10 for gradients in (ga, gb)]
    tangents = tuple(torch.randn_like(value) for value in inputs)
    torch.testing.assert_close(
        torch.func.jvp(module.contraction, inputs, tangents),
        torch.func.jvp(reference, inputs, tangents),
        atol=2e-10,
        rtol=2e-10,
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("FullyConnectedTensorProduct", ("2x0e+2x1o", "2x0e+2x1o", "2x0e+2x1o")),
        ("FullTensorProduct", ("2x0e+2x1o", "2x0e+2x1o")),
        ("ElementwiseTensorProduct", ("2x0e+2x1o", "2x0e+2x1o")),
    ],
)
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_factories_and_state(name, args, backend, device, double_precision):
    torch.compiler.reset()
    if backend == "cuda" and device != "cuda":
        pytest.skip("CUDA unavailable")
    reference = getattr(o3, name)(*args).to(device)
    module = getattr(eqx_o3, name)(*args, backend=backend).to(device)
    assert isinstance(module, eqx_o3.TensorProduct)
    module.load_state_dict(reference.state_dict())
    x = torch.randn(4, module.irreps_in1.dim, device=device)
    y = torch.randn(4, module.irreps_in2.dim, device=device)
    expected = reference(x, y)
    torch.testing.assert_close(module(x, y), expected, atol=2e-12, rtol=2e-12)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(x, y), expected, atol=2e-12, rtol=2e-12)
    assert compiled(x[:0], y[:0]).shape == (0, module.irreps_out.dim)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_precision(dtype):
    module = eqx_o3.FullyConnectedTensorProduct("0e", "0e", "0e")
    with pytest.raises(TypeError, match="float32 or float64"):
        module(torch.ones(2, 1, dtype=dtype), torch.ones(2, 1, dtype=dtype))


def test_dtype_roundtrip(double_precision):
    module = eqx_o3.FullyConnectedTensorProduct("2x1o", "2x1e", "2x2o")
    x, y = torch.randn(2, 3, 6)
    weights = module.weight.detach().clone()
    expected = module(x, y)
    module.float().double()
    with torch.no_grad():
        module.weight.copy_(weights)
    torch.testing.assert_close(module(x, y), expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        module.contraction(x, y, module.weight), expected, atol=2e-12, rtol=2e-12
    )


def test_grouped_reference_switch(monkeypatch, device, double_precision):
    if device != "cuda":
        pytest.skip("Compiled CUDA schedule")
    from equivariantx.ops import config
    from torch.fx.experimental.proxy_tensor import make_fx

    module = eqx_o3.FullyConnectedTensorProduct(
        "2x0e+2x1o", "2x0e+2x1o", "2x0e+2x1o", internal_weights=False
    ).cuda()
    x = torch.randn(512, module.irreps_in1.dim, device="cuda")
    y = torch.randn_like(x)
    weights = torch.randn(module.weight_numel, device="cuda")
    expected = module(x, y, weights)
    graph = make_fx(module)(x, y, weights)
    torch.testing.assert_close(graph(x, y, weights), expected, atol=2e-12, rtol=2e-12)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    reference = make_fx(module)(x, y, weights)
    torch.testing.assert_close(
        reference(x, y, weights), expected, atol=2e-12, rtol=2e-12
    )
    assert str(graph.graph) != str(reference.graph)
