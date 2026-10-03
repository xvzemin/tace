"""ACE coefficients, recursive derivatives, and CUDA graph replay."""

import pytest
import torch
from e3nn import o3

from eqx.ace import TACE


def test_native_cuda_graph_ace(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import cuda_graph

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
