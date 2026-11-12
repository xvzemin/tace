"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "inputs", [("0o", "1m"), ("1m", "0o"), ("1m", "1m"), ("2m", "5m"), ("16m", "12m")]
)
def test_o2_coupling_generators(inputs, double_precision):
    from equivariantx.o2.clebsch_gordan import clebsch_gordan_product

    ir1, ir2 = map(o2.Irrep, inputs)
    x, y = torch.randn(ir1.dim, 3), torch.randn(ir2.dim, 3)

    def matrices(ir):
        if ir.m:
            return ir.m * torch.tensor([[0.0, -1.0], [1.0, 0.0]]), torch.diag(
                torch.tensor([1.0, -1.0])
            )
        return torch.zeros(1, 1), torch.tensor([[float(ir.p)]])

    g1, f1 = matrices(ir1)
    g2, f2 = matrices(ir2)
    for ir in ir1 * ir2:
        g3, f3 = matrices(ir)

        def couple(a, b):
            return clebsch_gordan_product(a, ir1, b, ir2, ir, elementwise=True)

        value = couple(x, y)
        torch.testing.assert_close(g3 @ value, couple(g1 @ x, y) + couple(x, g2 @ y))
        torch.testing.assert_close(f3 @ value, couple(f1 @ x, f2 @ y))


@pytest.mark.parametrize("mode", ["u1u", "uuu", "uvw"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_o2_shared_couplings(mode, order, backend, double_precision):
    from equivariantx.o2.clebsch_gordan import clebsch_gordan_product

    if backend == "cuda" and DEVICE.type != "cuda":
        pytest.skip("CUDA requires a CUDA device")
    torch._dynamo.reset()
    ir1, ir2 = o2.Irrep("1mo"), o2.Irrep(order, 0, 1)
    mul1, mul2, mul_out = 3, 1 if mode == "u1u" else 3, 3
    outputs = list(ir1 * ir2) * 2
    module = o2.TensorProduct(
        mul1 * ir1,
        mul2 * ir2,
        [(ir, mul_out) for ir in outputs],
        [(0, 0, i, mode, True) for i in range(len(outputs))],
        internal_weights=False,
        shared_weights=False,
        path_normalization="none",
        backend=backend,
    ).to(DEVICE)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch in (4, 0):
        x = torch.randn(batch, ir1.dim * mul1, device=DEVICE, requires_grad=True)
        y = torch.randn(1, ir2.dim * mul2, device=DEVICE, requires_grad=True)
        weights = torch.randn(1, module.weight_numel, device=DEVICE, requires_grad=True)
        expected = []
        for i, ir in enumerate(outputs):
            value = clebsch_gordan_product(
                x.reshape(batch, ir1.dim, mul1),
                ir1,
                y.reshape(1, ir2.dim, mul2),
                ir2,
                ir,
                elementwise=mode != "uvw",
            )
            weight = module.weight_view_for_instruction(i, weights)
            value = (
                torch.einsum("...duv,...uvw->...dw", value, weight)
                if mode == "uvw"
                else value * weight.unsqueeze(-2)
            )
            expected.append((value * module.instructions[i].path_weight).flatten(-2))
        expected = torch.cat(expected, dim=-1)
        actual = compiled(x, y, weights)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        if batch:
            ga = torch.autograd.grad(actual.sin().sum(), (x, y, weights))
            gb = torch.autograd.grad(
                expected.sin().sum(), (x, y, weights), retain_graph=True
            )
            for a, b in zip(ga, gb):
                torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
            actual = module(x, y, weights)
            for _ in range(3):
                actual, expected = [
                    torch.cat(
                        [
                            g.flatten()
                            for g in torch.autograd.grad(
                                value.sin().sum(), (x, y, weights), create_graph=True
                            )
                        ]
                    )
                    for value in (actual, expected)
                ]
                torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
