"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import co2, o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_gate_derivatives_and_compile(dtype, backend):
    if backend == "cuda" and DEVICE.type != "cuda":
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    args = (
        "2x0e+1x0oo",
        [torch.nn.functional.silu, torch.tanh],
        "2x0e+3x0oo",
        [torch.sigmoid, torch.tanh],
        "1x0o+2x1mo+2x4m",
    )
    reference = o2.Gate(*args).to(device=DEVICE, dtype=dtype)
    gate = co2.Gate(*args, backend=backend).to(device=DEVICE, dtype=dtype)
    embed = co2.ChangeOfBasis(gate.irreps_in).to(device=DEVICE, dtype=dtype)
    output = co2.ChangeOfBasis(gate.irreps_out).to(device=DEVICE, dtype=dtype)
    compiled = torch.compile(gate, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch in ((3,), (2, 3), (0,)):
        x = torch.randn(
            *batch,
            reference.irreps_in.dim,
            device=DEVICE,
            dtype=dtype,
            requires_grad=True,
        )
        for method in (gate, compiled):
            values = [output(reference(x)), method(embed(x))]
            for _ in range(3 if method is gate else 1):
                torch.testing.assert_close(
                    *values,
                    atol=2e-5 if dtype == torch.float32 else 2e-12,
                    rtol=2e-5 if dtype == torch.float32 else 2e-12,
                )
                values = [
                    torch.autograd.grad(
                        value.sin().sum(), x, create_graph=method is gate
                    )[0]
                    for value in values
                ]
                torch.testing.assert_close(
                    *values,
                    atol=2e-5 if dtype == torch.float32 else 2e-12,
                    rtol=2e-5 if dtype == torch.float32 else 2e-12,
                )
