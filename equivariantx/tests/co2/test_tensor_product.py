"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import co2, o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("mode", ["u1u", "uuu", "uvw"])
@pytest.mark.parametrize("irrep_normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_tensor_product(
    double_precision, mode, irrep_normalization, path_normalization, backend
):
    if backend == "cuda" and DEVICE.type != "cuda":
        pytest.skip("CUDA requires a CUDA device")
    v = 1 if mode == "u1u" else 2
    a, b = co2.Irreps("2x0oo+2x1mo+2x2m"), co2.Irreps(f"{v}x0o+{v}x1m+{v}x2mo")
    output, ins = [], []
    for i, (_, ir) in enumerate(a):
        for j, (_, jr) in enumerate(b):
            for kr in ir * jr:
                ins.append((i, j, len(output), mode, True, 0.7))
                output.append((3 if mode == "uvw" else 2, kr))
    c = co2.Irreps(output)
    ins.append((*ins[0][:5], 1.3))
    kwargs = dict(
        internal_weights=False,
        shared_weights=False,
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        in1_var=[0.7, 1.2, 1.1],
        in2_var=[1.2, 0.9, 1.3],
    )
    ref = o2.TensorProduct(a.circular(), b.circular(), c.circular(), ins, **kwargs).to(
        DEVICE
    )
    module = co2.TensorProduct(
        a, b, c, ins, project=False, backend=backend, **kwargs
    ).to(DEVICE)
    x = a.circular().randn(2, -1, device=DEVICE, requires_grad=True)
    y = b.circular().randn(2, -1, device=DEVICE, requires_grad=True)
    weight = torch.randn(2, module.weight_numel, device=DEVICE, requires_grad=True)
    embed_a, embed_b = co2.ChangeOfBasis(a).to(DEVICE), co2.ChangeOfBasis(b).to(DEVICE)
    extract = co2.ChangeOfBasis(c, True).to(DEVICE)
    expected = ref(x, y, weight)
    projected = co2.Projector(c).to(DEVICE)(module(embed_a(x), embed_b(y), weight))
    torch.testing.assert_close(
        projected, co2.ChangeOfBasis(c).to(DEVICE)(expected), atol=3e-13, rtol=3e-13
    )
    actual = extract(projected)
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)
    for _ in range(2):
        probe = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (value * probe).sum(),
                        (x, y, weight),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    assert module(embed_a(x[:0]), embed_b(y[:0]), weight[:0]).shape == (0, c.dim)


@pytest.mark.parametrize("space", [o2, co2])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_tensor_product_export(space, backend, double_precision):
    torch.compiler.reset()
    if backend == "cuda" and DEVICE.type != "cuda":
        pytest.skip("CUDA is unavailable")
    module = space.TensorProduct(
        "2x1m+3x2m",
        "2x1m+3x2m",
        "2x0e+3x4m+3x0o",
        [(0, 0, 0, "uuu", True), (1, 1, 1, "uuu", True), (1, 1, 2, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
        backend=backend,
    ).to(DEVICE)
    inputs = tuple(
        torch.randn(3, dim, device=DEVICE, requires_grad=True)
        for dim in (module.irreps_in1.dim, module.irreps_in2.dim, module.weight_numel)
    )
    expected = module(*inputs)
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    assert set(module.state_dict()) == {"weight"}
    # Static coefficients must not retain intermediate fp32 rounding.
    module.float().double()
    for model in (
        module,
        torch.export.export(module, inputs).module(),
        torch.compile(module, fullgraph=True, backend="aot_eager"),
    ):
        actual = model(*inputs)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), inputs), expected_grads
        ):
            torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("space", [o2, co2])
@pytest.mark.parametrize("mode", ["uuu", "uvw"])
def test_tensor_product_cpu_fallback(space, mode):
    module = space.TensorProduct(
        "2x1m", "2x1m", "2x0e", [(0, 0, 0, mode, True)], backend="cuda"
    )
    reference = space.TensorProduct(
        "2x1m", "2x1m", "2x0e", [(0, 0, 0, mode, True)], backend="torch"
    )
    reference.load_state_dict(module.state_dict())
    x, y = (torch.randn(3, 4, requires_grad=True) for _ in range(2))
    actual, expected = module(x, y), reference(x, y)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), (x, y)),
        torch.autograd.grad(expected.square().sum(), (x, y)),
    )


def test_native_compile_and_empty(double_precision):
    module = co2.FullyConnectedTensorProduct("2x1m", "2x2m", "3x1m+3x3m").to(DEVICE)
    x = module.irreps_in1.randn(3, -1, device=DEVICE)
    y = module.irreps_in2.randn(3, -1, device=DEVICE)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, y), module(x, y))
    linear = co2.Linear(module.irreps_out, "2x1m+2x3m", output_basis="circular").to(
        DEVICE
    )
    value = module(x, y)
    torch.testing.assert_close(
        torch.compile(linear, backend="aot_eager", fullgraph=True)(value), linear(value)
    )
    elementwise = co2.ElementwiseTensorProduct("1x0o+2x1m", "2x2m+1x0o").to(DEVICE)
    assert elementwise.irreps_out.num_irreps == 4
    for count in (0, 3):
        empty = co2.TensorProduct("", "", "1x0e", []).to(DEVICE)
        x = torch.empty(count, 0, device=DEVICE, requires_grad=True)
        y = torch.empty(1, 0, device=DEVICE, requires_grad=True)
        z = empty(x, y)
        assert z.shape == (count, 1)
        assert not z.count_nonzero()
        assert torch.autograd.grad(z.sum(), x)[0].shape == x.shape
        linear = co2.Linear("2x1m", "1x0e").to(DEVICE)
        assert linear(torch.randn(count, 4, device=DEVICE)).shape == (count, 1)
