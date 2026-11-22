"""Shared equivariant normalization."""

import pytest
import torch
from e3nn import o3

from equivariantx import co2, co3, nn, o2


def pack(blocks, layout):
    return torch.cat(
        [
            (x.transpose(-1, -2) if layout == "ir_mul" else x).flatten(-2)
            for x in blocks
        ],
        dim=-1,
    )


@pytest.mark.parametrize("cls", [nn.MergeRmsNorm, nn.MergeLayerNorm])
@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("planar", [False, True])
def test_norm_blocks(cls, layout, planar, device, double_precision):
    irreps = (
        o2.Irreps("2x2mo + 4x0e + 3x0o + 3x0e + 2x0eo")
        if planar
        else o3.Irreps("2x2e + 4x0e + 3x0o + 3x0e")
    )
    blocks = [
        torch.randn(2, 3, b.mul, b.ir.dim, device=device, requires_grad=True)
        for b in irreps
    ]
    norm = cls(irreps, layout=layout).to(device)
    assert not list(norm.buffers())
    assert set(norm.state_dict()) == (
        {"weight", "bias"} if cls is nn.MergeLayerNorm else {"weight"}
    )
    with torch.no_grad():
        for p in norm.parameters():
            p.normal_()
    features = pack(blocks, layout)
    observed = norm(features)
    centered = blocks.copy()
    if cls is nn.MergeLayerNorm:
        assert norm.bias.shape == (7,)
        for i in (1, 3):
            centered[i] = blocks[i] - blocks[i].mean(-2, keepdim=True)
    variance = sum(b.square().mean((-1, -2), keepdim=True) for b in centered) / len(
        blocks
    )
    expected = [
        x * (variance + norm.eps).rsqrt() * w[:, None]
        for x, w in zip(centered, norm.weight.split([b.mul for b in irreps]))
    ]
    if norm.bias is not None:
        expected[1] = expected[1] + norm.bias[:4, None]
        expected[3] = expected[3] + norm.bias[4:, None]
    expected = pack(expected, layout)
    torch.testing.assert_close(observed, expected)
    assert norm(features[0, 0]).shape == (irreps.dim,)
    assert norm(features[:, :0]).shape == (2, 0, irreps.dim)
    with pytest.raises(ValueError, match="uniform"):
        norm(features.new_empty(3, 7))

    variables = (*blocks, *norm.parameters())

    def derivatives(y):
        first = torch.autograd.grad(y.sin().sum(), variables, create_graph=True)
        return (
            *first,
            *torch.autograd.grad(sum(g.square().sum() for g in first), blocks),
        )

    for x, y in zip(derivatives(observed), derivatives(expected)):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("cls", [nn.MergeRmsNorm, nn.MergeLayerNorm, nn.SoftCapRmsNorm])
@pytest.mark.parametrize("planar", [False, True])
def test_norm_change_of_basis(cls, planar, device, double_precision):
    group = co2 if planar else co3
    irreps = (
        o2.Irreps("3x0e + 2x0o + 4x3mo + 1x0eo")
        if planar
        else o3.Irreps("3x0e + 2x0o + 4x3o")
    )
    compact = cls(irreps).to(device)
    cartesian = cls(group.Irreps(irreps)).to(device)
    change = group.ChangeOfBasis(irreps, backend="torch").to(device)
    with torch.no_grad():
        for p in compact.parameters():
            p.normal_()
    cartesian.load_state_dict(compact.state_dict(), strict=True)
    assert not list(cartesian.buffers())
    x = torch.randn(2, 3, irreps.dim, device=device, requires_grad=True)
    y, ref = cartesian(change(x)), change(compact(x))
    torch.testing.assert_close(y, ref)
    seed = torch.randn_like(y)
    for _ in range(2):
        y, ref = (
            torch.autograd.grad((z * seed).sum(), x, create_graph=True)[0]
            for z in (y, ref)
        )
        torch.testing.assert_close(y, ref)
        seed = torch.randn_like(y)
    if planar:
        for norm, v in ((compact, x), (cartesian, change(x))):
            matrix = norm.irreps.D_from_angle(
                x.new_tensor(0.47), reflected=True, time_reversal=True
            )
            torch.testing.assert_close(norm(v @ matrix.T), norm(v) @ matrix.T)


@pytest.mark.parametrize("cls", [nn.MergeRmsNorm, nn.MergeLayerNorm, nn.SoftCapRmsNorm])
def test_uniform_storage(cls, device, double_precision):
    for irreps in (
        o3.Irreps("3x0e + 3x2e"),
        co3.Irreps("3x0e + 3x2e"),
        o2.Irreps("3x0e + 3x2m"),
        co2.Irreps("3x0e + 3x2m"),
    ):
        norm = cls(irreps, affine=False, layout="ir_mul").to(device)
        x = torch.randn(2, 5, irreps.dim, device=device)
        assert not norm.state_dict() and not list(norm.buffers())
        torch.testing.assert_close(norm(x.unflatten(-1, (-1, 3))).flatten(-2), norm(x))
    scalar = torch.randn(2, 5, 3, device=device)
    norm = cls("3x0e", affine=False).to(device)
    ref = (
        torch.nn.functional.layer_norm
        if cls is nn.MergeLayerNorm
        else torch.nn.functional.rms_norm
    )
    torch.testing.assert_close(norm(scalar), ref(scalar, (3,), eps=norm.eps))


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0])
def test_soft_cap_rms(p, device, double_precision):
    norm = nn.SoftCapRmsNorm("2x0e + 1x1o", p=p, max_rms=2, eps=0.25).to(device)
    assert isinstance(norm, nn.EquivariantLayerNorm)
    assert not norm.state_dict() and not list(norm.buffers())
    x = torch.randn(6, 5, device=device, requires_grad=True)
    v = (x[:, :2].square().mean(1) + x[:, 2:].square().mean(1)) / 2
    reference = x * (0.25**p + (v / 4).pow(p))[:, None].pow(-0.5 / p)
    actual = norm(x)
    torch.testing.assert_close(actual, reference)
    for _ in range(3):
        probe = torch.randn_like(x)
        actual, reference = (
            torch.autograd.grad((y * probe).sum(), x, create_graph=True)[0]
            for y in (actual, reference)
        )
        torch.testing.assert_close(actual, reference)
    large = norm(1e15 * torch.ones_like(x))
    torch.testing.assert_close(large, torch.full_like(x, 2))

    z = torch.zeros_like(x, requires_grad=True)
    for layer in (nn.EquivariantLayerNorm, nn.SoftCapRmsNorm):
        default = layer("2x0e + 1x1o", p=p).to(device)
        out = default(z)
        grad = torch.autograd.grad(out.sum(), z, create_graph=True)[0]
        curvature = torch.autograd.grad(grad.sum(), z)[0]
        torch.testing.assert_close(out, torch.zeros_like(z))
        torch.testing.assert_close(grad, torch.ones_like(z))
        torch.testing.assert_close(curvature, torch.zeros_like(z))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (nn.MergeRmsNorm, {}),
        (nn.MergeLayerNorm, {}),
        (nn.SoftCapRmsNorm, {"p": 2.0, "max_rms": 1.5}),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalization_derivatives(cls, kwargs, dtype, double_precision):
    irreps = co2.Irreps("40x0e+20x0o+18x2m+12x0e")
    layer = cls(irreps, backend="cuda", **kwargs).cuda().to(dtype)
    reference = cls(irreps, backend="torch", **kwargs).cuda().to(dtype)
    x = torch.randn(3, irreps.dim, device="cuda", dtype=dtype)
    x[0] = 0
    x[1] *= 1e-4
    x.requires_grad_()
    actual, expected = layer(x), reference(x)
    torch.testing.assert_close(actual, expected)
    for _ in range(4):
        probe = torch.randn_like(x) / x.shape[-1]
        actual, expected = (
            torch.autograd.grad((v * probe).sum(), x, create_graph=True)[0]
            for v in (actual, expected)
        )
        if dtype == torch.float32:
            scale = expected.abs().amax(-1, keepdim=True).clamp_min(1)
            torch.testing.assert_close(
                actual / scale, expected / scale, rtol=2e-5, atol=2e-5
            )
        else:
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-9)
