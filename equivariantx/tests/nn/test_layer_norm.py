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


@pytest.mark.parametrize("cls", [nn.MergeRmsNorm, nn.MergeLayerNorm])
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


@pytest.mark.parametrize("cls", [nn.MergeRmsNorm, nn.MergeLayerNorm])
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
