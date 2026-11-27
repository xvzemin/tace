"""Feature permutations composed with nonlinear upstream operations."""

import pytest
import torch
from equivariantx import o3

from equivariantx.ops import config
from equivariantx.tests.ops.testing import assert_functional_derivatives
from equivariantx.utils.layout import permute


@pytest.mark.parametrize("space", ["o3", "co3", "o2", "co2"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_transpose_irreps_layout(space, backend):
    import itertools

    import equivariantx as eqx

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cpu" if backend == "torch" else "cuda"
    group = getattr(eqx, space)
    label = "2e" if space.endswith("3") else "2m"
    op = group.TransposeIrrepsLayout
    for widths in ((3, 3, 3), (3, 1, 2)):
        irreps = "+".join(f"{n}x{ir}" for n, ir in zip(widths, ("0e", label, label)))
        block_dim = {"o3": 5, "co3": 9, "o2": 2, "co2": 4}[space]
        x = torch.randn(
            2,
            3,
            2 * (widths[0] + sum(widths[1:]) * block_dim),
            dtype=torch.float64,
            device=device,
        )[..., ::2].requires_grad_()
        blocks = [
            v.unflatten(-1, (c, d))
            for v, c, d in zip(
                x.split((widths[0], widths[1] * block_dim, widths[2] * block_dim), -1),
                widths,
                (1, block_dim, block_dim),
            )
        ]
        layouts = {
            ("mul_ir", True): x,
            ("ir_mul", True): torch.cat(
                [v.transpose(-1, -2).flatten(-2) for v in blocks], -1
            ),
        }
        if len(set(widths)) == 1:
            layouts["mul_ir", False] = torch.cat(blocks, -1)
            layouts["ir_mul", False] = layouts["mul_ir", False].transpose(-1, -2)
        else:
            with pytest.raises(ValueError, match="common multiplicity"):
                op(irreps, flat_out=False)
        for (source, fi), (target, fo) in itertools.product(layouts, repeat=2):
            layer = op(
                irreps,
                source=source,
                target=target,
                flat_in=fi,
                flat_out=fo,
                backend=backend,
            ).to(device)
            result = layer(layouts[source, fi])
            expected = layouts[target, fo]
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                layer.inverse(result), layouts[source, fi], rtol=0, atol=0
            )
            a, b = result, expected
            for _ in range(2):
                a, b = (
                    torch.autograd.grad(
                        v.sin().sum(), x, create_graph=True, retain_graph=True
                    )[0]
                    for v in (a, b)
                )
                torch.testing.assert_close(a, b)
            assert not layer.state_dict()

    layer = op(
        f"3x{label}", source="mul_ir", target="ir_mul", flat_out=False, backend=backend
    ).to(device)
    x = torch.randn(2, 3 * block_dim, device=device)
    out = layer(x)
    assert out.data_ptr() == x.data_ptr()
    torch.testing.assert_close(out, x.unflatten(-1, (3, block_dim)).transpose(-1, -2))
    torch.testing.assert_close(layer.inverse(out), x, rtol=0, atol=0)


@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_derivatives(compiler):
    values = torch.randn(2, 3, dtype=torch.float64)
    index = torch.tensor([2, 0, 1])

    def function(value, index):
        return permute(value.sin(), index, index.argsort())

    def reference(value, index):
        return value.sin().index_select(-1, index)

    assert_functional_derivatives(function, reference, (values, index), compiler)


def test_layout_export_with_dynamic_rows():
    layer = o3.TransposeIrrepsLayout(
        "3x0e+2x2e", source="ir_mul", target="mul_ir", backend="torch"
    )
    exported = torch.export.export(
        layer,
        (torch.randn(3, 13),),
        dynamic_shapes=({0: torch.export.Dim("rows", max=32)},),
        strict=True,
    ).module()
    for rows in (2, 7):
        x = torch.randn(rows, 13)
        torch.testing.assert_close(exported(x), layer(x), rtol=0, atol=0)


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("rows", [0, 4])
def test_reverse_derivatives(reference, rows, monkeypatch):
    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    values = torch.randn(rows, 3, dtype=torch.float64, requires_grad=True)
    index = torch.tensor([2, 0, 1])
    actual = permute(values, index, index.argsort())
    expected = values.index_select(-1, index)
    for _ in range(4):
        torch.testing.assert_close(actual, expected)
        actual, expected = [
            torch.autograd.grad(value.sin().sum(), values, create_graph=True)[0]
            for value in (actual, expected)
        ]
