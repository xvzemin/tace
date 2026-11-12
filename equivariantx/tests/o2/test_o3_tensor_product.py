"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import o2

from equivariantx.tests.o2.testing import spherical_layout


@pytest.mark.parametrize("wigner_lmax", [2, 4])
def test_o3_tensor_product_compiles_with_dynamic_and_empty_batches(
    double_precision, wigner_lmax
):
    module = o2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        internal_weights=False,
        shared_weights=False,
    )
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch_size in (3, 1, 0):
        x = torch.randn(batch_size, 6, requires_grad=True)
        w = torch.randn(batch_size, module.weight_numel, requires_grad=True)
        d, di = o2.WignerD(2, wigner_lmax)(torch.randn(batch_size, 3))
        actual, expected = compiled(x, d, di, w), module(x, d, di, w)
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual.square().sum(), (x, w)),
            torch.autograd.grad(expected.square().sum(), (x, w)),
        ):
            torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
@pytest.mark.parametrize("batch_size", [0, 5])
def test_o3_tensor_product_matches_edge_cgtp(
    double_precision, mode, normalization, batch_size
):

    irreps_in = o3.Irreps("2x0e+2x1o+3x1e+3x2o+2x3e")
    irreps_sh = o3.Irreps.spherical_harmonics(4)
    irreps_out = o3.Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o+2x3e+2x3o")
    paths, instructions = [], []
    for _, ir_out in irreps_out:
        for i, (mul, ir_in) in enumerate(irreps_in):
            for j, (_, ir_sh) in enumerate(irreps_sh):
                if ir_out in ir_in * ir_sh:
                    instructions.append((i, j, len(paths), mode, True))
                    paths.append((mul, ir_out))
    irreps_out = o3.Irreps(paths)
    kwargs = dict(internal_weights=False, shared_weights=False)
    reference = o3.TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    module = o2.O3TensorProduct(
        irreps_in,
        irreps_sh,
        irreps_out,
        instructions,
        normalization=normalization,
        **kwargs,
    )
    assert not module.local_frame_in.basis_change
    assert not module.local_frame_out.basis_change
    x = torch.randn(batch_size, irreps_in.dim, requires_grad=True)
    r = torch.randn(batch_size, 3, requires_grad=True)
    w = torch.randn(batch_size, module.weight_numel, requires_grad=True)

    d, di = o2.WignerD(3, 3)(r)
    actual = spherical_layout(
        module(spherical_layout(x, irreps_in), d, di, w), irreps_out, inverse=True
    )
    expected = reference(
        x, o3.spherical_harmonics(irreps_sh, r, True, normalization), w
    )
    assert module.weight_numel == reference.weight_numel
    assert not any(isinstance(child, o3.TensorProduct) for child in module.modules())
    torch.testing.assert_close(module.output_mask, reference.output_mask)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    if batch_size:
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual.square().sum(), (x, r, w)),
            torch.autograd.grad(expected.square().sum(), (x, r, w)),
        ):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-9)


@pytest.mark.parametrize("irrep_normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_o3_tensor_product_normalization_and_second_derivatives(
    double_precision,
    irrep_normalization,
    path_normalization,
):
    irreps_in = o3.Irreps("2x1o")
    irreps_sh = o3.Irreps.spherical_harmonics(2)
    irreps_out = o3.Irreps("2x1o+2x1e+2x2e+2x0o")
    instructions = [
        (0, 0, 0, "uvu", True, 0.7),
        (0, 2, 0, "uvu", True, 1.3),
        (0, 1, 1, "uvu", False, 0.9),
        (0, 1, 2, "uvu", True, 1.1),
    ]
    kwargs = dict(
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        in1_var=[0.8],
        in2_var=[1.2, 0.9, 1.5],
        out_var=[0.7, 1.1, 1.3, 1.0],
        internal_weights=False,
        shared_weights=False,
    )
    reference = o3.TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    module = o2.O3TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    x = torch.randn(4, irreps_in.dim, requires_grad=True)
    r = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(1, module.weight_numel, requires_grad=True)

    d, di = o2.WignerD(2, 2)(r)
    # A nonunit input vector tests the optional per-degree amplitude and its gradient.
    scale = r.square().sum(-1, keepdim=True).sqrt().pow(torch.arange(3))
    actual = spherical_layout(
        module(spherical_layout(x, irreps_in), d, di, weight, scale),
        irreps_out,
        inverse=True,
    )
    expected = reference(
        x, o3.spherical_harmonics(irreps_sh, r, False, "component"), weight
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    grads = [
        torch.autograd.grad(value.square().sum(), (x, r, weight), create_graph=True)
        for value in (actual, expected)
    ]
    for actual_grad, expected_grad in zip(*grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-9)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(grads[0][1].square().sum(), (x, r, weight)),
        torch.autograd.grad(grads[1][1].square().sum(), (x, r, weight)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-8)


def test_o3_tensor_product_internal_weights_axes_and_truncated_frames(double_precision):
    module = o2.O3TensorProduct(
        "2x1o",
        "1x1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    reference = o3.TensorProduct(
        module.irreps_in1,
        module.irreps_in2,
        module.irreps_out,
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    r = torch.cat((torch.eye(3), -torch.eye(3)))
    x = torch.randn(6, 3, 6)

    d, di = o2.WignerD(2, 2)(r)
    actual = spherical_layout(
        module(spherical_layout(x, module.irreps_in1), d, di),
        module.irreps_out,
        inverse=True,
    )
    expected = reference(
        x, o3.spherical_harmonics([1], r, True, "component")[:, None], module.weight
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    d, di = o2.WignerD(1, 2)(r)
    with pytest.raises(ValueError, match="orders"):
        module(spherical_layout(x, module.irreps_in1), d, di)
    with pytest.raises(ValueError, match="spherical harmonics"):
        o2.O3TensorProduct("1o", "1e", "0o", [(0, 0, 0, "uvu", True)])


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed O(3) irreps do not expose time-reversal parity.",
)
def test_o3_tensor_product_time_reversal(double_precision):
    irreps_in = o3.Irreps("2x1eo")
    irreps_out = o3.Irreps("2x0oo+2x1oo+2x2oo")
    module = o2.O3TensorProduct(
        irreps_in,
        o3.Irreps.spherical_harmonics(1)[1:],
        irreps_out,
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    x = torch.randn(4, irreps_in.dim)
    d, di = o2.WignerD(2, 2)(torch.randn(4, 3))
    torch.testing.assert_close(module(-x, d, di), -module(x, d, di))
