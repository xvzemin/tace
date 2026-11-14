"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import o2
from equivariantx.tests.o2.testing import spherical_layout


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    from equivariantx.tests.utils.testing import check_method_precision

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = o2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    inputs = (
        torch.randn(3, layer.local_frame_in.output_dim, device=device),
        torch.randn(3, layer.weight_numel, device=device),
        torch.randn(3, layer.num_harmonics, device=device),
    )
    check_method_precision(layer, [("forward_local", inputs, {})])


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


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_local_kernel_derivatives(mode, backend, monkeypatch, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.o2.o3_tensor_product import handwritten, reference

    monkeypatch.setattr(config, "TORCH_ALIGNED_PRODUCT", True)
    module = o2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, mode, i != 0 if mode == "uvu" else True) for i in range(3)],
        internal_weights=False,
        shared_weights=False,
        backend=backend,
    ).to(device=device, dtype=torch.float64)
    x = (
        torch.randn(
            2, 1, module.local_frame_in.output_dim, device=device, dtype=torch.float64
        )
        * 0.1
    ).requires_grad_()
    w = (
        torch.randn(1, 3, module.weight_numel, device=device, dtype=torch.float64) * 0.1
    ).requires_grad_()
    h = (
        torch.randn(2, 3, 1, device=device, dtype=torch.float64) * 0.1 + 1
    ).requires_grad_()
    values = x, w, h
    run = (
        (lambda *args: handwritten(module, *args))
        if backend == "torch"
        else module.forward_local
    )
    torch.testing.assert_close(run(*values), module.forward_local(*values))
    actual = run(*values).sin().sum()
    expected = reference(module, *values).sin().sum()
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        da, db = [
            torch.autograd.grad(value, values, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(da, db):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        actual = sum(v.sin().sum() for v in da)
        expected = sum(v.sin().sum() for v in db)
    tangent = tuple(torch.randn_like(v) for v in values)
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(v, t) for v, t in zip(values, tangent)
        )
        primal, actual_tangent = torch.autograd.forward_ad.unpack_dual(run(*duals))
    expected_primal, expected_tangent = torch.autograd.functional.jvp(
        lambda *args: reference(module, *args), values, tangent
    )
    assert actual_tangent is not None
    torch.testing.assert_close(primal, expected_primal, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(actual_tangent, expected_tangent, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(
        torch.func.jvp(run, values, tangent),
        torch.autograd.functional.jvp(run, values, tangent),
        atol=3e-10,
        rtol=3e-10,
    )
    small = tuple(
        v.detach().reshape(-1, v.shape[-1])[:1].requires_grad_() for v in values
    )
    assert torch.autograd.gradcheck(run, small)
    assert torch.autograd.gradgradcheck(run, small)
    torch.testing.assert_close(torch.vmap(run)(*small), run(*small))
    empty = tuple(v[:0] for v in small)
    assert run(*empty).shape == (0, module.local_output_dim)
    del module.local_paths
    torch.testing.assert_close(module.forward_local(*small), reference(module, *small))


def test_local_precision_restoration(device):
    previous = torch.get_default_dtype()
    try:
        args = (
            "2x1o",
            "1o",
            "2x0e+2x1e+2x2e",
            [(0, 0, i, "uvw", True) for i in range(3)],
        )
        kwargs = dict(internal_weights=False, shared_weights=False)
        torch.set_default_dtype(torch.float32)
        promoted = o2.O3TensorProduct(*args, **kwargs).to(
            device=device, dtype=torch.float64
        )
        torch.set_default_dtype(torch.float64)
        precise = o2.O3TensorProduct(*args, **kwargs).to(device=device)
        for name, value in promoted.named_buffers():
            if name.startswith("scale_") or name.startswith("readout_"):
                torch.testing.assert_close(
                    value, precise.get_buffer(name), atol=0, rtol=0
                )
        x = torch.randn(2, promoted.local_frame_in.output_dim, device=device)
        w = torch.randn(2, promoted.weight_numel, device=device)
        torch.testing.assert_close(
            promoted.forward_local(x, w),
            precise.forward_local(x, w),
            atol=2e-12,
            rtol=2e-12,
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_local_strict_fp32(mode, backend, monkeypatch, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.o2.o3_tensor_product import reference

    monkeypatch.setattr(config, "TORCH_ALIGNED_PRODUCT", True)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        args = (
            "2x1o",
            "1o",
            "2x0e+2x1e+2x2e",
            [(0, 0, i, mode, True) for i in range(3)],
        )
        kwargs = dict(internal_weights=False, shared_weights=False)
        module = o2.O3TensorProduct(*args, **kwargs, backend=backend).to(
            device=device, dtype=torch.float32
        )
        oracle = o2.O3TensorProduct(*args, **kwargs).to(
            device=device, dtype=torch.float64
        )
        x = (
            torch.randn(
                2, module.local_frame_in.output_dim, device=device, dtype=torch.float32
            )
            * 0.1
        ).requires_grad_()
        w = (
            torch.randn(2, module.weight_numel, device=device, dtype=torch.float32)
            * 0.1
        ).requires_grad_()
        h = (
            torch.randn(2, 1, device=device, dtype=torch.float32) * 0.1 + 1
        ).requires_grad_()
        values = (x, w, h)
        precise = tuple(v.detach().double().requires_grad_() for v in values)
        actual = module.forward_local(*values)
        expected = reference(oracle, *precise)
        for _ in range(3):
            torch.testing.assert_close(actual.double(), expected, atol=3e-7, rtol=2e-5)
            actual = torch.autograd.grad(actual.sin().sum(), x, create_graph=True)[0]
            expected = torch.autograd.grad(
                expected.sin().sum(), precise[0], create_graph=True
            )[0]
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("active", [False, True])
def test_mixed_and_inactive_paths(backend, active, device):
    from equivariantx.ops.torch.o2.o3_tensor_product import handwritten, reference

    module = o2.O3TensorProduct(
        "2x0e+3x1o",
        "0e+1o",
        "2x0e+4x1o+3x2e",
        [
            (0, 0, 0, "uvu", False),
            (0, 1, 1, "uvw", True),
            (1, 0, 1, "uvw", True),
            (1, 1, 2, "uvu", True),
        ]
        if active
        else [],
        internal_weights=False,
        shared_weights=False,
        backend=backend,
    ).to(device=device, dtype=torch.float64)
    inputs = tuple(
        (torch.randn(*shape, device=device, dtype=torch.float64) * 0.1).requires_grad_()
        for shape in (
            (2, 1, module.local_frame_in.output_dim),
            (1, 3, module.weight_numel),
            (2, 3, module.num_harmonics),
        )
    )
    actual = module.forward_local(*inputs)
    expected = reference(module, *inputs)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        handwritten(module, *inputs), expected, atol=2e-12, rtol=2e-12
    )
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), inputs),
    ):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    import copy
    from unittest.mock import patch

    from equivariantx import o2
    from equivariantx.ops import config
    from equivariantx.ops.cuda.o2 import o3_tensor_product as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = o2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        backend="cuda",
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    reference = copy.deepcopy(module)
    reference.backend = "torch"
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        values = (
            torch.randn(count, module.local_frame_in.output_dim, device=device),
            torch.randn(count, module.weight_numel, device=device),
            torch.randn(count, module.num_harmonics, device=device),
        )
        with patch.object(kernel, "local", wraps=kernel.local) as selected:
            actual = module.forward_local(*values)
        assert selected.call_count == (device == "cuda")
        torch.testing.assert_close(actual, reference.forward_local(*values))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True
