"""Irreducible tensor decomposition tests."""

from itertools import product

import pytest
import torch
from e3nn import o3

from equivariantx import co3, o2
from equivariantx import o3 as eqx_o3
from equivariantx.tests.utils.testing import (
    DEVICE,
    SPACES,
    check_ictd_basis,
    check_ictd_equivariance,
    check_ictd_invalid_rank,
    check_ictd_method_precision,
    check_ictd_precision_and_derivatives,
    check_ictd_projectors,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_w3j_mixed_bases(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    degrees = (2, 3, 2)
    symbol = o3.wigner_3j(*degrees, dtype=dtype, device=device)
    assert torch.equal(eqx_o3.w3j(*degrees, dtype=dtype, device=device), symbol)
    assert co3.w3j is eqx_o3.w3j
    q = [co3.path_matrix(l) for l in degrees]
    a, b, c = [p.to(symbol) for p in q]
    x, y = (torch.randn(7, 2 * l + 1, dtype=dtype, device=device) for l in degrees[:2])
    expected = torch.einsum("bi,bj,ijk->bk", x, y, symbol)
    for selected in product((False, True), repeat=3):
        paths = tuple(p if use else None for p, use in zip(q, selected))
        tensor = eqx_o3.w3j(*degrees, paths=paths, dtype=dtype, device=device)
        assert tensor.is_contiguous()
        assert tensor.shape == tuple(
            3**l if use else 2 * l + 1 for l, use in zip(degrees, selected)
        )
        value = torch.einsum(
            "bi,bj,ijk->bk",
            x @ a.T if selected[0] else x,
            y @ b.T if selected[1] else y,
            tensor,
        )
        torch.testing.assert_close(value, expected @ c.T if selected[2] else expected)
        # Unit covariance in the independent output coordinates, not in all 3**l entries.
        projector = (
            c @ c.T if selected[2] else torch.eye(c.size(1), dtype=dtype, device=device)
        )
        flat = tensor.flatten(0, 1)
        torch.testing.assert_close((2 * degrees[2] + 1) * flat.T @ flat, projector)


@pytest.mark.parametrize("lout", [1, 2])
def test_w3j_cartesian_normalization(lout, double_precision):
    degrees = (2, 2, lout)
    weights = co3.w3j(*degrees, paths=tuple(co3.path_matrix(l) for l in degrees))
    x, y = (torch.randn(8, 5) @ co3.path_matrix(2).T for _ in range(2))
    expected = (2 * lout + 1) ** 0.5 * torch.einsum("bi,bj,ijk->bk", x, y, weights)
    for cls in (co3.ictd.TensorProduct, co3.ictp.TensorProduct):
        tp = cls("2e", "2e", f"{lout}e", [(0, 0, 0, "uuu", False)], backend="torch")
        torch.testing.assert_close(tp(x, y), expected, rtol=1e-13, atol=1e-13)


def test_w3j_repeated_irrep_paths(double_precision):
    decomp = eqx_o3.ICTD(3)
    indices = [i for i, path in enumerate(decomp.paths) if path[-1].l == 1]
    paths = [decomp.path_matrix(i) for i in indices]
    assert len(paths) == 3 and decomp._buffers["change_of_basis"] is None
    x, y = torch.randn(4, 27), torch.randn(4, 3)
    expected = decomp(x)
    for i, q in zip(indices, paths):
        c = eqx_o3.w3j(1, 1, 0, paths=(q, None, None))
        torch.testing.assert_close(
            torch.einsum("bi,bj,ij->b", x, y, c[..., 0]),
            (expected[:, decomp.slices[i]] * y).sum(-1) / 3**0.5,
        )
    with pytest.raises(ValueError, match="one entry"):
        eqx_o3.w3j(1, 1, 0, paths=(q, None))
    with pytest.raises(ValueError, match=r"paths\[2\]"):
        eqx_o3.w3j(1, 1, 0, paths=(None, None, q))
    with pytest.raises(TypeError, match=r"paths\[0\]"):
        eqx_o3.w3j(1, 1, 0, paths=(q.to(torch.complex128), None, None))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.ops import config
    from equivariantx.ops.cuda.o3 import ictd as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module, reference = (
        eqx_o3.ICTD(3, backend="cuda").to(device),
        eqx_o3.ICTD(3).to(device),
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        x = torch.randn(count, 27, device=device)
        for name, args in (("forward", (x,)), ("inverse", (x,)), ("project", (x, 0))):
            with patch.object(kernel, name, wraps=getattr(kernel, name)) as selected:
                actual = getattr(module, name)(*args)
            assert selected.call_count == (device == "cuda")
            torch.testing.assert_close(actual, getattr(reference, name)(*args))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    check_ictd_method_precision(3, device)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 6])
def test_ictd_basis(rank, double_precision):
    check_ictd_basis(rank, 3, double_precision)


def test_ictd_projectors(double_precision):
    check_ictd_projectors(3, double_precision)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("reflected", [False, True])
def test_ictd_equivariance(rank, reflected, double_precision):
    check_ictd_equivariance(rank, 3, reflected, double_precision)


def test_ictd_precision_and_derivatives():
    check_ictd_precision_and_derivatives(3)


@pytest.mark.parametrize("rank", [-1, 1.5])
def test_ictd_invalid_rank(rank):
    check_ictd_invalid_rank(rank, 3)


def test_ictd_rank_two_decomposition(double_precision):
    for d in (2, 3):
        module = SPACES[d].ICTD(2, device=DEVICE)
        x = torch.randn(4, d, d, device=DEVICE)
        transpose = x.transpose(-1, -2)
        trace = x.diagonal(dim1=-2, dim2=-1).sum(-1)
        isotropic = trace[:, None, None] * torch.eye(d, device=DEVICE) / d
        expected = {
            "stf": (x + transpose) / 2 - isotropic,
            "antisymmetric": (x - transpose) / 2,
            "scalar": isotropic,
        }
        for index, path in enumerate(module.paths):
            ir = path[-1]
            order = ir.l if d == 3 else ir.m
            kind = (
                "stf"
                if order == 2
                else "scalar"
                if order == 0 and ir.p == 1
                else "antisymmetric"
            )
            torch.testing.assert_close(
                module.project(x.flatten(-2), index).reshape_as(x),
                expected[kind],
                atol=2e-14,
                rtol=2e-14,
            )
    assert o2.ICTD(2).irreps_out == o2.Irreps("1x2m+1x0e+1x0o")
    assert eqx_o3.ICTD(2).irreps_out == o3.Irreps("1x2e+1x1e+1x0e")


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_sparse_backend(backend, device, double_precision, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_sparse_backend

    check_ictd_sparse_backend(3, backend, device, monkeypatch)


def test_lazy_reference_compile(device, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_lazy_compile

    check_ictd_lazy_compile(3, device, monkeypatch)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_path_cache(backend, device, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_path_cache

    check_ictd_path_cache(3, backend, device, monkeypatch)
