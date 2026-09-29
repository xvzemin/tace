"""Complete Cartesian decomposition and path projections."""

import pytest
import torch
from e3nn import o3

from eqx import co2, co3, o2
from eqx import o3 as eqx_o3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPACES = {2: o2, 3: eqx_o3}


@pytest.mark.parametrize(
    "rank,d", [(n, d) for d in (2, 3) for n in range(5)] + [(9, 2), (6, 3)]
)
def test_basis(rank, d, double_precision):
    module = SPACES[d].ICTD(rank, device=DEVICE)
    matrix = module.change_of_basis
    identity = torch.eye(d**rank, device=DEVICE)
    torch.testing.assert_close(matrix.T @ matrix, identity, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(matrix @ matrix.T, identity, atol=3e-14, rtol=3e-14)
    assert module.irreps_out.dim == module.dim == d**rank
    assert len(module.paths) == len(module.irreps_out)
    assert all(len(path) == rank + 1 for path in module.paths)
    basis = co3.path_matrix(rank) if d == 3 else co2.path_matrix(rank)
    torch.testing.assert_close(
        module.path_matrix(0), basis.to(DEVICE), atol=3e-15, rtol=3e-15
    )
    x = torch.randn(2, d**rank, 3, device=DEVICE).transpose(-1, -2)
    h = module(x)
    torch.testing.assert_close(module.inverse(h), x, atol=3e-14, rtol=3e-14)
    assert module(x[:0]).shape == x[:0].shape
    assert module.inverse(h[:0]).shape == h[:0].shape
    assert not tuple(module.parameters())


@pytest.mark.parametrize("d", [2, 3])
def test_projectors(d, double_precision):
    module = SPACES[d].ICTD(3, device=DEVICE)
    x = torch.randn(5, d**3, device=DEVICE)
    projections = [module.project(x, i) for i in range(len(module.paths))]
    torch.testing.assert_close(sum(projections), x, atol=3e-14, rtol=3e-14)
    for i, projected in enumerate(projections):
        torch.testing.assert_close(module.project(projected, i), projected)
        for j in range(i):
            torch.testing.assert_close(
                module.project(projected, j), torch.zeros_like(x), atol=3e-14, rtol=0
            )
    assert module.project(x[:0], 0).shape == x[:0].shape


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("reflected", [False, True])
def test_equivariance(rank, d, reflected, double_precision):
    module = SPACES[d].ICTD(rank, device=DEVICE)
    if d == 3:
        rotation = o3.rand_matrix(device=DEVICE) * (-1 if reflected else 1)
        output_rotation = module.irreps_out.D_from_matrix(rotation.cpu()).to(DEVICE)
    else:
        angle = torch.rand((), device=DEVICE)
        rotation = o2.Irrep("1m").D_from_angle(angle, reflected)
        output_rotation = module.irreps_out.D_from_angle(angle, reflected)
    cartesian_rotation = torch.ones(1, 1, device=DEVICE)
    for _ in range(rank):
        cartesian_rotation = torch.kron(cartesian_rotation, rotation.contiguous())
    x = torch.randn(3, d**rank, device=DEVICE)
    torch.testing.assert_close(
        module(x @ cartesian_rotation.T),
        module(x) @ output_rotation.T,
        atol=3e-12,
        rtol=3e-12,
    )


def test_rank_two_decomposition(double_precision):
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


@pytest.mark.parametrize("d", [2, 3])
def test_precision_and_derivatives(d):
    module = SPACES[d].ICTD(3, dtype=torch.float32, device=DEVICE).double()
    reference = SPACES[d].ICTD(3, dtype=torch.float64, device=DEVICE)
    torch.testing.assert_close(
        module.change_of_basis, reference.change_of_basis, atol=0, rtol=0
    )
    x = torch.randn(2, d**3, dtype=torch.float64, device=DEVICE, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: module(x).sin(), (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(lambda x: module(x).sin(), (x,), fast_mode=True)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), module(x))


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("rank", [-1, 1.5])
def test_invalid_rank(rank, d):
    with pytest.raises(ValueError):
        SPACES[d].ICTD(rank)
    with pytest.raises(ValueError):
        list(SPACES[d].path_matrices(rank))


@pytest.mark.parametrize("d", [2, 3])
def test_fixed_dimension(d):
    with pytest.raises(TypeError):
        SPACES[d].ICTD(2, d=d)
    with pytest.raises(TypeError):
        list(SPACES[d].path_matrices(2, d=d))
