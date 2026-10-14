"""Analytic radial spectrum, path routing, and energy/force training."""

from copy import deepcopy
from math import prod

import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.special import eval_jacobi, roots_legendre

from tace.lightning import create_model, export_tace, load_tace
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models.linear import mlpLinear
from tace.models.mlp import MLP, PathRadialMLP
from tace.models.radial import C2PolynomialCutoff, PoschlTellerBasis, RadialBasis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.usefixtures("double_precision")


@pytest.mark.parametrize("lmax, max_eigenvalue", [(3, 323.0), (12, 2400.0)])
def test_basis_values_spectrum_and_normalization(lmax, max_eigenvalue):
    basis = PoschlTellerBasis(lmax, max_eigenvalue, cutoff=3 ** (1 / 3)).to(DEVICE)
    r = torch.tensor([[0.0], [1e-8], [0.15], [1.0], [5.0], [30.0]], device=DEVICE)
    actual = basis(r)
    x = r.cpu().numpy()
    blocks = []
    for ell, count in enumerate(basis.num_basis_by_degree):
        b = (1 + np.sqrt(4 * ell * (ell + 1) + 25)) / 2
        prefactor = x**ell / (1 + x**2) ** ((ell + b + 2) / 2)
        blocks.append(
            np.concatenate(
                [
                    prefactor
                    * eval_jacobi(n, ell + 0.5, b - 0.5, (1 - x * x) / (1 + x * x))
                    for n in range(count)
                ],
                axis=-1,
            )
        )
    expected = (
        torch.tensor(np.concatenate(blocks, axis=-1), device=DEVICE) * basis.normalizer
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    assert (basis.eigenvalues <= max_eigenvalue).all()
    assert basis(torch.empty(0, 1, device=DEVICE)).shape == (0, basis.num_basis)
    points, weights = roots_legendre(256)
    angle = torch.tensor((points + 1) * np.pi / 4, device=DEVICE)[:, None]
    x = angle.tan()
    measure = (
        torch.tensor(weights * np.pi / 4, device=DEVICE)[:, None]
        * x.square()
        * (1 + x.square())
    )
    for block in basis(x).split(basis.num_basis_by_degree, dim=-1):
        torch.testing.assert_close(
            block.T @ (measure * block),
            torch.eye(block.size(-1), device=DEVICE),
            atol=3e-10,
            rtol=3e-10,
        )


def test_basis_equation_and_derivatives():
    basis = PoschlTellerBasis(3, 80.0, length_scale=1.7).to(DEVICE)
    assert basis.num_basis_by_degree == (3, 2, 2, 1)
    r = torch.tensor([[0.17], [0.7], [1.9], [4.0]], device=DEVICE, requires_grad=True)
    value = basis(r)
    x = r / 1.7
    offset = 0
    for ell, count in enumerate(basis.num_basis_by_degree):
        for n in range(count):
            radial = value[:, offset + n : offset + n + 1]
            first = (
                torch.autograd.grad(
                    radial.sum(), r, create_graph=True, retain_graph=True
                )[0]
                * 1.7
            )
            second = torch.autograd.grad(first.sum(), r, retain_graph=True)[0] * 1.7
            weight = (1 + x.square()).square()
            operator = (
                -weight * second
                - (2 * weight / x + 4 * x * (1 + x.square())) * first
                + ell * (ell + 1) * weight / x.square() * radial
            )
            torch.testing.assert_close(
                operator, basis.eigenvalues[offset + n] * radial, atol=2e-10, rtol=2e-10
            )
        offset += count
    r = torch.tensor([[0.0], [0.6]], device=DEVICE, requires_grad=True)
    assert torch.autograd.gradcheck(basis, (r,))
    assert torch.autograd.gradgradcheck(basis, (r,))


@pytest.mark.parametrize("hidden", [[], [8], [8, 8]])
def test_path_mlp_initialization_matches_mlp(hidden):
    tp = o3.FullyConnectedTensorProduct(
        "2x0e + 2x1o",
        "0e + 1o",
        "2x0e + 2x1o",
        internal_weights=False,
        shared_weights=False,
    )
    state = torch.get_rng_state()
    mlp = PathRadialMLP(tp, (3, 2), hidden).to(DEVICE)
    torch.set_rng_state(state)
    for ell, actual in zip(mlp.degrees, mlp.radial_mlps):
        expected = MLP(
            [mlp.num_basis[ell], *hidden, actual[-1].out_dim],
            bias=False,
            act="silu",
        ).mlp.to(DEVICE)
        assert len(actual) == len(expected)
        for layer, reference in zip(actual, expected):
            assert type(layer) is type(reference)
            if isinstance(layer, mlpLinear):
                assert layer.bias is None
                assert layer.alpha == reference.alpha
                torch.testing.assert_close(layer.weight, reference.weight, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_path_weights_cutoff_and_compile(dtype):
    torch.set_default_dtype(dtype)
    basis = RadialBasis(
        radial_basis="space_pt",
        lmax=2,
        max_eigenvalue=80,
        apply_cutoff=False,
        cutoff=5.0,
    ).to(device=DEVICE, dtype=dtype)
    tp = o3.TensorProduct(
        "2x0e + 2x1o",
        "0e + 1o + 2e",
        "2x0e + 2x1o + 2x2e",
        [
            (1, 1, 2, "uvu", True),
            (0, 0, 0, "uvu", True),
            (0, 2, 2, "uvu", True),
            (0, 1, 1, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    mlp = PathRadialMLP(tp, basis.num_basis_by_degree, [8, 8]).to(DEVICE)
    radial = torch.randn(4, basis.num_basis, device=DEVICE, requires_grad=True)
    actual = mlp(radial)
    degree_weights = dict(
        zip(
            mlp.degrees,
            [
                network(block)
                for network, block in zip(
                    mlp.radial_mlps, radial.split(mlp.num_basis, -1)
                )
            ],
        )
    )
    offsets = dict.fromkeys(mlp.degrees, 0)
    expected = []
    for instruction in tp.instructions:
        ell = tp.irreps_in2[instruction.i_in2].ir.l
        size = prod(instruction.path_shape)
        expected.append(degree_weights[ell][:, offsets[ell] : offsets[ell] + size])
        offsets[ell] += size
    torch.testing.assert_close(actual, torch.cat(expected, -1))
    assert actual.shape[-1] == tp.weight_numel
    compiled = torch.compile(mlp, backend="aot_eager", fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(radial), actual)
    torch.testing.assert_close(
        torch.autograd.grad(compiled(radial).sum(), radial)[0],
        torch.autograd.grad(actual.sum(), radial)[0],
    )
    compiled_basis = torch.compile(
        basis.radial_fn, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for size in (4, 7, 0):
        r = torch.linspace(0, 5, size, device=DEVICE).unsqueeze(-1).requires_grad_()
        actual = compiled_basis(r)
        torch.testing.assert_close(actual, basis.radial_fn(r))
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), r)[0],
            torch.autograd.grad(basis.radial_fn(r).sum(), r)[0],
        )
    cutoff = C2PolynomialCutoff(5.0, 5).to(DEVICE)
    r = torch.tensor([[5.0], [5.1]], device=DEVICE, requires_grad=True)
    output = mlp(basis.radial_fn(r)) * cutoff(r)
    for _ in range(3):
        torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
        output = torch.autograd.grad(output.sum(), r, create_graph=True)[0]


def test_model_force_training_and_reload(tmp_path):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=3,
        Lmax=2,
        lmax=2,
        mmax=2,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["radial_basis"].update(
        radial_basis="space_pt",
        hidden=[8, 8],
        max_eigenvalue=80.0,
        bias=False,
        apply_cutoff=False,
    )
    config["atomic_basis"].update(
        nonlinear="gate", scalar_act="silu", tensor_act="sigmoid"
    )
    config["scale_shift"]["enable"] = False
    statistics = [
        {"atomic_numbers": [1], "atomic_energy": {1: 0.0}, "avg_num_neighbors": 2.0}
    ]
    model = (
        create_model(config, statistics, ["energy", "forces"], []).to(DEVICE).train()
    )
    data = dict(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.3], [0.3, 1.5, -0.1]], device=DEVICE
        ),
        node_attrs=torch.ones(3, 1, device=DEVICE),
        edge_index=torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], device=DEVICE
        ),
        edge_shifts=torch.zeros(6, 3, device=DEVICE),
        batch=torch.zeros(3, dtype=torch.long, device=DEVICE),
        ptr=torch.tensor([0, 3], device=DEVICE),
        lattice=torch.eye(3, device=DEVICE).unsqueeze(0) * 10,
        fidelity_idx=torch.zeros(1, dtype=torch.long, device=DEVICE),
    )
    output = model(data)
    loss = output["energy"].square().mean() + output["forces"].square().mean()
    loss.backward()
    for layer in model.readout_fn.representation.interactions:
        assert layer.edge_info.weight_numel == layer.rejector.weight_numel
        for parameter in layer.edge_info.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    path = tmp_path / "model.pt"
    export_tace(model, str(path))
    loaded = load_tace(path, device=DEVICE, dtype=torch.float64).train()
    restored = loaded(data)
    for name in ("energy", "forces"):
        torch.testing.assert_close(restored[name], output[name])
    rotation = o3.rand_matrix(device=DEVICE)
    transformed = model({**data, "positions": data["positions"] @ rotation.T})
    torch.testing.assert_close(
        transformed["energy"], output["energy"], atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        transformed["forces"], output["forces"] @ rotation.T, atol=1e-10, rtol=1e-10
    )


def test_invalid_basis_configuration():
    with pytest.raises(ValueError, match="ell=3"):
        PoschlTellerBasis(3, max_eigenvalue=50)
    with pytest.raises(ValueError, match="cutoff must follow"):
        RadialBasis(radial_basis="space_pt", apply_cutoff=True)
