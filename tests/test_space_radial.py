"""SPACE basis fidelity, CGTP weight layout, and OEQ force training."""

import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from ase.data import covalent_radii
from e3nn import o3
from scipy.interpolate import CubicHermiteSpline

from tace.lightning import create_model, export_tace, load_tace
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG, check_model_config
from tace.models.linear import torchLinear
from tace.models.space import SpaceRadialBasis, SpaceRadialMLP

physical_basis = pytest.importorskip("physical_basis")
pytestmark = pytest.mark.usefixtures("double_precision")


@pytest.mark.parametrize(
    "lmax,eigenvalue,counts", [(3, 50.0, (3, 3, 2, 1)), (5, 100.0, (5, 5, 4, 3, 2, 2))]
)
def test_official_physical_basis_values_and_derivatives(lmax, eigenvalue, counts):
    module = SpaceRadialBasis(lmax, [1, 8], 5.0, max_eigenvalue=eigenvalue)
    assert module.n_max_l == counts
    source = physical_basis.PhysicalBasis()
    r = torch.linspace(0.01, 3.9, 97).reshape(-1, 1).requires_grad_()
    node_type = torch.tensor([0, 1])
    edges = torch.tensor([[0, 1], [1, 0]]).repeat(1, 49)[:, : len(r)]
    scale = 0.1 + 0.7 * (covalent_radii[1] + covalent_radii[8])
    x = r.detach().numpy().reshape(-1) / scale
    normalization = math.sqrt(5.0**3 / 3)
    value = torch.cat(module(r, node_type, edges), dim=-1)
    expected = torch.tensor(
        np.stack(
            [
                source.compute(n, l, x)
                for l, count in enumerate(counts)
                for n in range(count)
            ],
            axis=-1,
        )
        * normalization
    )
    # SPACE refines on mean error, not on the worst point near the origin.
    assert (value - expected).abs().mean() < 5e-8
    assert (value - expected).abs().max() < 5e-6
    knots = np.linspace(0, 10, len(module.spline_values))
    spline = CubicHermiteSpline(
        knots,
        np.stack(
            [
                source.compute(n, l, knots)
                for l, count in enumerate(counts)
                for n in range(count)
            ],
            axis=-1,
        )
        * normalization,
        np.stack(
            [
                source.compute_derivative(n, l, knots)
                for l, count in enumerate(counts)
                for n in range(count)
            ],
            axis=-1,
        )
        * normalization,
    )
    torch.testing.assert_close(value, torch.tensor(spline(x)), atol=2e-13, rtol=2e-13)
    adjoint = torch.randn_like(value)
    derivative = torch.autograd.grad((value * adjoint).sum(), r)[0]
    exact_derivatives = torch.tensor(spline(x, 1) / scale)
    torch.testing.assert_close(
        derivative,
        (exact_derivatives * adjoint).sum(-1, keepdim=True),
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.parametrize("bias", [False, True])
def test_cutoffs_empty_edges_and_higher_derivatives(bias):
    basis = SpaceRadialBasis(3, [1], 5.0)
    tp = o3.FullyConnectedTensorProduct(
        "2x0e+2x1o",
        o3.Irreps.spherical_harmonics(3),
        "2x0e+2x1o+2x2e",
        internal_weights=False,
        shared_weights=False,
    )
    network = SpaceRadialMLP(tp, basis.n_max_l, hidden=[5, 7], bias=bias)
    if bias:
        with torch.no_grad():
            for name, parameter in network.named_parameters():
                if name.endswith("bias"):
                    parameter.normal_()
    types = torch.zeros(2, dtype=torch.long)
    r = (
        torch.tensor([0.0, 4.0, 4.3, 4.9, 5.0, 5.1, 8.0])
        .reshape(-1, 1)
        .requires_grad_()
    )
    edges = torch.tensor([[0], [1]]).expand(2, len(r))
    weights = network(*basis.forward_with_cutoff(r, types, edges))
    assert torch.count_nonzero(weights[4:]) == 0
    gradients = torch.autograd.grad(weights.sum(), r, create_graph=True)[0]
    second = torch.autograd.grad(gradients.sum(), r)[0]
    assert torch.isfinite(gradients).all() and torch.isfinite(second).all()
    assert torch.count_nonzero(gradients[4:]) == 0
    assert torch.count_nonzero(second[4:]) == 0
    assert network(*basis.forward_with_cutoff(r[:0], types, edges[:, :0])).shape == (
        0,
        tp.weight_numel,
    )

    # The physical-domain cap is independent of the neighbor cutoff.
    long_basis = SpaceRadialBasis(3, [1], 12.0)
    cap = 10 * (0.1 + 1.4 * covalent_radii[1])
    capped = long_basis(torch.tensor([[cap], [cap + 0.1]]), types, edges[:, :2])
    assert torch.count_nonzero(torch.cat(capped, -1)) == 0

    interior = torch.tensor([[1.1357], [4.3678]], requires_grad=True)

    def evaluate(distances):
        return network(*basis.forward_with_cutoff(distances, types, edges[:, :2]))

    assert torch.autograd.gradcheck(evaluate, (interior,))
    assert torch.autograd.gradgradcheck(evaluate, (interior,))
    compiled = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(interior), evaluate(interior))


@pytest.mark.parametrize("hidden", [[], [5], [7, 3, 5]])
@pytest.mark.parametrize("bias", [False, True])
def test_instruction_order_keeps_all_path_and_channel_weights(hidden, bias):
    # Deliberately interleave angular degrees and repeat l2=1 on distinct paths.
    tp = o3.TensorProduct(
        "2x1o",
        "1x0e+1x1o+1x2e",
        "2x1o+2x0e+2x2e+2x1o",
        [
            (0, 2, 3, "uvu", True),
            (0, 0, 0, "uvu", True),
            (0, 1, 2, "uvu", True),
            (0, 1, 1, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    module = SpaceRadialMLP(tp, (3, 2, 1), hidden=hidden, bias=bias)
    assert module.hidden_widths == tuple(hidden)
    assert module.num_layers == len(hidden) + 1
    assert module.weight_numel_by_l == (2, 4, 2)
    assert module.weight_numel == tp.weight_numel == 8
    for l, mlp in module.radial_mlps.items():
        widths = [module.n_max_l[int(l)], *hidden, module.weight_numel_by_l[int(l)]]
        linears = [layer for layer in mlp if isinstance(layer, torchLinear)]
        assert [(layer.in_features, layer.out_features) for layer in linears] == list(
            zip(widths, widths[1:])
        )
        assert all((layer.bias is not None) == bias for layer in linears)
    basis = tuple(torch.randn(7, n) for n in module.n_max_l)
    by_l = [module.radial_mlps[str(l)](values) for l, values in enumerate(basis)]
    weights = module(basis, torch.ones(7, 1))
    expected = torch.cat((by_l[2], by_l[0], by_l[1][:, :2], by_l[1][:, 2:]), dim=-1)
    torch.testing.assert_close(weights, expected, atol=0, rtol=0)
    assert tp(torch.ones(7, 6), torch.randn(7, 9), weights).shape == (
        7,
        tp.irreps_out.dim,
    )


def _model_config(lmax=3, mlp_type="space"):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=2,
        lmax=lmax,
        cutoff=5.0,
        max_neighbors=None,
        target_property=["energy", "forces", "stress"],
        statistics=[
            dict(
                atomic_numbers=[1, 8],
                atomic_energy={1: 0.0, 8: 0.0},
                avg_num_neighbors=2.0,
            )
        ],
    )
    config["radial_basis"].update(mlp_type=mlp_type, hidden=[4, 7, 3])
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    return config


def _data(device):
    return dict(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [4.3, 0.1, -0.2]], device=device
        ),
        node_attrs=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device=device),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device=device
        ),
        edge_shifts=torch.zeros(6, 3, device=device),
        lattice=torch.eye(3, device=device).unsqueeze(0) * 8,
        batch=torch.zeros(3, dtype=torch.long, device=device),
        ptr=torch.tensor([0, 3], device=device),
        fidelity_idx=torch.zeros(1, dtype=torch.long, device=device),
    )


@pytest.mark.parametrize("lmax", [3, 5])
@pytest.mark.parametrize("backend", ["native", "oeq"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
def test_model_roundtrip_and_force_training(
    tmp_path, monkeypatch, lmax, backend, dtype, bias
):
    torch.set_default_dtype(dtype)
    if backend == "oeq":
        if not torch.cuda.is_available():
            pytest.skip("OEQ needs a CUDA device")
        pytest.importorskip("openequivariance")
    device = "cuda" if backend == "oeq" else "cpu"
    config = _model_config(lmax)
    config["radial_basis"]["bias"] = bias
    properties = ["energy", "forces", "stress"]
    reference = (
        create_model(config, config["statistics"], properties, []).to(device).train()
    )
    if bias:
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if ".radial_mlps." in name and name.endswith("bias"):
                    parameter.normal_(std=0.2)
    dense_config = _model_config(lmax, "dense")
    dense = create_model(dense_config, dense_config["statistics"], properties, [])
    for new, old in zip(
        reference.readout_fn.representation.interactions,
        dense.readout_fn.representation.interactions,
    ):
        assert new.rejector.tp.instructions == old.rejector.tp.instructions
        assert new.edge_info.weight_numel == old.rejector.weight_numel
        assert sum(new.edge_info.weight_numel_by_l) == old.rejector.weight_numel
        assert new.edge_info.hidden_widths == tuple(config["radial_basis"]["hidden"])
        assert new.edge_info.use_bias == bias

    path = tmp_path / "space.pt"
    export_tace(reference, str(path))
    if backend == "oeq":
        monkeypatch.setenv("TACE_USE_OEQ", "1")
    restored = load_tace(path, device=device, dtype=dtype, strict=True).train()
    if backend == "oeq":
        assert all(
            layer.rejector.use_oeq
            for layer in restored.readout_fn.representation.interactions
        )
    outputs = [model(_data(device)) for model in (reference, restored)]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-9
    for name in properties:
        torch.testing.assert_close(
            outputs[0][name], outputs[1][name], atol=tolerance, rtol=tolerance
        )
    for output in outputs:
        sum(output[name].square().sum() for name in properties).backward()
    restored_parameters = dict(restored.named_parameters())
    radial_parameters = 0
    for name, parameter in reference.named_parameters():
        other = restored_parameters[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(
                parameter.grad, other.grad, atol=10 * tolerance, rtol=10 * tolerance
            )
        if ".radial_mlps." in name:
            radial_parameters += 1
            assert parameter.grad is not None
    assert radial_parameters > 0


def test_rejects_incomplete_bases_and_incompatible_config():
    with pytest.raises(ValueError, match="l2=3"):
        SpaceRadialBasis(3, [1], 5.0, max_eigenvalue=25.0)
    for section, key, value, match in (
        ("atomic_basis", "type", "o2_cgtp", "atomic_basis.type"),
        ("edge_update", "type", "element2", "edge_update.type"),
        ("radial_basis", "mlp_type", "unknown", "mlp_type"),
    ):
        config = _model_config()
        config[section][key] = value
        with pytest.raises(ValueError, match=match):
            check_model_config(config)
    for key in ("mlp_depth", "mlp_expansion_ratio"):
        config = _model_config()
        config["radial_basis"]["space"][key] = 4
        with pytest.raises(ValueError, match="radial_basis.hidden"):
            check_model_config(config)
