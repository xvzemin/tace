from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from e3nn import o3
from e3nn.nn import Gate
from torch_geometric.data import Data

from tace.dataset.augmentation import AugmentedDataset, validate_augmentations
from tace.dataset.quantity import PROPERTIES, TIME_ODD_PROPERTIES
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG, check_model_config
from tace.models._e3nn.edge import EDGE_UPDATE
from tace.models._e3nn.fused import (
    O3TensorProductConv,
    UuuTensorProduct,
    UvuTensorProduct,
)
from tace.models._e3nn.layer_norm import get_normalization_layer
from tace.models._e3nn.magnetic import MagneticBasis
from tace.models._e3nn.node import ElementNodeUpdate
from tace.models._e3nn.nonlinear import get_nonlinear_layer
from tace.models._e3nn.paths import generate_paths
from tace.models._e3nn.readout import TensorReadOut
from tace.models._e3nn.tace import e3nnTACE
from tace.models._e3nn.ue import UniversalEquivariantEmbedding
from tace.models.adapter import TensorModel
from tace.models.angular import SolidHarmonics
from tace.models.layout import LayoutTransform
from tace.models.linear import e3nnLinear
from tace.models.time_reversal import (
    spherical_harmonics_irreps,
    supports_time_reversal,
    with_natural_parity,
    with_time_reversal,
)

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_config() -> dict:
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        statistics=[
            {
                "atomic_numbers": [1],
                "avg_num_neighbors": 2.0,
                "atomic_energy": {1: 0.0},
            }
        ],
        num_layers=1,
        num_channel=2,
        Lmax=1,
        lmax=1,
        target_property=["energy"],
    )
    config["fidelity"] = [{"name": "PBE", "atomic_energy": None}]
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [2]
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["scale_shift"]["enable"] = False
    return config


def _time_reversal_model_config() -> dict:
    config = _model_config()
    config["universal_embedding"]["magnetic_field"]["enable"] = True
    return config


@pytest.mark.parametrize(
    "edge_update,mlp_type", [("element2", "mlp_v1"), ("element2_v2", "mlp_v2")]
)
def test_magnetic_wigner6j_model(edge_update, mlp_type, monkeypatch, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = _model_config()
    config.update(num_layers=2, parity=True, mlp_type=mlp_type)
    config["statistics"][0].update(
        atomic_numbers=[1, 8], atomic_energy={1: 0.0, 8: 0.0}
    )
    config["target_property"] = ["energy", "forces", "noncollinear_magnetic_forces"]
    config["fidelity"][0]["magnetic_scale"] = 2.0
    config["atomic_basis"]["type"] = "w6j_mag"
    config["angular_basis"]["magnetic_Lmax"] = 1
    config["radial_basis"]["apply_cutoff"] = False
    config["edge_embedding"]["type"] = "linear"
    config["edge_update"]["type"] = edge_update
    model = TensorModel(e3nnTACE(**config)).cuda().train()
    representation = model.readout_fn.representation
    assert not hasattr(representation, "magnetic_edge_embedding")
    for node_update, spatial, magnetic in zip(
        representation.node_updates,
        representation.edge_updates,
        representation.magnetic_edge_updates,
    ):
        assert isinstance(node_update, ElementNodeUpdate)
        assert isinstance(spatial, EDGE_UPDATE[edge_update])
        assert isinstance(magnetic, EDGE_UPDATE["identity"])
        assert magnetic.out_dim == config["num_channel"]
    data = {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [-0.3, 1.1, 0.4]], device="cuda"
        ),
        "node_attrs": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device="cuda"),
        "edge_index": torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device="cuda"
        ),
        "edge_shifts": torch.zeros(6, 3, device="cuda"),
        "lattice": torch.eye(3, device="cuda").unsqueeze(0) * 10,
        "batch": torch.zeros(3, dtype=torch.long, device="cuda"),
        "ptr": torch.tensor([0, 3], device="cuda"),
        "initial_noncollinear_magmoms": torch.randn(3, 3, device="cuda") / 2,
    }
    properties = config["target_property"]
    radial, _, _ = representation.magnetic_basis(
        data["initial_noncollinear_magmoms"],
        data["node_attrs"],
        data["edge_index"],
        torch.zeros(3, dtype=torch.long, device="cuda"),
    )
    radial = radial.detach().requires_grad_()
    captured = []
    hooks = [
        interaction.magnetic_edge_info.register_forward_hook(
            lambda module, args, output: captured.append((args[0], output))
        )
        for interaction in representation.interactions
    ]
    model(dict(data))
    for hook in hooks:
        hook.remove()
    source = data["edge_index"][0]
    for node_update, interaction, (features, weights) in zip(
        representation.node_updates, representation.interactions, captured
    ):
        node_features = node_update(radial, data["node_attrs"])[0]
        expected_weights = interaction.magnetic_edge_info(node_features)[source]
        torch.testing.assert_close(features, node_features[source])
        torch.testing.assert_close(weights, expected_weights)
        # Outgoing edges with different targets use the same source weights.
        torch.testing.assert_close(weights[0], weights[2])
        gradients = torch.autograd.grad(expected_weights[0].sum(), radial)[0]
        assert gradients[1:].count_nonzero() == 0
        assert gradients[0].count_nonzero() > 0
    with monkeypatch.context() as patch:
        for interaction in representation.interactions:
            tp = interaction.rejector.eqx_tp
            patch.setattr(
                tp,
                "forward",
                lambda *args, reference=tp.forward_reference, **kwargs: reference(
                    *args
                ),
            )
        expected = model(dict(data))
        expected_grad = torch.autograd.grad(
            sum(expected[name].square().sum() for name in properties),
            tuple(model.parameters()),
            allow_unused=True,
        )
    for enabled in ("0", "1"):
        monkeypatch.setenv("TACE_USE_EQX", enabled)
        actual = model(dict(data))
        for name in properties:
            torch.testing.assert_close(
                actual[name], expected[name], atol=1e-10, rtol=1e-10
            )
        actual_grad = torch.autograd.grad(
            sum(actual[name].square().sum() for name in properties),
            tuple(model.parameters()),
            allow_unused=True,
        )
        for a, b in zip(actual_grad, expected_grad):
            if b is None:
                assert a is None
            else:
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    # An axial magnetic moment is unchanged by spatial inversion.
    baseline = model(dict(data))["energy"]
    for rotation, reverse_time in (
        (o3.rand_matrix().cuda(), False),
        (-torch.eye(3, device="cuda"), False),
        (torch.eye(3, device="cuda"), True),
    ):
        transformed = dict(data)
        transformed["positions"] = data["positions"] @ rotation.T
        transformed["initial_noncollinear_magmoms"] = (
            data["initial_noncollinear_magmoms"]
            @ rotation.T
            * rotation.det()
            * (-1 if reverse_time else 1)
        )
        if reverse_time and not supports_time_reversal():
            continue
        torch.testing.assert_close(
            model(transformed)["energy"], baseline, atol=1e-10, rtol=1e-10
        )


def _time_parities(irreps: o3.Irreps) -> list[int]:
    return [getattr(ir, "t", 1) for _, ir in irreps]


def _time_reversal_matrix(irreps: o3.Irreps) -> torch.Tensor:
    return irreps.D_from_matrix(
        -torch.eye(3),
        parity=False,
        time_reversal=True,
    )


def test_magnetic_property_metadata_is_separate_from_spatial_irreps():
    assert all("time_reversal" in quantity for quantity in PROPERTIES.values())
    assert PROPERTIES["initial_collinear_magmoms"]["irreps"] == "1x0e"
    assert PROPERTIES["initial_collinear_magmoms"]["time_reversal"] == -1
    assert PROPERTIES["initial_noncollinear_magmoms"]["irreps"] == "1x1e"
    assert PROPERTIES["initial_noncollinear_magmoms"]["time_reversal"] == -1
    assert PROPERTIES["abs_final_collinear_magmoms"]["time_reversal"] == 1

    assert set(TIME_ODD_PROPERTIES) == {
        "initial_collinear_magmoms",
        "final_collinear_magmoms",
        "collinear_magnetic_forces",
        "total_collinear_magmom",
        "initial_noncollinear_magmoms",
        "final_noncollinear_magmoms",
        "noncollinear_magnetic_forces",
        "total_noncollinear_magmom",
        "magnetic_field",
    }
    if supports_time_reversal():
        assert str(with_time_reversal("1x0e", -1)) == "1x0eo"
        assert str(with_time_reversal("1x1e", -1)) == "1x1eo"
        assert str(with_time_reversal("1x0e", 1)) == "1x0ee"


def test_spin_rotation_augmentation_uses_one_rotation_per_structure():
    moments = torch.eye(3, dtype=torch.float64)
    magnetic_forces = 2 * moments
    positions = torch.randn(3, 3, dtype=torch.float64)
    data = Data(
        positions=positions,
        initial_noncollinear_magmoms=moments,
        noncollinear_magnetic_forces=magnetic_forces,
    )

    transformed = AugmentedDataset([data], ["spin_rotation"])[0]
    rotation_transpose = transformed.initial_noncollinear_magmoms

    assert torch.allclose(rotation_transpose @ rotation_transpose.mT, moments)
    assert torch.allclose(
        transformed.noncollinear_magnetic_forces,
        2 * transformed.initial_noncollinear_magmoms,
    )
    assert torch.equal(transformed.positions, positions)
    assert torch.equal(data.initial_noncollinear_magmoms, moments)


def test_time_reversal_augmentation_flips_all_time_odd_properties(monkeypatch):
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.zeros(*size, **kwargs),
    )
    data = Data(
        energy=torch.tensor([2.0]),
        initial_collinear_magmoms=torch.tensor([1.0, -2.0]),
        initial_noncollinear_magmoms=torch.tensor([[1.0, 2.0, 3.0]]),
        noncollinear_magnetic_forces=torch.tensor([[4.0, 5.0, 6.0]]),
        magnetic_field=torch.tensor([[0.1, 0.2, 0.3]]),
    )

    transformed = AugmentedDataset([data], ["time_reversal"])[0]

    assert torch.equal(transformed.energy, data.energy)
    for name in ("initial_noncollinear_magmoms", "noncollinear_magnetic_forces"):
        assert torch.equal(getattr(transformed, name), -getattr(data, name))
    assert torch.equal(
        transformed.initial_collinear_magmoms,
        data.initial_collinear_magmoms,
    )
    assert torch.equal(transformed.magnetic_field, data.magnetic_field)


def test_dataset_augmentation_configuration_accepts_supported_forms():
    assert validate_augmentations(None) == ()
    assert validate_augmentations([]) == ()
    assert validate_augmentations("spin_rotation") == ("spin_rotation",)
    assert validate_augmentations(["spin_rotation", "time_reversal"]) == (
        "spin_rotation",
        "time_reversal",
    )


def test_time_reversal_helpers_support_both_e3nn_variants():
    irreps = with_time_reversal("1x0e + 1x1e", -1)
    magnetic_irreps = spherical_harmonics_irreps(
        2,
        p=1,
        time_reversal=-1,
    )

    if supports_time_reversal():
        assert _time_parities(irreps) == [-1, -1]
        assert _time_parities(magnetic_irreps) == [1, -1, 1]
    else:
        assert _time_parities(irreps) == [1, 1]
        assert _time_parities(magnetic_irreps) == [1, 1, 1]

    angular_basis = SolidHarmonics(magnetic_irreps)
    assert angular_basis.irreps_out == magnetic_irreps
    assert getattr(angular_basis.irreps_in[0].ir, "t", 1) == (
        -1 if supports_time_reversal() else 1
    )
    unnatural_irreps = with_time_reversal("1x0o + 1x1e + 1x2o", -1)
    natural_irreps = with_natural_parity(unnatural_irreps)

    assert [ir.p for _, ir in natural_irreps] == [1, -1, 1]
    assert _time_parities(natural_irreps) == _time_parities(unnatural_irreps)


def test_readout_uses_selected_spatial_parity():
    readout_kwargs = {
        "layer": 0,
        "num_layers": 2,
        "hidden_channel": [],
        "bias": False,
        "num_elements": 1,
        "num_fidelities": 1,
        "irreps_in": o3.Irreps("2x1o"),
        "irreps_out": o3.Irreps("1e"),
    }
    natural_readout = TensorReadOut(parity=False, **readout_kwargs)
    complete_readout = TensorReadOut(parity=True, **readout_kwargs)

    assert natural_readout.irreps_out[0].ir.p == -1
    assert complete_readout.irreps_out[0].ir.p == 1


def test_universal_embedding_uses_property_time_reversal_metadata():
    embedding = UniversalEquivariantEmbedding(
        irreps_in=o3.Irreps("2x0e"),
        num_channel=2,
        num_elements=2,
        config={"magnetic_field": {"normalizer": 1.0}},
        time_reversal=True,
    )
    input_irrep = embedding.uee["magnetic_field"].irreps_in[0].ir
    assert getattr(input_irrep, "t", 1) == (-1 if supports_time_reversal() else 1)

    legacy_embedding = UniversalEquivariantEmbedding(
        irreps_in=o3.Irreps("2x0e"),
        num_channel=2,
        num_elements=2,
        config={"magnetic_field": {"normalizer": 1.0}},
        time_reversal=False,
    )
    legacy_irrep = legacy_embedding.uee["magnetic_field"].irreps_in[0].ir
    assert getattr(legacy_irrep, "t", 1) == 1


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_magnetic_field_uses_time_odd_equivariant_embedding():
    config = _model_config()
    config["parity"] = False
    config["universal_embedding"]["magnetic_field"]["enable"] = True
    model = e3nnTACE(**config)
    representation = model.representation

    assert representation.invariant_property == []
    assert representation.equivariant_property == ["magnetic_field"]
    ir = representation.uee_embeddings[0].uee["magnetic_field"].irreps_in[0].ir
    assert ir.l == 1 and ir.p == -1 and ir.t == -1

    config["parity"] = True
    full_o3_model = e3nnTACE(**config)
    full_o3_ir = (
        full_o3_model.representation.uee_embeddings[0]
        .uee["magnetic_field"]
        .irreps_in[0]
        .ir
    )
    assert full_o3_ir.l == 1 and full_o3_ir.p == 1 and full_o3_ir.t == -1


def test_parity_selects_natural_or_complete_magnetic_paths():
    config = _model_config()
    config["parity"] = False
    config["atomic_basis"]["type"] = "o2_mag"
    config["angular_basis"]["magnetic_Lmax"] = 1
    config["fidelity"][0]["magnetic_scale"] = 2.0
    config["mmax"] = 1

    natural_model = e3nnTACE(**config)
    natural_representation = natural_model.representation
    natural_basis = natural_representation.magnetic_basis.angular_basis
    assert natural_basis.irreps_in[0].ir.p == -1
    assert natural_basis.irreps_out == natural_representation.magnetic_node_irreps_out
    for irreps in (
        natural_representation.magnetic_node_irreps_out,
        natural_representation.magnetic_edge_irreps_out,
        natural_representation.interactions[0].irreps_out,
        natural_representation.products[0].irreps_out,
    ):
        assert all(ir.p == (-1) ** ir.l for _, ir in irreps)

    config["parity"] = True
    complete_model = e3nnTACE(**config)
    complete_representation = complete_model.representation
    complete_basis = complete_representation.magnetic_basis.angular_basis
    assert complete_basis.irreps_in[0].ir.p == 1
    assert complete_basis.irreps_out == complete_representation.magnetic_node_irreps_out
    assert _time_parities(natural_basis.irreps_out) == _time_parities(
        complete_basis.irreps_out
    )
    assert any(
        ir.p != (-1) ** ir.l
        for _, ir in complete_representation.magnetic_node_irreps_out
    )
    if complete_representation.magnetic_edge_irreps_out is not None:
        assert any(
            ir.p != (-1) ** ir.l
            for _, ir in complete_representation.magnetic_edge_irreps_out
        )


@pytest.mark.parametrize("num_fidelities", [1, 2])
def test_magnetic_representation_uses_node_fidelity(num_fidelities):
    config = _model_config()
    config["atomic_basis"]["type"] = "o2_mag"
    config["angular_basis"]["magnetic_Lmax"] = 1
    config["radial_basis"]["apply_cutoff"] = False
    config["mmax"] = 1
    config["statistics"] *= num_fidelities
    config["fidelity"] = [
        {"name": "PBE", "magnetic_scale": 2.0},
        {"name": "SCAN", "magnetic_scale": 4.0},
    ][:num_fidelities]
    model = TensorModel(e3nnTACE(**config)).double().eval()
    representation = model.readout_fn.representation
    assert representation.magnetic_basis.magnetic_scale.shape == (num_fidelities, 1)
    data = {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * 2, dtype=torch.float64
        ),
        "node_attrs": torch.ones(4, 1, dtype=torch.float64),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        "edge_shifts": torch.zeros(4, 3, dtype=torch.float64),
        "lattice": torch.eye(3, dtype=torch.float64).repeat(2, 1, 1) * 10.0,
        "batch": torch.tensor([0, 0, 1, 1]),
        "ptr": torch.tensor([0, 2, 4]),
        "fidelity_idx": torch.tensor([1 % num_fidelities, 0]),
        "initial_noncollinear_magmoms": torch.tensor(
            [[1.0, 0.0, 0.0]] * 4, dtype=torch.float64
        ),
    }
    graph = model.prepare_graph(data)
    output = representation(data, graph)
    expected = representation.magnetic_basis.radial_basis(
        torch.tensor(
            [[0.5]] * 4 if num_fidelities == 1 else [[0.875], [0.875], [0.5], [0.5]],
            dtype=torch.float64,
        )
    )
    torch.testing.assert_close(output["magnetic_radial_basis"], expected)

    data.pop("fidelity_idx")
    for fidelity_idx in range(num_fidelities):
        model.reset_fidelity_idx(fidelity_idx)
        output = representation(data, model.prepare_graph(data))
        torch.testing.assert_close(
            output["magnetic_radial_basis"],
            expected[2 - 2 * fidelity_idx].expand(4, -1),
        )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_o2_magnetic_interactions_keep_complete_soc_irreps():
    config = _model_config()
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=2,
        lmax=3,
        mmax=2,
        parity=True,
    )
    config["fidelity"][0]["magnetic_scale"] = {1: 2.0}
    config["angular_basis"].update(
        magnetic_Lmax=2,
        use_spin_orbit_coupling=True,
    )
    config["atomic_basis"].update(
        type="o2_mag",
        nonlinear="gate",
        edge_nonlinear="gate",
        use_radial_rotary_attention=False,
    )
    config["product_basis"].update(type="cgtp", correlation=2)
    model = e3nnTACE(**config)

    for interaction in model.representation.interactions:
        output_lmax = (
            interaction.Lmax if interaction.correlation == 1 else interaction.lmax
        )
        expected = {
            ir_out
            for _, ir_node in interaction.irreps_in
            for _, ir_mag in interaction.magnetic_edge_irreps
            for _, ir_spatial in interaction.irreps_sh
            for ir_edge in ir_mag * ir_spatial
            for ir_out in ir_node * ir_edge
            if ir_out.l <= output_lmax
        }
        assert {ir for _, ir in interaction.irrreps_tp_out} == expected
        assert interaction.rejector.local_frame_out.global_irreps == (
            interaction.irreps_out
        )

    first_irreps = {
        str(ir) for _, ir in model.representation.interactions[0].irrreps_tp_out
    }
    assert first_irreps == {
        f"{l}{parity}{time_parity}"
        for l in range(4)
        for parity in "oe"
        for time_parity in "oe"
    }
    assert "1eo" in first_irreps


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_time_reversal_model_disables_automatic_eqt(monkeypatch):
    for name in ("TACE_USE_EQT", "TACE_USE_CUE", "TACE_USE_OEQ", "TACE_USE_EQX"):
        monkeypatch.delenv(name, raising=False)
    config = _time_reversal_model_config()
    config["product_basis"]["correlation"] = 3
    model = e3nnTACE(**config)
    representation = model.representation
    interaction = representation.interactions[0]

    assert model.use_time_reversal
    assert type(representation.o3_angular_basis).__module__.startswith("e3nn.")
    assert type(interaction.linear_up.linear).__module__.startswith("e3nn.")
    assert type(interaction.nonlinearity).__module__.startswith("e3nn.")
    assert type(interaction.rejector.tp).__module__.startswith("e3nn.")
    assert all(
        not tensor_product.use_eqt
        for product in model.representation.products
        for tensor_product in product.aces
    )
    assert all(
        type(tensor_product.tp).__module__.startswith("e3nn.")
        for product in representation.products
        for tensor_product in product.aces
    )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
@pytest.mark.parametrize("time_reversal", [False, True])
def test_o2_model_selects_time_reversal(monkeypatch, time_reversal):
    for name in ("TACE_USE_EQT", "TACE_USE_CUE", "TACE_USE_OEQ", "TACE_USE_EQX"):
        monkeypatch.delenv(name, raising=False)
    config = _time_reversal_model_config() if time_reversal else _model_config()
    config["mmax"] = 1
    config["atomic_basis"]["type"] = ["o2"]
    config["atomic_basis"]["edge_nonlinear"] = ["silu"]
    model = e3nnTACE(**config)

    assert model.representation.use_o2
    assert model.representation.use_time_reversal is time_reversal
    assert all(
        ir.t == 1
        for ir, _ in model.representation.interactions[0].rejector.local_irreps_out
    )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
@pytest.mark.parametrize("gate_m0", [False, True])
def test_gate_is_time_reversal_equivariant(gate_m0):
    irreps = o3.Irreps("2x0ee + 2x0eo + 2x1eo")
    gate, _, _ = get_nonlinear_layer(
        "gate",
        irreps,
        irreps,
        gate_m0=gate_m0,
    )
    assert isinstance(gate, Gate)
    features = torch.randn(5, gate.irreps_in.dim)
    input_matrix = _time_reversal_matrix(gate.irreps_in)
    output_matrix = _time_reversal_matrix(gate.irreps_out)

    expected = gate(features) @ output_matrix.T
    observed = gate(features @ input_matrix.T)
    torch.testing.assert_close(observed, expected)


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_linear_is_time_reversal_equivariant():
    irreps = o3.Irreps("2x0ee + 2x0eo + 2x1ee + 2x1eo")
    linear = e3nnLinear(irreps, irreps).to(dtype=torch.get_default_dtype())
    assert type(linear.linear).__module__.startswith("e3nn.")

    features = torch.randn(5, irreps.dim)
    matrix = _time_reversal_matrix(irreps)
    expected = linear(features) @ matrix.T
    observed = linear(features @ matrix.T)
    torch.testing.assert_close(observed, expected)


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_merge_layer_norm_is_time_reversal_equivariant():
    irreps = o3.Irreps("2x0oo + 2x0ee + 2x1eo")
    layout = LayoutTransform(irreps)
    norm = get_normalization_layer(
        "merge_layer_norm",
        ls=irreps.ls,
        num_channels=2,
        irreps=irreps,
    ).to(dtype=torch.get_default_dtype())
    with torch.no_grad():
        norm.affine_bias.fill_(0.25)

    features = torch.randn(5, irreps.dim)
    matrix = _time_reversal_matrix(irreps)
    expected = layout.inverse(norm(layout(features))) @ matrix.T
    observed = layout.inverse(norm(layout(features @ matrix.T)))
    torch.testing.assert_close(observed, expected)


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_time_odd_irreps_reject_eqt(monkeypatch):
    irreps_in = o3.Irreps("2x0ee + 2x1eo")
    target_irreps = o3.Irreps("2x0ee + 2x0eo + 2x1ee + 2x1eo + 2x2ee + 2x2eo")
    monkeypatch.setenv("TACE_USE_EQT", "1")
    with pytest.raises(ValueError, match="EQT does not support time-reversal"):
        UuuTensorProduct(
            irreps_in,
            irreps_in,
            target_irreps,
        )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
@pytest.mark.parametrize(
    ("kernel", "environment"),
    [("CUE", "TACE_USE_CUE"), ("OEQ", "TACE_USE_OEQ")],
)
def test_time_odd_irreps_reject_scatter_acceleration(
    monkeypatch,
    kernel,
    environment,
):
    node_irreps = o3.Irreps("2x0ee + 2x1eo")
    edge_irreps = spherical_harmonics_irreps(1, p=-1)
    target_irreps = o3.Irreps("2x0ee + 2x0oo + 2x1eo + 2x1oe + 2x1oo")

    monkeypatch.setenv(environment, "1")
    with pytest.raises(ValueError, match=f"{kernel} does not support"):
        O3TensorProductConv(
            node_irreps,
            edge_irreps,
            target_irreps,
        )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_time_odd_irreps_reject_oeq_tensor_product(monkeypatch):
    node_irreps = o3.Irreps("2x0ee + 2x1eo")
    edge_irreps = spherical_harmonics_irreps(1, p=-1)
    target_irreps = o3.Irreps("2x0ee + 2x0oo + 2x1eo + 2x1oe + 2x1oo")

    instructions, actual_irreps = generate_paths(
        target_irreps,
        node_irreps,
        edge_irreps,
        e3nn_mode="uvu",
    )
    monkeypatch.setenv("TACE_USE_OEQ", "1")
    with pytest.raises(ValueError, match="OEQ does not support time-reversal"):
        UvuTensorProduct(
            node_irreps,
            edge_irreps,
            actual_irreps,
            instructions,
            shared_weights=False,
        )


@pytest.mark.parametrize(
    ("magnetic_scales", "expected_scales"),
    [
        ([None], [2.5]),
        ([None, None], [2.5, 3.7]),
        ([2.0, {26: 4.0}], [2.0, 4.0]),
        ([None, {"26": 4.0}], [2.5, 4.0]),
    ],
)
def test_magnetic_scale_is_resolved_per_fidelity(magnetic_scales, expected_scales):
    config = check_model_config(
        {
            "statistics": [
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 2.0,
                    "max_noncollinear_magmoms_norm_by_element": {26: 2.0},
                },
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 4.0,
                    "max_noncollinear_magmoms_norm_by_element": {26: 3.0},
                },
            ][: len(magnetic_scales)],
            "target_property": [],
            "fidelity": [
                {"name": name, "magnetic_scale": scale}
                for name, scale in zip(("PBE", "SCAN"), magnetic_scales)
            ],
        }
    )

    assert [scale[26] for scale in config["magnetic_scale"]] == pytest.approx(
        expected_scales
    )


@pytest.mark.parametrize("magnetic_scale", [2.0, {26: 2.0}])
def test_manual_magnetic_scale_is_used_without_rescaling(magnetic_scale):
    config = check_model_config(
        {
            "statistics": [
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 2.0,
                    "max_noncollinear_magmoms_norm_by_element": {26: 9.0},
                }
            ],
            "target_property": [],
            "fidelity": [
                {"name": "PBE", "magnetic_scale": magnetic_scale},
            ],
        }
    )

    assert config["magnetic_scale"] == [{26: 2.0}]


@pytest.mark.parametrize("num_nodes", [0, 4])
@pytest.mark.parametrize("num_fidelities", [1, 2])
def test_magnetic_basis_selects_fidelity_and_element(num_nodes, num_fidelities):
    basis = MagneticBasis(
        [2.0, {26: 4.0, 28: 3.0}][:num_fidelities],
        num_mag_radial_basis=4,
        Lmax=1,
        atomic_numbers=[26, 28],
    ).to(DEVICE, DTYPE)
    magmoms = (
        torch.tensor(
            [[0.3, -0.4, 0.5]] * num_nodes,
            dtype=DTYPE,
            device=DEVICE,
        )
        .reshape(num_nodes, 3)
        .requires_grad_()
    )
    node_attrs = torch.eye(2, dtype=DTYPE, device=DEVICE)[
        torch.tensor([1, 0, 0, 1], device=DEVICE)[:num_nodes]
    ]
    node_fidelity = (
        torch.tensor([0, 1, 0, 1], device=DEVICE)[:num_nodes] % num_fidelities
    )
    edge_index = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]], device=DEVICE)[
        :, :num_nodes
    ]

    radial, node_attrs_out, edge_attrs_out = basis(
        magmoms, node_attrs, edge_index, node_fidelity
    )
    magnetic_scale = torch.tensor(
        [2.0] * 4 if num_fidelities == 1 else [2.0, 4.0, 2.0, 3.0],
        dtype=DTYPE,
        device=DEVICE,
    )[:num_nodes, None]
    squared_magnitude = (magmoms / magnetic_scale).square().sum(-1, keepdim=True)
    expected_radial = basis.radial_basis(1.0 - 2.0 * squared_magnitude.clamp(max=1.0))
    torch.testing.assert_close(radial, expected_radial)
    torch.testing.assert_close(
        torch.autograd.grad(radial.sum(), magmoms)[0],
        torch.autograd.grad(expected_radial.sum(), magmoms)[0],
    )
    torch.testing.assert_close(node_attrs_out, basis.angular_basis(magmoms))
    assert edge_attrs_out.shape == (num_nodes, basis.magnetic_edge_irreps_out.dim)
    assert basis.magnetic_scale.shape == (num_fidelities, 2)
    assert "magnetic_scale=[[" in repr(basis)


@pytest.mark.parametrize(
    ("num_fidelities", "legacy"), [(1, True), (1, False), (2, False)]
)
@pytest.mark.parametrize("nested", [False, True])
def test_magnetic_basis_loads_scale_fidelity_axis(num_fidelities, legacy, nested):
    basis = MagneticBasis(
        [1.0] * num_fidelities,
        num_mag_radial_basis=4,
        Lmax=1,
        atomic_numbers=[26, 28],
    ).to(DEVICE, DTYPE)
    module = torch.nn.ModuleDict({"magnetic_basis": basis}) if nested else basis
    key = "magnetic_basis.magnetic_scale" if nested else "magnetic_scale"
    expected = torch.arange(2, 2 + num_fidelities * 2, dtype=DTYPE, device=DEVICE).view(
        num_fidelities, 2
    )
    state_dict = module.state_dict()
    state_dict[key] = expected[0] if legacy else expected

    incompatible = module.load_state_dict(state_dict, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(basis.magnetic_scale, expected)


@pytest.mark.parametrize("Lmax", [1, 2])
@pytest.mark.parametrize("parity", [False, True])
def test_magnetic_basis_builds_regrouped_edge_attrs(Lmax, parity):
    time_reversal = hasattr(o3.Irrep("0e"), "t")
    basis = MagneticBasis(
        [{26: 2.0}],
        num_mag_radial_basis=4,
        Lmax=Lmax,
        atomic_numbers=[26],
        time_reversal=time_reversal,
        parity=parity,
    ).to(DEVICE, DTYPE)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        device=DEVICE,
    )
    magmoms = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    node_attrs = torch.ones(4, 1, dtype=DTYPE, device=DEVICE)
    node_fidelity = torch.zeros(4, dtype=torch.long, device=DEVICE)

    radial, magnetic_node_attrs, magnetic_edge_attrs = basis(
        magmoms,
        node_attrs,
        edge_index,
        node_fidelity,
    )
    source, target = edge_index
    expected = basis.magnetic_edge_tensor_product(
        magnetic_node_attrs[target],
        magnetic_node_attrs[source],
    )
    scaled_magmoms = magmoms / basis.magnetic_scale[0]
    squared_magnitude = scaled_magmoms.square().sum(dim=-1, keepdim=True)
    expected_radial_coordinate = 1.0 - 2.0 * torch.clamp(
        squared_magnitude,
        min=0.0,
        max=1.0,
    )

    assert basis.angular_basis.normalization == "integral"
    assert not basis.angular_basis.normalize
    assert basis.angular_basis.irreps_in[0].ir.p == (1 if parity else -1)
    assert basis.angular_basis.irreps_out == basis.magnetic_node_irreps_out
    for _, ir in basis.magnetic_node_irreps_out:
        assert ir.p == (1 if parity else (-1) ** ir.l)
        assert getattr(ir, "t", 1) == ((-1) ** ir.l if time_reversal else 1)
    if not parity:
        assert all(ir.p == (-1) ** ir.l for _, ir in basis.magnetic_edge_irreps_out)
    assert basis.radial_normalization == "clamp"
    torch.testing.assert_close(
        basis.magnetic_scale,
        torch.tensor([[2.0]], dtype=DTYPE, device=DEVICE),
    )
    assert basis.magnetic_edge_tensor_product.weight_numel == 0
    assert basis.magnetic_node_irreps_out.lmax == Lmax
    assert basis.magnetic_edge_irreps_out.lmax == Lmax
    assert basis.magnetic_edge_tensor_product.irreps_out.simplify() == (
        basis.magnetic_edge_irreps_out
    )
    assert radial.shape == (4, 4)
    assert magnetic_node_attrs.shape[-1] == basis.magnetic_node_irreps_out.dim
    assert magnetic_edge_attrs.shape == (
        edge_index.size(1),
        basis.magnetic_edge_irreps_out.dim,
    )
    torch.testing.assert_close(radial, basis.radial_basis(expected_radial_coordinate))
    torch.testing.assert_close(
        magnetic_node_attrs,
        basis.angular_basis(magmoms),
    )
    torch.testing.assert_close(magnetic_edge_attrs, expected)

    if time_reversal:
        reversed_radial, reversed_node, reversed_edge = basis(
            -magmoms,
            node_attrs,
            edge_index,
            node_fidelity,
        )
        torch.testing.assert_close(reversed_radial, radial)
        torch.testing.assert_close(
            reversed_node,
            time_reverse_features(magnetic_node_attrs, basis.magnetic_node_irreps_out),
        )
        torch.testing.assert_close(
            reversed_edge,
            time_reverse_features(magnetic_edge_attrs, basis.magnetic_edge_irreps_out),
        )


def test_magnetic_basis_clamps_radial_coordinate_smoothly_at_zero():
    basis = MagneticBasis(
        [{26: 2.0, 28: 4.0}],
        num_mag_radial_basis=4,
        Lmax=2,
        atomic_numbers=[26, 28],
    ).to(DEVICE, DTYPE)
    magmoms = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    node_attrs = torch.eye(2, dtype=DTYPE, device=DEVICE)
    edge_index = torch.tensor([[0, 1], [1, 0]], device=DEVICE)
    node_fidelity = torch.zeros(2, dtype=torch.long, device=DEVICE)

    radial, magnetic_node_attrs, _ = basis(
        magmoms, node_attrs, edge_index, node_fidelity
    )
    magnetic_scale = basis.magnetic_scale[0].unsqueeze(-1)
    scaled_magmoms = magmoms / magnetic_scale
    squared_magnitude = scaled_magmoms.square().sum(dim=-1, keepdim=True)
    radial_coordinate = 1.0 - 2.0 * torch.clamp(
        squared_magnitude,
        min=0.0,
        max=1.0,
    )

    assert basis.angular_basis.normalization == "integral"
    torch.testing.assert_close(radial, basis.radial_basis(radial_coordinate))
    torch.testing.assert_close(
        magnetic_node_attrs,
        basis.angular_basis(magmoms),
    )
    zero_gradient = torch.autograd.grad(radial[0].sum(), magmoms, create_graph=True)[0][
        0
    ]
    torch.testing.assert_close(zero_gradient, torch.zeros_like(zero_gradient))
    zero_hessian = torch.stack(
        [
            torch.autograd.grad(zero_gradient[axis], magmoms, retain_graph=True)[0][0]
            for axis in range(3)
        ]
    )
    orders = torch.arange(1, basis.num_mag_radial_basis + 1, dtype=DTYPE, device=DEVICE)
    expected_curvature = (
        -4.0 * orders.square().sum() / basis.magnetic_scale[0, 0].square()
    )
    torch.testing.assert_close(
        zero_hessian,
        expected_curvature * torch.eye(3, dtype=DTYPE, device=DEVICE),
    )


@pytest.mark.parametrize("Lmax", [1, 2, 3])
def test_magnetic_basis_without_soc_keeps_independent_spin_scalars(Lmax):
    basis = MagneticBasis(
        [{26: 2.0}],
        num_mag_radial_basis=4,
        Lmax=Lmax,
        atomic_numbers=[26],
        time_reversal=hasattr(o3.Irrep("0e"), "t"),
        use_spin_orbit_coupling=False,
    ).to(DEVICE, DTYPE)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        device=DEVICE,
    )
    magmoms = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    node_attrs = torch.ones(4, 1, dtype=DTYPE, device=DEVICE)
    node_fidelity = torch.zeros(4, dtype=torch.long, device=DEVICE)

    _, _, edge_scalars = basis(magmoms, node_attrs, edge_index, node_fidelity)
    rotation = o3.rand_matrix(dtype=DTYPE, device=DEVICE)
    _, _, rotated_edge_scalars = basis(
        magmoms @ rotation.T,
        node_attrs,
        edge_index,
        node_fidelity,
    )
    _, _, reversed_edge_scalars = basis(-magmoms, node_attrs, edge_index, node_fidelity)

    assert not basis.use_spin_orbit_coupling
    assert basis.magnetic_edge_irreps_out.num_irreps == Lmax + 1
    assert len(basis.magnetic_edge_tensor_product.instructions) == Lmax + 1
    assert all(
        ir.l == 0 and ir.p == 1 and getattr(ir, "t", 1) == 1
        for _, ir in basis.magnetic_edge_irreps_out
    )
    torch.testing.assert_close(rotated_edge_scalars, edge_scalars)
    torch.testing.assert_close(reversed_edge_scalars, edge_scalars)


def test_magnetic_radial_basis_is_bounded_and_has_no_constant_mode():
    basis = MagneticBasis(
        [{26: 2.0}],
        num_mag_radial_basis=4,
        Lmax=2,
        atomic_numbers=[26],
    ).to(DEVICE, DTYPE)
    node_attrs = torch.ones(2, 1, dtype=DTYPE, device=DEVICE)
    edge_index = torch.tensor([[0], [1]], device=DEVICE)
    node_fidelity = torch.zeros(2, dtype=torch.long, device=DEVICE)
    magmoms = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0e8, -2.0e8, 3.0e8]],
        dtype=DTYPE,
        device=DEVICE,
    )

    radial, angular, _ = basis(magmoms, node_attrs, edge_index, node_fidelity)

    assert not basis.radial_basis.include_constant
    torch.testing.assert_close(radial[0], torch.ones_like(radial[0]))
    assert torch.isfinite(angular).all()
    torch.testing.assert_close(angular, basis.angular_basis(magmoms))


def time_reverse_features(features: torch.Tensor, irreps) -> torch.Tensor:
    output = features.clone()
    for ir_mul, ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir_mul.ir.t
    return output
