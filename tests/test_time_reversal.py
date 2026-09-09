from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from e3nn import o3
from e3nn.nn import Gate
from torch_geometric.data import Data

from tace.dataset.augmentation import AugmentedDataset, validate_augmentations
from tace.dataset.quantity import PROPERTY, TIME_ODD_PROPERTIES
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.fused import (
    O3ScatterTensorProduct,
    uuuTensorProduct,
    uvuTensorProduct,
)
from tace.models._e3nn.layer_norm import get_normalization_layer
from tace.models._e3nn.node import (
    LinearSpinNodeEmbedding,
    NonLinearSpinNodeEmbedding,
)
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


def _time_parities(irreps: o3.Irreps) -> list[int]:
    return [getattr(ir, "t", 1) for _, ir in irreps]


def _time_reversal_matrix(irreps: o3.Irreps) -> torch.Tensor:
    return irreps.D_from_matrix(
        -torch.eye(3),
        parity=False,
        time_reversal=True,
    )


def test_magnetic_property_metadata_is_separate_from_spatial_irreps():
    assert all("time_reversal" in quantity for quantity in PROPERTY.values())
    assert PROPERTY["initial_collinear_magmoms"]["irreps"] == "1x0e"
    assert PROPERTY["initial_collinear_magmoms"]["time_reversal"] == -1
    assert PROPERTY["initial_noncollinear_magmoms"]["irreps"] == "1x1e"
    assert PROPERTY["initial_noncollinear_magmoms"]["time_reversal"] == -1
    assert PROPERTY["abs_final_collinear_magmoms"]["time_reversal"] == 1

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


def test_dataset_augmentation_configuration_is_a_unique_known_list():
    assert validate_augmentations(None) == ()
    assert validate_augmentations([]) == ()
    assert validate_augmentations("spin_rotation") == ("spin_rotation",)
    assert validate_augmentations(["spin_rotation", "time_reversal"]) == (
        "spin_rotation",
        "time_reversal",
    )
    with pytest.raises(TypeError, match="string or a list"):
        validate_augmentations(1)
    with pytest.raises(ValueError, match="Unknown"):
        validate_augmentations(["random_noise"])
    with pytest.raises(ValueError, match="duplicates"):
        validate_augmentations(["time_reversal", "time_reversal"])


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


def test_natural_parity_preserves_time_reversal_labels():
    irreps = with_time_reversal("1x0o + 1x1e + 1x2o", -1)
    natural_irreps = with_natural_parity(irreps)

    assert [ir.p for _, ir in natural_irreps] == [1, -1, 1]
    assert _time_parities(natural_irreps) == _time_parities(irreps)


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
    config["atomic_basis"]["type"] = "o2_mag"
    config["angular_basis"]["magnetic_Lmax"] = 1
    config["fidelity"][0]["magnetic_scale"] = 2.0
    config["mmax"] = 1

    natural_model = e3nnTACE(**config)
    natural_representation = natural_model.representation
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
    assert any(
        ir.p != (-1) ** ir.l
        for _, ir in complete_representation.magnetic_node_irreps_out
    )
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


@pytest.mark.parametrize(
    ("embedding_name", "embedding_type"),
    [
        ("linear_spin", LinearSpinNodeEmbedding),
        ("nonlinear_spin", NonLinearSpinNodeEmbedding),
    ],
)
def test_spin_node_embedding_registers_magnetic_input(
    embedding_name,
    embedding_type,
):
    config = _model_config()
    config["node_embedding"]["type"] = embedding_name
    config["fidelity"][0]["magnetic_scale"] = {1: 2.0}
    config["mmax"] = 1
    config["parity"] = True
    config["angular_basis"]["magnetic_Lmax"] = 1
    config["atomic_basis"]["type"] = ["o2_mag"]
    config["atomic_basis"]["edge_nonlinear"] = ["silu"]
    model = e3nnTACE(**config)

    assert isinstance(model.representation.node_embedding, embedding_type)
    assert "initial_noncollinear_magmoms" in model.embedding_property


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
            interaction.Lmax
            if interaction.correlation == 1
            else interaction.lmax
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

    assert model.representation.use_time_reversal
    assert all(
        not tensor_product.use_eqt
        for product in model.representation.products
        for tensor_product in product.aces
    )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_time_even_o2_model_keeps_automatic_acceleration(monkeypatch):
    for name in ("TACE_USE_EQT", "TACE_USE_CUE", "TACE_USE_OEQ", "TACE_USE_EQX"):
        monkeypatch.delenv(name, raising=False)
    config = _model_config()
    config["mmax"] = 1
    config["atomic_basis"]["type"] = ["o2"]
    config["atomic_basis"]["edge_nonlinear"] = ["silu"]
    model = e3nnTACE(**config)

    assert model.representation.use_o2
    assert not model.representation.use_time_reversal
    assert all(
        ir.t == 1
        for ir, _ in model.representation.interactions[0].rejector.local_irreps_out
    )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_time_reversal_model_supports_o2_interactions(monkeypatch):
    for name in ("TACE_USE_EQT", "TACE_USE_CUE", "TACE_USE_OEQ", "TACE_USE_EQX"):
        monkeypatch.delenv(name, raising=False)
    config = _time_reversal_model_config()
    config["mmax"] = 1
    config["atomic_basis"]["type"] = ["o2"]
    config["atomic_basis"]["edge_nonlinear"] = ["silu"]
    model = e3nnTACE(**config)

    assert model.representation.use_o2
    assert model.representation.use_time_reversal
    assert all(
        ir.t == 1
        for ir, _ in model.representation.interactions[0].rejector.local_irreps_out
    )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_time_reversal_model_uses_e3nn_equivariant_operations(monkeypatch):
    for name in ("TACE_USE_EQT", "TACE_USE_CUE", "TACE_USE_OEQ", "TACE_USE_EQX"):
        monkeypatch.delenv(name, raising=False)
    model = e3nnTACE(**_time_reversal_model_config())
    representation = model.representation
    interaction = representation.interactions[0]

    assert model.use_time_reversal
    assert type(representation.o3_angular_basis).__module__.startswith("e3nn.")
    assert type(interaction.linear_up.linear).__module__.startswith("e3nn.")
    assert type(interaction.nonlinearity).__module__.startswith("e3nn.")
    assert type(interaction.rejector.tp).__module__.startswith("e3nn.")
    assert all(
        type(tensor_product.tp).__module__.startswith("e3nn.")
        for product in representation.products
        for tensor_product in product.aces
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
    linear = e3nnLinear(irreps, irreps)
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
    )
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
        uuuTensorProduct(
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
        O3ScatterTensorProduct(
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
        uvuTensorProduct(
            node_irreps,
            edge_irreps,
            actual_irreps,
            instructions,
            shared_weights=False,
        )
