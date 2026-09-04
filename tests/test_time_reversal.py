from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from e3nn import o3
from e3nn.nn import Gate

from tace.dataset.quantity import PROPERTY
from tace.models._e3nn.base import _to_possible_tp_irreps
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.fused import (
    O3ScatterTensorProduct,
    uuuTensorProduct,
    uvuTensorProduct,
)
from tace.models._e3nn.layer_norm import get_normalization_layer
from tace.models._e3nn.nonlinear import get_nonlinear_layer
from tace.models._e3nn.paths import generate_paths
from tace.models._e3nn.tace import e3nnTACE
from tace.models._e3nn.ue import UniversalEquivariantEmbedding
from tace.models._e3nn.wigner6j import O3Wigner6jScatterTensorProduct
from tace.models.angular import SolidHarmonics
from tace.models.layout import LayoutTransform
from tace.models.linear import e3nnLinear
from tace.models.time_reversal import (
    spherical_harmonics_irreps,
    supports_time_reversal,
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
    config["universal_embedding"]["initial_collinear_magmoms"] = {
        "enable": True,
        "normalizer": 1.0,
    }
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

    time_odd = {
        name
        for name, quantity in PROPERTY.items()
        if quantity["time_reversal"] == -1
    }
    assert time_odd == {
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


def test_universal_embedding_uses_property_time_reversal_metadata():
    embedding = UniversalEquivariantEmbedding(
        irreps_in=o3.Irreps("2x0e"),
        num_channel=2,
        num_elements=2,
        config={"initial_noncollinear_magmoms": {"normalizer": 1.0}},
        time_reversal=True,
    )
    input_irrep = embedding.uee["initial_noncollinear_magmoms"].irreps_in[0].ir
    assert getattr(input_irrep, "t", 1) == (-1 if supports_time_reversal() else 1)

    legacy_embedding = UniversalEquivariantEmbedding(
        irreps_in=o3.Irreps("2x0e"),
        num_channel=2,
        num_elements=2,
        config={"initial_noncollinear_magmoms": {"normalizer": 1.0}},
        time_reversal=False,
    )
    legacy_irrep = legacy_embedding.uee["initial_noncollinear_magmoms"].irreps_in[0].ir
    assert getattr(legacy_irrep, "t", 1) == 1


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_time_odd_scalar_uses_equivariant_embedding():
    config = _model_config()
    config["universal_embedding"]["initial_collinear_magmoms"] = {
        "enable": True,
        "normalizer": 1.0,
    }
    model = e3nnTACE(**config)
    representation = model.representation

    assert "initial_collinear_magmoms" not in representation.invariant_property
    assert "initial_collinear_magmoms" in representation.equivariant_property
    ir = representation.uee_embeddings[0].uee[
        "initial_collinear_magmoms"
    ].irreps_in[0].ir
    assert ir.l == 0 and ir.p == 1 and ir.t == -1


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
@pytest.mark.parametrize(
    "environment",
    ["TACE_USE_EQT", "TACE_USE_CUE", "TACE_USE_OEQ", "TACE_USE_EQX"],
)
def test_time_reversal_model_rejects_acceleration_kernel(monkeypatch, environment):
    for name in ("TACE_USE_EQT", "TACE_USE_CUE", "TACE_USE_OEQ", "TACE_USE_EQX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(environment, "1")

    with pytest.raises(ValueError, match="Time-reversal models do not support"):
        e3nnTACE(**_time_reversal_model_config())


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


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_wigner6j_is_time_reversal_equivariant():
    node_irreps = o3.Irreps("2x0ee + 2x1eo")
    edge_irreps = spherical_harmonics_irreps(1, p=-1)
    magnetic_irreps = spherical_harmonics_irreps(
        1,
        p=1,
        time_reversal=-1,
    )
    intermediate_irreps = _to_possible_tp_irreps(
        node_irreps,
        edge_irreps,
        parity=True,
        lmax=2,
    )
    output_irreps = (
        _to_possible_tp_irreps(
            intermediate_irreps,
            magnetic_irreps,
            parity=True,
            lmax=1,
        )
        * 2
    ).regroup()
    tensor_product = O3Wigner6jScatterTensorProduct(
        node_irreps,
        edge_irreps,
        output_irreps,
        magnetic_irreps,
        weight_level="edge",
    )

    num_nodes = 4
    num_edges = 7
    edge_index = torch.randint(num_nodes, (2, num_edges))
    node_features = torch.randn(num_nodes, node_irreps.dim)
    edge_features = torch.randn(num_edges, edge_irreps.dim)
    magnetic_features = torch.randn(num_nodes, magnetic_irreps.dim)
    edge_weights = torch.randn(num_edges, tensor_product.edge_weight_numel)
    magnetic_weights = torch.randn(num_edges, tensor_product.extra_weight_numel)

    output = tensor_product(
        node_features,
        edge_features,
        magnetic_features,
        edge_weights,
        magnetic_weights,
        edge_index,
    )
    observed = tensor_product(
        node_features @ _time_reversal_matrix(node_irreps).T,
        edge_features @ _time_reversal_matrix(edge_irreps).T,
        magnetic_features @ _time_reversal_matrix(magnetic_irreps).T,
        edge_weights,
        magnetic_weights,
        edge_index,
    )
    expected = output @ _time_reversal_matrix(tensor_product.irreps_out).T
    torch.testing.assert_close(observed, expected)
