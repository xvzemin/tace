import json
import sys
from copy import deepcopy
from typing import Union

import pytest
import torch

from tace.dataset.quantity import get_need_property
from tace.lightning.torch_model import (
    _prune_removed_keys,
    _should_warn_without_aoti,
)
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models.compile.aot import _export_metadata, _graph_aoti_input_keys
from tace.models.compile.compile import trace_to_fx
from tace.models.compile.wrapper import CompileTensorModel, _FlatE3nnCompileModel


def test_model_loading_prunes_removed_architecture_keys():
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["atomic_basis"]["removed_atomic_option"] = True
    config["product_basis"]["removed_product_option"] = True
    config["radial_basis"]["unrelated_option"] = True

    cleaned = _prune_removed_keys(config)

    assert "removed_atomic_option" not in cleaned["atomic_basis"]
    assert "removed_product_option" not in cleaned["product_basis"]
    assert cleaned["radial_basis"]["unrelated_option"] is True
    assert "removed_atomic_option" in config["atomic_basis"]
    assert "removed_product_option" in config["product_basis"]


@pytest.mark.parametrize(
    ("target_property", "expected"),
    [
        (["energy", "forces"], True),
        (["noncollinear_magnetic_forces"], True),
        (["energy", "dipole"], False),
        (["dipole"], False),
        ([], False),
    ],
)
def test_should_warn_without_aoti_requires_supported_target_subset(
    target_property,
    expected,
):
    assert _should_warn_without_aoti(target_property) is expected


class _MagneticEmbeddingReadout(torch.nn.Module):
    def __init__(
        self,
        embedding_property: list[str],
        atomic_basis_type: Union[str, None],
    ) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_buffer("cutoff", torch.tensor(4.0))
        self.register_buffer("atomic_numbers", torch.tensor([1], dtype=torch.int64))
        self.target_property = ["energy", "forces"]
        self.embedding_property = embedding_property
        self.model_config = (
            {"atomic_basis": {"type": atomic_basis_type}}
            if atomic_basis_type is not None
            else {}
        )
        self.max_neighbors = None

    def forward(self, data, graph):
        node_energy = self.scale * (
            graph.positions.square().sum(dim=-1)
            + data["initial_noncollinear_magmoms"].square().sum(dim=-1)
        )
        energy = torch.zeros(
            graph.num_graphs,
            dtype=node_energy.dtype,
            device=node_energy.device,
        ).index_add(0, data["batch"], node_energy)
        return {"energy": energy, "node_energy": node_energy}


def _magnetic_embedding_sample() -> dict[str, torch.Tensor]:
    return {
        "positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "node_attrs": torch.ones(3, 1),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_shifts": torch.zeros(4, 3),
        "lattice": torch.eye(3).unsqueeze(0) * 10.0,
        "batch": torch.zeros(3, dtype=torch.int64),
        "ptr": torch.tensor([0, 3], dtype=torch.int64),
        "fidelity_idx": torch.zeros(1, dtype=torch.int64),
        "initial_noncollinear_magmoms": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        ),
    }


def _slice_update(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    output = x.clone()
    output[:, 1:2, :] = value
    return output


def test_trace_to_fx_removes_full_dynamic_slice_scatter():
    x = torch.randn(4, 3, 5)
    value = torch.randn(4, 1, 5)
    traced = trace_to_fx(_slice_update, (x, value), functionalize=True)

    for node in traced.graph.nodes:
        if node.op != "call_function":
            continue
        assert node.target != torch.ops.aten.copy_.default
        if node.target == torch.ops.aten.slice_scatter.default:
            start = node.args[3] if len(node.args) > 3 else None
            end = node.args[4] if len(node.args) > 4 else None
            assert (start, end) != (0, sys.maxsize)

    torch.testing.assert_close(traced(x, value), _slice_update(x, value))


@pytest.mark.parametrize(
    ("embedding_property", "atomic_basis_type"),
    [
        pytest.param([], "o3_w6j_mag", id="o3-wigner6j-magnetic"),
        pytest.param([], "o2_mag", id="o2-magnetic"),
        pytest.param(
            ["initial_noncollinear_magmoms"],
            "cgtp",
            id="universal-embedding",
        ),
    ],
)
def test_aoti_keeps_magnetic_embedding_without_magnetic_force_target(
    embedding_property,
    atomic_basis_type,
):
    model = CompileTensorModel(
        _MagneticEmbeddingReadout(embedding_property, atomic_basis_type)
    ).eval()
    sample = _magnetic_embedding_sample()

    assert "initial_noncollinear_magmoms" in get_need_property(
        model.get_target_property(),
        model.get_embedding_property(),
        training=True,
    )
    assert "noncollinear_magnetic_forces" not in get_need_property(
        model.get_target_property(),
        model.get_embedding_property(),
        training=True,
    )

    input_keys = _graph_aoti_input_keys(model)
    output_keys = model._output_keys()
    assert "initial_noncollinear_magmoms" in input_keys
    assert "noncollinear_magnetic_forces" not in output_keys

    flat_model = _FlatE3nnCompileModel(model, input_keys, output_keys).eval()
    output = flat_model(*(sample[key] for key in input_keys))
    zero_magmoms = dict(sample)
    zero_magmoms["initial_noncollinear_magmoms"] = torch.zeros_like(
        sample["initial_noncollinear_magmoms"]
    )
    output_without_magmoms = flat_model(*(zero_magmoms[key] for key in input_keys))
    assert output_keys == ("energy", "node_energy", "forces")
    assert not torch.equal(output[0], output_without_magmoms[0])
    torch.testing.assert_close(output[2], output_without_magmoms[2])
    torch.testing.assert_close(output[2], -2.0 * sample["positions"])

    metadata = _export_metadata(model, input_keys, output_keys)
    assert "initial_noncollinear_magmoms" in json.loads(metadata["tace_input_keys"])
    assert "initial_noncollinear_magmoms" in json.loads(
        metadata["tace_embedding_property"]
    )
    assert "noncollinear_magnetic_forces" not in json.loads(
        metadata["tace_output_keys"]
    )
