"""Regression tests for the optional TorchSim interface."""

import pytest
import torch

ts = pytest.importorskip("torch_sim")

from tace.interface.torchsim import TACETorchSimCalc
from tace.models.adapter import TensorModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


class PairReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(3.0, dtype=DTYPE))
        self.register_buffer("atomic_numbers", torch.tensor([13, 14]))
        self.target_property = ["energy", "forces", "stress"]
        self.embedding_property = []
        self.fidelity = {0: "first", 1: "second"}

    def forward(self, data, graph):
        node_energy = data["node_attrs"][:, 1] + 3 * graph.node_fidelity
        node_energy = node_energy.index_add(
            0, data["edge_index"][0], graph.edge_vector.square().sum(-1)
        )
        energy = node_energy.new_zeros(graph.num_graphs).index_add(
            0, data["batch"], node_energy
        )
        return {"energy": energy, "node_energy": node_energy}


@pytest.fixture
def state():
    return ts.SimState(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.5, 0.0]],
            dtype=DTYPE,
            device=DEVICE,
        ),
        masses=torch.ones(4, dtype=DTYPE, device=DEVICE),
        cell=8 * torch.eye(3, dtype=DTYPE, device=DEVICE).repeat(2, 1, 1),
        atomic_numbers=torch.full((4,), 14, dtype=torch.int64, device=DEVICE),
        system_idx=torch.tensor([0, 0, 1, 1], device=DEVICE),
        pbc=False,
    )


@pytest.fixture
def calculator():
    def create(**kwargs):
        return TACETorchSimCalc(
            TensorModel(PairReadout()), device=DEVICE, dtype=DTYPE, **kwargs
        )

    return create


@pytest.mark.parametrize("field", ["atomic_numbers", "system_idx"])
def test_cache_detects_in_place_changes(state, calculator, field):
    calc = calculator()
    calc(state)
    assert getattr(calc, field).data_ptr() != getattr(state, field).data_ptr()
    if field == "atomic_numbers":
        state.atomic_numbers[0] = 13
    else:
        state.system_idx[1] = 1
    actual = calc(state)
    expected = calculator()(state.clone())
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key])
    torch.testing.assert_close(
        calc.ptr,
        torch.tensor([0, 1, 4], device=DEVICE)
        if field == "system_idx"
        else torch.tensor([0, 2, 4], device=DEVICE),
    )


@pytest.mark.parametrize("field", ["atomic_numbers", "system_idx"])
def test_fixed_inputs_are_snapshots_and_checked(state, calculator, field):
    calc = calculator(**{field: getattr(state, field)})
    calc(state)
    if field == "atomic_numbers":
        state.atomic_numbers[0] = 13
    else:
        state.system_idx[1] = 1
    with pytest.raises(ValueError, match=field):
        calc(state)


@pytest.mark.parametrize("compute_forces", [False, True])
@pytest.mark.parametrize("compute_stress", [False, True])
def test_derivative_targets(state, calculator, compute_forces, compute_stress):
    calc = calculator(
        target_property=["energy", "virials"],
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )
    actual = calc(state)
    expected = calculator()(state.clone())
    torch.testing.assert_close(actual["energy"], expected["energy"])
    torch.testing.assert_close(actual["virials"], expected["virials"])
    assert actual["virials"].abs().max() > 0
    assert ("forces" in actual) == compute_forces
    assert ("stress" in actual) == compute_stress


def test_energy_only(state, calculator):
    calc = calculator(compute_forces=False, compute_stress=False)
    assert not calc.model.compute_first_derivative
    actual = calc(state)
    assert "forces" not in actual and "stress" not in actual
    assert not state.positions.requires_grad


@pytest.mark.parametrize("periodic", [False, True])
def test_boundary_conditions(state, calculator, periodic):
    state.pbc.fill_(periodic)
    if periodic:
        state.positions[1, 0] = 7.0
    actual = calculator()(state)
    energy = torch.tensor([4.0, 6.5], dtype=DTYPE, device=DEVICE)
    forces = torch.tensor(
        [[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, -6.0, 0.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    stress = torch.zeros(2, 3, 3, dtype=DTYPE, device=DEVICE)
    if periodic:
        forces[:2] *= -1
        stress[0, 0, 0] = 4.0 / 8**3
        stress[1, 1, 1] = 9.0 / 8**3
    torch.testing.assert_close(actual["energy"], energy, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual["forces"], forces, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual["stress"], stress, atol=1e-10, rtol=1e-10)


def test_derivative_dependencies_are_retained(state, calculator):
    calc = calculator(
        target_property=["energy", "hessian"],
        compute_forces=False,
        compute_stress=False,
    )
    assert calc.model.compute_first_derivative
    assert calc.model.compute_second_derivative
    actual = calc(state)
    expected = calculator(target_property=["energy", "forces", "hessian"])(state)
    torch.testing.assert_close(actual["hessian"], expected["hessian"])
    assert "forces" not in actual


@pytest.mark.parametrize("device", [DEVICE, torch.device("cpu")])
@pytest.mark.parametrize("dtype", [torch.float32, DTYPE])
def test_direct_call_converts_inputs_without_mutation(state, calculator, device, dtype):
    state = state.to(device=device, dtype=dtype)
    original = state.clone()
    calc = calculator()
    actual = calc(state)
    expected = calc(original.clone().to(device=DEVICE, dtype=DTYPE))
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key])
        assert actual[key].dtype == DTYPE
        assert actual[key].device.type == DEVICE.type
        assert not actual[key].requires_grad
    for key in ("positions", "cell", "atomic_numbers", "system_idx", "pbc"):
        torch.testing.assert_close(getattr(state, key), getattr(original, key))
        assert not getattr(state, key).requires_grad


def test_system_fidelity_overrides_default(state, calculator):
    calc = calculator(fidelity_idx=1)
    default = calc(state)["energy"]
    state.system_extras["fidelity_idx"] = torch.tensor([0, 1], device=DEVICE)
    actual = calc(state)["energy"]
    torch.testing.assert_close(actual, default - default.new_tensor([6, 0]))
    state.system_extras["fidelity_idx"].fill_(0)
    torch.testing.assert_close(calc(state)["energy"], default - 6)
    del state.system_extras["fidelity_idx"]
    torch.testing.assert_close(calc(state)["energy"], default)


@pytest.mark.parametrize(
    "value, error", [([[0], [1]], ValueError), ([0.5, 1.0], TypeError)]
)
def test_system_fidelity_validation(state, calculator, value, error):
    state.system_extras["fidelity_idx"] = torch.tensor(value, device=DEVICE)
    with pytest.raises(error, match="fidelity_idx"):
        calculator()(state)
