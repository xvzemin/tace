"""GPU integration tests for optional model packages."""

import os

import pytest
import torch

from eqx.models.convolution import copy_model
from eqx.models.nequip import convert_nequip_to_eqx
from eqx.models.prophet import convert_prophet_to_eqx
from eqx.models.sevennet import convert_sevennet_to_eqx
from eqx.models.equflash import convert_equflash_to_eqx


@pytest.mark.parametrize("implementation", ["o3", "o2"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_equflash_fullconv(implementation, backend, double_precision):
    pytest.importorskip("GGNN")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from e3nn import o3
    from GGNN.model.EquFlashV2.nn.convolution import FullConv

    torch.manual_seed(17)
    reference = FullConv(
        o3.Irreps("4x0e+4x1o+4x2e"),
        o3.Irreps("0e+1o+2e"),
        o3.Irreps("0e+1o+2e"),
        [8],
        4,
    ).cuda()
    converted = convert_equflash_to_eqx(
        reference,
        implementation=implementation,
        backend=backend,
    )
    x = torch.randn(5, reference.irreps_in.dim, device="cuda", requires_grad=True)
    vectors = torch.randn(12, 3, device="cuda", requires_grad=True)
    radial = torch.randn(12, 4, device="cuda", requires_grad=True)
    edges = torch.randint(5, (2, 12), device="cuda", dtype=torch.int32)
    harmonics = o3.spherical_harmonics(
        reference.irreps_filter,
        vectors,
        True,
        normalization="component",
    )
    expected = reference(x, edges, radial, harmonics)
    actual = converted(x, edges, radial, harmonics)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
    gradients = []
    for output in (expected, actual):
        gradients.append(
            torch.autograd.grad(
                output.square().mean(),
                (x, vectors, radial),
                create_graph=True,
                retain_graph=True,
            )
        )
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
    expected = torch.autograd.grad(
        gradients[0][1].square().mean(),
        reference.weight_nn[-1].weight,
        retain_graph=True,
    )[0]
    actual = torch.autograd.grad(
        gradients[1][1].square().mean(),
        converted.convolution.projection.weight,
    )[0]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)


def test_equflash_reject_efficient():
    pytest.importorskip("GGNN")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from e3nn import o3
    from GGNN.model.EquFlashV2.nn.convolution import EfficientConv

    module = EfficientConv(
        o3.Irreps("4x0e"),
        o3.Irreps("0e"),
        o3.Irreps("4x0e"),
        [4],
        2,
    ).cuda()
    with pytest.raises(NotImplementedError, match="Only EquFlash FullConv"):
        convert_equflash_to_eqx(module)


@pytest.mark.parametrize("implementation", ["o3", "o2"])
def test_nequip_package(implementation, atoms):
    path = os.environ.get("EQX_NEQUIP_CHECKPOINT")
    if not path:
        pytest.skip("Set EQX_NEQUIP_CHECKPOINT to validate a packaged checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("nequip")
    from nequip.model import ModelFromPackage
    from nequip.utils.global_state import set_global_state
    from nequip.integrations.ase import NequIPCalculator
    from nequip.data.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
        NeighborListTransform,
    )

    set_global_state()
    model = ModelFromPackage(path)["sole_model"].cuda().eval()
    reference = []
    for convert in (False, True):
        if convert:
            parameters = {id(p) for p in model.parameters()}
            model = convert_nequip_to_eqx(
                model, implementation=implementation, inplace=True
            )
            assert {id(p) for p in model.parameters()} == parameters
        atoms.calc = NequIPCalculator(
            model,
            device="cuda",
            transforms=[
                ChemicalSpeciesToAtomTypeMapper(model_type_names=model.type_names),
                NeighborListTransform(r_max=float(model.metadata["r_max"])),
            ],
        )
        reference.append(
            (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
        )
    for expected, actual in zip(*reference):
        torch.testing.assert_close(
            torch.as_tensor(actual), torch.as_tensor(expected), atol=5e-5, rtol=2e-5
        )


@pytest.fixture
def atoms():
    from ase import Atoms

    return Atoms(
        "OH2",
        positions=[[0.2, 0.4, 0.1], [1.1, 0.2, 0.3], [-0.1, 1.3, 0.5]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )


@pytest.fixture(params=["nequip", "sevennet", "prophet"])
def model_case(request, double_precision, atoms):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import e3nn
    from ase.neighborlist import neighbor_list

    defaults = e3nn.get_optimization_defaults()

    source, target, shift = neighbor_list("ijS", atoms, 3.0)
    edge_index = torch.tensor([source.tolist(), target.tolist()], device="cuda")
    positions = torch.tensor(atoms.positions, device="cuda")
    cell = torch.tensor(atoms.cell.array[None], device="cuda")
    shifts = torch.tensor(shift, device="cuda", dtype=torch.float64)
    species = torch.tensor([1, 0, 0], device="cuda")

    if request.param == "nequip":
        pytest.importorskip("nequip")
        from nequip.model import NequIPGNNModel
        from nequip.utils.global_state import set_global_state

        set_global_state()
        model = NequIPGNNModel(
            seed=12,
            model_dtype="float64",
            type_names=["H", "O"],
            r_max=3.0,
            num_layers=2,
            l_max=2,
            parity=True,
            num_features=2,
            radial_mlp_depth=1,
            radial_mlp_width=4,
            avg_num_neighbors=2.0,
            per_type_energy_shifts=0.0,
            per_type_energy_scales=1.0,
            compile_mode="eager",
        ).cuda()
        data = dict(
            pos=positions,
            cell=cell,
            edge_index=edge_index,
            edge_cell_shift=shifts,
            atom_types=species[:, None],
        )

        def evaluate(module):
            output = module({key: value.clone() for key, value in data.items()})
            return tuple(output[key] for key in ("total_energy", "forces", "stress"))

        convert = convert_nequip_to_eqx
    elif request.param == "sevennet":
        pytest.importorskip("sevenn")
        from sevenn.atom_graph_data import AtomGraphData
        from sevenn.model_build import build_E3_equivariant_model
        from sevenn.train.dataload import unlabeled_atoms_to_graph
        from sevenn.util import chemical_species_preprocess

        config = dict(
            cutoff=3.0,
            channel=2,
            radial_basis={"radial_basis_name": "bessel"},
            cutoff_function={"cutoff_function_name": "poly_cut"},
            interaction_type="nequip",
            lmax=2,
            is_parity=True,
            num_convolution_layer=2,
            weight_nn_hidden_neurons=[4],
            act_radial="silu",
            act_scalar={"e": "silu", "o": "tanh"},
            act_gate={"e": "silu", "o": "tanh"},
            conv_denominator=2.0,
            train_denominator=True,
            self_connection_type="nequip",
            shift=0.0,
            scale=1.0,
            train_shift_scale=False,
            irreps_manual=False,
            lmax_edge=-1,
            lmax_node=-1,
            readout_as_fcn=False,
            use_bias_in_linear=False,
            _normalize_sph=True,
        )
        config.update(chemical_species_preprocess(["H", "O"]))
        # SevenNet's native one-hot embedding explicitly produces float32.
        torch.set_default_dtype(torch.float32)
        model = build_E3_equivariant_model(config).cuda()
        model.set_is_batch_data(False)
        data = AtomGraphData.from_numpy_dict(
            unlabeled_atoms_to_graph(atoms, 3.0)
        ).cuda()

        def evaluate(module):
            output = module(data.clone())
            return tuple(
                output[key]
                for key in (
                    "inferred_total_energy",
                    "inferred_force",
                    "inferred_stress",
                )
            )

        convert = convert_sevennet_to_eqx
    else:
        pytest.importorskip("prophet")
        from prophet.model import Prophet

        model = Prophet(
            n_species=2,
            lmax=2,
            hidden_irreps="2x0e+2x1o+2x2e",
            n_layers=2,
            radial_basis_size=3,
            radial_mlp_size=4,
            radial_mlp_layers=1,
            mlp_init_scale=1.0,
            avg_n_neighbors=2.0,
            cutoffs=[1.2, 3.0],
            atom_energies=[0.0, 0.0],
            kernel=False,
        ).cuda()

        def evaluate(module):
            return module(
                species,
                positions.clone(),
                shifts,
                edge_index,
                cell,
                torch.tensor([3], device="cuda"),
                torch.tensor([edge_index.size(1)], device="cuda"),
                torch.zeros(3, dtype=torch.long, device="cuda"),
            )

        convert = convert_prophet_to_eqx
    yield model, convert, evaluate
    e3nn.set_optimization_defaults(**defaults)


@pytest.fixture
def tolerance(model_case):
    if next(model_case[0].parameters()).dtype == torch.float32:
        return dict(atol=2e-6, rtol=2e-5)
    return dict(atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("implementation", ["o3", "o2"])
@pytest.mark.parametrize("backend", ["cuda", "torch"])
def test_conversion_training(model_case, implementation, backend, tolerance):
    reference, convert, evaluate = model_case
    converted = copy_model(reference)
    parameters = tuple(converted.parameters())
    assert (
        convert(converted, inplace=True, implementation=implementation, backend=backend)
        is converted
    )
    assert set(converted.parameters()) == set(parameters)
    assert type(converted) is type(reference)
    if convert is convert_prophet_to_eqx:
        for layer in converted.layers:
            for convolution in layer.tpconv:
                assert isinstance(convolution.sort, torch.nn.Identity)
                assert (
                    convolution.tp_conv.irreps_out
                    == convolution.tp_conv.irreps_out.sort().irreps
                )
    expected, actual = evaluate(reference), evaluate(converted)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, **tolerance)
    for output in (actual, expected):
        sum(value.square().sum() for value in output).backward()
    for a, b in zip(parameters, reference.parameters()):
        if b.grad is None:
            assert a.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, **tolerance)
    before = [p.detach().clone() for p in parameters]
    torch.optim.SGD(converted.parameters(), lr=0.01).step()
    assert any(not torch.equal(a, b) for a, b in zip(parameters, before))


@pytest.mark.parametrize("implementation", ["o3", "o2"])
def test_conversion_checkpoint(model_case, implementation, tmp_path, tolerance):
    reference, convert, evaluate = model_case
    reference.eval()
    converted = convert(reference, implementation=implementation)
    assert converted is not reference
    assert not converted.training
    assert convert(converted, implementation=implementation, inplace=True) is converted
    torch.save(converted.state_dict(), tmp_path / "state.pt")
    restored = convert(reference, implementation=implementation)
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    for a, b in zip(evaluate(restored), evaluate(reference)):
        torch.testing.assert_close(a, b, **tolerance)


@pytest.mark.parametrize("implementation", ["o3", "o2"])
def test_conversion_ase(model_case, implementation, atoms, monkeypatch, tolerance):
    reference, convert, _ = model_case
    reference.eval()
    if convert is convert_prophet_to_eqx:
        # The upstream ASE graph builder fixes its floating-point data to float32.
        reference.float()
        torch.set_default_dtype(torch.float32)
        tolerance = dict(atol=2e-6, rtol=2e-5)
    converted = convert(reference, implementation=implementation)
    results = []
    for model in (reference, converted):
        if convert is convert_nequip_to_eqx:
            from nequip.data.transforms import (
                ChemicalSpeciesToAtomTypeMapper,
                NeighborListTransform,
            )
            from nequip.integrations.ase import NequIPCalculator

            calculator = NequIPCalculator(
                model,
                device="cuda",
                transforms=[
                    ChemicalSpeciesToAtomTypeMapper(model_type_names=model.type_names),
                    NeighborListTransform(r_max=float(model.metadata["r_max"])),
                ],
            )
        elif convert is convert_sevennet_to_eqx:
            from sevenn.calculator import SevenNetCalculator

            calculator = SevenNetCalculator(
                model=model, file_type="model_instance", device="cuda"
            )
        else:
            import prophet.calculator

            # Exercise ASE independently of Prophet's JSON/checkpoint loader.
            monkeypatch.setattr(
                prophet.calculator,
                "load_model",
                lambda *args, **kwargs: (
                    model,
                    {"atomic_numbers": [1, 8], "cutoffs": [1.2, 3.0]},
                ),
            )
            calculator = prophet.calculator.KairosCalculator(
                "unused",
                use_kernel=False,
                use_compile=False,
                device="cuda",
            )
        atoms.calc = calculator
        results.append(
            (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
        )
    for a, b in zip(*results):
        torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b), **tolerance)


@pytest.mark.parametrize("implementation", ["o3", "o2"])
def test_sevennet_pretrained(implementation):
    path = os.environ.get("EQX_SEVENNET_CHECKPOINT")
    if not path:
        pytest.skip("Set EQX_SEVENNET_CHECKPOINT to validate a pretrained checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("sevenn")
    from ase.build import bulk
    from sevenn.calculator import SevenNetCalculator
    from sevenn.util import load_checkpoint

    torch.set_default_dtype(torch.float32)
    original = (
        load_checkpoint(path)
        .build_model(
            enable_cueq=False,
            enable_flash=False,
            enable_oeq=False,
        )
        .cuda()
        .eval()
    )
    converted = convert_sevennet_to_eqx(original, implementation=implementation)
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    atoms.positions[1] += [0.05, -0.03, 0.02]
    for modal in ("omat24", "matpes_r2scan"):
        results = []
        for model in (original, converted):
            atoms.calc = SevenNetCalculator(
                model=model, file_type="model_instance", device="cuda", modal=modal
            )
            results.append(
                (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
            )
        errors = []
        for a, b in zip(*results):
            a, b = torch.as_tensor(a), torch.as_tensor(b)
            errors.append((a - b).abs().max().item())
            torch.testing.assert_close(a, b, atol=5e-5, rtol=2e-5)
        print(f"SevenNet {implementation} {modal}: max |delta E,F,S| = {errors}")
