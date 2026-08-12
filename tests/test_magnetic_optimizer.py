import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from tace.interface.ase import GeneralTACEAseCalc, MagneticFIRE


class QuadraticMagneticCalculator(Calculator):
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "noncollinear_magnetic_forces",
        "stress",
    ]

    def __init__(self, target_positions, target_magmoms, *, provide_stress=True):
        super().__init__()
        self.target_positions = np.asarray(target_positions)
        self.target_magmoms = np.asarray(target_magmoms)
        if not provide_stress:
            self.implemented_properties = [
                name for name in self.implemented_properties if name != "stress"
            ]

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms)
        delta_positions = atoms.positions - self.target_positions
        delta_magmoms = atoms.arrays["spin"] - self.target_magmoms
        energy = 0.5 * (
            np.square(delta_positions).sum() + np.square(delta_magmoms).sum()
        )
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -delta_positions,
            "noncollinear_magnetic_forces": -delta_magmoms,
        }
        if "stress" in self.implemented_properties:
            self.results["stress"] = np.zeros(6)


def make_atoms(*, provide_stress=True):
    atoms = Atoms(
        "H2",
        positions=[[0.4, -0.2, 0.1], [1.3, 0.3, -0.2]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    atoms.set_array("spin", np.array([[1.0, 0.5, -0.2], [-0.4, 0.3, 0.8]]))
    target_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target_magmoms = np.array([[0.2, 0.1, 0.0], [-0.1, 0.0, 0.3]])
    atoms.calc = QuadraticMagneticCalculator(
        target_positions, target_magmoms, provide_stress=provide_stress
    )
    return atoms, target_positions, target_magmoms


def test_magnetic_scf_precedes_every_structural_step():
    atoms, _, _ = make_atoms()
    original_positions = atoms.positions.copy()
    optimizer = MagneticFIRE(
        atoms,
        magmom_key="spin",
        magmom_fmax=1e-12,
        magmom_steps=1,
        magmom_maxstep=0.1,
        magmom_dt=0.1,
        logfile=None,
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        optimizer.run(steps=1)

    np.testing.assert_allclose(atoms.positions, original_positions, atol=0.0)


def test_nested_magnetic_and_structural_optimization():
    atoms, target_positions, target_magmoms = make_atoms()
    optimizer = MagneticFIRE(
        atoms,
        magmom_key="spin",
        position_maxstep=0.1,
        magmom_maxstep=0.08,
        position_fmax=1e-5,
        magmom_fmax=1e-5,
        magmom_steps=500,
        position_dt=0.1,
        magmom_dt=0.1,
        downhill_check=True,
        logfile=None,
    )

    converged = optimizer.run(steps=500)

    assert converged
    np.testing.assert_allclose(atoms.positions, target_positions, atol=2e-5)
    np.testing.assert_allclose(atoms.arrays["spin"], target_magmoms, atol=2e-5)
    maxima = optimizer.get_force_maxima()
    assert maxima["positions"] < 1e-5
    assert maxima["magmoms"] < 1e-5


def test_optional_cell_optimization_requires_stress():
    atoms, _, _ = make_atoms(provide_stress=False)
    optimizer = MagneticFIRE(
        atoms,
        optimize_cell=True,
        magmom_key="spin",
        magmom_fmax=1e-5,
        magmom_steps=500,
        magmom_dt=0.1,
        logfile=None,
    )
    with pytest.raises(ValueError, match="stress"):
        optimizer.run(steps=1)


def test_general_calculator_rejects_non_embedding_keys():
    with pytest.raises(ValueError, match="not an embedding property"):
        GeneralTACEAseCalc("unused.ckpt", keys={"energy": "energy"})
