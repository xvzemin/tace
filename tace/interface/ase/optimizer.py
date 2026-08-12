################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""ASE optimization over positions, noncollinear moments, and the cell."""

import time
from typing import Dict, Optional

import numpy as np
from ase.filters import FrechetCellFilter, OptimizableFilter
from ase.optimize import FIRE
from ase.optimize.optimize import DEFAULT_MAX_STEPS


def _clip_rows(values: np.ndarray, maximum: Optional[float]) -> None:
    if maximum is None:
        return
    if maximum <= 0.0:
        raise ValueError("Maximum step sizes must be positive")
    norms = np.linalg.norm(values, axis=1)
    scale = np.maximum(norms / maximum, 1.0)
    values /= scale[:, None]


class _MagneticFilter:
    """Expose selected structural and magnetic variables to an ASE optimizer.

    By default only noncollinear magnetic moments are optimized. Positions and
    cell degrees of freedom can be enabled independently. Magnetic moments are
    represented internally as ``magmom_scale * magmom`` so that their numerical
    scale can be balanced against coordinates measured in Angstrom.
    """

    def __init__(
        self,
        atoms,
        *,
        optimize_positions: bool = False,
        optimize_magmoms: bool = True,
        optimize_cell: bool = False,
        magmom_key: str = "initial_noncollinear_magmoms",
        magnetic_forces_key: str = "noncollinear_magnetic_forces",
        magmom_scale: float = 1.0,
        cell_filter_kwargs: Optional[dict] = None,
    ):
        if not any((optimize_positions, optimize_magmoms, optimize_cell)):
            raise ValueError("At least one degree-of-freedom block must be enabled")
        if magmom_scale <= 0.0:
            raise ValueError("magmom_scale must be positive")

        self.atoms = atoms
        self.info = atoms.info
        self.optimize_positions = optimize_positions
        self.optimize_magmoms = optimize_magmoms
        self.optimize_cell = optimize_cell
        self.magmom_key = magmom_key
        self.magnetic_forces_key = magnetic_forces_key
        self.magmom_scale = float(magmom_scale)
        self.num_atoms = len(atoms)

        if optimize_magmoms:
            if magmom_key not in atoms.arrays:
                raise ValueError(
                    f"atoms.arrays does not contain magnetic moments {magmom_key!r}"
                )
            magmoms = np.asarray(atoms.arrays[magmom_key])
            if magmoms.shape != (self.num_atoms, 3):
                raise ValueError(
                    f"{magmom_key!r} must have shape {(self.num_atoms, 3)}, "
                    f"got {magmoms.shape}"
                )

        self.cell_filter = None
        if optimize_cell:
            self.cell_filter = FrechetCellFilter(
                atoms, **({} if cell_filter_kwargs is None else cell_filter_kwargs)
            )

        row = 0
        self.block_slices: Dict[str, slice] = {}
        if optimize_positions:
            self.block_slices["positions"] = slice(row, row + self.num_atoms)
            row += self.num_atoms
        if optimize_magmoms:
            self.block_slices["magmoms"] = slice(row, row + self.num_atoms)
            row += self.num_atoms
        if optimize_cell:
            self.block_slices["cell"] = slice(row, row + 3)
            row += 3
        self.num_rows = row

    def _require_property(self, name: str) -> None:
        if self.atoms.calc is None:
            raise RuntimeError("The Atoms object has no calculator")
        if name not in self.atoms.calc.implemented_properties:
            raise ValueError(f"Calculator does not provide required property {name!r}")

    def get_positions(self) -> np.ndarray:
        blocks = []
        cell_coordinates = None
        if self.cell_filter is not None:
            cell_coordinates = self.cell_filter.get_positions()
        if self.optimize_positions:
            positions = (
                self.atoms.get_positions()
                if cell_coordinates is None
                else cell_coordinates[: self.num_atoms]
            )
            blocks.append(positions)
        if self.optimize_magmoms:
            blocks.append(
                np.asarray(self.atoms.arrays[self.magmom_key]) * self.magmom_scale
            )
        if self.optimize_cell:
            blocks.append(cell_coordinates[self.num_atoms :])
        return np.concatenate(blocks, axis=0)

    def set_positions(self, coordinates: np.ndarray, **kwargs) -> None:
        coordinates = np.asarray(coordinates)
        if coordinates.shape != (self.num_rows, 3):
            raise ValueError(
                f"Optimization coordinates must have shape {(self.num_rows, 3)}, "
                f"got {coordinates.shape}"
            )

        if self.cell_filter is not None:
            cell_coordinates = self.cell_filter.get_positions()
            if self.optimize_positions:
                cell_coordinates[: self.num_atoms] = coordinates[
                    self.block_slices["positions"]
                ]
            cell_coordinates[self.num_atoms :] = coordinates[self.block_slices["cell"]]
            self.cell_filter.set_positions(cell_coordinates, **kwargs)
        elif self.optimize_positions:
            self.atoms.set_positions(
                coordinates[self.block_slices["positions"]], **kwargs
            )

        if self.optimize_magmoms:
            current = np.asarray(self.atoms.arrays[self.magmom_key])
            magmoms = coordinates[self.block_slices["magmoms"]] / self.magmom_scale
            self.atoms.set_array(
                self.magmom_key, np.asarray(magmoms, dtype=current.dtype)
            )

        if self.atoms.calc is not None:
            self.atoms.calc.reset()

    positions = property(get_positions, set_positions)

    def get_forces(self, **kwargs) -> np.ndarray:
        blocks = []
        cell_forces = None
        if self.optimize_cell:
            self._require_property("stress")
            cell_forces = self.cell_filter.get_forces(**kwargs)
        if self.optimize_positions:
            self._require_property("forces")
            forces = (
                self.atoms.get_forces(**kwargs)
                if cell_forces is None
                else cell_forces[: self.num_atoms]
            )
            blocks.append(forces)
        if self.optimize_magmoms:
            self._require_property(self.magnetic_forces_key)
            magnetic_forces = self.atoms.calc.get_property(
                self.magnetic_forces_key, self.atoms
            )
            magnetic_forces = np.asarray(magnetic_forces)
            if magnetic_forces.shape != (self.num_atoms, 3):
                raise ValueError(
                    f"{self.magnetic_forces_key!r} must have shape "
                    f"{(self.num_atoms, 3)}, got {magnetic_forces.shape}"
                )
            blocks.append(magnetic_forces / self.magmom_scale)
        if self.optimize_cell:
            blocks.append(cell_forces[self.num_atoms :])
        return np.concatenate(blocks, axis=0)

    def get_potential_energy(self, **kwargs) -> float:
        self._require_property("energy")
        if self.cell_filter is not None:
            return self.cell_filter.get_potential_energy(**kwargs)
        return self.atoms.get_potential_energy(**kwargs)

    def force_maxima(self, generalized_forces: np.ndarray) -> Dict[str, float]:
        forces = np.asarray(generalized_forces).reshape(self.num_rows, 3)
        maxima = {}
        if self.optimize_positions:
            block = forces[self.block_slices["positions"]]
            maxima["positions"] = float(np.linalg.norm(block, axis=1).max())
        if self.optimize_magmoms:
            block = forces[self.block_slices["magmoms"]] * self.magmom_scale
            maxima["magmoms"] = float(np.linalg.norm(block, axis=1).max())
        if self.optimize_cell:
            maxima["cell"] = float(np.abs(self.cell_filter.stress).max())
        return maxima

    def limit_step(
        self,
        step: np.ndarray,
        *,
        position_maxstep: Optional[float],
        magmom_maxstep: Optional[float],
        cell_maxstrain: Optional[float],
    ) -> np.ndarray:
        limited = np.asarray(step).reshape(self.num_rows, 3).copy()
        if self.optimize_positions:
            _clip_rows(limited[self.block_slices["positions"]], position_maxstep)
        if self.optimize_magmoms:
            maximum = (
                None if magmom_maxstep is None else magmom_maxstep * self.magmom_scale
            )
            _clip_rows(limited[self.block_slices["magmoms"]], maximum)
        if self.optimize_cell and cell_maxstrain is not None:
            if cell_maxstrain <= 0.0:
                raise ValueError("cell_maxstrain must be positive")
            block = limited[self.block_slices["cell"]]
            maximum = cell_maxstrain * self.cell_filter.exp_cell_factor
            norm = np.linalg.norm(block)
            if norm > maximum:
                block *= maximum / norm
        return limited.ravel()

    def get_cell(self):
        return self.atoms.get_cell()

    def get_pbc(self):
        return self.atoms.get_pbc()

    def get_calculator(self):
        return self.atoms.calc

    @property
    def calc(self):
        return self.atoms.calc

    def iterimages(self):
        return self.atoms.iterimages()

    def __len__(self) -> int:
        return self.num_rows

    def __ase_optimizable__(self):
        return OptimizableFilter(self)


class _BlockFIRE(FIRE):
    """FIRE over one selected block of magnetic or structural ASE DOFs."""

    def __init__(
        self,
        atoms: _MagneticFilter,
        *,
        log_name: str,
        position_maxstep: Optional[float] = 0.2,
        magmom_maxstep: Optional[float] = 0.1,
        cell_maxstrain: Optional[float] = 0.01,
        **kwargs,
    ):
        if not isinstance(atoms, _MagneticFilter):
            raise TypeError("_BlockFIRE requires a _MagneticFilter")
        if "maxstep" in kwargs or "maxmove" in kwargs:
            raise TypeError("Use position_maxstep, magmom_maxstep, and cell_maxstrain")
        self.position_maxstep = position_maxstep
        self.magmom_maxstep = magmom_maxstep
        self.cell_maxstrain = cell_maxstrain
        self.log_name = log_name
        self.position_fmax = 0.05
        self.magmom_fmax = 0.01
        self.cell_fmax = 0.005
        super().__init__(atoms, maxstep=float("inf"), **kwargs)

    def _set_fmax(
        self,
        fmax: Optional[float],
        position_fmax: Optional[float],
        magmom_fmax: Optional[float],
        cell_fmax: Optional[float],
    ) -> None:
        if fmax is not None:
            if position_fmax is None:
                position_fmax = fmax
            if magmom_fmax is None:
                magmom_fmax = fmax
            if cell_fmax is None:
                cell_fmax = fmax
        if position_fmax is not None:
            self.position_fmax = float(position_fmax)
        if magmom_fmax is not None:
            self.magmom_fmax = float(magmom_fmax)
        if cell_fmax is not None:
            self.cell_fmax = float(cell_fmax)
        if min(self.position_fmax, self.magmom_fmax, self.cell_fmax) <= 0.0:
            raise ValueError("Convergence thresholds must be positive")

    def run(
        self,
        fmax: Optional[float] = None,
        steps: int = DEFAULT_MAX_STEPS,
        *,
        position_fmax: Optional[float] = None,
        magmom_fmax: Optional[float] = None,
        cell_fmax: Optional[float] = None,
    ) -> bool:
        self._set_fmax(fmax, position_fmax, magmom_fmax, cell_fmax)
        return super().run(fmax=1.0, steps=steps)

    def irun(
        self,
        fmax: Optional[float] = None,
        steps: int = DEFAULT_MAX_STEPS,
        *,
        position_fmax: Optional[float] = None,
        magmom_fmax: Optional[float] = None,
        cell_fmax: Optional[float] = None,
    ):
        self._set_fmax(fmax, position_fmax, magmom_fmax, cell_fmax)
        return super().irun(fmax=1.0, steps=steps)

    def gradient_converged(self, gradient: np.ndarray) -> bool:
        maxima = self.atoms.force_maxima(-np.asarray(gradient))
        thresholds = {
            "positions": self.position_fmax,
            "magmoms": self.magmom_fmax,
            "cell": self.cell_fmax,
        }
        return all(maxima[name] < thresholds[name] for name in maxima)

    def log(self, gradient: np.ndarray) -> None:
        maxima = self.atoms.force_maxima(-np.asarray(gradient))
        energy = self.optimizable.get_value()
        now = time.localtime()
        if self.nsteps == 0:
            labels = {
                "positions": "pos_fmax",
                "magmoms": "mag_fmax",
                "cell": "cell_fmax",
            }
            names = " ".join(f"{labels.get(name, name):>12s}" for name in maxima)
            self.logfile.write(
                f"{self.log_name:>13s}  Step     Time          Energy {names}\n"
            )
        values = " ".join(f"{value:12.6f}" for value in maxima.values())
        self.logfile.write(
            f"{self.log_name:>13s}: {self.nsteps:4d} "
            f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d} "
            f"{energy:15.6f} {values}\n"
        )

    def step(self, f=None) -> None:
        forces = -self._get_gradient(f)
        optimizable = self.optimizable

        if self.vel is None:
            self.vel = np.zeros(optimizable.ndofs())
            if self.downhill_check:
                self.e_last = optimizable.get_value()
                self.r_last = optimizable.get_x()
                self.vel_last = self.vel.copy()
        else:
            is_uphill = False
            if self.downhill_check:
                energy = optimizable.get_value()
                if energy > self.e_last:
                    if self.position_reset_callback is not None:
                        self.position_reset_callback(
                            optimizable, self.r_last, energy, self.e_last
                        )
                    optimizable.set_x(self.r_last)
                    is_uphill = True
                self.e_last = optimizable.get_value()
                self.r_last = optimizable.get_x()
                self.vel_last = self.vel.copy()

            velocity_force = np.vdot(forces, self.vel)
            force_squared = np.vdot(forces, forces)
            if velocity_force > 0.0 and not is_uphill and force_squared > 0.0:
                self.vel = (1.0 - self.a) * self.vel + self.a * forces / np.sqrt(
                    force_squared
                ) * np.sqrt(np.vdot(self.vel, self.vel))
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.vel[:] *= 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0

        self.vel += self.dt * forces
        step = self.atoms.limit_step(
            self.dt * self.vel,
            position_maxstep=self.position_maxstep,
            magmom_maxstep=self.magmom_maxstep,
            cell_maxstrain=self.cell_maxstrain,
        )
        optimizable.set_x(optimizable.get_x() + step)
        self.dump((self.vel, self.dt))


class MagneticFIRE:
    """Nested magnetic and structural FIRE relaxation.

    At every structural configuration, magnetic moments are first converged
    with positions and cell fixed. One structural step is then taken with the
    converged moments held fixed, after which magnetic relaxation is repeated.
    Cell degrees of freedom are optional and require calculator stress output.
    """

    def __init__(
        self,
        atoms,
        *,
        optimize_cell: bool = False,
        magmom_key: str = "initial_noncollinear_magmoms",
        magnetic_forces_key: str = "noncollinear_magnetic_forces",
        magmom_scale: float = 1.0,
        cell_filter_kwargs: Optional[dict] = None,
        position_maxstep: float = 0.1,
        magmom_maxstep: float = 0.02,
        cell_maxstrain: float = 0.01,
        position_fmax: float = 0.05,
        magmom_fmax: float = 0.01,
        cell_fmax: float = 0.005,
        magmom_steps: int = 1000,
        position_dt: float = 0.1,
        magmom_dt: float = 0.05,
        dtmax: float = 1.0,
        downhill_check: bool = True,
        trajectory=None,
        logfile="-",
    ):
        if magmom_steps <= 0:
            raise ValueError("magmom_steps must be positive")

        self.atoms = atoms
        self.optimize_cell = optimize_cell
        self.position_fmax = float(position_fmax)
        self.magmom_fmax = float(magmom_fmax)
        self.cell_fmax = float(cell_fmax)
        self.magmom_steps = int(magmom_steps)
        self.magmom_maxstep = magmom_maxstep
        self.magmom_dt = magmom_dt
        self.dtmax = dtmax
        self.downhill_check = downhill_check
        self.logfile = logfile

        self.magnetic_degrees = _MagneticFilter(
            atoms,
            optimize_positions=False,
            optimize_magmoms=True,
            optimize_cell=False,
            magmom_key=magmom_key,
            magnetic_forces_key=magnetic_forces_key,
            magmom_scale=magmom_scale,
        )
        self.structural_degrees = _MagneticFilter(
            atoms,
            optimize_positions=True,
            optimize_magmoms=False,
            optimize_cell=optimize_cell,
            magmom_key=magmom_key,
            magnetic_forces_key=magnetic_forces_key,
            cell_filter_kwargs=cell_filter_kwargs,
        )
        self.structural_optimizer = _BlockFIRE(
            self.structural_degrees,
            log_name="StructureFIRE",
            position_maxstep=position_maxstep,
            magmom_maxstep=None,
            cell_maxstrain=cell_maxstrain,
            dt=position_dt,
            dtmax=dtmax,
            downhill_check=downhill_check,
            trajectory=trajectory,
            logfile=logfile,
        )

    def _relax_magmoms(self) -> bool:
        optimizer = _BlockFIRE(
            self.magnetic_degrees,
            log_name="MagneticSCF",
            position_maxstep=None,
            magmom_maxstep=self.magmom_maxstep,
            cell_maxstrain=None,
            dt=self.magmom_dt,
            dtmax=self.dtmax,
            downhill_check=self.downhill_check,
            logfile=self.logfile,
        )
        return optimizer.run(
            magmom_fmax=self.magmom_fmax,
            steps=self.magmom_steps,
        )

    def _require_magnetic_convergence(self) -> None:
        if not self._relax_magmoms():
            raise RuntimeError(
                "Magnetic moments did not converge before the structural step; "
                "increase magmom_steps or adjust the magnetic FIRE settings"
            )

    def run(self, steps: int = DEFAULT_MAX_STEPS) -> bool:
        """Run at most ``steps`` structural updates with nested magnetic SCF."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")

        for _ in range(steps):
            self._require_magnetic_convergence()
            previous_steps = self.structural_optimizer.nsteps
            self.structural_optimizer.run(
                position_fmax=self.position_fmax,
                cell_fmax=self.cell_fmax,
                steps=1,
            )
            if self.structural_optimizer.nsteps == previous_steps:
                return True

        self._require_magnetic_convergence()
        forces = self.structural_degrees.get_forces()
        maxima = self.structural_degrees.force_maxima(forces)
        converged = maxima["positions"] < self.position_fmax
        if self.optimize_cell:
            converged = converged and maxima["cell"] < self.cell_fmax
        return converged

    def get_force_maxima(self) -> Dict[str, float]:
        """Return physical force maxima at the current nested state."""
        maxima = self.structural_degrees.force_maxima(
            self.structural_degrees.get_forces()
        )
        maxima.update(
            self.magnetic_degrees.force_maxima(self.magnetic_degrees.get_forces())
        )
        return maxima
