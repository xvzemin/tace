################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
except ModuleNotFoundError as e:
    if e.name != "torch_sim":
        raise
    raise ImportError(
        "Install the TorchSim interface with `pip install tace[torchsim]`."
    ) from e

from tace.dataset.quantity import PROPERTIES
from tace.lightning import load_tace
from tace.utils.env import DEVICE, enable_acceleration


class TACETorchSimCalc(ModelInterface):
    """Evaluate a TACE model on a batched TorchSim state.

    Parameters
    ----------
    model : str, pathlib.Path or torch.nn.Module
        Foundation model name, model file, or loaded TACE model.
    device : torch.device or str, optional
        Execution device. Defaults to CUDA when available, otherwise CPU.
    dtype : torch.dtype, optional
        Model precision. Defaults to ``torch.float32``.
    matmul : {"ieee", "tf32"}, optional
        FP32 matrix multiplication precision.
    neighbor_list_fn : callable, optional
        Neighbor search with the signature of `torch_sim.neighbors.torchsim_nl`.
    compute_forces, compute_stress : bool, optional
        Include forces and stress in the output. Both default to True.
    atomic_numbers, system_idx : torch.Tensor, optional
        Fixed atomic numbers and system indices, each of shape ``(n_atoms,)``.
    fidelity_idx : int, optional
        Default fidelity. ``state.system_extras["fidelity_idx"]`` overrides it per system.
    target_property : list of str, optional
        Additional model outputs.
    enable_oeq, enable_cue, enable_eqx, enable_luna : bool, optional
        Enable the corresponding acceleration backend when loading the model.
    enable_compile : bool, optional
        Enable the compiled model path.

    Examples
    --------
    >>> from tace.interface.torchsim import TACETorchSimCalc
    >>> calc = TACETorchSimCalc("TACE-OAM-L", compute_stress=False)  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str | Path | torch.nn.Module,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        matmul: str | None = None,
        neighbor_list_fn: Callable | None = None,
        compute_forces: bool = True,
        compute_stress: bool = True,
        atomic_numbers: torch.Tensor | None = None,
        system_idx: torch.Tensor | None = None,
        fidelity_idx: int | None = None,
        target_property: list[str] | None = None,
        enable_oeq: bool = False,
        enable_cue: bool = False,
        enable_compile: bool = False,
        enable_eqx: bool | None = None,
        enable_luna: bool = False,
    ) -> None:
        super().__init__()
        enable_acceleration(
            enable_oeq=enable_oeq,
            enable_cue=enable_cue,
            enable_compile=enable_compile,
            enable_eqx=enable_eqx,
            enable_luna=enable_luna,
        )

        self._device = DEVICE[device] or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.neighbor_list_fn = (
            torchsim_nl if neighbor_list_fn is None else neighbor_list_fn
        )

        model = load_tace(
            model,
            self._device,
            strict=True,
            use_ema=True,
            target_property=target_property,
            dtype=dtype,
            matmul=matmul,
        )
        self._dtype = model.get_model_dtype()
        target_property = model.get_target_property()
        for name, enabled in (
            ("forces", self._compute_forces),
            ("stress", self._compute_stress),
        ):
            required = any(
                name in PROPERTIES[p]["must_be_with"] for p in target_property
            )
            if enabled or required:
                if name not in target_property:
                    target_property.append(name)
            elif name in target_property:
                target_property.remove(name)
        model.reset_target_property(target_property)
        model.reset_fidelity_idx(fidelity_idx)
        self.model = model.eval().requires_grad_(False)
        self.r_max = model.get_cutoff()
        self.torch_element = model.get_torch_element()
        self.atomic_numbers = None
        self.system_idx = None

        self.atomic_numbers_in_init = atomic_numbers is not None
        self.system_idx_in_init = system_idx is not None

        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers.to(
                device=self.device, dtype=torch.int64, copy=True
            )
            self._setup_node_type(self.atomic_numbers)

        if system_idx is not None:
            self.system_idx = system_idx.to(
                device=self.device, dtype=torch.int64, copy=True
            )
            self._setup_ptr(self.system_idx)

        if (
            atomic_numbers is not None
            and system_idx is not None
            and system_idx.shape[0] != atomic_numbers.shape[0]
        ):
            raise ValueError(
                f"system_idx length {system_idx.shape[0]} must match "
                f"atomic_numbers length {atomic_numbers.shape[0]}."
            )

    def _setup_ptr(self, system_idx: torch.Tensor) -> None:
        counts = torch.bincount(system_idx)
        self.n_systems = len(counts)
        self.ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    @property
    def n_atoms_per_system(self) -> list[int]:
        """Number of atoms in each system of the current batch."""
        return self.ptr.diff().tolist()

    def _setup_node_type(self, atomic_numbers: torch.Tensor) -> None:
        if (atomic_numbers < 0).any() or (
            atomic_numbers >= self.torch_element.lookup_table.numel()
        ).any():
            raise ValueError("State contains atomic numbers unsupported by the model.")
        self.node_type = self.torch_element.z2idx(atomic_numbers)
        if (self.node_type < 0).any():
            raise ValueError("State contains atomic numbers unsupported by the model.")

    def forward(self, state: ts.SimState, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Evaluate one or more systems.

        Parameters
        ----------
        state : torch_sim.SimState
            Batched atomic positions, cells, boundary conditions and species.

        Returns
        -------
        dict of str to torch.Tensor
            Energy ``(n_systems,)``, requested forces ``(n_atoms, 3)``,
            stress ``(n_systems, 3, 3)``, and other requested properties.
        """
        atomic_numbers = state.atomic_numbers.to(device=self.device, dtype=torch.int64)
        if self.atomic_numbers_in_init:
            if not torch.equal(atomic_numbers, self.atomic_numbers):
                raise ValueError(
                    "State atomic_numbers must match those provided at initialization."
                )
        elif self.atomic_numbers is None or not torch.equal(
            atomic_numbers, self.atomic_numbers
        ):
            self._setup_node_type(atomic_numbers)
            self.atomic_numbers = atomic_numbers.clone()

        system_idx = state.system_idx.to(device=self.device, dtype=torch.int64)
        if self.system_idx_in_init:
            if not torch.equal(system_idx, self.system_idx):
                raise ValueError(
                    "State system_idx must match those provided at initialization."
                )
        elif self.system_idx is None or not torch.equal(system_idx, self.system_idx):
            self._setup_ptr(system_idx)
            self.system_idx = system_idx.clone()

        # Detach before the model sets requires_grad, leaving the state unchanged.
        positions = state.positions.detach().to(device=self.device, dtype=torch.float64)
        lattice = state.row_vector_cell.detach().to(
            device=self.device, dtype=torch.float64
        )
        pbc = state.pbc.to(device=self.device)
        edge_index, _, unit_shifts = self.neighbor_list_fn(
            positions,
            lattice,
            pbc,
            self.r_max,
            self.system_idx,
        )

        data_dict = dict(
            ptr=self.ptr,
            node_type=self.node_type,
            atomic_numbers=self.atomic_numbers,
            batch=self.system_idx,
            pbc=pbc.reshape(1, 3).expand(self.n_systems, 3),
            lattice=lattice,
            positions=positions,
            edge_index=edge_index,
            edge_shifts=unit_shifts.to(dtype=torch.int64),
        )
        fidelity_idx = state.system_extras.get("fidelity_idx")
        if fidelity_idx is not None:
            if fidelity_idx.shape != (self.n_systems,):
                raise ValueError("fidelity_idx must have shape (n_systems,).")
            if fidelity_idx.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError("fidelity_idx must contain integer indices.")
            data_dict["fidelity_idx"] = fidelity_idx.to(
                device=self.device, dtype=torch.int64
            )

        with torch.enable_grad():
            output = self.model(data_dict)

        results = {
            name: value.detach()
            for name, value in output.items()
            if isinstance(value, torch.Tensor)
            and (name != "forces" or self.compute_forces)
            and (name != "stress" or self.compute_stress)
        }
        if output["energy"] is None:
            results["energy"] = positions.new_zeros(self.n_systems)
        return results
