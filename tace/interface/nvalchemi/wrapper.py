"""NValCHEMI wrapper for an energy-based TACE model.

The numerical mapping in this file is intentionally small: eager TACE predicts
the total energy, while forces and stress are obtained as conservative
derivatives.
"""

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from nvalchemi._typing import ModelOutputs
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.models.base import (
        BaseModelMixin,
        ModelConfig,
        NeighborConfig,
        NeighborListFormat,
    )
except ImportError as e:
    raise ImportError(
        "The TACE NValCHEMI interface requires the optional "
        "'nvalchemi-toolkit' and 'nvalchemi-toolkit-ops' dependencies. "
        "Install them with "
        "`pip install nvalchemi-toolkit nvalchemi-toolkit-ops`."
    ) from e

from tace.lightning import export_tace, load_tace
from tace.models.adapter import TensorModel
from tace.models.utils import compute_symmetric_displacement


def _enable_tace_acceleration(
    *,
    enable_oeq: bool,
    enable_cue: bool,
    enable_eqt: bool,
    enable_compile: bool,
    enable_eqx: bool,
    force: bool = False,
) -> None:
    """Configure the supported TACE acceleration flags before model loading.

    TACE acceleration has three main categories: OEQ/CUE accelerate edge
    computation, EQT accelerates node computation, and Eqx provides TACE's
    custom operators. AOTInductor is used for compiled deployment. OEQ and CUE
    are mutually exclusive, while CUE currently conflicts with AOTI. For a
    typical model, OEQ + AOTI is currently recommended.
    """

    if sum((enable_oeq, enable_cue)) > 1:
        raise ValueError(
            "TACE OpenEquivariance (OEQ) and cuEquivariance (CUEQ) are "
            "mutually exclusive."
        )

    if enable_compile:
        raise ValueError(
            "In-process TACE compile is not supported by this wrapper. "
            "Export an AOTI .pt2 package and load it with load_tace instead."
        )
    os.environ["TACE_USE_COMPILE"] = "0"

    flags = {
        "TACE_USE_OEQ": enable_oeq,
        "TACE_USE_CUE": enable_cue,
        "TACE_USE_EQT": enable_eqt,
        "TACE_USE_COMPILE": False,
        "TACE_USE_EQX": enable_eqx,
    }
    try:
        from tace.utils.env import enable_acceleration
    except ImportError:
        for name, enabled in flags.items():
            if enabled or force:
                os.environ[name] = "1" if enabled else "0"
    else:
        enable_acceleration(
            enable_oeq=enable_oeq,
            enable_cue=enable_cue,
            enable_eqt=enable_eqt,
            enable_compile=False,
            enable_eqx=enable_eqx,
            force=force,
        )


class TACEWrapper(nn.Module, BaseModelMixin):
    """Adapt TACE tensors and conservative derivatives to NValCHEMI."""

    def __init__(self, model: TensorModel, fidelity_idx: int | None = None) -> None:
        super().__init__()
        self.model = model
        self._is_aoti = hasattr(model, "compiled_model") and hasattr(
            model, "compile_device"
        )
        self.fidelity_idx = int(
            model.get_fidelity_idx() if fidelity_idx is None else fidelity_idx
        )
        self._original_targets = model.get_target_property()

        if "energy" not in self._original_targets:
            raise ValueError(
                "This wrapper requires a TACE checkpoint with an energy target."
            )
        num_fidelities = getattr(model, "num_fidelities", None)
        if num_fidelities is not None and not 0 <= self.fidelity_idx < num_fidelities:
            raise ValueError(
                f"fidelity_idx={self.fidelity_idx} is outside "
                f"[0, {num_fidelities - 1}]."
            )

        # Eager derivatives are controlled here; AOTI uses its exported outputs.
        if not self._is_aoti:
            self.model.reset_target_property(["energy"])

        atomic_numbers = model.get_atomic_numbers()
        node_emb = torch.zeros(max(atomic_numbers) + 1, len(atomic_numbers))
        for index, atomic_number in enumerate(atomic_numbers):
            node_emb[atomic_number, index] = 1.0
        self.register_buffer(
            "_node_emb",
            node_emb.to(device=self._model_device, dtype=self._model_dtype),
            persistent=False,
        )

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=(
                frozenset() if self._is_aoti else frozenset({"forces", "stress"})
            ),
            autograd_inputs=(
                frozenset() if self._is_aoti else frozenset({"positions"})
            ),
            optional_inputs=frozenset(
                {"cell", "pbc", "neighbor_list_shifts", "fidelity_idx"}
            ),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=model.get_cutoff(),
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    @property
    def cutoff(self) -> float:
        return self.model.get_cutoff()

    @property
    def _model_dtype(self) -> torch.dtype:
        return self.model.get_model_dtype()

    @property
    def _model_device(self) -> torch.device:
        if self._is_aoti:
            return torch.device(self.model.compile_device)
        return next(self.model.parameters()).device

    def _node_attrs(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        z = atomic_numbers.long()
        invalid_range = (z < 0) | (z >= self._node_emb.shape[0])
        if invalid_range.any():
            unsupported = torch.unique(z[invalid_range]).detach().cpu().tolist()
            raise ValueError(f"TACE does not support atomic numbers {unsupported}.")

        node_attrs = self._node_emb.index_select(0, z)
        unsupported_mask = node_attrs.sum(dim=-1) == 0
        if unsupported_mask.any():
            unsupported = torch.unique(z[unsupported_mask]).detach().cpu().tolist()
            raise ValueError(f"TACE does not support atomic numbers {unsupported}.")
        return node_attrs

    def _prepare_input(
        self, data: AtomicData | Batch
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        model_device = self._model_device
        same_cuda_type = data.positions.device.type == model_device.type == "cuda"
        if data.positions.device != model_device and not same_cuda_type:
            raise ValueError(
                f"Input is on {data.positions.device}, but TACE is on "
                f"{model_device}. Move the Batch to the model device."
            )

        dtype = self._model_dtype
        device = data.positions.device
        num_graphs = data.num_graphs
        requested = self.output_data()
        need_forces = "forces" in requested
        need_stress = "stress" in requested

        # Clone only when this wrapper computes derivatives itself. In an
        # Alchemi shared-autograd group, the pipeline owns the input leaves.
        positions = data.positions.to(dtype=dtype)
        if (need_forces or need_stress) and not self._is_aoti:
            positions = positions.clone().requires_grad_(True)
        positions_leaf = positions

        edge_index = data.neighbor_list.long().T
        num_edges = edge_index.shape[1]
        edge_shifts_raw = getattr(data, "neighbor_list_shifts", None)
        edge_shifts = (
            torch.zeros(num_edges, 3, dtype=dtype, device=device)
            if edge_shifts_raw is None
            else edge_shifts_raw.to(device=device, dtype=dtype)
        )

        cell_raw = getattr(data, "cell", None)
        cell = (
            torch.zeros(num_graphs, 3, 3, dtype=dtype, device=device)
            if cell_raw is None
            else cell_raw.to(device=device, dtype=dtype)
        )
        if need_stress:
            if cell_raw is None:
                raise ValueError("Stress requires a periodic system with a cell.")
            volumes = torch.linalg.det(cell).abs()
            if (volumes <= 1.0e-12).any():
                raise ValueError("Stress requires a non-singular cell.")

        fidelity_raw = getattr(data, "fidelity_idx", None)
        if fidelity_raw is None:
            fidelity = torch.full(
                (num_graphs,),
                self.fidelity_idx,
                dtype=torch.long,
                device=device,
            )
        else:
            fidelity = fidelity_raw.to(device=device, dtype=torch.long).reshape(-1)
            if fidelity.numel() != num_graphs:
                raise ValueError(
                    "fidelity_idx must contain exactly one value per graph."
                )
        num_fidelities = getattr(self.model, "num_fidelities", None)
        if (
            num_fidelities is not None
            and ((fidelity < 0) | (fidelity >= num_fidelities)).any()
        ):
            raise ValueError(
                f"fidelity_idx values must be in [0, {num_fidelities - 1}]."
            )

        model_input = {
            "positions": positions,
            "atomic_numbers": data.atomic_numbers.long(),
            "node_attrs": self._node_attrs(data.atomic_numbers),
            "edge_index": edge_index,
            "edge_shifts": edge_shifts,
            "lattice": cell,
            "batch": data.batch_idx.long(),
            "ptr": data.batch_ptr.long(),
            "fidelity_idx": fidelity,
            "entropy": torch.ones(num_graphs, dtype=dtype, device=device),
        }

        displacement = None
        if need_stress and not self._is_aoti:
            displacement = compute_symmetric_displacement(model_input, num_graphs)
        return model_input, positions_leaf, displacement

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        model_input, _, _ = self._prepare_input(data)
        return model_input

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        requested = self.output_data()
        need_forces = "forces" in requested
        need_stress = "stress" in requested
        if (
            not self._is_aoti
            and (need_forces or need_stress)
            and not torch.is_grad_enabled()
        ):
            if torch.is_inference_mode_enabled():
                raise RuntimeError(
                    "Eager TACE forces and stress require autograd and cannot be "
                    "computed inside torch.inference_mode(). Use torch.no_grad() "
                    "or load an AOTI .pt2 package."
                )
            with torch.enable_grad():
                outputs = self.forward(data, **kwargs)
            return {name: value.detach() for name, value in outputs.items()}

        model_input, positions, displacement = self._prepare_input(data)
        raw_output = self.model(model_input)
        energy = raw_output["energy"]

        mapped: dict[str, torch.Tensor] = {"energy": energy}

        if self._is_aoti:
            for output, needed in (("forces", need_forces), ("stress", need_stress)):
                if needed:
                    value = raw_output.get(output)
                    if value is None:
                        raise ValueError(
                            f"The AOTI package was not exported with {output}."
                        )
                    mapped[output] = value
            return self.adapt_output(mapped, data)

        grad_inputs: list[torch.Tensor] = []
        if need_forces:
            grad_inputs.append(positions)
        if need_stress:
            assert displacement is not None
            grad_inputs.append(displacement)

        if grad_inputs:
            training = self.training and torch.is_grad_enabled()
            gradients = torch.autograd.grad(
                energy,
                grad_inputs,
                grad_outputs=torch.ones_like(energy),
                create_graph=training,
                retain_graph=training,
            )
            index = 0
            if need_forces:
                mapped["forces"] = -gradients[index]
                index += 1
            if need_stress:
                volumes = torch.linalg.det(model_input["lattice"].detach()).abs()
                mapped["stress"] = gradients[index] / volumes[:, None, None]

        return self.adapt_output(mapped, data)

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        raise NotImplementedError("This TACE wrapper does not expose embeddings.")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        fidelity_idx: int | None = None,
        *,
        enable_oeq: bool = False,
        enable_cue: bool = False,
        enable_eqt: bool = False,
        enable_compile: bool = False,
        enable_eqx: bool = False,
    ) -> "TACEWrapper":
        """Load TACE and expose it through the NValCHEMI model contract.

        Acceleration flags must be applied before ``load_tace`` constructs the
        model, so they belong to this factory rather than ``__init__``.
        """

        # Checkpoint_path must be end with .ckpt, .pt, .pth or .pt2
        checkpoint_path = Path(checkpoint_path)
        cue_enabled = enable_cue or os.environ.get("TACE_USE_CUE") == "1"
        if cue_enabled and checkpoint_path.suffix.lower() == ".pt2":
            raise ValueError(
                "TACE cuEquivariance (CUEQ) currently conflicts with AOTI."
            )
        _enable_tace_acceleration(
            enable_oeq=enable_oeq,
            enable_cue=enable_cue,
            enable_eqt=enable_eqt,
            enable_compile=enable_compile,
            enable_eqx=enable_eqx,
        )
        # Let TACE manage the dtype to avoid runtime dtype conversions,
        # which are unsupported by AOTI and similar backends.
        model = load_tace(checkpoint_path, device=device, dtype=dtype)
        model.eval()
        wrapper = cls(model, fidelity_idx=fidelity_idx)
        wrapper.eval()
        return wrapper

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        if self._is_aoti:
            raise RuntimeError(
                "An AOTI .pt2 package is already a compiled deployment artifact "
                "and cannot be re-exported through TACEWrapper."
            )
        if as_state_dict:
            torch.save(self.model.state_dict(), path)
            return

        self.model.reset_target_property(self._original_targets)
        try:
            export_tace(self.model, str(path))
        finally:
            self.model.reset_target_property(["energy"])
