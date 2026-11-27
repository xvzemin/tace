################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import sys
from pathlib import Path

import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from torch.nn.modules import module as module_hooks
from torch_geometric.data import Batch, Data

from tace.dataset.graph import from_record
from tace.dataset.neighbour_list import VerletNeighborList
from tace.dataset.quantity import (
    KEYS,
    PROPERTIES,
    KeySpecification,
    update_keyspec_from_kwargs,
)
from tace.dataset.records import record_from_atoms
from tace.lightning import load_tace
from tace.models.adapter import TensorModel
from tace.utils.env import DEVICE, enable_acceleration


def _single_model_input(model, graph, device):
    """Stage one stock graph without building PyG's unbatching metadata."""
    # The two native entry points both consume dict(data.items()). Looking
    # up the loaded AOTI type avoids importing the compiler for eager users.
    aoti = sys.modules.get("tace.models.compile.aot")
    if (
        type(graph) is Data
        and not {"__cat_dim__", "__inc__"}.intersection(vars(graph))
        and type(model) in (TensorModel, getattr(aoti, "AOTICompiledTensorModel", None))
        and device.type in ("cpu", "cuda")
        and not {"forward", "prepare_property_data"}.intersection(vars(model))
        and model._compiled_call_impl is None
        and not (
            model._forward_hooks
            or model._forward_pre_hooks
            or model._backward_hooks
            or model._backward_pre_hooks
            or module_hooks._global_forward_hooks
            or module_hooks._global_forward_pre_hooks
            or module_hooks._global_backward_hooks
            or module_hooks._global_backward_pre_hooks
        )
    ):
        fields = dict(graph.items())
        n = fields.get("num_nodes")
        if type(n) is int and all(
            k == "num_nodes"
            or (
                type(t) is torch.Tensor
                and t.layout == torch.strided
                and not t.is_nested
            )
            for k, t in fields.items()
        ):
            for k, t in fields.items():
                if k != "num_nodes":
                    # Singleton PyG cat adds a leading dimension to scalars
                    # and creates independent, contiguous storage even on CPU.
                    fields[k] = (t[None] if t.ndim == 0 else t).to(
                        device=device,
                        non_blocking=True,
                        copy=True,
                        memory_format=torch.contiguous_format,
                    )
            fields["batch"] = torch.zeros(n, dtype=torch.long, device=device)
            fields["ptr"] = torch.tensor([0, n], dtype=torch.long).to(
                device, non_blocking=True
            )
            return fields
    # Custom models and hooks can inspect num_graphs or call to_data_list().
    return Batch.from_data_list([graph]).to(device)


def _host_results(outputs, properties):
    """Copy the calculator's requested tensors to CPU before exposing them to ASE."""
    values = {key: outputs[key].detach() for key in properties}
    if len(values) > 1:
        device = next(iter(values.values())).device
        if device.type == "cuda" and all(t.device == device for t in values.values()):
            last = max(values, key=lambda key: values[key].numel())
            for key, t in values.items():
                if key != last:
                    # Only scalar/small tensor copies may allocate pinned host memory.
                    values[key] = t.to("cpu", non_blocking=t.numel() <= 9)
            # This blocking copy fences earlier transfers on the same CUDA stream.
            values[last] = values[last].cpu()
            return values
    return {key: t.cpu() for key, t in values.items()}


class TACEAseCalc(Calculator):
    """Evaluate a TACE model on ASE atoms.

    Parameters
    ----------
    model : str, pathlib.Path or torch.nn.Module
        Foundation model name, model file, or loaded model.
    dtype : str or torch.dtype, optional
        Model precision.
    matmul : {"ieee", "tf32"}, optional
        FP32 matrix multiplication precision.
    device : str or torch.device, optional
        Execution device. Defaults to CUDA when available, otherwise CPU.
    fidelity_idx : int, optional
        Model fidelity index.
    target_property : list of str, optional
        Model outputs to evaluate. Defaults to the model's target properties.
    neighborlist_backend : str, optional
        Neighbor-list backend. Defaults to "matscipy".
    neighbor_skin : float, optional
        Extra radius in angstroms for reusing neighbors across MD steps.
        Zero rebuilds the neighbor list on every call. Per-edge properties
        always use the original neighbor ordering.
    enable_oeq, enable_cue, enable_eqx, enable_luna : bool, optional
        Enable the corresponding acceleration backend when loading the model.
    enable_compile : bool, optional
        Enable the compiled model path.
    **kwargs
        Arguments to the ASE calculator.

    Examples
    --------
    >>> from tace.interface.ase import TACEAseCalc
    >>> calc = TACEAseCalc("TACE-OAM-L", dtype="float32")  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str | Path | torch.nn.Module,
        *,
        dtype: str | torch.dtype | None = None,
        matmul: str | None = None,
        device: str | torch.device | None = None,
        fidelity_idx: int | None = None,
        target_property: list[str] | None = None,
        neighborlist_backend: str = "matscipy",
        neighbor_skin: float = 0.0,
        enable_oeq: bool = False,
        enable_cue: bool = False,
        enable_compile: bool = False,
        enable_eqx: bool | None = None,
        enable_luna: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        enable_acceleration(
            enable_oeq=enable_oeq,
            enable_cue=enable_cue,
            enable_compile=enable_compile,
            enable_eqx=enable_eqx,
            enable_luna=enable_luna,
        )
        self.device = DEVICE[device] or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = load_tace(
            model,
            self.device,
            strict=True,
            use_ema=True,
            target_property=target_property,
            dtype=dtype,
            matmul=matmul,
        )
        model.eval().requires_grad_(False)

        self.dtype = model.get_model_dtype()
        self.target_property = model.get_target_property()
        self.embedding_property = model.get_embedding_property()
        self.max_neighbors = model.get_max_neighbors()
        self.cutoff = model.get_cutoff()
        self.element = model.get_torch_element()
        self.neighborlist_backend = neighborlist_backend
        self.neighbors = (
            VerletNeighborList(neighbor_skin) if neighbor_skin != 0 else None
        )
        self.implemented_properties = []
        for p in self.target_property:
            ase_name = PROPERTIES[p]["ase_name"]
            save_name = ase_name if ase_name else p
            if save_name == "energy":
                self.implemented_properties.extend(["energy", "free_energy"])
            else:
                self.implemented_properties.append(save_name)
        if fidelity_idx is not None:
            self.fidelity_idx = fidelity_idx
            model.reset_fidelity_idx(fidelity_idx)
        else:
            self.fidelity_idx = model.get_fidelity_idx()
        self.keySpecification = KeySpecification()
        update_keyspec_from_kwargs(self.keySpecification, KEYS)
        self.model = model

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        record = record_from_atoms(self.atoms if atoms is None else atoms)
        record.info["fidelity_idx"] = self.fidelity_idx
        graph = from_record(
            self.element,
            record,
            self.cutoff,
            max_neighbors=self.max_neighbors,
            target_property=self.target_property,
            embedding_property=self.embedding_property,
            keyspec=self.keySpecification,
            training=False,
            neighborlist_backend=self.neighborlist_backend,
            neighborhood=(
                self.neighbors
                if self.neighbors is not None
                and not any(
                    PROPERTIES[k]["scope"] == "per-edge"
                    for k in (*self.target_property, *self.embedding_property)
                )
                else None
            ),
        )
        batch = _single_model_input(self.model, graph, self.device)
        with torch.enable_grad():
            outs = self.model(batch)
        self.results = {}
        values = _host_results(outs, self.target_property)
        for p in self.target_property:
            p_rank = PROPERTIES[p]["rank"]
            p_scope = PROPERTIES[p]["scope"]
            ase_name = PROPERTIES[p]["ase_name"]
            save_name = ase_name if ase_name else p
            if p_scope == "per-system":
                if p_rank == 0:
                    if p == "energy":
                        energy = values[p].item()
                        self.results["energy"] = energy
                        self.results["free_energy"] = self.results["energy"]
                    else:
                        self.results[save_name] = values[p].item()
                else:
                    self.results[save_name] = values[p].numpy().squeeze(0)
            elif p_scope in {"per-atom", "per-edge", "per-pair"}:
                self.results[save_name] = values[p].numpy()
            else:
                raise ValueError(f"Unsupported property scope: {p_scope}")


def add_dispersion(
    base_calc: Calculator,
    damping: str = "bj",
    dispersion_xc: str = "pbe",
    dispersion_cutoff: float = 40.0 * units.Bohr,
    **kwargs,
) -> SumCalculator:
    """Add a D3 energy and force correction to a calculator.

    Parameters
    ----------
    base_calc : ase.calculators.calculator.Calculator
        Calculator with ``device`` and ``dtype`` attributes.
    damping : {"bj", "zero", "bjm", "zerom"}, optional
        Damping function.
    dispersion_xc : str, optional
        Exchange-correlation functional for the D3 parameters.
    dispersion_cutoff : float, optional
        Pair cutoff in angstroms.
    **kwargs
        Additional arguments to TorchDFTD3Calculator.

    Examples
    --------
    >>> calc = add_dispersion(TACEAseCalc("TACE-OAM-L"))  # doctest: +SKIP
    """
    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ModuleNotFoundError as e:
        if e.name != "torch_dftd":
            raise
        raise ImportError("Install D3 with `pip install tace[d3]`.") from e

    d3_calc = TorchDFTD3Calculator(
        dtype=base_calc.dtype,
        device=base_calc.device,
        damping=damping,
        xc=dispersion_xc,
        cutoff=dispersion_cutoff,
        **kwargs,
    )
    return SumCalculator([base_calc, d3_calc])
