################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path

import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from torch_geometric.data import Batch

from tace.dataset.graph import from_record
from tace.dataset.quantity import (
    KEYS,
    PROPERTIES,
    KeySpecification,
    update_keyspec_from_kwargs,
)
from tace.dataset.records import record_from_atoms
from tace.lightning import load_tace
from tace.utils.env import DEVICE, enable_acceleration


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
    enable_oeq, enable_cue, enable_eqt, enable_eqx, enable_luna : bool, optional
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
        enable_oeq: bool = False,
        enable_cue: bool = False,
        enable_eqt: bool = False,
        enable_compile: bool = False,
        enable_eqx: bool = False,
        enable_luna: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        enable_acceleration(
            enable_oeq=enable_oeq,
            enable_cue=enable_cue,
            enable_eqt=enable_eqt,
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
        )
        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.enable_grad():
            outs = self.model(batch)
        self.results = {}
        for p in self.target_property:
            p_rank = PROPERTIES[p]["rank"]
            p_scope = PROPERTIES[p]["scope"]
            ase_name = PROPERTIES[p]["ase_name"]
            save_name = ase_name if ase_name else p
            if p_scope == "per-system":
                if p_rank == 0:
                    if p == "energy":
                        energy = outs[p].detach().cpu().item()
                        self.results["energy"] = energy
                        self.results["free_energy"] = self.results["energy"]
                    else:
                        self.results[save_name] = outs[p].detach().cpu().item()
                else:
                    self.results[save_name] = outs[p].detach().cpu().numpy().squeeze(0)
            elif p_scope in {"per-atom", "per-edge", "per-pair"}:
                self.results[save_name] = outs[p].detach().cpu().numpy()
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
