################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import importlib
import json
import logging
import shutil
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Sequence, Set, Union
from uuid import uuid4

import torch

from tace.dataset.element import TorchElement

from ..utils import prepare_direct_hessian_inputs
from .compile import inductor_options, trace_to_fx
from .wrapper import (
    _LES_FIELD_OR_SPIN_KEYS,
    CompileTensorModel,
    FlatLammpsModel,
    FlatTorchModel,
    _GraphTensorModel,
    _valid_periodic_les_cell,
)

TACE_AOTI_FORMAT = "tace_graph_v1"
LAMMPS_AOTI_FORMAT = "tace_lammps_v1"
TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY = "tace_custom_ops_libs.txt"
TACE_AOTI_LUNA_OPS_ENTRY = "tace_luna_ops.json"
TACE_AOTI_INPUT_KEYS = (
    "positions",
    "node_type",
    "edge_index",
    "edge_shifts",
    "lattice",
    "pbc",
    "batch",
    "ptr",
    "fidelity_idx",
)
LAMMPS_AOTI_INPUT_KEYS = (
    "edge_vector",
    "node_type",
    "edge_index",
    "batch",
    "ptr",
    "fidelity_idx",
)
LAMMPS_AOTI_OUTPUT_KEYS = ("energy", "node_energy", "edge_forces")


def _package_device(metadata, device):
    compiled = torch.device(metadata.get("AOTI_DEVICE_KEY") or device or "cpu")
    if device is None:
        return compiled
    requested = torch.device(device)
    if compiled == requested or compiled.type == requested.type == "cuda":
        return requested
    raise RuntimeError(
        f"TACE .pt2 was compiled for {compiled}, but device={requested} was requested."
    )


class AOTICompiledTensorModel(torch.nn.Module):
    def __init__(
        self,
        compiled_model,
        metadata: Dict[str, str],
        device: Union[str, torch.device, None],
    ) -> None:
        super().__init__()
        self.compiled_model = compiled_model
        self.metadata = metadata
        self.periodic_les = metadata.get("tace_periodic_les") == "charge_only_v1"
        self.input_keys = tuple(metadata_json(metadata, "tace_input_keys"))
        self.output_keys = tuple(metadata_json(metadata, "tace_output_keys"))
        self.exported_target_property = list(
            metadata_json(metadata, "tace_target_property")
        )
        self.target_property = list(self.exported_target_property)
        self.embedding_property = list(
            metadata_json(metadata, "tace_embedding_property")
        )
        self.atomic_numbers = [
            int(z) for z in metadata_json(metadata, "tace_atomic_numbers")
        ]
        self.cutoff = float(metadata["tace_cutoff"])
        self.direct_hessian_cutoff = (
            float(metadata["tace_direct_hessian_cutoff"])
            if "direct_hessian" in self.exported_target_property
            else None
        )
        self.max_neighbors = metadata_json(metadata, "tace_max_neighbors")
        self.fidelity_idx = int(metadata["tace_fidelity_idx"])
        self.model_dtype = dtype_from_name(metadata["tace_dtype"])
        self.compile_device = _package_device(metadata, device)
        self.elements = TorchElement(self.atomic_numbers, device=self.compile_device)
        self.fixed_inputs = tuple(
            (key, torch.Size(shape), dtype_from_name(dtype))
            for key, shape, dtype in json.loads(metadata.get("tace_fixed_inputs", "[]"))
        )
        if self.fixed_inputs and self.compile_device == torch.device("cuda"):
            self.compile_device = torch.device("cuda", torch.cuda.current_device())

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, None]]:
        if self.periodic_les:
            _validate_periodic_les_inputs(data)
        data = canonicalize_inputs(
            data,
            self.input_keys,
            self.fidelity_idx,
            self.direct_hessian_cutoff,
            elements=self.elements,
        )
        if self.direct_hessian_cutoff is not None and data["positions"].size(0) < 2:
            raise ValueError(
                "Direct Hessian AOTI inference requires at least two atoms per batch. "
                "Use the PyTorch model for a single isolated atom."
            )
        missing = [key for key in self.input_keys if key not in data]
        if missing:
            raise KeyError(f"missing TACE graph .pt2 inputs: {missing}")
        for key, shape, dtype in self.fixed_inputs:
            value = data[key]
            if (
                value.shape != shape
                or value.dtype != dtype
                or value.device.type != self.compile_device.type
                or (
                    self.compile_device.index is not None
                    and value.device != self.compile_device
                )
            ):
                raise ValueError(
                    f"Fixed AOTI input {key!r} requires shape {tuple(shape)}, "
                    f"{dtype} on {self.compile_device}; received {tuple(value.shape)}, "
                    f"{value.dtype} on {value.device}."
                )
        outputs = self.compiled_model(*(data[key] for key in self.input_keys))
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        result: Dict[str, Union[torch.Tensor, None]] = {
            "energy": None,
            "node_energy": None,
            "forces": None,
            "virials": None,
            "stress": None,
            "direct_forces": None,
            "direct_virials": None,
            "direct_stress": None,
            "noncollinear_magnetic_forces": None,
            "charges": None,
            "direct_hessian": None,
        }
        result.update(zip(self.output_keys, outputs))
        return result

    def reset_target_property(self, target_property: list[str]) -> None:
        missing = set(target_property) - set(self.exported_target_property)
        if missing:
            raise ValueError(
                "The TACE graph .pt2 model was not exported with target "
                f"properties {sorted(missing)}."
            )
        self.target_property = list(target_property)

    def reset_fidelity_idx(self, fidelity_idx: Union[int, None] = 0) -> None:
        if fidelity_idx is not None:
            self.fidelity_idx = int(fidelity_idx)

    def get_fidelity_idx(self) -> int:
        return int(self.fidelity_idx)

    def get_embedding_property(self) -> list[str]:
        return list(self.embedding_property)

    def get_target_property(self) -> list[str]:
        return list(self.target_property)

    def get_model_dtype(self) -> torch.dtype:
        return self.model_dtype

    def get_max_neighbors(self) -> Union[int, None]:
        return self.max_neighbors

    def get_cutoff(self) -> float:
        return self.cutoff

    def get_atomic_numbers(self) -> list[int]:
        return list(self.atomic_numbers)

    def get_torch_element(self) -> TorchElement:
        return TorchElement(self.atomic_numbers)


class AOTICompiledLammpsModel(torch.nn.Module):
    def __init__(
        self,
        compiled_model,
        metadata: Dict[str, str],
        device: Union[str, torch.device, None],
    ) -> None:
        super().__init__()
        self.compiled_model = compiled_model
        self.metadata = metadata
        self.input_keys = tuple(metadata_json(metadata, "tace_input_keys"))
        self.output_keys = tuple(metadata_json(metadata, "tace_output_keys"))
        self.atomic_numbers = [
            int(z) for z in metadata_json(metadata, "tace_atomic_numbers")
        ]
        self.cutoff = float(metadata["tace_cutoff"])
        self.fidelity_idx = int(metadata["tace_fidelity_idx"])
        self.model_dtype = dtype_from_name(metadata["tace_dtype"])
        self.compile_device = _package_device(metadata, device)
        self.elements = TorchElement(self.atomic_numbers, device=self.compile_device)

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = canonicalize_inputs(
            data, self.input_keys, self.fidelity_idx, elements=self.elements
        )
        missing = [key for key in self.input_keys if key not in data]
        if missing:
            raise KeyError(f"missing TACE LAMMPS .pt2 inputs: {missing}")
        outputs = self.compiled_model(*(data[key] for key in self.input_keys))
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if len(outputs) != len(self.output_keys):
            raise RuntimeError(
                "TACE LAMMPS .pt2 returned "
                f"{len(outputs)} outputs, expected {len(self.output_keys)}."
            )
        result = dict(zip(self.output_keys, outputs))
        return result["energy"], result["node_energy"], result["edge_forces"]


def export_aotinductor(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    sample_data: Union[Dict[str, torch.Tensor], None] = None,
    *,
    fixed_shapes: bool = False,
) -> str:
    """Export a frozen graph with dynamic sizes or the exact sample input shapes.

    Dynamic export rejects observed trace guards that specialize a relationship
    between independent input sizes (such as the node and edge counts) unless
    guaranteed by the declared ranges. This is not a general check of Python
    control flow: custom readouts should express size-dependent math in tensors.

    With ``fixed_shapes=True``, all tensor shapes, including the edge count
    and any reciprocal-pair count, must match the supplied sample. Atom
    types, coordinates, connectivity and other input values remain variable.
    """
    if fixed_shapes and sample_data is None:
        raise ValueError("Fixed AOTI export requires sample_data.")
    logging.info("[AOTI 1/5] Preparing model and example inputs...")
    model.eval()
    compile_model = _frozen_graph_model(model)
    input_keys = graph_aoti_input_keys(compile_model)
    output_keys = compile_model._output_keys()
    if not output_keys:
        raise ValueError("TACE graph .pt2 export needs at least one output property.")

    if sample_data is None:
        sample_data = synthetic_graph_sample(compile_model)
    else:
        sample_data = {key: value for key, value in sample_data.items()}
    periodic_les = getattr(compile_model, "_periodic_les_graph", False)
    if periodic_les:
        _validate_periodic_les_inputs(sample_data, sample=True)
    ensure_sample_inputs(sample_data, input_keys, compile_model)

    flat_model = FlatTorchModel(compile_model, input_keys, output_keys)
    flat_model.eval()
    custom_ops_libs = custom_ops_libs_from_model(flat_model)
    inputs = tuple(sample_data[key] for key in input_keys)
    with size_oblivious_export():
        dynamic_shapes = (
            None
            if fixed_shapes
            else graph_dynamic_shapes(
                input_keys, num_graphs=sample_data["ptr"].numel() - 1
            )
        )
        logging.info("[AOTI 2/5] Tracing the forward and derivative graph...")
        traced = trace_to_fx(
            flat_model,
            inputs,
            functionalize=compile_model.flags.compute_noncollinear_magnetic_forces,
            fixed_shapes=fixed_shapes,
            dynamic_shapes=dynamic_shapes,
        )
        logging.info(
            "[AOTI 3/5] Exporting the %s graph...",
            "fixed-shape" if fixed_shapes else "dynamic",
        )
        exported = torch.export.export(
            traced,
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    custom_ops_libs.update(custom_ops_libs_from_model(exported.graph_module))
    output_path = normalize_pt2_path(output_path)
    metadata = export_metadata(
        compile_model,
        input_keys,
        output_keys,
        export_num_graphs=sample_data["ptr"].numel() - 1,
    )
    if periodic_les:
        metadata["tace_periodic_les"] = "charge_only_v1"
    if fixed_shapes:
        metadata["tace_fixed_inputs"] = json.dumps(
            [
                (key, list(value.shape), str(value.dtype).removeprefix("torch."))
                for key, value in zip(input_keys, inputs, strict=True)
            ]
        )
    inductor_configs = {**inductor_options(), "aot_inductor.metadata": metadata}
    ensure_cxx_compiler()
    logging.info(
        "[AOTI 4/5] Compiling AOTInductor kernels; this may take several minutes..."
    )
    out_path = compile_and_package(
        exported,
        package_path=output_path,
        inductor_configs=inductor_configs,
    )
    logging.info("[AOTI 5/5] Finalizing the package...")
    embed_custom_ops_libs(out_path, custom_ops_libs, flat_model)
    return str(out_path)


def _load_package(model_path, device, single_threaded):
    import_custom_ops_libs(str(model_path))
    options = {"run_single_threaded": True} if single_threaded else {}
    if device is not None and torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            return torch._inductor.aoti_load_package(
                str(model_path), device_index=torch.cuda.current_device(), **options
            )
    return torch._inductor.aoti_load_package(str(model_path), **options)


def load_aotinductor(
    model_path: Union[str, Path],
    device: Union[str, torch.device, None],
    *,
    single_threaded: bool = False,
) -> AOTICompiledTensorModel:
    compiled_model = _load_package(model_path, device, single_threaded)
    metadata = dict(compiled_model.get_metadata())
    if metadata.get("tace_format") != TACE_AOTI_FORMAT:
        raise ValueError(
            f"{model_path} is not a TACE graph .pt2 package "
            f"({metadata.get('tace_format')!r})."
        )
    return AOTICompiledTensorModel(compiled_model, metadata, device)


def export_lammps_aotinductor(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    sample_data: Union[Dict[str, torch.Tensor], None] = None,
    *,
    local_owners: bool = False,
) -> str:
    logging.info("[AOTI 1/5] Preparing model and example inputs...")
    model.eval()
    compile_model = as_compile_tensor_model(model)
    CompileTensorModel._validate_compile_properties(compile_model.readout_fn)
    input_keys = LAMMPS_AOTI_INPUT_KEYS

    if sample_data is None:
        sample_data = synthetic_lammps_sample(compile_model)
        if local_owners:
            n = sample_data["batch"].numel()
            sample_data["node_type"] = sample_data["node_type"][:n]
            sample_data["edge_index"] = sample_data["edge_index"] % n
    else:
        sample_data = {key: value for key, value in sample_data.items()}
    ensure_sample_inputs(sample_data, input_keys, compile_model)

    flat_model = FlatLammpsModel(compile_model, input_keys, local_owners=local_owners)
    flat_model.eval()
    custom_ops_libs = custom_ops_libs_from_model(flat_model)
    inputs = tuple(sample_data[key] for key in input_keys)
    with size_oblivious_export():
        logging.info("[AOTI 2/5] Tracing the forward and derivative graph...")
        traced = trace_to_fx(flat_model, inputs)
        logging.info("[AOTI 3/5] Exporting the dynamic graph...")
        exported = torch.export.export(
            traced,
            inputs,
            dynamic_shapes=lammps_dynamic_shapes(local_owners=local_owners),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    custom_ops_libs.update(custom_ops_libs_from_model(exported.graph_module))
    output_path = normalize_pt2_path(output_path)
    metadata = export_metadata(
        compile_model,
        input_keys,
        LAMMPS_AOTI_OUTPUT_KEYS,
        aoti_format=LAMMPS_AOTI_FORMAT,
        target="lammps",
    )
    if local_owners:
        metadata["tace_lammps_local_owners"] = "1"
    inductor_configs = {**inductor_options(), "aot_inductor.metadata": metadata}
    ensure_cxx_compiler()
    logging.info(
        "[AOTI 4/5] Compiling AOTInductor kernels; this may take several minutes..."
    )
    with size_oblivious_export():
        out_path = compile_and_package(
            exported,
            package_path=output_path,
            inductor_configs=inductor_configs,
        )
    logging.info("[AOTI 5/5] Finalizing the package...")
    embed_custom_ops_libs(out_path, custom_ops_libs, flat_model)
    return str(out_path)


def load_lammps_aotinductor(
    model_path: Union[str, Path],
    device: Union[str, torch.device],
    *,
    single_threaded: bool = False,
) -> AOTICompiledLammpsModel:
    compiled_model = _load_package(model_path, device, single_threaded)
    metadata = dict(compiled_model.get_metadata())
    if metadata.get("tace_format") != LAMMPS_AOTI_FORMAT:
        raise ValueError(
            f"{model_path} is not a TACE LAMMPS .pt2 package "
            f"({metadata.get('tace_format')!r})."
        )
    return AOTICompiledLammpsModel(compiled_model, metadata, device)


def as_compile_tensor_model(model: torch.nn.Module) -> CompileTensorModel:
    if isinstance(model, CompileTensorModel):
        return model
    if hasattr(model, "readout_fn"):
        wrapped = CompileTensorModel(model.readout_fn)
        wrapped.reset_fidelity_idx(model.get_fidelity_idx())
        # The readout is the caller's live module. Keep its individual
        # training/eval modes (e.g. frozen statistics during fine-tuning).
        wrapped.training = model.training
        return wrapped
    raise TypeError("TACE graph .pt2 export requires a TensorModel-like model.")


def _frozen_graph_model(model):
    """The live compiler and LAMMPS never receive the graph-only LES adapter."""
    readout = getattr(model, "readout_fn", None)
    if not hasattr(readout, "les"):
        wrapped = as_compile_tensor_model(model)
        CompileTensorModel._validate_compile_properties(wrapped.readout_fn)
        return wrapped
    return _periodic_les_graph_model(model)


def as_joint_export_model(model, sample):
    """Build the explicit-gradient export adapter, never a live-compile exception."""
    if not hasattr(getattr(model, "readout_fn", None), "les"):
        return as_compile_tensor_model(model)
    wrapped = _periodic_les_graph_model(model, joint=True)
    device = model.readout_fn.atomic_numbers.device.type
    if device not in ("cpu", "cuda") or (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        raise ValueError(
            "Periodic LES joint AOTI requires single-rank CPU or CUDA training."
        )
    _validate_periodic_les_inputs(sample, sample=True)
    # Standalone loaders enforce these without importing TACE or the LES solver.
    wrapped._joint_local_only = True
    wrapped._joint_forbidden_inputs = _LES_FIELD_OR_SPIN_KEYS
    if device == "cuda":
        wrapped._joint_serial_trace = True
        wrapped._joint_positive_periodic_cell = ("pbc", "lattice")
    return wrapped


def _periodic_les_graph_model(model, *, joint=False):
    from ..adapter import TensorModel
    from ..cartesian.tace import CartesianTACE
    from ..spherical.les import is_periodic_charge_only_les
    from ..spherical.tace import SphericalTACE

    readout = model.readout_fn
    if joint and type(readout) is CartesianTACE:
        raise ValueError(
            "Cartesian periodic LES is supported for frozen inference, not joint training."
        )
    if type(model) is not TensorModel or type(readout) not in (
        SphericalTACE,
        CartesianTACE,
    ):
        raise ValueError(
            "LES graph export requires a standard spherical or Cartesian TACE and TensorModel adapter."
        )
    config = readout.model_config
    targets = set(model.get_target_property())
    if not (
        not any(
            m._forward_hooks
            or m._forward_pre_hooks
            or m._backward_hooks
            or m._backward_pre_hooks
            or m._compiled_call_impl is not None
            for m in model.modules()
        )
        # These adapter methods are replaced by the export wrapper; don't
        # silently export the stock derivatives of a customized live model.
        and not {
            "forward",
            "prepare_property_data",
            "prepare_graph",
            "first_derivative_fn",
            "second_derivative_fn",
        }.intersection(vars(model))
        and not {"forward", "readout_fn"}.intersection(vars(readout))
        and is_periodic_charge_only_les(readout.les)
        and "energy" in targets
        and targets <= {"energy", "forces", "stress"}
        and not model.get_embedding_property()
        and not config.get("embedding_property")
        and not any(
            v.get("enable", False)
            for v in config.get("universal_embedding", {}).values()
        )
        and not readout.representation.use_magnetic_interaction
        and not model.lmp
    ):
        raise ValueError(
            "LES graph export requires a field-free, nonmagnetic TACE with a fixed periodic LES 0.2.0 charge-only solver and energy/forces/stress targets."
        )
    wrapped = _GraphTensorModel(readout)
    wrapped._periodic_les_graph = True
    wrapped.reset_fidelity_idx(model.get_fidelity_idx())
    # Share live state; preserve each child module's individual mode.
    wrapped.training = model.training
    return wrapped


def _validate_periodic_les_inputs(data, *, sample=False):
    if any(data.get(k) is not None for k in _LES_FIELD_OR_SPIN_KEYS):
        raise ValueError(
            "LES graph export does not accept external fields or spin inputs."
        )
    pbc, cell, ptr = (data.get(k) for k in ("pbc", "lattice", "ptr"))
    msg = "LES requires fully periodic finite cells with positive volume"
    if not (
        torch.is_tensor(ptr)
        and ptr.ndim == 1
        and ptr.numel() >= 2
        and torch.is_tensor(cell)
        and cell.is_floating_point()
        and cell.shape == (ptr.numel() - 1, 3, 3)
        and torch.is_tensor(pbc)
        and pbc.dtype == torch.bool
        and pbc.device == cell.device
        and pbc.shape == (ptr.numel() - 1, 3)
    ):
        raise ValueError(msg + "; expected boolean pbc[G,3] and lattice[G,3,3].")
    valid = _valid_periodic_les_cell(pbc, cell.detach().to(torch.float64))
    if sample or cell.device.type == "cpu":
        if not bool(valid):
            raise ValueError(msg)
    else:
        torch._assert_async(valid, msg)


def synthetic_graph_sample(model: CompileTensorModel) -> Dict[str, torch.Tensor]:
    dtype = torch.float64
    device = next(model.parameters()).device
    lattice = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
    lattice = lattice.repeat(2, 1, 1) * max(model.get_cutoff() * 4.0, 1.0)
    sample = {
        "positions": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5 * model.get_cutoff(), 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5 * model.get_cutoff(), 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        ),
        "node_type": torch.zeros(4, dtype=torch.int64, device=device),
        "edge_index": torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]],
            dtype=torch.int64,
            device=device,
        ),
        "edge_shifts": torch.zeros((4, 3), dtype=torch.int64, device=device),
        "lattice": lattice,
        "pbc": torch.ones((2, 3), dtype=torch.bool, device=device),
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device),
        "ptr": torch.tensor([0, 2, 4], dtype=torch.int64, device=device),
        "fidelity_idx": torch.full(
            (2,),
            model.get_fidelity_idx(),
            dtype=torch.int64,
            device=device,
        ),
    }
    if model._requires_noncollinear_magmoms():
        sample["initial_noncollinear_magmoms"] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
    if model._requires_total_charge():
        sample["total_charge"] = torch.zeros(2, dtype=dtype, device=device)
    if "direct_hessian" in model.get_target_property():
        # Keep both readout and backbone edges present in the tracing sample.
        distance = 0.5 * min(
            model.get_cutoff(), model.readout_fn.pair_representation.cutoff
        )
        sample["positions"][[1, 3], 0] = distance
    return sample


def synthetic_lammps_sample(model: CompileTensorModel) -> Dict[str, torch.Tensor]:
    dtype = torch.float64
    device = next(model.parameters()).device
    edge_vector = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=dtype,
        device=device,
    )
    return {
        "edge_vector": edge_vector,
        "node_type": torch.zeros(6, dtype=torch.int64, device=device),
        "edge_index": torch.tensor(
            [[0, 4, 1, 5, 2, 3], [1, 0, 2, 1, 3, 2]],
            dtype=torch.int64,
            device=device,
        ),
        "batch": torch.zeros(4, dtype=torch.int64, device=device),
        "ptr": torch.tensor([0, 4], dtype=torch.int64, device=device),
        "fidelity_idx": torch.full(
            (1,),
            model.get_fidelity_idx(),
            dtype=torch.int64,
            device=device,
        ),
    }


def ensure_sample_inputs(
    sample_data: Dict[str, torch.Tensor],
    input_keys: Sequence[str],
    model: CompileTensorModel,
) -> None:
    device = next(model.parameters()).device
    cutoff = (
        model.readout_fn.pair_representation.cutoff
        if "direct_hessian_edge_pairs" in input_keys
        else None
    )
    sample_data.update(
        canonicalize_inputs(
            sample_data,
            input_keys,
            model.get_fidelity_idx(),
            cutoff,
            elements=TorchElement(model.get_atomic_numbers(), device=device),
            device=device,
        )
    )
    missing = [key for key in input_keys if key not in sample_data]
    if missing:
        raise KeyError(f"missing TACE graph .pt2 sample inputs: {missing}")


def graph_aoti_input_keys(model: CompileTensorModel) -> tuple[str, ...]:
    keys = list(TACE_AOTI_INPUT_KEYS)
    if model.uses_reciprocal_weights():
        keys.extend(("edge_pair_rows", "edge_pair_inverse"))
    if model._requires_noncollinear_magmoms():
        keys.append("initial_noncollinear_magmoms")
    if model._requires_total_charge():
        keys.append("total_charge")
    if "direct_hessian" in model.get_target_property():
        keys.extend(
            [
                "direct_hessian_pair_index",
                "direct_hessian_edge_pairs",
                "direct_hessian_edge_shifts",
            ]
        )
    return tuple(keys)


def graph_dynamic_shapes(
    input_keys: Sequence[str],
    num_graphs: int,
) -> tuple[Dict[int, object], ...]:
    direct_hessian = "direct_hessian_edge_pairs" in input_keys
    num_nodes = torch.export.Dim("num_nodes", min=2)
    num_edges = torch.export.Dim("num_edges", min=0 if direct_hessian else 1)
    num_hessian_edges = torch.export.Dim("num_hessian_edges", min=0)
    num_graphs_dim = torch.export.Dim("num_graphs", min=1) if num_graphs != 1 else None
    shapes = {
        "positions": {0: num_nodes},
        "node_type": {0: num_nodes},
        "edge_index": {1: num_edges},
        "edge_shifts": {0: num_edges},
        "edge_pair_inverse": {0: num_edges},
        "edge_pair_rows": {0: torch.export.Dim("num_edge_pairs", min=0)},
        "lattice": {} if num_graphs_dim is None else {0: num_graphs_dim},
        "pbc": {} if num_graphs_dim is None else {0: num_graphs_dim},
        "batch": {0: num_nodes},
        "ptr": {} if num_graphs_dim is None else {0: num_graphs_dim + 1},
        "fidelity_idx": {} if num_graphs_dim is None else {0: num_graphs_dim},
        "initial_noncollinear_magmoms": {0: num_nodes},
        "total_charge": {} if num_graphs_dim is None else {0: num_graphs_dim},
        "direct_hessian_pair_index": {1: torch.export.Dim("num_pairs", min=1)},
        "direct_hessian_edge_pairs": {0: num_hessian_edges},
        "direct_hessian_edge_shifts": {0: num_hessian_edges},
    }
    return tuple(shapes[key] for key in input_keys)


def size_oblivious_export():
    config = torch.fx.experimental._config
    if hasattr(config, "backed_size_oblivious"):
        return config.patch(backed_size_oblivious=True)
    return nullcontext()


def lammps_dynamic_shapes(
    *, local_owners: bool = False
) -> tuple[Dict[int, object], ...]:
    num_edges = torch.export.Dim("num_edges", min=2)
    num_total = torch.export.Dim("num_total", min=2)
    num_local = torch.export.Dim("num_local", min=1)
    return (
        {0: num_edges},
        {0: num_local if local_owners else num_total},
        {1: num_edges},
        {0: num_local},
        {},
        {},
    )


def export_metadata(
    model: CompileTensorModel,
    input_keys: Sequence[str],
    output_keys: Sequence[str],
    *,
    aoti_format: str = TACE_AOTI_FORMAT,
    target: str = "graph",
    export_num_graphs: Union[int, None] = None,
) -> Dict[str, str]:
    embedding_property = list(model.get_embedding_property())
    if (
        model._requires_noncollinear_magmoms()
        and "initial_noncollinear_magmoms" not in embedding_property
    ):
        embedding_property.append("initial_noncollinear_magmoms")
    if model._requires_total_charge() and "total_charge" not in embedding_property:
        embedding_property.append("total_charge")
    metadata = {
        "tace_format": aoti_format,
        "tace_aoti_target": target,
        "tace_input_keys": json.dumps(list(input_keys)),
        "tace_output_keys": json.dumps(list(output_keys)),
        "tace_target_property": json.dumps(model.get_target_property()),
        "tace_embedding_property": json.dumps(embedding_property),
        "tace_atomic_numbers": json.dumps(model.get_atomic_numbers()),
        "tace_cutoff": str(model.get_cutoff()),
        "tace_max_neighbors": json.dumps(model.get_max_neighbors()),
        "tace_fidelity_idx": str(model.get_fidelity_idx()),
        "tace_dtype": str(model.get_model_dtype()).replace("torch.", ""),
        "tace_geometry_dtype": "float64",
        "tace_matmul": "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee",
    }
    if export_num_graphs is not None:
        metadata["tace_export_num_graphs"] = str(export_num_graphs)
    if "direct_hessian" in model.get_target_property():
        metadata["tace_direct_hessian_cutoff"] = str(
            model.readout_fn.pair_representation.cutoff
        )
    return metadata


def ensure_cxx_compiler() -> None:
    from torch._inductor import config

    configured = config.cpp.cxx
    configured_candidates = (
        configured if isinstance(configured, (tuple, list)) else (configured,)
    )
    for candidate in configured_candidates:
        if candidate and shutil.which(str(candidate)):
            return

    candidates = ["g++", "clang++"]
    candidates.extend(f"g++-{version}" for version in range(15, 6, -1))
    candidates.extend(f"clang++-{version}" for version in range(20, 9, -1))
    for candidate in candidates:
        compiler = shutil.which(candidate)
        if compiler:
            config.cpp.cxx = compiler
            return


def canonicalize_inputs(
    data: Dict[str, torch.Tensor],
    input_keys: Sequence[str],
    fidelity_idx: int,
    direct_hessian_cutoff=None,
    *,
    elements: TorchElement | None = None,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    data = as_tensor_dict(data)
    for key in input_keys:
        value = data.get(key)
        if not torch.is_tensor(value):
            continue
        dtype = value.dtype
        if key == "edge_shifts":
            dtype = torch.int64
        elif value.is_floating_point():
            dtype = torch.float64
        if dtype != value.dtype or device is not None:
            data[key] = value.to(device=device, dtype=dtype)
    if elements is not None:
        if "node_type" in data:
            indices = data["node_type"]
            if indices.ndim != 1 or indices.dtype != torch.int64:
                raise TypeError("node_type must be a one-dimensional int64 tensor.")
            torch._assert_async(
                ((indices >= 0) & (indices < len(elements))).all(),
                "node_type is outside the model element table.",
            )
        else:
            numbers = data["atomic_numbers"]
            if numbers.ndim != 1 or numbers.dtype not in (torch.int32, torch.int64):
                raise TypeError(
                    "atomic_numbers must be a one-dimensional integer tensor."
                )
            if device is not None:
                numbers = numbers.to(device=device)
            indices = elements.lookup_table.index_select(0, numbers)
            torch._assert_async(
                (indices >= 0).all(),
                "The structure contains an element not supported by this model.",
            )
            data["node_type"] = indices
    if "edge_pair_rows" in input_keys:
        from ..utils import prepare_edge_pairs

        prepare_edge_pairs(data)
    if direct_hessian_cutoff is not None:
        data = prepare_direct_hessian_inputs(data, direct_hessian_cutoff)
    if "fidelity_idx" in input_keys and "fidelity_idx" not in data:
        if "ptr" not in data:
            raise KeyError("missing TACE graph .pt2 inputs: ['ptr']")
        data["fidelity_idx"] = torch.full(
            (data["ptr"].numel() - 1,),
            fidelity_idx,
            dtype=torch.int64,
            device=data["ptr"].device,
        )
    for key in input_keys:
        value = data.get(key)
        if torch.is_tensor(value):
            data[key] = value.resolve_conj().resolve_neg().contiguous()
    return data


def as_tensor_dict(data) -> Dict[str, torch.Tensor]:
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "items"):
        return dict(data.items())
    if hasattr(data, "keys"):
        return {key: data[key] for key in data.keys()}
    return dict(data)


def normalize_pt2_path(output_path: Union[str, Path]) -> str:
    output_path = str(output_path)
    if not output_path.endswith(".pt2"):
        output_path += ".pt2"
    return output_path


def compile_and_package(exported, *, package_path, inductor_configs):
    from torch._inductor import aoti_compile_and_package, config

    options = dict(inductor_configs or {})
    if hasattr(config.aot_inductor, "model_name_for_generated_files") and any(
        isinstance(value := node.meta.get("val"), torch.Tensor) and value.is_cuda
        for node in exported.graph.nodes
        if node.op == "placeholder"
    ):
        options.setdefault(
            "aot_inductor.model_name_for_generated_files", f"tace_{uuid4().hex}"
        )
    return aoti_compile_and_package(
        exported, package_path=package_path, inductor_configs=options
    )


def custom_ops_libs_from_model(model: torch.nn.Module) -> Set[str]:
    """Collect registration modules from model layers and traced operators."""
    libs: Set[str] = set()
    for module in model.modules():
        module_name = type(module).__module__.lower()
        if module_name.startswith(
            ("openequivariance", "tace.models.backends.openequivariance.")
        ):
            libs.add("openequivariance")
        if module_name.startswith(
            ("cuequivariance", "tace.models.backends.cuequivariance.")
        ):
            libs.update({"cuequivariance", "cuequivariance_torch"})
        if module_name.startswith("tace.models.backends.lunaequivariance."):
            libs.add("tace.models.backends.lunaequivariance")
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                if isinstance(node.target, torch._ops.OpOverload):
                    if node.target.namespace == "eqx":
                        libs.add("equivariantx")
                    elif node.target.namespace.startswith("eqx_"):
                        from equivariantx_foundation.registry import registration_module

                        libs.add(registration_module(node.target))
    return libs


def embed_custom_ops_libs(
    pt2_path: Union[str, Path],
    custom_ops_libs: Set[str],
    model: Union[torch.nn.Module, None] = None,
) -> None:
    if not custom_ops_libs:
        return
    with zipfile.ZipFile(pt2_path, "a") as archive:
        archive_root = archive.namelist()[0].split("/", 1)[0]
        archive.writestr(
            f"{archive_root}/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}",
            " ".join(sorted(custom_ops_libs)),
        )
        if (
            model is not None
            and "tace.models.backends.lunaequivariance" in custom_ops_libs
        ):
            configs = {
                json.dumps(module.aoti_config): module.aoti_config
                for module in model.modules()
                if type(module).__module__.startswith(
                    "tace.models.backends.lunaequivariance."
                )
            }
            archive.writestr(
                f"{archive_root}/{TACE_AOTI_LUNA_OPS_ENTRY}",
                json.dumps(list(configs.values())),
            )


def import_custom_ops_libs(pt2_path: Union[str, Path]) -> None:
    """Register external operators before constructing the AOTI runner."""
    with zipfile.ZipFile(pt2_path, "r") as archive:
        archive_root = archive.namelist()[0].split("/", 1)[0]
        entry = f"{archive_root}/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}"
        libs = (
            set(archive.read(entry).decode().split())
            if entry in archive.namelist()
            else set()
        )
        luna_entry = f"{archive_root}/{TACE_AOTI_LUNA_OPS_ENTRY}"
        luna_configs = (
            json.loads(archive.read(luna_entry))
            if luna_entry in archive.namelist()
            else []
        )
    for lib in sorted(libs):
        importlib.import_module(lib)
    if luna_configs:
        from ..backends.lunaequivariance import register_luna_ops

        for config in luna_configs:
            register_luna_ops(config)


def metadata_json(metadata: Dict[str, str], key: str):
    value = metadata[key]
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value)


def dtype_from_name(name: str) -> torch.dtype:
    if name.startswith("torch."):
        name = name[len("torch.") :]
    if not hasattr(torch, name):
        raise ValueError(f"unsupported TACE graph .pt2 dtype {name!r}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported TACE graph .pt2 dtype {name!r}")
    return dtype
