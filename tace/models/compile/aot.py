import importlib
import json
import shutil
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Sequence, Set, Union

import torch

from tace.dataset.element import TorchElement

from .compile import trace_to_fx
from .wrapper import (
    CompileTensorModel,
    _FlatE3nnCompileModel,
    _FlatE3nnLammpsCompileModel,
)

TACE_AOTI_FORMAT = "tace_graph_v1"
ASE_AOTI_FORMAT = "tace_ase_v1"
LAMMPS_AOTI_FORMAT = "tace_lammps_v1"
TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY = "tace_custom_ops_libs.txt"
TACE_AOTI_INPUT_KEYS = (
    "positions",
    "node_attrs",
    "edge_index",
    "edge_shifts",
    "lattice",
    "batch",
    "ptr",
    "fidelity_idx",
)
ASE_AOTI_INPUT_KEYS = TACE_AOTI_INPUT_KEYS
LAMMPS_AOTI_INPUT_KEYS = (
    "edge_vector",
    "node_attrs",
    "edge_index",
    "batch",
    "ptr",
    "fidelity_idx",
)
LAMMPS_AOTI_OUTPUT_KEYS = ("energy", "node_energy", "edge_forces")


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
        self.input_keys = tuple(_metadata_json(metadata, "tace_input_keys"))
        self.output_keys = tuple(_metadata_json(metadata, "tace_output_keys"))
        self.exported_target_property = list(
            _metadata_json(metadata, "tace_target_property")
        )
        self.target_property = list(self.exported_target_property)
        self.embedding_property = list(
            _metadata_json(metadata, "tace_embedding_property")
        )
        self.atomic_numbers = [
            int(z) for z in _metadata_json(metadata, "tace_atomic_numbers")
        ]
        self.cutoff = float(metadata["tace_cutoff"])
        self.max_neighbors = _metadata_json(metadata, "tace_max_neighbors")
        self.fidelity_idx = int(metadata["tace_fidelity_idx"])
        self.model_dtype = _dtype_from_name(metadata["tace_dtype"])
        compile_device = metadata.get("AOTI_DEVICE_KEY") or device or "cpu"
        self.compile_device = torch.device(compile_device)
        self._check_device(device)

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, None]]:
        data = _canonicalize_inputs(data, self.input_keys, self.fidelity_idx)
        missing = [key for key in self.input_keys if key not in data]
        if missing:
            raise KeyError(f"missing TACE graph .pt2 inputs: {missing}")
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

    def _check_device(self, device: Union[str, torch.device, None]) -> None:
        if device is None:
            return
        requested = torch.device(device)
        if self.compile_device == requested:
            return
        if self.compile_device.type == "cuda" and requested.type == "cuda":
            return
        raise RuntimeError(
            f"TACE graph .pt2 was compiled for {self.compile_device}, "
            f"but device={requested} was requested."
        )


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
        self.input_keys = tuple(_metadata_json(metadata, "tace_input_keys"))
        self.output_keys = tuple(_metadata_json(metadata, "tace_output_keys"))
        self.atomic_numbers = [
            int(z) for z in _metadata_json(metadata, "tace_atomic_numbers")
        ]
        self.cutoff = float(metadata["tace_cutoff"])
        self.fidelity_idx = int(metadata["tace_fidelity_idx"])
        self.model_dtype = _dtype_from_name(metadata["tace_dtype"])
        compile_device = metadata.get("AOTI_DEVICE_KEY") or device or "cpu"
        self.compile_device = torch.device(compile_device)
        self._check_device(device)

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = _canonicalize_inputs(data, self.input_keys, self.fidelity_idx)
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

    def _check_device(self, device: Union[str, torch.device, None]) -> None:
        if device is None:
            return
        requested = torch.device(device)
        if self.compile_device == requested:
            return
        if self.compile_device.type == "cuda" and requested.type == "cuda":
            return
        raise RuntimeError(
            f"TACE LAMMPS .pt2 was compiled for {self.compile_device}, "
            f"but device={requested} was requested."
        )


def export_aotinductor(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    sample_data: Union[Dict[str, torch.Tensor], None] = None,
) -> str:
    model.eval()
    compile_model = _as_compile_tensor_model(model)
    CompileTensorModel._validate_compile_properties(compile_model.readout_fn)
    input_keys = _graph_aoti_input_keys(compile_model)
    output_keys = compile_model._output_keys()
    if not output_keys:
        raise ValueError("TACE graph .pt2 export needs at least one output property.")

    if sample_data is None:
        sample_data = _synthetic_graph_sample(compile_model)
    else:
        sample_data = {key: value for key, value in sample_data.items()}
    _ensure_sample_inputs(sample_data, input_keys, compile_model)

    flat_model = _FlatE3nnCompileModel(compile_model, input_keys, output_keys)
    flat_model.eval()
    custom_ops_libs = _custom_ops_libs_from_model(flat_model)
    inputs = tuple(sample_data[key] for key in input_keys)
    with _size_oblivious_export():
        traced = trace_to_fx(
            flat_model,
            inputs,
            functionalize=compile_model.flags.compute_noncollinear_magnetic_forces,
        )
        dynamic_shapes = _graph_dynamic_shapes(
            input_keys,
            num_graphs=sample_data["ptr"].numel() - 1,
        )
        exported = torch.export.export(
            traced,
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    output_path = _normalize_pt2_path(output_path)
    metadata = _export_metadata(
        compile_model,
        input_keys,
        output_keys,
        export_num_graphs=sample_data["ptr"].numel() - 1,
    )
    inductor_configs = _valid_inductor_configs(
        {
            "aot_inductor.metadata": metadata,
            "max_autotune": False,
            "shape_padding": True,
            "epilogue_fusion": False,
            "triton.cudagraphs": False,
            "max_fusion_size": 8,
            "triton.persistent_reductions": False,
            "triton.max_tiles": 1,
        }
    )
    _ensure_cxx_compiler()
    out_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
        inductor_configs=inductor_configs,
    )
    _embed_custom_ops_libs(out_path, custom_ops_libs)
    return str(out_path)


def load_aotinductor(
    model_path: Union[str, Path],
    device: Union[str, torch.device, None],
) -> AOTICompiledTensorModel:
    _import_custom_ops_libs(str(model_path))
    compiled_model = torch._inductor.aoti_load_package(str(model_path))
    metadata = dict(compiled_model.get_metadata())
    if metadata.get("tace_format") not in {TACE_AOTI_FORMAT, ASE_AOTI_FORMAT}:
        raise ValueError(
            f"{model_path} is not a TACE graph .pt2 package "
            f"({metadata.get('tace_format')!r})."
        )
    return AOTICompiledTensorModel(compiled_model, metadata, device)


def export_lammps_aotinductor(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    sample_data: Union[Dict[str, torch.Tensor], None] = None,
) -> str:
    model.eval()
    compile_model = _as_compile_tensor_model(model)
    CompileTensorModel._validate_compile_properties(compile_model.readout_fn)
    input_keys = LAMMPS_AOTI_INPUT_KEYS

    if sample_data is None:
        sample_data = _synthetic_lammps_sample(compile_model)
    else:
        sample_data = {key: value for key, value in sample_data.items()}
    _ensure_sample_inputs(sample_data, input_keys, compile_model)

    flat_model = _FlatE3nnLammpsCompileModel(compile_model, input_keys)
    flat_model.eval()
    custom_ops_libs = _custom_ops_libs_from_model(flat_model)
    inputs = tuple(sample_data[key] for key in input_keys)
    with _size_oblivious_export():
        traced = trace_to_fx(flat_model, inputs)
        exported = torch.export.export(
            traced,
            inputs,
            dynamic_shapes=_lammps_dynamic_shapes(),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    output_path = _normalize_pt2_path(output_path)
    metadata = _export_metadata(
        compile_model,
        input_keys,
        LAMMPS_AOTI_OUTPUT_KEYS,
        aoti_format=LAMMPS_AOTI_FORMAT,
        target="lammps",
    )
    inductor_configs = _valid_inductor_configs(
        {
            "aot_inductor.metadata": metadata,
            "max_autotune": False,
            "shape_padding": True,
            "epilogue_fusion": False,
            "triton.cudagraphs": False,
            "max_fusion_size": 8,
            "triton.persistent_reductions": False,
            "triton.max_tiles": 1,
        }
    )
    _ensure_cxx_compiler()
    with _size_oblivious_export():
        out_path = torch._inductor.aoti_compile_and_package(
            exported,
            package_path=output_path,
            inductor_configs=inductor_configs,
        )
    _embed_custom_ops_libs(out_path, custom_ops_libs)
    return str(out_path)


def load_lammps_aotinductor(
    model_path: Union[str, Path],
    device: Union[str, torch.device],
) -> AOTICompiledLammpsModel:
    _import_custom_ops_libs(str(model_path))
    compiled_model = torch._inductor.aoti_load_package(str(model_path))
    metadata = dict(compiled_model.get_metadata())
    if metadata.get("tace_format") != LAMMPS_AOTI_FORMAT:
        raise ValueError(
            f"{model_path} is not a TACE LAMMPS .pt2 package "
            f"({metadata.get('tace_format')!r})."
        )
    return AOTICompiledLammpsModel(compiled_model, metadata, device)


def export_ase_aotinductor(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    sample_data: Union[Dict[str, torch.Tensor], None] = None,
) -> str:
    return export_aotinductor(model, output_path, sample_data)


def load_ase_aotinductor(
    model_path: Union[str, Path],
    device: Union[str, torch.device, None],
) -> AOTICompiledTensorModel:
    return load_aotinductor(model_path, device)


def _as_compile_tensor_model(model: torch.nn.Module) -> CompileTensorModel:
    if isinstance(model, CompileTensorModel):
        return model
    if hasattr(model, "readout_fn"):
        wrapped = CompileTensorModel(model.readout_fn)
        wrapped.reset_fidelity_idx(model.get_fidelity_idx())
        wrapped.train(model.training)
        return wrapped
    raise TypeError("TACE graph .pt2 export requires a TensorModel-like model.")


def _synthetic_graph_sample(model: CompileTensorModel) -> Dict[str, torch.Tensor]:
    dtype = model.get_model_dtype()
    device = next(model.parameters()).device
    num_elements = len(model.get_atomic_numbers())
    node_attrs = torch.zeros((4, num_elements), dtype=dtype, device=device)
    node_attrs[:, 0] = 1.0
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
        "node_attrs": node_attrs,
        "edge_index": torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]],
            dtype=torch.int64,
            device=device,
        ),
        "edge_shifts": torch.zeros((4, 3), dtype=dtype, device=device),
        "lattice": lattice,
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
    return sample


def _synthetic_lammps_sample(model: CompileTensorModel) -> Dict[str, torch.Tensor]:
    dtype = model.get_model_dtype()
    device = next(model.parameters()).device
    num_elements = len(model.get_atomic_numbers())
    node_attrs = torch.zeros((6, num_elements), dtype=dtype, device=device)
    node_attrs[:, 0] = 1.0
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
        "node_attrs": node_attrs,
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


def _ensure_sample_inputs(
    sample_data: Dict[str, torch.Tensor],
    input_keys: Sequence[str],
    model: CompileTensorModel,
) -> None:
    sample_data.update(
        _canonicalize_inputs(sample_data, input_keys, model.get_fidelity_idx())
    )
    missing = [key for key in input_keys if key not in sample_data]
    if missing:
        raise KeyError(f"missing TACE graph .pt2 sample inputs: {missing}")
    device = next(model.parameters()).device
    dtype = model.get_model_dtype()
    for key, value in list(sample_data.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if torch.is_floating_point(value):
            sample_data[key] = value.to(device=device, dtype=dtype)
        else:
            sample_data[key] = value.to(device=device)


def _graph_aoti_input_keys(model: CompileTensorModel) -> tuple[str, ...]:
    keys = list(TACE_AOTI_INPUT_KEYS)
    if model._requires_noncollinear_magmoms():
        keys.append("initial_noncollinear_magmoms")
    return tuple(keys)


def _graph_dynamic_shapes(
    input_keys: Sequence[str],
    num_graphs: int,
) -> tuple[Dict[int, object], ...]:
    num_nodes = torch.export.Dim("num_nodes", min=2)
    num_edges = torch.export.Dim("num_edges", min=1)
    num_graphs_dim = torch.export.Dim("num_graphs", min=1) if num_graphs != 1 else None
    shapes = {
        "positions": {0: num_nodes},
        "node_attrs": {0: num_nodes},
        "edge_index": {1: num_edges},
        "edge_shifts": {0: num_edges},
        "lattice": {} if num_graphs_dim is None else {0: num_graphs_dim},
        "batch": {0: num_nodes},
        "ptr": {} if num_graphs_dim is None else {0: num_graphs_dim + 1},
        "fidelity_idx": {} if num_graphs_dim is None else {0: num_graphs_dim},
        "initial_noncollinear_magmoms": {0: num_nodes},
    }
    return tuple(shapes[key] for key in input_keys)


def _size_oblivious_export():
    config = torch.fx.experimental._config
    if hasattr(config, "backed_size_oblivious"):
        return config.patch(backed_size_oblivious=True)
    return nullcontext()


def _lammps_dynamic_shapes() -> tuple[Dict[int, object], ...]:
    num_edges = torch.export.Dim("num_edges", min=2)
    num_total = torch.export.Dim("num_total", min=2)
    num_local = torch.export.Dim("num_local", min=1)
    return (
        {0: num_edges},
        {0: num_total},
        {1: num_edges},
        {0: num_local},
        {},
        {},
    )


def _export_metadata(
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
    }
    if export_num_graphs is not None:
        metadata["tace_export_num_graphs"] = str(export_num_graphs)
    return metadata


def _valid_inductor_configs(configs: Dict[str, object]) -> Dict[str, object]:
    try:
        from torch._inductor import config

        valid = config.get_config_copy()
        return {
            key: value
            for key, value in configs.items()
            if key.replace("-", "_") in valid
        }
    except Exception:
        return configs


def _ensure_cxx_compiler() -> None:
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


def _canonicalize_inputs(
    data: Dict[str, torch.Tensor],
    input_keys: Sequence[str],
    fidelity_idx: int,
) -> Dict[str, torch.Tensor]:
    data = _as_tensor_dict(data)
    if "fidelity_idx" in input_keys and "fidelity_idx" not in data:
        if "ptr" not in data:
            raise KeyError("missing TACE graph .pt2 inputs: ['ptr']")
        data["fidelity_idx"] = torch.full(
            (data["ptr"].numel() - 1,),
            fidelity_idx,
            dtype=torch.int64,
            device=data["ptr"].device,
        )
    return {
        key: value.contiguous() if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def _as_tensor_dict(data) -> Dict[str, torch.Tensor]:
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "items"):
        return dict(data.items())
    if hasattr(data, "keys"):
        return {key: data[key] for key in data.keys()}
    return dict(data)


def _normalize_pt2_path(output_path: Union[str, Path]) -> str:
    output_path = str(output_path)
    if not output_path.endswith(".pt2"):
        output_path += ".pt2"
    return output_path


def _custom_ops_libs_from_model(model: torch.nn.Module) -> Set[str]:
    libs: Set[str] = set()
    for module in model.modules():
        module_name = type(module).__module__.lower()
        if module_name.startswith("openequivariance") or ".models.oeq" in module_name:
            libs.add("openequivariance")
        if module_name.startswith("cuequivariance") or ".models.cue" in module_name:
            libs.update({"cuequivariance", "cuequivariance_torch"})
    return libs


def _embed_custom_ops_libs(
    pt2_path: Union[str, Path], custom_ops_libs: Set[str]
) -> None:
    if not custom_ops_libs:
        return
    with zipfile.ZipFile(pt2_path, "a") as archive:
        archive_root = archive.namelist()[0].split("/", 1)[0]
        archive.writestr(
            f"{archive_root}/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}",
            " ".join(sorted(custom_ops_libs)),
        )


def _import_custom_ops_libs(pt2_path: Union[str, Path]) -> None:
    with zipfile.ZipFile(pt2_path, "r") as archive:
        archive_root = archive.namelist()[0].split("/", 1)[0]
        entry = f"{archive_root}/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}"
        if entry not in archive.namelist():
            return
        libs = archive.read(entry).decode().split()
    for lib in libs:
        importlib.import_module(lib)


def _metadata_json(metadata: Dict[str, str], key: str):
    value = metadata[key]
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value)


def _dtype_from_name(name: str) -> torch.dtype:
    if name.startswith("torch."):
        name = name[len("torch.") :]
    if not hasattr(torch, name):
        raise ValueError(f"unsupported TACE graph .pt2 dtype {name!r}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported TACE graph .pt2 dtype {name!r}")
    return dtype
