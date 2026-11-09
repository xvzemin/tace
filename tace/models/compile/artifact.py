"""Persist jointly compiled predictions, loss and parameter gradients."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import pickle
import platform
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from functools import lru_cache, partial
from pathlib import Path

import torch
from filelock import FileLock

FORMAT_VERSION = "v1"


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def source_fingerprint():
    files = {}
    for package in ("tace", "equivariantx"):
        root = Path(importlib.util.find_spec(package).submodule_search_locations[0])
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".cu", ".cuh", ".h"):
                files[f"{package}/{path.relative_to(root).as_posix()}"] = digest(path)
    return {
        "sha256": hashlib.sha256(json_bytes(files)).hexdigest(),
        "file_count": len(files),
    }


def module_context(module):
    """Hash Python configuration while keeping registered tensor state live."""
    from omegaconf import OmegaConf

    state = {
        id(value): name
        for name, value in (*module.named_parameters(), *module.named_buffers())
    }
    modules = {id(item): name for name, item in module.named_modules()}

    def encode(value):
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, torch.nn.Module):
            return {"module": modules[id(value)]}
        if isinstance(value, torch.Tensor):
            if id(value) in state:
                return {"state": state[id(value)]}
            if value.requires_grad or value.layout != torch.strided:
                raise ValueError("Unregistered trainable or non-strided tensor state")
            data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            return {
                "tensor": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "stride": list(value.stride()),
            }
        if OmegaConf.is_config(value):
            return encode(OmegaConf.to_container(value, resolve=True))
        if isinstance(value, dict):
            return {
                type_name(value): [(encode(k), encode(v)) for k, v in value.items()]
            }
        if isinstance(value, (tuple, list)):
            return {type_name(value): [encode(item) for item in value]}
        if isinstance(value, (set, frozenset)):
            return {"set": sorted((encode(item) for item in value), key=json_bytes)}
        if isinstance(value, (slice, range)):
            return {type_name(value): encode((value.start, value.stop, value.step))}
        if isinstance(value, (torch.dtype, torch.device, torch.layout, Path)):
            return {type_name(value): str(value)}
        if isinstance(value, torch.fx.Graph):
            return {"graph": str(value)}
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": type_name(value),
                "fields": encode(
                    {field.name: getattr(value, field.name) for field in fields(value)}
                ),
            }
        if isinstance(value, type):
            return {"class": f"{value.__module__}.{value.__qualname__}"}
        if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
            import marshal

            result = {"function": f"{value.__module__}.{value.__qualname__}"}
            if isinstance(value, types.FunctionType):
                result.update(
                    code=hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
                    defaults=encode(value.__defaults__),
                    kwdefaults=encode(value.__kwdefaults__),
                    closure=encode(
                        tuple(c.cell_contents for c in value.__closure__ or ())
                    ),
                )
            return result
        if isinstance(value, partial):
            return {
                "partial": encode(value.func),
                "args": encode(value.args),
                "kwargs": encode(value.keywords),
            }
        if type(value).__module__.startswith("numpy"):
            return {"numpy": type_name(value), "value": encode(value.tolist())}
        raise TypeError(f"Unsupported compiled module attribute: {type_name(value)}")

    def type_name(value):
        return f"{type(value).__module__}.{type(value).__qualname__}"

    descriptions = []
    for name, item in module.named_modules():
        attributes = {
            key: encode(value)
            for key, value in vars(item).items()
            if key not in ("_parameters", "_buffers", "_modules", "_compiled_cache")
        }
        descriptions.append((name, type_name(item), attributes))
    return hashlib.sha256(json_bytes(descriptions)).hexdigest()


def storage_aliases(values):
    """Describe alias topology without including addresses in a cache key."""
    identities, groups = {}, []
    for i, value in enumerate(values):
        storage = value.untyped_storage()
        identity = (
            (str(value.device), storage.data_ptr())
            if storage.nbytes()
            else ("empty", i)
        )
        if identity not in identities:
            identities[identity] = len(identities)
        groups.append(identities[identity])
    return groups


def pack_constants(values):
    """Save complete independent storages while retaining each constant's ABI."""
    storages, identities, specs = [], {}, []
    for value in values:
        storage = value.untyped_storage()
        identity = (
            str(value.device),
            storage.data_ptr() if storage.nbytes() else id(value),
        )
        if identity not in identities:
            identities[identity] = len(storages)
            flat = torch.as_strided(
                value.detach(), (storage.nbytes() // value.element_size(),), (1,), 0
            )
            storages.append(flat.cpu())
        index = identities[identity]
        if storages[index].dtype != value.dtype:
            raise ValueError("Aliased constants with different dtypes are unsupported")
        specs.append(
            {
                "storage": index,
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        )
    return tuple(storages), specs


def unpack_constants(storages, specs):
    """Restore CPU scalars and device tensors without changing layout or aliasing."""
    restored, values = {}, []
    for spec in specs:
        index = spec["storage"]
        raw = storages[index]
        if not isinstance(raw, torch.Tensor) or str(raw.dtype) != spec["dtype"]:
            raise ValueError("Saved constant storage dtype is invalid")
        identity = (index, spec["device"])
        if identity not in restored:
            restored[identity] = raw.to(torch.device(spec["device"]))
        value = torch.as_strided(
            restored[identity], spec["shape"], spec["stride"], spec["storage_offset"]
        )
        if (
            str(value.device) != spec["device"]
            or list(value.shape) != spec["shape"]
            or list(value.stride()) != spec["stride"]
            or value.storage_offset() != spec["storage_offset"]
        ):
            raise ValueError("Restored constant ABI differs")
        values.append(value)
    return tuple(values)


def compiled_source_identity(graph):
    """Identify the exact generated wrapper, including its runtime assertions."""
    source = graph.source_code
    if not isinstance(source, str) or not source:
        raise ValueError("Compiled graph has no generated source")
    return {
        "cache_key": graph.cache_key,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "guards_expr": graph.guards_expr,
    }


def runner_source_identities(runner):
    """Inspect only callable wrappers, without running a graph or reading tensors.

    Unknown wrapper forms fail closed. This deliberately avoids a generic walk
    through object attributes, which could visit unrelated compiler cache entries.
    """
    from torch._functorch._aot_autograd.runtime_wrappers import (
        SerializableCompiledFunction,
    )
    from torch._inductor.output_code import CompiledFxGraph
    from torch._inductor.standalone_compile import CacheCompiledArtifact

    pending, seen, result = [runner], set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if len(seen) > 10000:
            raise ValueError("Compiled wrapper traversal exceeded its limit")
        if isinstance(value, CompiledFxGraph):
            identity = compiled_source_identity(value)
            from torch._inductor.codecache import get_path

            source_path = Path(get_path(value.cache_key, "py")[2])
            if (
                not source_path.is_file()
                or digest(source_path) != identity["source_sha256"]
            ):
                raise ValueError(
                    "Actual generated source file differs from the saved wrapper"
                )
            result.append(identity)
        elif isinstance(value, CacheCompiledArtifact):
            pending.append(value._compiled_fn)
        elif isinstance(value, SerializableCompiledFunction):
            pending.append(value.compiled_fn)
            if hasattr(value, "__wrapped__"):
                pending.append(value.__wrapped__)
        elif isinstance(value, types.FunctionType):
            pending.extend(cell.cell_contents for cell in value.__closure__ or ())
            pending.extend(value.__defaults__ or ())
            pending.extend((value.__kwdefaults__ or {}).values())
            if hasattr(value, "__wrapped__"):
                pending.append(value.__wrapped__)
            # AOT code-generated wrappers capture compiled_fn in their globals
            # rather than a Python closure. Visit only referenced callables.
            for name in value.__code__.co_names:
                candidate = value.__globals__.get(name)
                if isinstance(candidate, CompiledFxGraph) or (
                    name
                    in {"compiled_fn", "compiled_fw", "compiled_fw_func", "inner_fn"}
                    and isinstance(
                        candidate, (types.FunctionType, types.MethodType, partial)
                    )
                ):
                    pending.append(candidate)
        elif isinstance(value, types.MethodType):
            pending.extend((value.__self__, value.__func__))
        elif isinstance(value, partial):
            pending.extend((value.func, *value.args))
            pending.extend((value.keywords or {}).values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return sorted(result, key=json_bytes)


def audit_compiler_guards(runner):
    """Audit the saved AOT entry and its exact generated forward program.

    Empty cache guards do not mean every size is accepted. Unbacked ShapeVars
    carry their constraints in the generated wrapper's runtime assertions.
    """
    try:
        from torch.compiler._cache import CacheArtifactManager

        payload, key = runner._validate_and_unpack()
        artifacts = CacheArtifactManager.deserialize(payload)
        if artifacts is None:
            raise RuntimeError("Cannot decode captured cache artifacts")
        autograd, forward, inductor, forward_keys, source_records = [], [], [], [], []
        for kind, entries in artifacts.items():
            for artifact in entries:
                if kind not in ("aot_autograd", "inductor"):
                    continue
                # These are trusted compiler-produced bytes, not external pickles.
                entry = pickle.loads(artifact.content)
                if kind == "aot_autograd":
                    if artifact.key != key or entry.compiled_bw is not None:
                        raise ValueError(
                            "Expected one inference AOT entry for the explicit Joint graph"
                        )
                    autograd.append(entry.guards_expr)
                    forward.append(entry.compiled_fw.fx_graph_guard_expr)
                    forward_keys.append(entry.compiled_fw.fx_graph_cache_info[0])
                else:
                    inductor.append(entry.guards_expr)
                    source_records.append(
                        (artifact.key, compiled_source_identity(entry))
                    )
        confirmed = len(autograd) == len(forward) == len(forward_keys) == 1 and bool(
            inductor
        )
        sources = [
            source for cache_key, source in source_records if cache_key in forward_keys
        ]
        return {
            "status": "recorded",
            "autograd_cache_key": key,
            "aot_guards": autograd,
            "forward_fx_guards": forward,
            "inductor_guards": inductor,
            "all_recorded_guards_none": confirmed
            and all(value is None for value in (*autograd, *forward, *inductor)),
            "forward_sources": sorted(sources, key=json_bytes),
            "complete_forward_source": confirmed and len(sources) == 1,
            "policy": "Original compiled Joint domain only; all generated runtime assertions remain authoritative.",
        }
    except Exception as error:
        return {
            "status": "unavailable",
            "reason": str(error),
            "policy": "recompile",
        }


def tensor_contract(value, axes, contiguous):
    shape = [axes.get(i, int(size)) for i, size in enumerate(value.shape)]
    if any(isinstance(dim, str) and value.shape[i] < 2 for i, dim in enumerate(shape)):
        raise ValueError("A dynamic Joint dimension must be at least two")
    return {
        "shape": shape,
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "is_conj": value.is_conj(),
        "is_neg": value.is_neg(),
        "contiguous": contiguous,
        "stride": None if contiguous else list(value.stride()),
        "storage_offset": None if contiguous else value.storage_offset(),
        "alignment_mod16": value.data_ptr() % 16 if value.numel() else 0,
    }


def runtime_signature(device, options):
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    result = {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "python": list(sys.version_info[:3]),
        "machine": platform.machine(),
        "cuda_build": torch.version.cuda,
        "triton": triton_version,
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "options": options,
        "platform": platform.system(),
        "device": str(device),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("e3nn", "numpy", "sympy")
        },
        "assertion_configuration": {
            name: options.get(name, getattr(torch._inductor.config, name))
            for name in (
                "scalar_asserts",
                "size_asserts",
                "unsafe_skip_scalar_range_asserts",
            )
        },
        "backend_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if (
                key.startswith(("TACE_", "EQX_", "E3NN_", "TRITON_", "TORCHINDUCTOR_"))
                or key
                in ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "PYTORCH_JIT")
            )
            and not ("CACHE" in key and (key.endswith("DIR") or key.endswith("PATH")))
        },
    }
    if device.type == "cuda":
        result["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    else:
        result["cpu"] = {
            "capability": torch.backends.cpu.get_cpu_capability(),
            "threads": torch.get_num_threads(),
        }
    return result


class JointArtifactCache:
    """Guard and persist the existing explicit loss/parameter-gradient runner.

    Parameters
    ----------
    directory : str or pathlib.Path
        Local directory for compiled artifacts.
    contract : dict
        Complete source/context/ABI contract. Dynamic dimensions are explicit
        named axes shared between inputs; every other dimension is static.
    """

    def __init__(self, directory, contract):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.tensor_guard = None
        self.tensor_guard_attempted = False
        self.dynamic_tensor_specs = ()
        self.tensor_device_layout = ()
        self.key = hashlib.sha256(json_bytes(contract)).hexdigest()
        self.binary = self.directory / f"{self.key}.bin"
        self.constants_path = self.directory / f"{self.key}.constants.pt"
        self.manifest = self.directory / f"{self.key}.json"
        self.info = {
            "status": "miss",
            "key": self.key,
            "directory": str(self.directory),
        }

    @classmethod
    def create(
        cls,
        *,
        module,
        keys,
        parameter_names,
        buffer_names,
        trainable,
        entries,
        values,
        dynamic_axes,
        options,
    ):
        """Build the cache contract from module configuration and tensor inputs."""
        if not hasattr(torch._inductor, "CompiledArtifact"):
            return None
        try:
            from torch._inductor.runtime.cache_dir_utils import cache_dir

            directory = Path(cache_dir()) / "tace" / "joint" / FORMAT_VERSION
            devices = {value.device for value in values}
            if len(devices) != 1:
                raise ValueError("Mixed-device Joint artifacts are unsupported")
            specs = [
                tensor_contract(value, dynamic_axes.get(key, {}), True)
                for key, value in zip(keys, values[: len(keys)], strict=True)
            ]
            specs.extend(
                tensor_contract(value, {}, False) for value in values[len(keys) :]
            )
            contract = {
                "format_version": FORMAT_VERSION,
                "context_sha256": module_context(module),
                "source": source_fingerprint(),
                "runtime": runtime_signature(next(iter(devices)), options),
                "input_keys": list(keys),
                "parameter_names": list(parameter_names),
                "buffer_names": list(buffer_names),
                "trainable": list(trainable),
                "state_entries": [list(entry) for entry in entries],
                "prediction_keys": list(module.model.output_keys),
                "module_types": [
                    (name, f"{type(item).__module__}.{type(item).__qualname__}")
                    for name, item in module.named_modules()
                ],
                "tensor_inputs": specs,
                "storage_aliases": storage_aliases(values),
                "dynamic_min": 2,
            }
            cache = cls(directory, contract)
            cache.validate_inputs(values)
            return cache
        except Exception as error:
            logging.warning(
                "Joint artifact disabled; original compilation retained: %s", error
            )
            return None

    @contextmanager
    def transaction(self):
        """Serialize lookup, cold compilation, and atomic publication by key."""
        begin = time.perf_counter()
        with FileLock(self.directory / f"{self.key}.lock").acquire(poll_interval=1.0):
            self.info["lock_wait_seconds"] = time.perf_counter() - begin
            yield self

    def configure_domain(self, runner, audit):
        """Widen only after verifying original code identity and enabled assertions."""
        config = self.contract["runtime"].get("assertion_configuration", {})
        guard_lists = [
            audit.get(name)
            for name in ("aot_guards", "forward_fx_guards", "inductor_guards")
        ]
        guards_empty = all(
            isinstance(values, list)
            and values
            and all(value is None for value in values)
            for values in guard_lists
        )
        checks = {
            "all_cache_guards_none": audit.get("status") == "recorded"
            and audit.get("all_recorded_guards_none") is True
            and guards_empty,
            "complete_saved_forward": audit.get("complete_forward_source") is True,
            "scalar_asserts": config.get("scalar_asserts") is True,
            "size_asserts": config.get("size_asserts") is True,
            "range_asserts": config.get("unsafe_skip_scalar_range_asserts") is False,
        }
        try:
            actual = runner_source_identities(runner)
            expected = audit.get("forward_sources", [])
            checks["actual_forward_source_matches"] = (
                bool(expected) and actual == expected
            )
        except Exception as error:
            actual = []
            checks["actual_forward_source_matches"] = False
            self.info["source_verification_error"] = str(error)
        self.info.update(
            dynamic_domain_checks=checks,
            actual_forward_sources=actual,
            compiler_guard_audit=audit,
        )
        return all(checks.values())

    def validate_inputs(self, values):
        """Enforce the explicit ABI before calling an artifact with skipped guards."""
        if self.tensor_guard is not None:
            try:
                matches = self.tensor_guard.check(*values)
            except Exception:
                matches = False
            if matches:
                return self.validate_input_relations(values)

        dimensions = self.validate_inputs_full(values)
        if not self.tensor_guard_attempted:
            self.tensor_guard_attempted = True
            try:
                from torch._C._dynamo.guards import TensorGuards

                # Only build from a fully validated, detached runtime ABI. The
                # C++ guard has additional type/dispatch/requires_grad checks;
                # their failure always returns to the original validator.
                if all(
                    type(value) is torch.Tensor and not value.requires_grad
                    for value in values
                ):
                    specs = self.contract["tensor_inputs"]
                    guard = TensorGuards(
                        *values,
                        dynamic_dims_sizes=[
                            [
                                None if isinstance(dim, str) else dim
                                for dim in spec["shape"]
                            ]
                            for spec in specs
                        ],
                        dynamic_dims_strides=[
                            [None] * len(spec["shape"])
                            if spec["contiguous"]
                            else spec["stride"]
                            for spec in specs
                        ],
                    )
                    if guard.check(*values):
                        self.tensor_device_layout = tuple(
                            (
                                torch.device(spec["device"]),
                                getattr(torch, spec["layout"].removeprefix("torch.")),
                            )
                            for spec in specs
                        )
                        self.dynamic_tensor_specs = tuple(
                            (index, spec)
                            for index, spec in enumerate(specs)
                            if spec["contiguous"]
                            or any(isinstance(dim, str) for dim in spec["shape"])
                        )
                        self.tensor_guard = guard
            except Exception:
                # The private PyTorch interface is optional. Unsupported
                # versions and tensor kinds retain every original check.
                self.tensor_guard = None
        return dimensions

    def validate_input_relations(self, values):
        """Retain checks that the batched C++ metadata guard does not express."""
        dimensions = {}
        for index, spec in self.dynamic_tensor_specs:
            value = values[index]
            if spec["contiguous"] and not value.is_contiguous():
                raise ValueError(f"Persistent Joint input {index} must be contiguous")
            for actual, expected in zip(value.shape, spec["shape"], strict=True):
                if isinstance(expected, str):
                    if actual < self.contract["dynamic_min"]:
                        raise ValueError(
                            f"Persistent Joint dynamic axis {expected} is below its minimum"
                        )
                    if expected in dimensions and dimensions[expected] != actual:
                        raise ValueError(
                            f"Persistent Joint dynamic axis {expected} is inconsistent"
                        )
                    dimensions[expected] = actual
        for index, (value, spec, (device, layout)) in enumerate(
            zip(
                values,
                self.contract["tensor_inputs"],
                self.tensor_device_layout,
                strict=True,
            )
        ):
            # TensorGuards applies thread-local dispatch modifiers. Keep the
            # properties represented by dispatch keys explicit, so a masked
            # device/layout/conjugate/negative key cannot weaken the ABI.
            if (
                value.device != device
                or value.layout != layout
                or value.is_conj() != spec["is_conj"]
                or value.is_neg() != spec["is_neg"]
            ):
                raise ValueError(
                    f"Persistent Joint device/layout/view changed at input {index}"
                )
            if value.numel() and value.data_ptr() % 16 != spec["alignment_mod16"]:
                raise ValueError(
                    f"Persistent Joint input alignment changed at input {index}"
                )
            if (
                not spec["contiguous"]
                and value.storage_offset() != spec["storage_offset"]
            ):
                raise ValueError(
                    f"Persistent Joint state storage offset changed at input {index}"
                )
        if storage_aliases(values) != self.contract["storage_aliases"]:
            raise ValueError("Persistent Joint input/state alias topology changed")
        return dimensions

    def validate_inputs_full(self, values):
        """Original complete validator, also used for every C++ guard fallback."""
        specs = self.contract["tensor_inputs"]
        if len(values) != len(specs):
            raise ValueError("Persistent Joint input count changed")
        dimensions = {}
        for index, (value, spec) in enumerate(zip(values, specs, strict=True)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Persistent Joint input {index} must be a tensor")
            if (
                str(value.dtype) != spec["dtype"]
                or str(value.device) != spec["device"]
                or str(value.layout) != spec["layout"]
                or value.ndim != len(spec["shape"])
                or value.is_conj() != spec["is_conj"]
                or value.is_neg() != spec["is_neg"]
            ):
                raise ValueError(
                    f"Persistent Joint dtype/device/rank/layout changed at input {index}"
                )
            if spec["contiguous"]:
                if not value.is_contiguous():
                    raise ValueError(
                        f"Persistent Joint input {index} must be contiguous"
                    )
            elif (
                list(value.stride()) != spec["stride"]
                or value.storage_offset() != spec["storage_offset"]
            ):
                raise ValueError(
                    f"Persistent Joint state stride or storage offset changed at input {index}"
                )
            if value.numel() and value.data_ptr() % 16 != spec["alignment_mod16"]:
                raise ValueError(
                    f"Persistent Joint input alignment changed at input {index}"
                )
            for actual, expected in zip(value.shape, spec["shape"], strict=True):
                if isinstance(expected, str):
                    if actual < self.contract["dynamic_min"]:
                        raise ValueError(
                            f"Persistent Joint dynamic axis {expected} is below its minimum"
                        )
                    if expected in dimensions and dimensions[expected] != actual:
                        raise ValueError(
                            f"Persistent Joint dynamic axis {expected} is inconsistent"
                        )
                    dimensions[expected] = actual
                elif actual != expected:
                    raise ValueError(
                        f"Persistent Joint static shape changed at input {index}"
                    )
        if storage_aliases(values) != self.contract["storage_aliases"]:
            raise ValueError("Persistent Joint input/state alias topology changed")
        return dimensions

    def lift_constants(self, graph, live_values):
        """Make GraphModule constants explicit so PyTorch's saved ABI is complete.

        aot_module_simplified prepends GraphModule state through an outer Python
        closure that CompiledArtifact does not restore. Eliminate that closure's
        state, preserving any aliases to existing live parameter/buffer inputs.
        """
        placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
        last = placeholders[-1]
        constants, descriptions, records = [], [], []
        for node in list(graph.graph.nodes):
            if node.op != "get_attr":
                continue
            value = graph
            for field in node.target.split("."):
                value = getattr(value, field)
            if not isinstance(value, torch.Tensor) or value.is_meta:
                raise ValueError(f"Unsupported Joint constant: {node.target}")
            if value.layout != torch.strided or value.is_conj() or value.is_neg():
                raise ValueError(
                    f"Unsupported Joint constant layout/view flags: {node.target}"
                )
            aliases = [
                i
                for i, live in enumerate(live_values)
                if torch._C._is_alias_of(value, live)
            ]
            if aliases:
                exact = [
                    i
                    for i in aliases
                    if (
                        value.dtype == live_values[i].dtype
                        and value.device == live_values[i].device
                        and value.layout == live_values[i].layout
                        and value.shape == live_values[i].shape
                        and value.stride() == live_values[i].stride()
                        and value.storage_offset() == live_values[i].storage_offset()
                        and value.is_conj() == live_values[i].is_conj()
                        and value.is_neg() == live_values[i].is_neg()
                    )
                ]
                if not exact:
                    raise ValueError(
                        f"Joint constant {node.target} aliases a live input with a different view"
                    )
                index = exact[0]
            else:
                if value.requires_grad:
                    raise ValueError(
                        f"Trainable Joint get_attr cannot be persisted: {node.target}"
                    )
                index = None
            records.append((node, value, index))
        # Preflight is complete before changing the graph, so unsupported state
        # can safely fall back to exactly the original compiler path.
        for node, value, index in records:
            if index is not None:
                replacement = placeholders[index]
                kind = "existing-live-input"
            else:
                with graph.graph.inserting_after(last):
                    replacement = graph.graph.placeholder(
                        f"artifact_constant_{len(constants)}"
                    )
                replacement.meta.update(node.meta)
                last = replacement
                constants.append(value.detach())
                kind = "immutable-trace-constant"
            descriptions.append(
                {
                    "name": node.target,
                    "kind": kind,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "numel": value.numel(),
                    "device": str(value.device),
                    "stride": list(value.stride()),
                    "storage_offset": value.storage_offset(),
                }
            )
            node.replace_all_uses_with(replacement)
            graph.graph.erase_node(node)
        graph.graph.lint()
        graph = torch.fx.GraphModule({}, graph.graph)
        if list(graph.named_parameters()) or list(graph.named_buffers()):
            raise AssertionError("Persistent Joint still has implicit module state")
        self.info["lifted_constants"] = descriptions
        return graph, tuple(constants)

    def wrap(self, runner, output_schema, constants=()):
        """Guard calls while preserving None gradients and output order."""
        none_mask = output_schema["none_mask"]
        expected = (
            1 + len(self.contract["trainable"]) + len(self.contract["prediction_keys"])
        )
        if (
            len(none_mask) != expected
            or none_mask[0]
            or any(none_mask[1 + len(self.contract["trainable"]) :])
        ):
            raise ValueError("Persistent Joint saved output schema is invalid")

        def run(*values):
            self.validate_inputs(values)
            outputs = runner(*values, *constants)
            if not isinstance(outputs, (tuple, list)) or len(outputs) != expected:
                raise RuntimeError("Persistent Joint output count changed")
            for index, (output, is_none) in enumerate(
                zip(outputs, none_mask, strict=True)
            ):
                if (output is None) != is_none or (
                    output is not None and not isinstance(output, torch.Tensor)
                ):
                    raise RuntimeError(
                        f"Persistent Joint None/tensor output changed at {index}"
                    )
            return outputs

        return run

    def load(self):
        """Load without FX tracing; a corrupt/unavailable artifact is a miss."""
        if not self.manifest.exists() or not self.binary.exists():
            return None
        begin = time.perf_counter()
        try:
            manifest = json.loads(self.manifest.read_text())
            if manifest["contract"] != json.loads(json_bytes(self.contract)):
                raise ValueError("Artifact contract differs")
            if manifest["binary_sha256"] != digest(self.binary):
                raise ValueError("Artifact binary hash differs")
            if manifest["constants_sha256"] != digest(self.constants_path):
                raise ValueError("Artifact constant payload hash differs")
            constant_storages = torch.load(
                self.constants_path, map_location="cpu", weights_only=True
            )
            constants = unpack_constants(constant_storages, manifest["constant_specs"])
            runner = torch._inductor.CompiledArtifact.load(
                path=str(self.binary), format="binary"
            )
            if not self.configure_domain(
                runner, manifest.get("compiler_guard_audit", {})
            ):
                raise ValueError(
                    "Loaded code or dynamic shape guards could not be verified"
                )
            wrapped = self.wrap(runner, manifest["output_schema"], constants)
            self.info.update(status="hit", load_seconds=time.perf_counter() - begin)
            self.info["lifted_constants"] = manifest["lifted_constants"]
            logging.info(
                "Joint artifact hit %s: loaded in %.3f s without make_fx",
                self.key[:12],
                self.info["load_seconds"],
            )
            return wrapped
        except Exception as error:
            self.info.update(status="invalid", reason=str(error))
            logging.warning(
                "Joint artifact load failed; compiling original graph: %s", error
            )
            return None

    def save(self, runner, output_schema, constants=()):
        """Atomically publish PyTorch's binary, or explicitly report fallback."""
        binary_temp = self.directory / f".{self.key}.{os.getpid()}.bin.tmp"
        manifest_temp = self.directory / f".{self.key}.{os.getpid()}.json.tmp"
        constants_temp = self.directory / f".{self.key}.{os.getpid()}.constants.tmp"
        begin = time.perf_counter()
        try:
            runner.save(path=str(binary_temp), format="binary")
            constant_storages, constant_specs = pack_constants(constants)
            torch.save(constant_storages, constants_temp)
            manifest = {
                "contract": self.contract,
                "binary_sha256": digest(binary_temp),
                "output_schema": output_schema,
                "constants_sha256": digest(constants_temp),
                "constant_specs": constant_specs,
                "lifted_constants": self.info.get("lifted_constants", []),
                "compiler_guard_audit": audit_compiler_guards(runner),
            }
            if not self.configure_domain(runner, manifest["compiler_guard_audit"]):
                raise ValueError(
                    "Compiled code or dynamic shape guards could not be verified"
                )
            with open(manifest_temp, "wb") as stream:
                stream.write(json_bytes(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(binary_temp, self.binary)
            os.replace(constants_temp, self.constants_path)
            os.replace(manifest_temp, self.manifest)
            self.info.update(
                status="saved",
                save_seconds=time.perf_counter() - begin,
                binary_bytes=self.binary.stat().st_size,
            )
            logging.info(
                "Joint artifact saved %s (%d bytes)",
                self.key[:12],
                self.info["binary_bytes"],
            )
            return self.wrap(runner, output_schema, constants)
        except Exception as error:
            self.info.update(status="unsupported", reason=str(error))
            logging.warning(
                "Joint artifact cannot be saved; original compiled runner retained: %s",
                error,
            )
            return self.wrap(runner, output_schema, constants) if constants else runner
        finally:
            binary_temp.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)
            constants_temp.unlink(missing_ok=True)
