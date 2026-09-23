"""Generated CUDA execution for aligned-frame convolutions."""

import hashlib
import os
import tempfile
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import torch

from .codegen import convolution_source, rotation_source
from .graph import prepare_graph
from .schedule import (
    contraction_groups,
    project,
    register_estimate,
    schedule,
    split_program,
)

WORKSPACE_BYTES = 1024 << 20
CHUNK_SIZE = 65536
ROW_SIZE = 64
THREADS = 128
_KERNELS = OrderedDict()
_RUNTIME_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=min(16, len(os.sched_getaffinity(0))))


@lru_cache(maxsize=1)
def runtime():
    """Build the model-independent launcher only when CUDA is requested."""
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise ImportError(
            "The EQX CUDA backend requires a CUDA toolkit; set CUDA_HOME."
        )
    cuda_home = Path(CUDA_HOME)
    with _RUNTIME_LOCK:
        return load(
            name="eqx_cuda_runtime",
            sources=[str(Path(__file__).parent / "csrc" / "runtime.cpp")],
            extra_include_paths=[str(cuda_home / "include")],
            extra_cflags=["-O3"],
            extra_ldflags=[
                f"-L{cuda_home / 'lib64'}",
                f"-Wl,-rpath,{cuda_home / 'lib64'}",
                f"-L{cuda_home / 'lib64' / 'stubs'}",
                "-lnvrtc",
                "-lcuda",
            ],
            with_cuda=False,
        )


def compile_binary(source, options, compiler):
    """Cache CUDA binaries independently of model tensors and graph sizes."""
    from filelock import FileLock

    digest = hashlib.sha256(
        (source + repr(options) + repr(compiler.version())).encode()
    ).hexdigest()
    root = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "eqx" / "cuda"
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest}.cubin"
    with FileLock(str(path) + ".lock"):
        if path.exists():
            return path.read_bytes()
        binary = compiler.compile(source, options)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(binary)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return binary


def kernels(sources, device):
    """Compile independent phases concurrently and load on the active device."""
    result, pending = {}, {}
    with torch.cuda.device(device):
        capability = torch.cuda.get_device_capability(device)
        options = (
            "--std=c++17",
            f"--gpu-architecture=sm_{capability[0]}{capability[1]}",
            "--fmad=true",
        )
        missing = list(
            dict.fromkeys(
                source for source in sources if (device, source) not in _KERNELS
            )
        )
        if missing:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up the EQX CUDA forward and required derivatives before CUDA Graph capture."
                )
            compiler = runtime()
            pending = {
                source: _POOL.submit(compile_binary, source, options, compiler)
                for source in missing
            }
        for source in sources:
            key = (device, source)
            if key not in _KERNELS:
                _KERNELS[key] = runtime().Kernel(pending[source].result(), "run")
            _KERNELS.move_to_end(key)
            result[source] = _KERNELS[key]
        # Captured CUDA Graphs can outlive the host execution-plan cache. Keep
        # their CUDA modules resident so replay never references unloaded code.
    return result


@lru_cache(maxsize=256)
def execution_plan(plan, specification, dimensions, shared, dtype, grouped, device):
    """Cache tensor-free schedules, using compiled resource usage to bound tiles."""
    count = 1 + max(
        i for _, operands, results, _ in specification for i in (*operands, *results)
    )
    tokens = tuple(object() for _ in range(count))
    locations = {id(token): i for i, token in enumerate(tokens)}
    calls = tuple(
        (
            outputs,
            tuple(tokens[i] for i in operands),
            tuple(tokens[i] for i in results),
            weighted,
        )
        for outputs, operands, results, weighted in specification
    )
    phases = []
    for indices in contraction_groups(plan.path_data):
        entries = [(*plan.path_data[i], plan.sparse_paths[i], i) for i in indices]
        entries.sort(key=lambda entry: (entry[1][7], entry[1][0]))
        paths = tuple((mode, path, cg) for mode, path, cg, _ in entries)
        pending = deque([(paths, calls)])
        while pending:
            paths, terms = pending.popleft()
            inputs, results, *operations = schedule(terms)
            if (
                register_estimate(
                    [(path, cg) for _, path, cg in paths],
                    operations,
                    4 if dtype == "float" else 8,
                )
                > 256
                and sum(len(term[0]) for term in terms) > 1
            ):
                pending.extend((paths, group) for group in split_program(terms))
                continue
            roles = {role for outputs, *_ in terms for role in outputs}
            owner = (
                -1 if not grouped else 1 if roles == {6} else 0 if roles == {0} else -1
            )
            # Bound the live receiver state without introducing degree thresholds.
            if (
                owner == 1
                and sum(path[5] for _, path, _ in paths) * len(operations[8]) > 80
            ):
                owner = -1
            code, width = convolution_source(
                paths,
                tuple(operations),
                dimensions,
                shared,
                len(inputs),
                len(results),
                dtype,
                owner,
            )
            phases.append((code, width, owner, inputs, results, paths, terms))
    # Compile candidates together, then refine only kernels with excessive live
    # state. Splitting is independent of angular degree and never changes paths.
    accepted = []
    while phases:
        compiled = kernels([phase[0] for phase in phases], device)
        pending = []
        for code, width, owner, inputs, results, paths, terms in phases:
            kernel = compiled[code]
            if kernel.registers > 160 or kernel.local_bytes:
                boundaries = [
                    i
                    for i in range(1, len(paths))
                    if paths[i - 1][1][7] != paths[i][1][7]
                ]
                if sum(len(term[0]) for term in terms) > 1:
                    parts = tuple((paths, group) for group in split_program(terms))
                elif boundaries:
                    midpoint = min(
                        boundaries,
                        key=lambda i: abs(2 * i - len(paths)),
                    )
                    parts = ((paths[:midpoint], terms), (paths[midpoint:], terms))
                elif len(paths) > 1:
                    midpoint = len(paths) // 2
                    parts = ((paths[:midpoint], terms), (paths[midpoint:], terms))
                else:
                    parts = ()
                if parts:
                    for subset, group in parts:
                        operands, destinations, *operations = schedule(group)
                        source, width = convolution_source(
                            subset,
                            tuple(operations),
                            dimensions,
                            shared,
                            len(operands),
                            len(destinations),
                            dtype,
                            owner,
                        )
                        pending.append(
                            (
                                source,
                                width,
                                owner,
                                operands,
                                destinations,
                                subset,
                                group,
                            )
                        )
                    continue
            accepted.append(
                (
                    code,
                    width,
                    owner,
                    tuple(locations[id(value)] for value in inputs),
                    tuple(locations[id(value)] for value in results),
                )
            )
        phases = pending
    return tuple(accepted)


def contract(plan, source, target, calls, shared=None):
    """Execute a cached mixed-adjoint program without edge message storage."""
    x, radial, _, din, dout, amplitudes, y = calls[0][1]
    if shared is None:
        shared = tuple(value.size(0) == 1 for value in (radial, din, dout, amplitudes))
    dimensions = (x.size(1), y.size(1), din.size(1), radial.size(1), amplitudes.size(1))
    values, indices = [], {}
    specification = []
    for outputs, operands, results, weighted in calls:
        slots = []
        for value in (*operands, *results):
            if id(value) not in indices:
                indices[id(value)] = len(values)
                values.append(value)
            slots.append(indices[id(value)])
        specification.append((outputs, tuple(slots[:7]), tuple(slots[7:]), weighted))
    phases = execution_plan(
        plan,
        tuple(specification),
        dimensions,
        shared,
        "float" if x.dtype == torch.float32 else "double",
        source.numel() >= 1024,
        x.device,
    )
    compiled = kernels([phase[0] for phase in phases], x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    graphs = {}
    for code, width, owner, inputs, results in phases:
        if owner >= 0:
            if owner not in graphs:
                graphs[owner] = prepare_graph(
                    source,
                    target,
                    x.size(0) if owner == 0 else y.size(0),
                    owner,
                    ROW_SIZE,
                )
            order, tasks = graphs[owner]
            count = tasks.size(0)
        else:
            order, tasks, count = source, source, source.numel()
        args = [
            value.data_ptr()
            for value in (
                *(values[i] for i in (*inputs, *results)),
                source,
                target,
                order,
                tasks,
            )
        ]
        args += [source.numel(), count]
        compiled[code].launch(
            args,
            (count + THREADS // 32 - 1) // (THREADS // 32),
            (width + 31) // 32,
            THREADS,
            stream,
        )


def contract_many(plan, source, target, calls):
    """Consume projected radial weights and their adjoints in bounded chunks."""
    contiguous = {}
    prepared = []
    for outputs, operands, results, weighted in calls:
        if operands[0].dtype not in (torch.float32, torch.float64):
            raise TypeError("EQX CUDA convolutions support float32 and float64.")
        values = []
        for role, value in enumerate(operands):
            if outputs != (role,) and not value.is_contiguous():
                if id(value) not in contiguous:
                    contiguous[id(value)] = value.contiguous()
                value = contiguous[id(value)]
            values.append(value)
        prepared.append((outputs, tuple(values), results, weighted))
    source, target = source.contiguous(), target.contiguous()
    if prepared[0][1][2].numel():
        project(plan, source, target, prepared, contract, CHUNK_SIZE, WORKSPACE_BYTES)
    else:
        contract(plan, source, target, prepared)


def rotate(cg, output, values, result, rotation_plan):
    """Evaluate a recursive Wigner contraction using CUDA, including transposes."""
    inputs = [value for i, value in enumerate(values) if i != output]
    indices, coefficients = rotation_plan(cg)[output]
    widths = (9, cg.size(1) ** 2, cg.size(2) ** 2)
    code = rotation_source(
        widths,
        coefficients.size(1),
        "float" if cg.dtype == torch.float32 else "double",
        output,
    )
    kernel = kernels([code], result.device)[code]
    arguments = [value.data_ptr() for value in (*inputs, result, indices, coefficients)]
    arguments += [result.size(0), *inputs[0].stride(), *inputs[1].stride()]
    kernel.launch(
        arguments,
        (result.numel() + 127) // 128,
        1,
        128,
        torch.cuda.current_stream(result.device).cuda_stream,
    )
