"""Shared CUDA compilation, binary caching and launch configuration."""

import hashlib
import os
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import torch

_KERNELS = OrderedDict()
_LAUNCH_CONFIGS = OrderedDict()
_RUNTIME_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(
    max_workers=max(
        1, min(int(os.environ.get("MAX_JOBS", "16")), len(os.sched_getaffinity(0)))
    )
)


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


def launch_config(kernel, width, count, storage, threads, tensors, scalars, outputs):
    """Choose a cached block size using private outputs and measured latency."""
    common, per_warp = storage
    key = kernel, width, (max(1, count) - 1).bit_length(), storage
    if key in _LAUNCH_CONFIGS:
        _LAUNCH_CONFIGS.move_to_end(key)
        return _LAUNCH_CONFIGS[key]
    candidates = [
        n
        for n in (32, 64, 128, 256)
        if (width <= 32 or n <= max(32, (width + 31) // 32 * 32))
        and common + per_warp * (n // 32) <= 49152
        and kernel.active_blocks(n, common + per_warp * (n // 32))
    ]
    scratch_bytes = sum(tensors[i].numel() * tensors[i].element_size() for i in outputs)
    if (
        count * width >= 4096
        and scratch_bytes <= 64 << 20
        and not torch.cuda.is_current_stream_capturing()
    ):
        # Tuning must not accumulate into the actual results, including when
        # the same destination is shared by several derivative terms.
        scratch = {slot: torch.zeros_like(tensors[slot]) for slot in outputs}
        arguments = [
            scratch.get(i, value).data_ptr() for i, value in enumerate(tensors)
        ] + list(scalars)
        stream = torch.cuda.current_stream(tensors[0].device)
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        timings = []
        for n in candidates:
            edges_per_block = 1 if width > 32 else n // 32
            channels_per_block = n if width > 32 else 32
            candidate = (
                kernel,
                arguments,
                (count + edges_per_block - 1) // edges_per_block,
                (width + channels_per_block - 1) // channels_per_block,
                n,
                common + per_warp * (n // 32),
            )
            runtime().launch([candidate] * 2, stream.cuda_stream)
            start.record(stream)
            runtime().launch([candidate] * 5, stream.cuda_stream)
            stop.record(stream)
            stop.synchronize()
            timings.append((start.elapsed_time(stop), n))
        threads = min(timings)[1]
    result = threads, common + per_warp * (threads // 32)
    if not torch.cuda.is_current_stream_capturing():
        _LAUNCH_CONFIGS[key] = result
        if len(_LAUNCH_CONFIGS) > 1024:
            _LAUNCH_CONFIGS.popitem(last=False)
    return result
