"""Select a live TF32 GEMM algorithm for wide radial parameter reductions.

This opt-in post-autodiff operator retains FP32 tensors and accumulation.
Plans contain layouts/algorithm choices, never cached parameter values or
output buffers. Calibration runs before CUDA Graph capture.
"""

import ctypes as ct
import json
import os
from functools import lru_cache
from pathlib import Path

import torch


class Algorithm(ct.Structure):
    _fields_ = [("data", ct.c_uint64 * 8)]


class Heuristic(ct.Structure):
    _fields_ = [
        ("algo", Algorithm),
        ("workspace", ct.c_size_t),
        ("status", ct.c_int),
        ("waves", ct.c_float),
        ("reserved", ct.c_int * 4),
    ]


def checked(status, operation):
    if status:
        raise RuntimeError(f"{operation}: cuBLASLt status {status}")


@lru_cache(maxsize=1)
def library():
    directory = Path(torch.__file__).resolve().parent.parent / "nvidia/cu13/lib"
    lib = ct.CDLL(str(directory / "libcublasLt.so.13"))
    P, I, Q, S = ct.c_void_p, ct.c_int, ct.c_uint64, ct.c_size_t
    signatures = {
        "cublasLtCreate": [ct.POINTER(P)],
        "cublasLtMatmulDescCreate": [ct.POINTER(P), I, I],
        "cublasLtMatmulDescSetAttribute": [P, I, P, S],
        "cublasLtMatrixLayoutCreate": [ct.POINTER(P), I, Q, Q, ct.c_int64],
        "cublasLtMatrixLayoutSetAttribute": [P, I, P, S],
        "cublasLtMatmulPreferenceCreate": [ct.POINTER(P)],
        "cublasLtMatmulPreferenceSetAttribute": [P, I, P, S],
        "cublasLtMatmulAlgoGetHeuristic": [
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            I,
            ct.POINTER(Heuristic),
            ct.POINTER(I),
        ],
        "cublasLtMatmul": [
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            ct.POINTER(Algorithm),
            P,
            S,
            P,
        ],
        "cublasLtDestroy": [P],
        "cublasLtMatmulDescDestroy": [P],
        "cublasLtMatrixLayoutDestroy": [P],
        "cublasLtMatmulPreferenceDestroy": [P],
    }
    for name, types in signatures.items():
        getattr(lib, name).argtypes = types
        getattr(lib, name).restype = I
    return lib


class MatrixPlan:
    def __init__(
        self, left, right, workspace_bytes=128 * 1024 * 1024, candidates=32, sm_count=0
    ):
        self.closed = False
        if left.dtype != torch.float32 or right.dtype != torch.float32:
            raise TypeError("The calibrated TF32 plan requires FP32 operands")
        if left.ndim != right.ndim or left.ndim not in (2, 3):
            raise ValueError("Supply matrices or matching matrix batches")
        if left.ndim == 3 and left.shape[0] != right.shape[0]:
            raise ValueError("Matrix batch counts differ")
        self.shape = (*left.shape[:-1], right.shape[-1])
        self.lib = lib = library()
        self.handle = ct.c_void_p()
        self.desc = ct.c_void_p()
        checked(lib.cublasLtCreate(ct.byref(self.handle)), "create")
        checked(lib.cublasLtMatmulDescCreate(ct.byref(self.desc), 77, 0), "desc")
        if sm_count:
            value = ct.c_int(sm_count)
            checked(
                lib.cublasLtMatmulDescSetAttribute(
                    self.desc, 15, ct.byref(value), ct.sizeof(value)
                ),
                "SM target",
            )
        self.layouts = []

        def layout(shape, strides):
            if strides[-1] == 1:
                rows, cols = shape[-2:]
                leading = strides[-2]
                trans = 0
            elif strides[-2] == 1:
                cols, rows = shape[-2:]
                leading = strides[-1]
                trans = 1
            else:
                raise ValueError("Dense matrices or transposes required")
            pointer = ct.c_void_p()
            checked(
                lib.cublasLtMatrixLayoutCreate(
                    ct.byref(pointer), 0, rows, cols, leading
                ),
                "layout",
            )
            order = ct.c_int(1)
            checked(
                lib.cublasLtMatrixLayoutSetAttribute(
                    pointer, 1, ct.byref(order), ct.sizeof(order)
                ),
                "order",
            )
            self.layouts.append(pointer)
            if len(shape) == 3:
                for attribute, value in (
                    (5, ct.c_int(shape[0])),
                    (6, ct.c_int64(strides[0])),
                ):
                    checked(
                        lib.cublasLtMatrixLayoutSetAttribute(
                            pointer, attribute, ct.byref(value), ct.sizeof(value)
                        ),
                        "batch layout",
                    )
            return pointer, trans

        self.a, ta = layout(left.shape, left.stride())
        self.b, tb = layout(right.shape, right.stride())
        strides = (self.shape[-1], 1)
        if left.ndim == 3:
            strides = (self.shape[-2] * self.shape[-1], *strides)
        self.d, _ = layout(self.shape, strides)
        for attribute, value in ((3, ta), (4, tb)):
            datum = ct.c_int(value)
            checked(
                lib.cublasLtMatmulDescSetAttribute(
                    self.desc, attribute, ct.byref(datum), ct.sizeof(datum)
                ),
                "transpose",
            )
        self.pref = ct.c_void_p()
        checked(lib.cublasLtMatmulPreferenceCreate(ct.byref(self.pref)), "preference")
        capacity = ct.c_size_t(workspace_bytes)
        checked(
            lib.cublasLtMatmulPreferenceSetAttribute(
                self.pref, 1, ct.byref(capacity), ct.sizeof(capacity)
            ),
            "workspace size",
        )
        self.workspace = torch.empty(
            workspace_bytes, device=left.device, dtype=torch.uint8
        )
        self.algorithms = (Heuristic * candidates)()
        count = ct.c_int()
        checked(
            lib.cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                self.desc,
                self.a,
                self.b,
                self.d,
                self.d,
                self.pref,
                candidates,
                self.algorithms,
                ct.byref(count),
            ),
            "heuristics",
        )
        self.count = count.value
        self.alpha, self.beta = ct.c_float(1.0), ct.c_float(0.0)

    def run(self, left, right, index, out=None):
        if out is None:
            out = left.new_empty(self.shape, dtype=torch.float32)
        algorithm = self.algorithms[index]
        checked(
            self.lib.cublasLtMatmul(
                self.handle,
                self.desc,
                ct.byref(self.alpha),
                left.data_ptr(),
                self.a,
                right.data_ptr(),
                self.b,
                ct.byref(self.beta),
                out.data_ptr(),
                self.d,
                out.data_ptr(),
                self.d,
                ct.byref(algorithm.algo),
                self.workspace.data_ptr(),
                self.workspace.numel(),
                torch.cuda.current_stream(left.device).cuda_stream,
            ),
            "matmul",
        )
        return out

    def close(self):
        if self.closed:
            return
        self.closed = True
        for pointer in self.layouts:
            checked(self.lib.cublasLtMatrixLayoutDestroy(pointer), "destroy layout")
        checked(
            self.lib.cublasLtMatmulPreferenceDestroy(self.pref), "destroy preference"
        )
        checked(self.lib.cublasLtMatmulDescDestroy(self.desc), "destroy desc")
        checked(self.lib.cublasLtDestroy(self.handle), "destroy handle")

    def __del__(self):
        try:
            self.close()
        except (AttributeError, RuntimeError):
            pass


PLANS = {}
BATCHED_PLANS = {}


def graph_time(call, repeats=8):
    """Measure replay on the same stream used to initialize library workspaces."""
    parent = torch.cuda.current_stream()
    stream = parent if parent.cuda_stream else torch.cuda.Stream()
    stream.wait_stream(parent)
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            call()
        for _ in range(5):
            graph.replay()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
    parent.wait_stream(stream)
    return start.elapsed_time(end) / repeats


def calibrate_batched(left, right):
    """Use a live-validated TF32 batch plan only when it beats torch.bmm."""
    expected = torch.bmm(left, right)
    if not torch.isfinite(expected).all():
        return None
    plan = MatrixPlan(left, right, workspace_bytes=32 * 1024**2, candidates=8)
    output = torch.empty_like(expected)
    best = graph_time(lambda: torch.bmm(left, right, out=output))
    plan.choice = -1
    scale = expected.norm().clamp_min(1e-30)
    for index in range(plan.count):
        if plan.algorithms[index].status:
            continue
        try:
            plan.run(left, right, index, output)
            if (
                not torch.isfinite(output).all()
                or (output - expected).norm() / scale > 0.001
            ):
                continue
            elapsed = graph_time(lambda: plan.run(left, right, index, output))
            if elapsed < best * 0.98:
                best, plan.choice = elapsed, index
        except RuntimeError:
            continue
    if plan.choice < 0:
        plan.close()
        return None
    return plan


@torch.library.custom_op(
    "eqx_oam_xl::lt_radial_bmm", mutates_args=(), device_types="cuda"
)
def lt_radial_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Read current operands with a graph-owned, stream-local TF32 plan."""
    from equivariantx.ops.common.cuda.graph import CAPTURE_CONSTANTS

    from .precision import require_tensors

    require_tensors((left, right))
    if (
        left.dtype != torch.float32
        or torch.backends.cuda.matmul.fp32_precision != "tf32"
    ):
        return torch.bmm(left, right)
    retained = CAPTURE_CONSTANTS.get()
    plans = BATCHED_PLANS if retained is None else retained
    key = (
        "oam_batched_matrix",
        left.device,
        torch.cuda.current_stream(left.device).cuda_stream,
        tuple(left.shape),
        tuple(right.shape),
        left.stride(),
        right.stride(),
    )
    if key not in plans:
        if torch.cuda.is_current_stream_capturing():
            return torch.bmm(left, right)
        plans[key] = calibrate_batched(left, right)
    plan = plans[key]
    return (
        torch.bmm(left, right) if plan is None else plan.run(left, right, plan.choice)
    )


@lt_radial_bmm.register_fake
def fake_lt_radial_bmm(left, right):
    return left.new_empty((left.shape[0], left.shape[1], right.shape[2]))


def eligible(left, right):
    """Limit calibration to the measured long-reduction radial shape family."""
    return (
        left.dtype == right.dtype == torch.float32
        and left.ndim == right.ndim == 2
        and left.shape[0] == 128
        and right.shape[1] == 5376
        and left.shape[1] == right.shape[0]
        and left.shape[1] >= 8192
        and left.stride() == (1, 128)
        and right.stride() == (5376, 1)
        and left.data_ptr() % 256 == 0
        and right.data_ptr() % 256 == 0
        and torch.backends.cuda.matmul.fp32_precision == "tf32"
    )


def calibrate(left, right):
    """Choose among valid algorithms, checking their current live outputs."""
    target = int(os.environ.get("EQX_XL_LT_SM_TARGET", "0"))
    requested = int(os.environ.get("EQX_XL_LT_ALGORITHM", "-1"))
    plan = MatrixPlan(
        left, right, workspace_bytes=32 * 1024 * 1024, candidates=8, sm_count=target
    )
    expected = torch.mm(left, right)
    output = torch.empty_like(expected)

    def timing(call):
        call()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(8):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 8

    baseline = timing(lambda: torch.mm(left, right, out=output))
    choices = []
    for index in range(plan.count):
        if plan.algorithms[index].status:
            continue
        try:
            actual = plan.run(left, right, index, output)
            error = (
                (actual.double() - expected.double()).norm()
                / expected.double().norm().clamp_min(1e-30)
            ).item()
            if not error <= 1e-4:
                continue
            elapsed = timing(lambda: plan.run(left, right, index, output))
            choices.append((elapsed, index, error))
        except RuntimeError:
            continue
    plan.choice = (
        min(choices)[1] if choices and min(choices)[0] < baseline * 0.97 else -1
    )
    if requested >= 0:
        if requested not in {index for _, index, _ in choices}:
            raise RuntimeError(
                "The requested overlapped GEMM algorithm failed live validation"
            )
        plan.choice = requested
    plan.measurements = dict(
        reference_ms=baseline,
        candidates=choices,
        selected=plan.choice,
        sm_target=target,
        requested_algorithm=requested,
    )
    return plan


def selected_mm(left, right):
    """Reuse only algorithm plans; every call reads current operand values."""
    from equivariantx.ops.common.cuda.graph import CAPTURE_CONSTANTS

    from .precision import require_tensors

    require_tensors((left, right))
    if os.environ.get("EQX_XL_LT_RADIAL", "0") != "1" or not eligible(left, right):
        return torch.mm(left, right)
    stream = torch.cuda.current_stream(left.device)
    retained = CAPTURE_CONSTANTS.get()
    plans = PLANS if retained is None else retained
    # Separate workspaces for independent streams; repeated calls on a stream
    # are ordered. Including stream avoids concurrent scratchpad reuse.
    key = (
        "oam_matrix",
        left.device.index,
        stream.cuda_stream,
        tuple(left.shape),
        tuple(right.shape),
        int(os.environ.get("EQX_XL_LT_SM_TARGET", "0")),
        int(os.environ.get("EQX_XL_LT_ALGORITHM", "-1")),
    )
    plan = plans.get(key)
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up the live radial GEMM plan on this stream before capture"
            )
        plan = calibrate(left, right)
        plans[key] = plan
        manifest = os.environ.get("EQX_XL_LT_MANIFEST")
        if manifest:
            Path(manifest).write_text(
                json.dumps(
                    [
                        dict(key=list(key), **value.measurements)
                        for key, value in plans.items()
                        if isinstance(key, tuple) and key[0] == "oam_matrix"
                    ],
                    indent=2,
                )
                + "\n"
            )
    return (
        torch.mm(left, right) if plan.choice < 0 else plan.run(left, right, plan.choice)
    )


@torch.library.custom_op(
    "eqx_oam_xl::lt_radial_mm", mutates_args=(), device_types="cuda"
)
def lt_radial_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Execute an already differentiated matrix product under its TF32 policy."""
    return selected_mm(left, right)


@lt_radial_mm.register_fake
def fake_lt_radial_mm(left, right):
    return left.new_empty((left.shape[0], right.shape[1]))


def select_radial_matrices(graph):
    """Expose only the measured static output dimensions to the custom plan."""
    from .parallel import hint
    from .program import refresh_inputs

    count = 0
    for node in graph.nodes:
        if (
            node.target not in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default)
            or node.kwargs
        ):
            continue
        a, b = (item.meta.get("val") for item in node.args)
        if node.target == torch.ops.aten.bmm.default:
            if (
                isinstance(a, torch.Tensor)
                and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype == torch.float32
                and a.device.type == "cuda"
                and a.ndim == b.ndim == 3
                and isinstance(a.shape[0], int)
                and 1 < a.shape[0] <= 32
                and isinstance(a.shape[2], int)
                and a.shape[2] == 128
                and isinstance(b.shape[2], int)
                and b.shape[2] >= 128
                and hint(a.shape[1]) >= 4096
                and a.stride(-1) == b.stride(-1) == 1
            ):
                node.target = torch.ops.eqx_oam_xl.lt_radial_bmm.default
                refresh_inputs(node)
                count += 1
            continue
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype == torch.float32
            and a.ndim == b.ndim == 2
            and hint(a.shape[0]) == 128
            and hint(b.shape[1]) == 5376
            and hint(a.shape[1]) >= 8192
        ):
            node.target = torch.ops.eqx_oam_xl.lt_radial_mm.default
            refresh_inputs(node)
            count += 1
    graph.lint()
    return count


def warm_capacity(edges, device):
    """Calibrate new matrix sizes without retaining an eager whole-model tape.

    Only streams that already executed an eligible radial product participate.
    Private seeded probe operands are discarded; training RNG is untouched.
    """
    if os.environ.get("EQX_XL_LT_RADIAL", "0") != "1" or edges < 8192:
        return
    from .parallel import matrix_stream

    parent = torch.cuda.current_stream(device)
    streams = (parent, matrix_stream(device, parent.cuda_stream))
    for stream in streams:
        previous = [
            key for key in PLANS if key[:2] == (device.index, stream.cuda_stream)
        ]
        key = (
            device.index,
            stream.cuda_stream,
            (128, edges),
            (edges, 5376),
            int(os.environ.get("EQX_XL_LT_SM_TARGET", "0")),
            int(os.environ.get("EQX_XL_LT_ALGORITHM", "-1")),
        )
        if not previous or key in PLANS:
            continue
        stream.wait_stream(parent)
        with torch.cuda.stream(stream):
            generator = torch.Generator(device=device).manual_seed(829)
            left = (
                torch.empty(edges, 128, device=device)
                .normal_(std=0.05, generator=generator)
                .T
            )
            right = torch.empty(edges, 5376, device=device).normal_(
                std=0.05, generator=generator
            )
            selected_mm(left, right)
        parent.wait_stream(stream)
