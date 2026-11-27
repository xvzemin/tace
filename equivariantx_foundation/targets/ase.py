"""ASE deployment for calculators with a single model."""

from pathlib import Path

import torch

from .. import optimize
from ..execution.replay import ReplayContext, TensorProgramReplay
from ..registry import require_cuda


def optimize_calculator(calculator, *, family, **options):
    """Replace an ASE calculator's model and invalidate its cached results.

    Parameters
    ----------
    calculator : ase.calculators.calculator.Calculator
        Calculator with a writable ``model`` attribute.
    family : str
        Model adapter passed to :func:`equivariantx_foundation.optimize`.
    **options
        Keyword arguments for the adapter.

    Examples
    --------
    >>> from equivariantx_foundation.targets.ase import optimize_calculator
    >>> atoms.calc = optimize_calculator(calc, family="tace")  # doctest: +SKIP
    """
    calculator.model = optimize(calculator.model, family=family, **options)
    calculator.reset()
    return calculator


class _SingleShapeRunner:
    """Own exactly one single-threaded AOTI runner and at most one CUDA graph."""

    def __init__(self, runner, device, input_keys):
        self.runner, self.device = runner, device
        self.context = ReplayContext(device)
        self.positions = input_keys.index("positions")
        self.edge_index = input_keys.index("edge_index")
        self.allow_empty_edges = "direct_hessian_edge_pairs" in input_keys
        self.replay = self.signature = None

    def get_metadata(self):
        return self.runner.get_metadata()

    @torch.no_grad()
    def __call__(self, *inputs):
        self.context.check()
        if any(
            not isinstance(x, torch.Tensor)
            or x.device != self.device
            or x.layout != torch.strided
            for x in inputs
        ):
            raise ValueError(
                "All canonical ASE inputs must be strided tensors on the selected CUDA device."
            )
        if inputs[self.positions].size(0) < 2 or (
            not self.allow_empty_edges and not inputs[self.edge_index].size(1)
        ):
            raise ValueError(
                "This TACE graph package requires at least two atoms and a supported edge count; use an eager calculator for isolated atoms."
            )
        signature = tuple((x.shape, x.stride(), x.dtype) for x in inputs)
        if self.replay is None:
            self.replay = TensorProgramReplay(self.runner, inputs, context=self.context)
            self.signature = signature
        if signature == self.signature:
            return self.replay(*inputs)
        # N/E or a derived reciprocal-pair count may change even during MD.
        # Do not reuse stale edges, capture more graphs or load another runner.
        return self.runner(*inputs)


def replay_calculator(path, *, device="cuda", **calculator_options):
    """Evaluate ASE frames using one frozen model and one CUDA Graph.

    The first frame is captured. New frames with matching tensor signatures
    reuse the capture with updated inputs. Other shapes within the exported
    bounds use the same model without replay. Use the creating thread and CUDA
    stream. Capture reserves a private GPU memory pool.

    Parameters
    ----------
    path : str or pathlib.Path
        Deterministic TACE graph ``.pt2`` supporting single-structure input.
    device : str or torch.device, optional
        CUDA device for the compiled model.
    **calculator_options
        Additional arguments to ``TACEAseCalc``.

    Examples
    --------
    >>> atoms.calc = replay_calculator("potential.pt2", device="cuda")  # doctest: +SKIP
    """
    from tace.interface.ase import TACEAseCalc
    from tace.models.compile.aot import load_aotinductor

    if not isinstance(path, (str, Path)) or Path(path).suffix.lower() != ".pt2":
        raise TypeError(
            "Provide a TACE graph .pt2 path, not a loaded calculator or model."
        )
    device = torch.device(device)
    require_cuda(device)
    with torch.cuda.device(device):
        device = torch.device("cuda", torch.cuda.current_device())
        model = load_aotinductor(path, device, single_threaded=True)
        calc = TACEAseCalc(model, device=device, **calculator_options)
        model.compiled_model = _SingleShapeRunner(
            model.compiled_model, device, model.input_keys
        )
    return calc
