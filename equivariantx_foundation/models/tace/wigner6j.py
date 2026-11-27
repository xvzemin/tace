"""Select the provider of an existing spin--bond Wigner-6j coupling schedule."""

from functools import partial

import torch
from e3nn import o3

from equivariantx.o3 import O3TensorProductConv
from equivariantx.o3 import Wigner6jTensorProductConv as EQXWigner6j
from equivariantx.ops.common.sparse_coefficients import AngularMap, SparseMap
from equivariantx.ops.torch.o3.wigner6j.tensor_product import Schedule
from tace.models.spherical.fused import Wigner6jTensorProductConv
from tace.models.spherical.mtace import O3MagneticWigner6jInteraction
from tace.utils.env import acceleration_enabled

from .matching import custom_forward


class _SelectedWigner6j(Wigner6jTensorProductConv):
    def forward(self, x1, x2, x3, weight1, weight2, edge_index):
        # The host owns both sets of paths, recoupling coefficients and CG
        # schedules. Only its environment-based backend argument changes.
        return self.eqx_tp(
            x1, x2, x3, edge_index, weight1, weight2, backend=self.selected_backend
        )


def _select(module, *, backend):
    module.rejector.__class__ = _SelectedWigner6j
    module.rejector.selected_backend = backend
    return module


def _ordinary(module):
    if (
        type(module) is not O3MagneticWigner6jInteraction
        or type(module.rejector) not in (Wigner6jTensorProductConv, _SelectedWigner6j)
        or type(module.rejector.eqx_tp) is not EQXWigner6j
        or custom_forward(module, recursive=True)
        or "_compute_messages" in vars(module)
    ):
        return False
    op = module.rejector.eqx_tp
    # Torch and CUDA may consult different parts of the recoupling tree.
    # Do not bypass an overridden intermediate or a customized schedule.
    return (
        not op.internal_weights
        and not op.shared_weights
        and type(op.schedule) is Schedule
        and type(op.conv13) is type(op.conv132) is O3TensorProductConv
        and all(
            type(child) is O3TensorProductConv
            for name, child in op.named_children()
            if name in ("standard12", "standard123")
        )
        and all(type(m) is AngularMap for m in op.schedule.angular.values())
        and type(op.weight_product) is SparseMap
        and all(
            type(tp) is o3.TensorProduct for tp in (op.tp1, op.tp2, op.tp13, op.tp132)
        )
        and not any(
            name != "forward" and callable(getattr(type(child), name, None))
            for child in op.modules()
            for name in vars(child)
        )
    )


def _host_backend(p):
    native_cuda = p.is_cuda and acceleration_enabled("eqx", kernel="conv")
    return "cuda" if native_cuda else "torch"


def host_wigner6j_provider(module):
    """Unchanged provider of the Wigner convolution, not the rest of mTACE."""
    if not _ordinary(module) or type(module.rejector) is _SelectedWigner6j:
        return None
    p = next(module.parameters())
    if p.dtype not in (torch.float32, torch.float64):
        return None
    return "eqx_" + _host_backend(p)


def wigner6j_rule(module, *, backend, task):
    """Keep the native Wigner-6j operator and all its angular and weight inputs."""
    if not _ordinary(module):
        return None, "preserved custom or unsupported magnetic Wigner-6j execution"
    if task not in ("training", "inference") or backend not in ("cuda", "torch"):
        return None, "Wigner-6j requires a Torch or CUDA training/inference provider"
    p = next(module.parameters())
    if p.dtype not in (torch.float32, torch.float64) or (
        backend == "cuda" and (not p.is_cuda or torch.version.hip)
    ):
        return (
            None,
            "requested Wigner-6j backend requires supported device and precision",
        )
    if type(module.rejector) is _SelectedWigner6j:
        if module.rejector.selected_backend == backend:
            return None, f"already uses the selected {backend} Wigner-6j schedule"
    elif _host_backend(p) == backend:
        return (
            None,
            f"host already evaluates the Wigner-6j convolution with EQX {backend}",
        )
    return partial(
        _select, backend=backend
    ), f"select EQX {backend} for the host spin--bond Wigner-6j convolution"
