################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import threading
from contextlib import contextmanager
from typing import Any, Iterator, NamedTuple, Tuple, Union

import torch

AOTI_LAMMPS_GHOST_EXCHANGE = object()
_LAMMPS_MLIAP_CONTEXT = threading.local()


def _current_lammps_mliap_data() -> Any:
    data = getattr(_LAMMPS_MLIAP_CONTEXT, "data", None)
    if data is None:
        raise RuntimeError(
            "LAMMPS AOTI ghost exchange was called without active MLIAP data."
        )
    return data


@contextmanager
def use_lammps_mliap_data(data: Any) -> Iterator[None]:
    previous = getattr(_LAMMPS_MLIAP_CONTEXT, "data", None)
    _LAMMPS_MLIAP_CONTEXT.data = data
    try:
        yield
    finally:
        if previous is None:
            del _LAMMPS_MLIAP_CONTEXT.data
        else:
            _LAMMPS_MLIAP_CONTEXT.data = previous


@torch.library.custom_op("tace::lammps_forward_exchange", mutates_args=())
def _lammps_forward_exchange(node_features: torch.Tensor) -> torch.Tensor:
    lmp_data = _current_lammps_mliap_data()
    original_shape = node_features.shape
    node_features_flat = node_features.reshape(node_features.size(0), -1)
    out_flat = torch.empty_like(node_features_flat)
    lmp_data.forward_exchange(node_features_flat, out_flat, out_flat.size(-1))
    return out_flat.reshape(original_shape)


@_lammps_forward_exchange.register_fake
def _lammps_forward_exchange_fake(node_features: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(node_features)


@torch.library.custom_op("tace::lammps_reverse_exchange", mutates_args=())
def _lammps_reverse_exchange(grad_output: torch.Tensor) -> torch.Tensor:
    lmp_data = _current_lammps_mliap_data()
    original_shape = grad_output.shape
    grad_output_flat = grad_output.reshape(grad_output.size(0), -1)
    grad_input_flat = torch.empty_like(grad_output_flat)
    lmp_data.reverse_exchange(
        grad_output_flat,
        grad_input_flat,
        grad_input_flat.size(-1),
    )
    return grad_input_flat.reshape(original_shape)


@_lammps_reverse_exchange.register_fake
def _lammps_reverse_exchange_fake(grad_output: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(grad_output)


def _lammps_exchange_backward(ctx, grad_output: torch.Tensor):
    return _lammps_reverse_exchange(grad_output)


torch.library.register_autograd(
    _lammps_forward_exchange,
    _lammps_exchange_backward,
)


class GhostExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        node_features, lmp_data = args
        original_shape = node_features.shape
        node_features_flat = node_features.view(node_features.size(0), -1)
        out_flat = torch.empty_like(node_features_flat)
        lmp_data.forward_exchange(node_features_flat, out_flat, out_flat.size(-1))

        # save for backward
        ctx.original_shape = original_shape
        ctx.lmp_data = lmp_data

        return out_flat.view(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_flat = grad_output.view(grad_output.size(0), -1)
        gout_flat = torch.empty_like(grad_output_flat)
        ctx.lmp_data.reverse_exchange(grad_output_flat, gout_flat, gout_flat.size(-1))
        return gout_flat.view(ctx.original_shape), None


class e3nnGhostExchangeMixin:
    def handle_lammps(
        self,
        node_feats: torch.Tensor,
        lmp_data: Any,
        lmp_natoms: Tuple[int, int],
        layer: int,
    ) -> Union[torch.Tensor, None]:
        nlocal, nghosts = lmp_natoms
        first_layer = layer == 0
        if lmp_data is None or first_layer or torch.jit.is_scripting():
            return node_feats
        node_feats = node_feats.contiguous()
        expected_total = nlocal + nghosts
        if node_feats.shape[0] == expected_total:
            node_feats = self._exchange_lammps(node_feats, lmp_data)
            return node_feats
        pad = torch.zeros(
            (nghosts, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = torch.cat((node_feats, pad), dim=0)
        node_feats = self._exchange_lammps(node_feats, lmp_data)
        return node_feats

    @staticmethod
    def _exchange_lammps(node_feats: torch.Tensor, lmp_data: Any) -> torch.Tensor:
        if lmp_data is AOTI_LAMMPS_GHOST_EXCHANGE:
            return _lammps_forward_exchange(node_feats)
        return GhostExchange.apply(node_feats, lmp_data)

    def truncate_ghosts(
        self, tensor: Union[torch.Tensor, None], nlocal: Union[int, None] = None
    ) -> torch.Tensor:
        if tensor is None:
            return tensor
        return tensor[:nlocal] if nlocal is not None else tensor


class Graph(NamedTuple):
    lmp: bool
    lmp_data: Any
    lmp_natoms: Tuple[int, int]
    num_graphs: int
    displacement: Union[torch.Tensor, None]
    positions: torch.Tensor
    edge_vector: torch.Tensor
    edge_length: torch.Tensor
    lattice: torch.Tensor
    node_fidelity: torch.Tensor
    num_atoms_arange: torch.Tensor
