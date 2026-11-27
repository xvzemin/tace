################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import threading
from contextlib import contextmanager
from math import prod
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


def exchange_rows(features, exchange):
    """Exchange packed atom rows with a Kokkos ML-IAP callback."""
    width = prod(features.shape[1:])
    rows = features.contiguous().resolve_neg().view(features.shape[0], width)
    result = features.new_empty(features.shape)
    exchange(rows, result.view(features.shape[0], width), width)
    return result


@torch.library.custom_op("tace::lammps_forward_exchange", mutates_args=())
def _lammps_forward_exchange(node_features: torch.Tensor) -> torch.Tensor:
    return exchange_rows(node_features, _current_lammps_mliap_data().forward_exchange)


@_lammps_forward_exchange.register_fake
def _lammps_forward_exchange_fake(node_features: torch.Tensor) -> torch.Tensor:
    return node_features.new_empty(node_features.shape)


@torch.library.custom_op("tace::lammps_reverse_exchange", mutates_args=())
def _lammps_reverse_exchange(grad_output: torch.Tensor) -> torch.Tensor:
    return exchange_rows(grad_output, _current_lammps_mliap_data().reverse_exchange)


@_lammps_reverse_exchange.register_fake
def _lammps_reverse_exchange_fake(grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output.new_empty(grad_output.shape)


def _lammps_exchange_backward(ctx, grad_output: torch.Tensor):
    return _lammps_reverse_exchange(grad_output)


torch.library.register_autograd(
    _lammps_forward_exchange,
    _lammps_exchange_backward,
)


class GhostExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, node_features, lmp_data):
        ctx.lmp_data = lmp_data
        return exchange_rows(node_features, lmp_data.forward_exchange)

    @staticmethod
    def backward(ctx, grad_output):
        return exchange_rows(grad_output, ctx.lmp_data.reverse_exchange), None


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
        # Release the zero-filled ghost staging rows before exchange allocates
        # its distinct full-size Kokkos output.
        node_feats = torch.cat(
            (node_feats, node_feats.new_zeros((nghosts, node_feats.shape[1]))), dim=0
        )
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
    node_type: Union[torch.Tensor, None] = None
