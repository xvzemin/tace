################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from e3nn import o3

from ..time_reversal import contains_time_odd_irreps


def check_cuda_toolkit() -> None:
    """Check the platform and CUDA toolkit required by LunaEquivariance."""
    if sys.platform != "linux":
        raise RuntimeError("TACE_USE_LUNA requires Linux and an NVIDIA GPU.")
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None or torch.version.cuda is None:
        raise RuntimeError(
            "TACE_USE_LUNA requires CUDA-enabled PyTorch and a CUDA toolkit."
        )
    nvcc = os.environ.get("PYTORCH_NVCC", str(Path(CUDA_HOME) / "bin" / "nvcc"))
    version = re.search(
        r"release (\d+\.\d+)",
        subprocess.check_output([nvcc, "--version"], text=True),
    )
    if version and version[1].split(".")[0] != torch.version.cuda.split(".")[0]:
        raise RuntimeError(
            f"Luna CUDA toolkit {version[1]} does not match PyTorch CUDA "
            f"{torch.version.cuda}. Set CUDA_HOME to a matching toolkit."
        )


@torch.library.custom_op("tace::luna_graph", mutates_args=(), device_types="cuda")
def luna_graph(
    source: torch.Tensor, target: torch.Tensor, num_nodes: int, index_bits: int
) -> list[torch.Tensor]:
    """Build destination-sorted graph tables for a Luna convolution."""
    from LunaEquivariance import make_graph
    from LunaEquivariance.lunatp.graph import get_graph_with_int64_offsets

    graph = make_graph(source, target, num_nodes)
    if index_bits == 64:
        graph = get_graph_with_int64_offsets(graph)
    segments = graph.source_segments
    return [
        graph.src,
        graph.dst,
        graph.dst_rowptr,
        graph.src_rowptr,
        graph.edge_ids_by_src,
        graph.source_node_order,
        graph.node_order,
        segments.nodes,
        segments.begin,
        segments.end,
        segments.rowptr,
        segments.order,
        graph.node_tiers,
    ]


@luna_graph.register_fake
def luna_graph_fake(
    source: torch.Tensor, target: torch.Tensor, num_nodes: int, index_bits: int
) -> list[torch.Tensor]:
    num_edges = source.size(0)
    num_segments = torch.library.get_ctx().new_dynamic_size()
    offset_dtype = torch.int64 if index_bits == 64 else torch.int32
    sizes = (
        num_edges,
        num_edges,
        num_nodes + 1,
        num_nodes + 1,
        num_edges,
        num_nodes,
        num_nodes,
        num_segments,
        num_segments,
        num_segments,
        num_nodes + 1,
        num_segments,
        3,
    )
    return [
        source.new_empty(
            (size,), dtype=offset_dtype if i in (2, 3, 4, 8, 9, 10, 11) else torch.int32
        )
        for i, size in enumerate(sizes)
    ]


def register_luna_ops(config: dict) -> None:
    """Restore convolution and derivative operators from an exported configuration."""
    from LunaEquivariance import TensorProductConv

    check_cuda_toolkit()
    config = dict(config)
    dtype = getattr(torch, config.pop("dtype"))
    config["instructions"] = [tuple(path) for path in config["instructions"]]
    module = TensorProductConv(**config, dtype=dtype)
    module.op.register_ops()
    module.op.prepare_indexing()
    module.op.index64_operator.register_ops()


class LunaTensorProductConv(torch.nn.Module):
    """Sum externally weighted tensor products at destination nodes.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : e3nn.o3.Irreps
        Node, edge and output representations in ``mul_ir`` layout.
    instructions : list of tuple
        Weighted ``uvu`` tensor-product paths.
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: list[tuple],
    ) -> None:
        super().__init__()
        if contains_time_odd_irreps(irreps_in1, irreps_in2, irreps_out):
            raise ValueError(
                "LunaEquivariance does not support time-reversal-odd irreps."
            )
        try:
            from LunaEquivariance import TensorProductConv
        except ImportError as exc:
            raise ImportError(
                "TACE_USE_LUNA=1 requires LunaEquivariance; install tace[luna]."
            ) from exc

        check_cuda_toolkit()
        self.tp = TensorProductConv(
            [(mul, (ir.l, ir.p)) for mul, ir in irreps_in1],
            [(mul, (ir.l, ir.p)) for mul, ir in irreps_in2],
            [(mul, (ir.l, ir.p)) for mul, ir in irreps_out],
            instructions,
            internal_weights=False,
            shared_weights=False,
            dtype=torch.get_default_dtype(),
        )
        # The empty external-weight buffer must not change checkpoint keys.
        self.tp.tp.register_buffer("weight", self.tp.weight, persistent=False)
        self.aoti_config = {
            "irreps_in1": str(self.tp.irreps_in1),
            "irreps_in2": str(self.tp.irreps_in2),
            "irreps_out": str(self.tp.irreps_out),
            "instructions": instructions,
            "internal_weights": False,
            "shared_weights": False,
        }
        self.register_ops()
        self.irreps_out = irreps_out
        self.weight_numel = self.tp.weight_numel

    def register_ops(self) -> None:
        """Register convolution operators for the current precision."""
        self.operator_name = self.tp.op.register_ops()
        self.tp.op.prepare_indexing()
        self.operator_name64 = self.tp.op.index64_operator.register_ops()
        self.aoti_config["dtype"] = str(self.tp.op.compute_dtype).removeprefix("torch.")

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.register_ops()
        return self

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if not node_feats.is_cuda:
            raise RuntimeError("TACE_USE_LUNA requires CUDA tensors.")
        order = torch.argsort(edge_index[1], stable=True)
        source, target = edge_index[:, order]
        wide = (
            node_feats.size(0) > self.tp.op.index_node_limit
            or source.size(0) > self.tp.op.index_edge_limit
        )
        tables = luna_graph(source, target, node_feats.size(0), 64 if wide else 32)
        name = self.operator_name64 if wide else self.operator_name
        output, _ = getattr(torch.ops.lunatp, name)(
            node_feats.contiguous(),
            edge_attrs[order].contiguous(),
            conv_weights[order].contiguous(),
            None,
            tables,
            node_feats.size(0),
            False,
        )
        return output
