################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import math
from typing import Union

import torch
from e3nn import o3

from eqx.conv import Convolution
from eqx.o2 import O3TensorProduct
from tace.utils.env import acceleration_enabled
from tace.utils.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..time_reversal import contains_time_odd_irreps
from .paths import generate_paths


class uuuTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: Union[str, None] = None,
        l2l3: Union[str, None] = None,
        l3l1: Union[str, None] = None,
        trainable: bool = False,
        identical_inputs: bool = False,
        warning: bool = False,
        use_fused: bool = False,
    ) -> None:
        super().__init__()

        instructions, actual_irreps_out = generate_paths(
            irreps_out=irreps_out,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            l1l2=l1l2,
            l2l3=l2l3,
            l3l1=l3l1,
            e3nn_mode="uuu",
            trainable=trainable,
            identical_inputs=identical_inputs,
        )

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            actual_irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = actual_irreps_out
        self.instructions = instructions
        self.weight_numel = self.tp.weight_numel

        use_eqt = acceleration_enabled("eqt")
        self.use_eqt = use_fused if use_eqt is None else use_eqt

        # self.use_cue = acceleration_enabled("cue")

        if self.use_eqt:
            if contains_time_odd_irreps(
                irreps_in1,
                irreps_in2,
                actual_irreps_out,
            ):
                raise ValueError(
                    "EQT does not support time-reversal irreps. Disable EQT and "
                    "use the native e3nn tensor product."
                )
            from ..eqt import e3nnEqtTensorProduct

            self.fused_tp = e3nnEqtTensorProduct(
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                irreps_out=actual_irreps_out,
                num_channel=irreps_in2.count("1o"),
                path=instructions,
                trainable=trainable,
            )
        # elif self.use_cue and not trainable:
        #     from ..cue import e3nnCueTensorProduct
        #     self.fused_tp = e3nnCueTensorProduct(
        #         irreps_in1=irreps_in1,
        #         irreps_in2=irreps_in2,
        #         irreps_out=irreps_out,
        #         l1l2=l1l2,
        #         l2l3=l2l3,
        #         l3l1=l3l1,
        #         trainable=trainable,
        #     )
        elif warning:
            logging.warning(
                "Correlation >= 3 is running without Equitorch. "
                "For acceleration options, see "
                "https://tace.readthedocs.io/en/latest/guide/acceleration.html"
            )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, ws: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if hasattr(self, "fused_tp"):
            return self.fused_tp(x, y, ws)
        return self.tp(x, y, ws)


class uvuTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: list[tuple],
        *,
        shared_weights: bool,
    ) -> None:
        super().__init__()

        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)
        if any(ins[3] != "uvu" for ins in instructions):
            raise ValueError("uvuTensorProduct only accepts uvu instructions")

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights=shared_weights,
            internal_weights=False,
        )

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.instructions = self.tp.instructions
        self.weight_numel = self.tp.weight_numel
        self.shared_weights = shared_weights
        use_oeq = acceleration_enabled("oeq")
        oeq_compatible = all(mul == 1 for mul, _ in irreps_in2)
        oeq_compatible = oeq_compatible and all(
            ins[4] for ins in instructions
        )
        self.use_oeq = use_oeq and oeq_compatible

        if use_oeq and not oeq_compatible:
            logging.warning(
                "OEQ uvu tensor products require weighted instructions and "
                "multiplicity-one irreps_in2 for e3nn-compatible weight ordering. "
                "Falling back to e3nn."
            )

        if self.use_oeq:
            if contains_time_odd_irreps(
                irreps_in1,
                irreps_in2,
                irreps_out,
            ):
                raise ValueError(
                    "OEQ does not support time-reversal irreps. Disable OEQ and "
                    "use the native e3nn tensor product."
                )
            from ..oeq import e3nnOeqTensorProduct

            self.fused_tp = e3nnOeqTensorProduct(
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                irreps_out=irreps_out,
                instructions=instructions,
                shared_weights=shared_weights,
            )
            if self.fused_tp.weight_numel != self.weight_numel:
                raise RuntimeError(
                    "OEQ and e3nn generated different uvu tensor-product paths: "
                    f"{self.fused_tp.weight_numel} != {self.weight_numel}."
                )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, "fused_tp"):
            return self.fused_tp(x, y, weights)
        return self.tp(x, y, weights)


class O3ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: Union[str, None] = None,
        l2l3: Union[str, None] = None,
        l3l1: Union[str, None] = None,
        instructions: Union[list[tuple], None] = None,
    ) -> None:
        super().__init__()

        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        explicit_instructions = instructions is not None
        if instructions is None:
            instructions, actual_irreps_out = generate_paths(
                irreps_out=irreps_out,
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                l1l2=l1l2,
                l2l3=l2l3,
                l3l1=l3l1,
                e3nn_mode="uvu",
            )
        else:
            actual_irreps_out = irreps_out

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            actual_irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = actual_irreps_out
        self.instructions = instructions
        self.weight_numel = self.tp.weight_numel

        self.use_oeq = acceleration_enabled("oeq")
        self.use_cue = acceleration_enabled("cue")
        self.use_aoti = acceleration_enabled("compile")
        if self.use_aoti and self.use_cue:
            logging.warning(
                "CUE and AOTI cannot be used simultaneously in Scatter Tensor Product. "
                "Falling back to AOTI with OEQ instead. "
                "If execution fails, install OpenEquivariance with: pip install openequivariance"
            )
            self.use_oeq = True
            self.use_cue = False

        if self.use_oeq:
            if contains_time_odd_irreps(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
            ):
                raise ValueError(
                    "OEQ does not support time-reversal irreps. Disable OEQ and "
                    "use the native e3nn tensor product."
                )
            from ..oeq import e3nnOeqScatterTensorProduct

            self.fused_tp = e3nnOeqScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=self.instructions,
            )
        elif self.use_cue and not explicit_instructions:
            if contains_time_odd_irreps(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
            ):
                raise ValueError(
                    "CUE does not support time-reversal irreps. Disable CUE and "
                    "use the native e3nn tensor product."
                )
            from ..cue import e3nnCueScatterTensorProduct

            self.fused_tp = e3nnCueScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                l1l2=l1l2,
                l2l3=l2l3,
                l3l1=l3l1,
            )
        elif self.use_cue:
            logging.warning(
                "CUE scatter tensor products do not support explicit instructions. "
                "Falling back to e3nn for this tensor product."
            )
            self.use_cue = False

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        if hasattr(self, "fused_tp"):
            return self.fused_tp(x, y, w, edge_index)
        return scatter_sum(
            self.tp(x[edge_index[0]], y, w), edge_index[1], dim=0, dim_size=x.size(0)
        )


class O2CgtpScatterTensorProduct(torch.nn.Module):
    """Evaluate the same CGTP paths in an aligned frame and sum at target nodes."""

    def __init__(self, irreps_in1, irreps_in2, irreps_out, *, l1l2=None):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.instructions, self.irreps_out = generate_paths(
            o3.Irreps(irreps_out),
            self.irreps_in1,
            self.irreps_in2,
            l1l2=l1l2,
            e3nn_mode="uvu",
        )
        self.tp = O3TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.eqx_tp = Convolution(self.tp, backend="triton")
        self.weight_numel = self.tp.weight_numel
        self.reshape_in = LayoutTransform(
            self.irreps_in1,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.reshape_out = LayoutTransform(
            self.irreps_out.simplify(),
            layout_in="flatten_ir_mul",
            layout_out="flatten_mul_ir",
        )
        self.reshape_streamed = LayoutTransform(
            self.irreps_out,
            layout_in="flatten_ir_mul",
            layout_out="flatten_mul_ir",
        )
        self.register_buffer(
            "harmonic_degrees",
            torch.tensor([ir.l for _, ir in self.irreps_in2]),
            persistent=False,
        )

    def forward(self, node_feats, conv_weights, edge_index, wigner, wigner_inv, graph):
        # Preserve the magnitude and derivatives of the reference harmonic input.
        harmonic_scale = (
            graph.edge_vector.norm(dim=-1, keepdim=True) / graph.edge_length
        ).pow(self.harmonic_degrees)
        node_feats = self.reshape_in(node_feats)
        message = self.tp.local_frame_in.to_local(node_feats[edge_index[0]], wigner)
        message = self.tp.forward_local(message, conv_weights, harmonic_scale)
        message = self.tp.local_frame_out.to_global(message, wigner_inv)
        message = scatter_sum(
            message, edge_index[1], dim=0, dim_size=node_feats.size(0)
        )
        return self.reshape_out(message)

    def forward_stream(
        self, node_feats, radial, projection, edge_index, wigner, edge_cutoff, graph
    ):
        """Evaluate the radial projection and fused angular convolution."""
        if wigner.ndim == 3:
            # Other local interactions may still require the shared dense
            # frame. Extract its full degree blocks for the streamed CGTP.
            lmax = math.isqrt(wigner.size(-1)) - 1
            blocks = []
            for l in range(self.tp.lmax + 1):
                rows = [
                    l
                    if m == 0
                    else l + (2 * m - 1) * (lmax + 1) - m * m
                    if m > 0
                    else l + 2 * (-m) * (lmax + 1) - (-m) * (-m + 1)
                    for m in range(-l, l + 1)
                ]
                blocks.append(wigner[:, rows, l * l : (l + 1) ** 2].flatten(1))
            wigner = torch.cat(blocks, dim=1)
        harmonic_scale = (
            graph.edge_vector.norm(dim=-1, keepdim=True) / graph.edge_length
        ).pow(self.harmonic_degrees)
        if edge_cutoff is not None:
            harmonic_scale = harmonic_scale * edge_cutoff
        message = self.eqx_tp(
            self.reshape_in(node_feats),
            radial,
            projection,
            wigner,
            harmonic_scale,
            edge_index,
            node_feats.size(0),
        )
        return self.reshape_streamed(message)
