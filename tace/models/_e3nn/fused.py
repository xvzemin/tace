################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Union

import torch
from e3nn import o3

from tace.utils.env import acceleration_enabled
from tace.utils.torch_scatter import scatter_sum

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
        if self.use_eqt and contains_time_odd_irreps(
            irreps_in1,
            irreps_in2,
            actual_irreps_out,
        ):
            raise ValueError(
                "EQT does not support time-reversal irreps. Disable EQT and "
                "use the native e3nn tensor product."
            )

        # self.use_cue = acceleration_enabled("cue")

        if self.use_eqt:
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
        if any(instruction[3] != "uvu" for instruction in instructions):
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
        oeq_compatible = all(multiplicity == 1 for multiplicity, _ in irreps_in2)
        oeq_compatible = oeq_compatible and all(
            instruction[4] for instruction in instructions
        )
        uses_time_reversal = contains_time_odd_irreps(
            irreps_in1,
            irreps_in2,
            irreps_out,
        )
        if use_oeq and uses_time_reversal:
            raise ValueError(
                "OEQ does not support time-reversal irreps. Disable OEQ and "
                "use the native e3nn tensor product."
            )
        self.use_oeq = use_oeq and oeq_compatible

        if use_oeq and not oeq_compatible:
            logging.warning(
                "OEQ uvu tensor products require weighted instructions and "
                "multiplicity-one irreps_in2 for e3nn-compatible weight ordering. "
                "Falling back to e3nn."
            )

        if self.use_oeq:
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
        uses_time_reversal = contains_time_odd_irreps(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
        )

        enabled_time_reversal_kernels = [
            name
            for name, enabled in (("OEQ", self.use_oeq), ("CUE", self.use_cue))
            if enabled
        ]
        if uses_time_reversal and enabled_time_reversal_kernels:
            raise ValueError(
                f"{', '.join(enabled_time_reversal_kernels)} does not support "
                "time-reversal irreps. Disable the accelerated kernel and use "
                "the native e3nn tensor product."
            )
        if self.use_aoti and self.use_cue:
            logging.warning(
                "CUE and AOTI cannot be used simultaneously in Scatter Tensor Product. "
                "Falling back to AOTI with OEQ instead. "
                "If execution fails, install OpenEquivariance with: pip install openequivariance"
            )
            self.use_oeq = True
            self.use_cue = False
        else:
            pass

        if self.use_oeq:
            from ..oeq import e3nnOeqScatterTensorProduct

            self.fused_tp = e3nnOeqScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=self.instructions,
            )
        elif self.use_cue and not explicit_instructions:
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
