"""Scalar activations and gated spatial representations."""

import math

import torch
from e3nn.nn import Gate as E3nnGate


class Gate(E3nnGate):
    """Apply normalized scalar activations and gates.

    Parameters
    ----------
    irreps_scalars : o3.Irreps
        Scalar representations passed through activations.
    act_scalars : list of callable or None
        Activation for each scalar term.
    irreps_gates : o3.Irreps
        Scalar representations producing the gates.
    act_gates : list of callable or None
        Activation for each gate term.
    irreps_gated : o3.Irreps
        Representations multiplied by the gates, one gate per multiplicity.
    backend : {"cuda", "torch"}, optional
        CUDA fuses supported activations and products. CPU inputs and
        unsupported activations use torch operations.
    """

    def __init__(
        self,
        irreps_scalars,
        act_scalars,
        irreps_gates,
        act_gates,
        irreps_gated,
        backend="cuda",
    ):
        super().__init__(
            irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
        )
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be 'cuda' or 'torch'")
        self.backend = backend
        self.input_dim = self.irreps_in.dim
        self.output_dim = self.irreps_out.dim
        self.metadata = None
        if backend == "torch" or not self.irreps_out.dim:
            return

        from ..conv.program import Program, activate

        program = Program()
        features = program.input(0, "edge", self.irreps_in.dim)
        slices = self.sc.cut.irreps_in.slices()
        extracted = [
            program.gather(
                features,
                [k for i in indices for k in range(slices[i].start, slices[i].stop)],
            )
            for indices in self.sc.cut.instructions
        ]
        activated = []
        try:
            for values, module in zip(extracted, (self.act_scalars, self.act_gates)):
                parts = []
                for (mul, _), s, act in zip(
                    module.irreps_in, module.irreps_in.slices(), module.acts
                ):
                    parts.append(
                        activate(program, program.slice(values, s.start, mul), act)
                    )
                activated.append(program.concatenate(parts))
        except NotImplementedError:
            return
        outputs = [activated[0]]
        for ins in self.mul.instructions:
            mul, ir = self.mul.irreps_in1[ins.i_in1]
            s = self.mul.irreps_in1.slices()[ins.i_in1]
            t = self.mul.irreps_in2.slices()[ins.i_in2]
            values = program.slice(extracted[2], s.start, mul * ir.dim)
            gates = program.gather(
                activated[1], [t.start + u for u in range(mul) for _ in range(ir.dim)]
            )
            outputs.append(
                program.scale(
                    program.binary("mul", values, gates),
                    ins.path_weight / math.sqrt(ir.dim),
                )
            )
        output = program.concatenate(outputs)
        self.metadata = repr((tuple(program.nodes), ((output, 0, "edge"),)))

    def forward(self, features):
        """Return gated features with shape ``(..., irreps_out.dim)``."""
        if (
            self.metadata is None
            or not features.is_cuda
            or features.dtype not in (torch.float32, torch.float64)
        ):
            return super().forward(features)
        from ..conv.edge import evaluate

        if features.shape[-1] != self.input_dim:
            raise ValueError("The last feature dimension must equal irreps_in.dim.")
        values = features.reshape(-1, self.input_dim)
        indices = torch.zeros(values.shape[0], dtype=torch.int64, device=values.device)
        output = evaluate(self.metadata, [values], indices, indices, values.shape[0])[0]
        return output.reshape(*features.shape[:-1], self.output_dim)
