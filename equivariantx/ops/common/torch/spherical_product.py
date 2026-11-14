"""Grouped spherical couplings with channel GEMMs and exact CG coefficients."""

import torch
from e3nn import o3

from .tensor_product import TensorProduct, instruction_groups


class SphericalProduct(TensorProduct):
    """Lower e3nn instructions to grouped Torch or native angular products.

    Parameters
    ----------
    irreps : sequence of e3nn.o3.Irreps
        Two input representations and the output, in ``mul_ir`` layout.
    instructions : sequence
        Normalized e3nn instructions, retaining their independent weights.
    backend : {"torch", "cuda"}, optional
        Angular execution provider. Channel reductions use matrix products.
    """

    def __init__(self, irreps, instructions, backend="torch"):
        couplings = tuple(
            tuple(
                irs[index].ir.l
                for irs, index in zip(irreps, (p.i_in1, p.i_in2, p.i_out))
            )
            for p in instructions
        )
        order, groups = instruction_groups(instructions, couplings)
        super().__init__(
            [tuple((mul, ir.dim) for mul, ir in irs) for irs in irreps],
            groups,
            self.angular,
        )
        self.backend = backend
        self.degrees = tuple(sorted(set(couplings)))
        self.coefficients = {}
        for degrees in self.degrees:
            cg = o3.wigner_3j(*degrees, dtype=torch.float64, device="cpu")
            self.register_buffer(
                "cg_" + "_".join(map(str, degrees)), cg, persistent=False
            )
            self.coefficients[degrees] = tuple(
                (i, j, k, float(cg[i, j, k])) for i, j, k in cg.nonzero().tolist()
            )
        self.scales = tuple(instructions[i].path_weight for i in order)
        self.register_buffer("path_scales", torch.tensor(self.scales), persistent=False)

    def angular(self, x, y, *degrees):
        """Contract angular coordinates while broadcasting channel pairs."""
        if self.backend == "cuda" and x.is_cuda:
            from ..channel_product import angular_product

            return angular_product(
                x, y, self.coefficients[degrees], 2 * degrees[-1] + 1
            )
        coefficient = getattr(self, "cg_" + "_".join(map(str, degrees)))
        value = (x @ coefficient.flatten(1)).unflatten(-1, coefficient.shape[1:])
        return (value * y.unsqueeze(-1)).sum(-2)

    def forward(self, x, y, weight):
        return super().forward(x, y, weight, self.path_scales)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for degrees in self.degrees:
            name = "cg_" + "_".join(map(str, degrees))
            setattr(
                self,
                name,
                o3.wigner_3j(*degrees, dtype=torch.float64, device="cpu").to(
                    getattr(self, name)
                ),
            )
        self.path_scales = torch.tensor(self.scales, dtype=torch.float64).to(
            self.path_scales
        )
        return self
