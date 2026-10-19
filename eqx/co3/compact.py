"""Channel maps and products in independent Cartesian coordinates."""

import math

import torch

from .trace_free import trace_free_coupling


class Linear(torch.nn.Module):
    """Apply channel coefficients to independent Cartesian components.

    Parameters
    ----------
    linear : eqx.co3.Linear
        External-weight linear map without bias instructions.
    """

    def __init__(self, linear):
        super().__init__()
        self.irreps_in = linear.irreps_in.spherical()
        self.irreps_out = linear.irreps_out.spherical()
        self.output_widths = tuple(item.dim for item in self.irreps_out)
        self.paths = []
        offset = 0
        for ins in linear.instructions:
            if ins.i_in < 0:
                raise ValueError("Pass coefficient biases separately.")
            mul, ir = self.irreps_in[ins.i_in]
            cout = self.irreps_out[ins.i_out].mul
            self.paths.append(
                (
                    self.irreps_in.slices()[ins.i_in],
                    ins.i_out,
                    slice(offset, offset + mul * cout),
                    mul,
                    cout,
                    ir.dim,
                    ins.path_weight,
                )
            )
            offset += mul * cout

    def forward(self, features, weights):
        """Mix channels with externally supplied weights."""
        outputs = [None] * len(self.output_widths)
        for section, dest, sw, ci, co, dim, scale in self.paths:
            x = features[..., section].reshape(*features.shape[:-1], ci, dim)
            w = weights[..., sw].reshape(*weights.shape[:-1], ci, co)
            value = (w.transpose(-1, -2) @ x) * scale
            outputs[dest] = value if outputs[dest] is None else outputs[dest] + value
        zero = features[..., :0].sum() + weights[..., :0].sum()
        return torch.cat(
            [
                features.new_zeros((*features.shape[:-1], width)) + zero
                if value is None
                else value.flatten(-2)
                for width, value in zip(self.output_widths, outputs)
            ],
            dim=-1,
        )


class Contraction(torch.nn.Module):
    """Evaluate unweighted uuu paths without full Cartesian intermediates.

    Parameters
    ----------
    tensor_product : eqx.co3.TensorProduct
        Channel-wise paths, representation labels and normalization.
    """

    def __init__(self, tensor_product):
        super().__init__()
        self.irreps_in1 = tensor_product.irreps_in1.spherical()
        self.irreps_in2 = tensor_product.irreps_in2.spherical()
        self.irreps_out = tensor_product.irreps_out.spherical()
        self.instructions = tuple(tensor_product.instructions)
        self.groups = tensor_product.path_groups
        self.factors = tensor_product.path_factors
        self.widths_in1 = tuple(item.dim for item in self.irreps_in1)
        self.widths_in2 = tuple(item.dim for item in self.irreps_in2)
        self.layout_in1 = tuple((mul, ir.dim) for mul, ir in self.irreps_in1)
        self.layout_in2 = tuple((mul, ir.dim) for mul, ir in self.irreps_in2)
        self.layout_out = tuple((mul, ir.dim) for mul, ir in self.irreps_out)
        self.degrees = []
        for group, indices in enumerate(self.groups):
            ins = self.instructions[indices[0]]
            degrees = (
                self.irreps_in1[ins.i_in1].ir.l,
                self.irreps_in2[ins.i_in2].ir.l,
                self.irreps_out[ins.i_out].ir.l,
            )
            self.degrees.append(degrees)
            self.register_buffer(
                f"coupling_{group}",
                self.coefficients(degrees).to(torch.get_default_dtype()),
                persistent=False,
            )
            self.register_buffer(
                f"scale_{group}",
                torch.tensor([self.factors[i] for i in indices]).reshape(-1, 1, 1),
                persistent=False,
            )

    @staticmethod
    def coefficients(degrees):
        """Construct Cartesian contractions without spherical CG coefficients."""
        result = torch.zeros(
            *(2 * l + 1 for l in degrees), dtype=torch.float64, device="cpu"
        )
        k, odd = divmod(degrees[0] + degrees[1] - degrees[2], 2)
        scale = 1 / math.sqrt(3**k * (2 if odd else 1))
        for a, b, c, coefficient in trace_free_coupling(*degrees):
            result[a, b, c] = float(coefficient) * scale
        return result

    def forward(self, x, y):
        """Couple packed STF features with automatic differentiation."""
        inputs = [
            [
                block.reshape(*value.shape[:-1], mul, dim)
                for block, (mul, dim) in zip(value.split(widths, -1), layout)
            ]
            for value, layout, widths in (
                (x, self.layout_in1, self.widths_in1),
                (y, self.layout_in2, self.widths_in2),
            )
        ]
        outputs = [None] * len(self.layout_out)
        for group, indices in enumerate(self.groups):
            left = [inputs[0][self.instructions[i].i_in1] for i in indices]
            left = left[0].unsqueeze(-3) if len(left) == 1 else torch.stack(left, -3)
            right = inputs[1][self.instructions[indices[0]].i_in2]
            coefficient = getattr(self, f"coupling_{group}")
            if 0 in self.degrees[group][:2]:
                value = left * right.unsqueeze(-3)
            elif coefficient.shape[0] <= coefficient.shape[2]:
                value = (
                    left.unsqueeze(-1) * right.unsqueeze(-3).unsqueeze(-2)
                ).flatten(-2)
                value = value @ coefficient.flatten(0, 1)
            else:
                value = (left @ coefficient.flatten(1)).unflatten(
                    -1, coefficient.shape[1:]
                )
                value = (value * right.unsqueeze(-3).unsqueeze(-1)).sum(-2)
            value = value * getattr(self, f"scale_{group}")
            for i, block in zip(indices, value.unbind(-3)):
                dest = self.instructions[i].i_out
                outputs[dest] = (
                    block if outputs[dest] is None else outputs[dest] + block
                )
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
        zero = x[..., :0].sum() + y[..., :0].sum()
        return torch.cat(
            [
                x.new_zeros((*shape, mul, dim)).flatten(-2) + zero
                if block is None
                else block.flatten(-2)
                for block, (mul, dim) in zip(outputs, self.layout_out)
            ],
            dim=-1,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for group, degrees in enumerate(self.degrees):
            name = f"coupling_{group}"
            self._buffers[name] = self.coefficients(degrees).to(self._buffers[name])
            name = f"scale_{group}"
            self._buffers[name] = (
                torch.tensor(
                    [self.factors[i] for i in self.groups[group]],
                    dtype=torch.float64,
                    device="cpu",
                )
                .reshape(-1, 1, 1)
                .to(self._buffers[name])
            )
        return self
