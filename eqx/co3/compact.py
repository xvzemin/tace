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
        self.output_layout = tuple((mul, ir.dim) for mul, ir in self.irreps_out)
        groups = [[] for _ in self.irreps_out]
        offset = 0
        for ins in linear.instructions:
            if ins.i_in < 0:
                raise ValueError("Pass coefficient biases separately.")
            mul, ir = self.irreps_in[ins.i_in]
            cout = self.irreps_out[ins.i_out].mul
            if not mul or not cout:
                continue
            groups[ins.i_out].append(
                (
                    self.irreps_in.slices()[ins.i_in],
                    slice(offset, offset + mul * cout),
                    mul,
                    cout,
                    ir.dim,
                    ins.path_weight,
                )
            )
            offset += mul * cout
        self.groups = tuple(tuple(group) for group in groups)
        self.input_sections = []
        self.weight_sections = []
        self.scales = []
        for group in self.groups:
            for sections, position in (
                (self.input_sections, 0),
                (self.weight_sections, 1),
            ):
                slices = [path[position] for path in group]
                sections.append(
                    slice(slices[0].start, slices[-1].stop)
                    if slices
                    and all(a.stop == b.start for a, b in zip(slices, slices[1:]))
                    else None
                )
            scales = {path[-1] for path in group}
            self.scales.append(next(iter(scales)) if len(scales) == 1 else None)

    def forward(self, features, weights, node_type=None):
        """Mix channels with external weights.

        Parameters
        ----------
        features : torch.Tensor
            Independent Cartesian components in flattened ``mul_ir`` order.
        weights : torch.Tensor
            Shared or batched path weights. With ``node_type``, the shape is
            ``(elements, weight_numel)``.
        node_type : torch.Tensor, optional
            Element indices of shape ``(nodes,)`` for two-dimensional features.
        """
        outputs = []
        shape = (
            features.shape[:-1]
            if node_type is not None
            else torch.broadcast_shapes(features.shape[:-1], weights.shape[:-1])
        )
        zero = features[..., :0].sum() + weights[..., :0].sum()
        for i, ((co, dim), group) in enumerate(zip(self.output_layout, self.groups)):
            if not group:
                outputs.append(features.new_zeros((*shape, co * dim)) + zero)
                continue
            ci = sum(path[2] for path in group)
            section = self.input_sections[i]
            x = (
                features[..., section].reshape(*features.shape[:-1], ci, dim)
                if section is not None
                else torch.cat(
                    [
                        features[..., sx].reshape(*features.shape[:-1], c, dim)
                        for sx, _, c, _, _, _ in group
                    ],
                    dim=-2,
                )
            )
            section, scale = self.weight_sections[i], self.scales[i]
            w = (
                weights[..., section].reshape(*weights.shape[:-1], ci, co)
                if section is not None and scale is not None
                else torch.cat(
                    [
                        weights[..., sw].reshape(*weights.shape[:-1], c, co) * s
                        for _, sw, c, _, _, s in group
                    ],
                    dim=-2,
                )
            )
            if node_type is not None and weights.shape[0] * dim <= min(ci, co):
                # Contract shared element matrices before selecting nodes,
                # rather than expanding each matrix over the whole batch.
                value = x.transpose(-1, -2).reshape(-1, ci) @ w.permute(
                    1, 0, 2
                ).reshape(ci, -1)
                value = value.reshape(features.shape[0], dim, weights.shape[0], co)
                index = node_type[:, None, None, None].expand(-1, dim, 1, co)
                value = value.gather(2, index).squeeze(2).transpose(-1, -2)
            else:
                if node_type is not None:
                    w = w[node_type]
                value = w.transpose(-1, -2) @ x
            if section is not None and scale is not None:
                value = value * scale
            outputs.append(value.flatten(-2))
        return (
            torch.cat(outputs, -1)
            if outputs
            else features.new_zeros((*shape, 0)) + zero
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
