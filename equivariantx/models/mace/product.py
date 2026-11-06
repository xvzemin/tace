"""MACE symmetric-contraction conversion."""

import torch

from ...ace import SymmetricContraction as PolynomialContraction


class SymmetricContraction(torch.nn.Module):
    """Preserve MACE coefficients while replacing their polynomial evaluation.

    Parameters
    ----------
    module : torch.nn.Module
        MACE or CuEquivariance symmetric contraction.
    backend : {"cuda", "torch"}, optional
        Execution backend.
    layout : {"mul_ir", "ir_mul"}, optional
        Output layout used by the following MACE Linear.
    """

    def __init__(self, module, backend="cuda", layout="mul_ir"):
        super().__init__()
        self.indexed = not hasattr(module, "contractions")
        self.irreps_in = module.irreps_in
        self.irreps_out = module.irreps_out
        if not self.indexed:
            self.contractions = torch.nn.ModuleList()
            tensors = []
            for output, original in enumerate(module.contractions):
                coefficients = torch.nn.Module()
                coefficients.register_parameter("weights_max", original.weights_max)
                coefficients.weights = torch.nn.ParameterList(original.weights)
                self.contractions.append(coefficients)
                for degree in range(original.correlation, 0, -1):
                    tensors.append((output, degree, original.U_tensors(degree)))
            self.contraction = PolynomialContraction(
                str(module.irreps_in), str(module.irreps_out), tensors, backend, layout
            )
        else:
            self.weight = module.weight
            self.register_buffer("projection", module.projection)
            entries = []
            polynomial = module.etp.polynomial
            for operation, descriptor in polynomial.operations:
                if operation.buffers != (0, *([1] * (len(operation.buffers) - 2)), 2):
                    raise ValueError("Unsupported symmetric-contraction operands.")
                for path in descriptor.paths:
                    weight, *indices, output = path.indices
                    entries.append(
                        (tuple(indices), weight, output, float(path.coefficients))
                    )
            self.contraction = PolynomialContraction.from_entries(
                str(module.irreps_in),
                str(module.irreps_out),
                entries,
                polynomial.inputs[0].num_segments,
                backend,
                layout,
            )
        self.num_channels = self.contraction.num_channels
        self.train(module.training)

    def forward(self, features, node_type):
        if self.indexed:
            weight = self.weight
            if self.projection is not None:
                weight = torch.einsum("zau,ab->zbu", weight, self.projection)
            features = features.reshape(
                features.shape[0], self.contraction.input_dim, self.num_channels
            ).transpose(1, 2)
        else:
            weight = torch.cat(
                [
                    weight
                    for part in self.contractions
                    for weight in (part.weights_max, *part.weights)
                ],
                dim=1,
            )
            node_type = node_type.argmax(dim=-1)
        return self.contraction(features, weight, node_type)
