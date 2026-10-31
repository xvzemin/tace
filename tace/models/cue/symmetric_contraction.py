"""Symmetric contractions with shared coupling paths."""

import torch


class CueSymmetricContraction(torch.nn.Module):
    """Evaluate a symmetric polynomial using CuEquivariance.

    Parameters
    ----------
    module : eqx.ace.SymmetricContraction
        Contraction supplying angular coefficients and weight ordering.
    """

    def __init__(self, module):
        super().__init__()
        import cuequivariance as cue
        import cuequivariance_torch as cuet

        self.num_channels = module.num_channels
        self.output_dims = module.output_dims
        self.output_dim = module.output_dim
        operations = []
        for degree in range(1, module.correlation + 1):
            indices = getattr(module, f"indices_{degree}").tolist()
            if not indices:
                continue
            descriptor = cue.SegmentedTensorProduct.from_subscripts(
                ",".join(["u"] * (degree + 2))
            )
            sizes = (module.num_paths, *([module.input_dim] * degree), module.output_dim)
            for operand, size in enumerate(sizes):
                descriptor.add_segments(operand, [(self.num_channels,)] * size)
            weights = getattr(module, f"weights_{degree}").tolist()
            outputs = getattr(module, f"outputs_{degree}").tolist()
            coefficients = getattr(module, f"coefficients_{degree}").tolist()
            for index, weight, output, coefficient in zip(
                indices, weights, outputs, coefficients
            ):
                descriptor.add_path(weight, *index, output, c=coefficient)
            operations.append(((0, *([1] * degree), 2), descriptor))
        descriptor = operations[0][1]
        polynomial = cue.SegmentedPolynomial(
            [descriptor.operands[0], descriptor.operands[1]],
            [descriptor.operands[-1]],
            operations,
        )
        self.contraction = cuet.SegmentedPolynomial(polynomial, method="uniform_1d")

    def forward(
        self, features: torch.Tensor, weight: torch.Tensor, node_type: torch.Tensor
    ) -> torch.Tensor:
        output = self.contraction(
            [weight.flatten(1), features.transpose(1, 2).flatten(1)],
            input_indices={0: node_type},
        )[0].reshape(features.shape[0], self.output_dim, self.num_channels)
        return torch.cat(
            [part.transpose(1, 2).flatten(1) for part in output.split(self.output_dims, 1)],
            dim=1,
        )
