################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3


class LayoutTransform(torch.nn.Module):

    _LAYOUTS = (
        "ir_mul",
        "mul_ir",
        "flatten_ir_mul",
        "flatten_mul_ir",
    )

    def __init__(
        self,
        irreps: o3.Irreps,
        *,
        layout_in: str = "flatten_mul_ir",
        layout_out: str = "ir_mul",
    ) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        if layout_in not in self._LAYOUTS:
            raise ValueError(f"layout_in must be one of {self._LAYOUTS}.")
        if layout_out not in self._LAYOUTS:
            raise ValueError(f"layout_out must be one of {self._LAYOUTS}.")
        self.layout_in = layout_in
        self.layout_out = layout_out
        self.muls = []
        self.dims = []

        for mul, ir in self.irreps:
            self.muls.append(mul)
            self.dims.append(ir.dim)

        if not self.muls:
            raise ValueError("LayoutTransform requires at least one irrep.")
        common_multiplicity = all(mul == self.muls[0] for mul in self.muls[1:])
        if (
            layout_in in ("ir_mul", "mul_ir") or layout_out in ("ir_mul", "mul_ir")
        ) and not common_multiplicity:
            raise ValueError(
                "Explicit ir_mul and mul_ir layouts require one common multiplicity."
            )

        self.num_channels = self.muls[0] if common_multiplicity else None
        self.num_components = sum(self.dims)
        self.dim = self.irreps.dim
        self._setup_indices()

    def _setup_indices(self) -> None:
        indices = []
        offset = 0
        for mul, dim in zip(self.muls, self.dims):
            index = torch.arange(mul * dim, dtype=torch.long).reshape(mul, dim)
            indices.append(index.transpose(0, 1).reshape(-1) + offset)
            offset += mul * dim
        forward_index = torch.cat(indices)
        self.register_buffer("_forward_index", forward_index, persistent=False)
        self.register_buffer(
            "_inverse_index",
            torch.argsort(forward_index),
            persistent=False,
        )

    def _transform(
        self,
        features: torch.Tensor,
        layout_in: str,
        layout_out: str,
    ) -> torch.Tensor:
        if layout_in.startswith("flatten_"):
            if features.ndim < 1 or features.size(-1) != self.dim:
                raise ValueError(
                    f"{layout_in} input must have trailing dimension "
                    f"{self.dim}, got {tuple(features.shape)}."
                )
        else:
            expected = (
                (self.num_components, self.num_channels)
                if layout_in == "ir_mul"
                else (self.num_channels, self.num_components)
            )
            if features.ndim < 2 or tuple(features.shape[-2:]) != expected:
                raise ValueError(
                    f"{layout_in} input must have trailing dimensions "
                    f"{expected}, got {tuple(features.shape)}."
                )

        if layout_in == "flatten_ir_mul":
            flattened = features
        elif layout_in == "flatten_mul_ir":
            flattened = features.index_select(-1, self._forward_index)
        elif layout_in == "ir_mul":
            flattened = features.flatten(-2)
        else:
            flattened = features.transpose(-1, -2).reshape(
                *features.shape[:-2], self.dim
            )

        if layout_out == "flatten_ir_mul":
            return flattened
        if layout_out == "flatten_mul_ir":
            return flattened.index_select(-1, self._inverse_index)

        output = flattened.reshape(
            *flattened.shape[:-1],
            self.num_components,
            self.num_channels,
        )
        return output if layout_out == "ir_mul" else output.transpose(-1, -2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self._transform(features, self.layout_in, self.layout_out)

    def inverse(self, features: torch.Tensor) -> torch.Tensor:
        return self._transform(features, self.layout_out, self.layout_in)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.irreps.simplify()}, "
            f"{self.layout_in} -> {self.layout_out})"
        )

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        if not hasattr(self, "layout_in"):
            self.layout_in = "flatten_mul_ir"
            self.layout_out = "ir_mul"
        if not hasattr(self, "dim"):
            self.dim = self.irreps.dim
        if "_forward_index" not in self._buffers:
            self._setup_indices()
