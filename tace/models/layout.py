################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3


class LayoutTransform2(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.muls = []
        self.dims = []

        for mul, ir in self.irreps:
            d = ir.dim
            self.muls.append(mul)
            self.dims.append(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = 0
        out = []
        batch = x.size(0)
        for mul, d in zip(self.muls, self.dims):
            field = x[:, start : start + mul * d]
            start += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        start = 0
        out = []
        batch = x.size(0)
        for _, d in zip(self.muls, self.dims):
            field = x[:, :, start : start + d]
            start += d
            field = field.reshape(batch, -1)
            out.append(field)
        return torch.cat(out, dim=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps})"

# class LayoutTransform(torch.nn.Module):
#     def __init__(self, irreps: o3.Irreps) -> None:
#         super().__init__()
#         self.irreps = o3.Irreps(irreps)
#         self.muls = []
#         self.dims = []

#         for mul, ir in self.irreps:
#             d = ir.dim
#             self.muls.append(mul)
#             self.dims.append(d)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """MulIr (flatten) to IrMul"""
#         start = 0
#         out = []
#         batch = x.size(0)
#         for mul, d in zip(self.muls, self.dims):
#             field = x[:, start : start + mul * d]
#             start += mul * d
#             field = field.reshape(batch, mul, d)
#             out.append(field)
#         return torch.cat(out, dim=-1).transpose(-1, -2).contiguous()

#     def inverse(self, x: torch.Tensor) -> torch.Tensor:
#         """IrMul to MulIr (flatten)"""
#         x = x.transpose(-1, -2).contiguous()
#         start = 0
#         out = []
#         batch = x.size(0)
#         for _, d in zip(self.muls, self.dims):
#             field = x[:, :, start : start + d]
#             start += d
#             field = field.reshape(batch, -1)
#             out.append(field)
#         return torch.cat(out, dim=-1)

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}({self.irreps})"


class LayoutTransform(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.muls = []
        self.dims = []

        for mul, ir in self.irreps:
            self.muls.append(mul)
            self.dims.append(ir.dim)

        self._setup_indices()

    def _setup_indices(self) -> None:
        if not self.muls:
            raise ValueError("LayoutTransform requires at least one irrep.")
        if any(mul != self.muls[0] for mul in self.muls[1:]):
            raise ValueError("LayoutTransform requires one common multiplicity.")

        self.num_channels = self.muls[0]
        self.num_components = sum(self.dims)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MulIr (flatten) to IrMul"""
        batch = x.size(0)
        return x.index_select(-1, self._forward_index).reshape(
            batch,
            self.num_components,
            self.num_channels,
        )

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """IrMul to MulIr (flatten)"""
        batch = x.size(0)
        return x.reshape(batch, -1).index_select(-1, self._inverse_index)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps.simplify()})"

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        if "_forward_index" not in self._buffers:
            self._setup_indices()
