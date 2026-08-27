################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from functools import lru_cache
from math import sqrt
from typing import Callable, List, Union

import torch
import torch.nn.functional as F

from .linear import mlpLinear

def get_activation_scale_factor(
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(0)
    samples = torch.randn(
        1_000_000,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    with torch.no_grad():
        second_moment = activation(samples).square().mean()
    return second_moment.rsqrt().item()


class ScaledSiLU(torch.nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.scale_factor = 1.6791767923989418

    def forward(self, inputs):
        return F.silu(inputs, inplace=self.inplace) * self.scale_factor


class ScaledSigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1.8467055342154763

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale_factor


class ScaledTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1.5937334472592695

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) * self.scale_factor


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2


ACTIVATION = {
    None: torch.nn.Identity,
    "none": torch.nn.Identity,
    "None": torch.nn.Identity,
    "null": torch.nn.Identity,
    "identity": torch.nn.Identity,
    "silu": torch.nn.SiLU,
    "scaled_silu": ScaledSiLU,
    "sigmoid": torch.nn.Sigmoid,
    "scaled_sigmoid": ScaledSigmoid,
    "tanh": torch.nn.Tanh,
    "scaled_tanh": ScaledTanh,
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "smooth_leaky_relu": SmoothLeakyReLU,
    "prelu": torch.nn.PReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "mish": torch.nn.Mish,
    "softplus": torch.nn.Softplus,
    "softsign": torch.nn.Softsign,
    "hardtanh": torch.nn.Hardtanh,
    "hardswish": torch.nn.Hardswish,
    "hardsigmoid": torch.nn.Hardsigmoid,
    "tanhshrink": torch.nn.Tanhshrink,
}


class ScaledActivation(torch.nn.Module):
    """Scale an activation to unit second moment for normal inputs."""

    def __init__(
        self,
        activation: torch.nn.Module,
        scale_factor: float,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.scale_factor = scale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.activation(input) * self.scale_factor


@lru_cache(maxsize=None)
def _activation_scale_factor_from_name(name: str) -> float:
    return get_activation_scale_factor(ACTIVATION[name]())


def get_scaled_activation(name: str) -> torch.nn.Module:
    if not isinstance(name, str):
        raise TypeError("Activation name must be a string.")
    if name not in ACTIVATION:
        raise ValueError(f"Unknown activation: {name!r}.")
    if name.startswith("scaled_"):
        return ACTIVATION[name]()

    scaled_name = f"scaled_{name}"
    if scaled_name in ACTIVATION:
        return ACTIVATION[scaled_name]()
    return ScaledActivation(
        ACTIVATION[name](),
        _activation_scale_factor_from_name(name),
    )


class MLP(torch.nn.Module):
    def __init__(
        self,
        channels: List[int],
        bias: bool = False,
        act: Union[str, torch.nn.Module, None] = "silu",
        forward_weight_init: bool = True,
        layer_norm: bool = False,
        rms_norm: bool = False,
    ):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("MLP must have at least 2 layers")

        self.num_layers = len(channels) - 1
        self.dims = channels
        self.is_nonlinear = False

        mlp = []
        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):
            if forward_weight_init:
                norm_dim = h_in
                gain = 1.0 if act is None or (layer == 0) else sqrt(2)
            else:
                norm_dim = h_out
                gain = 1.0 if act is None or (layer == self.num_layers - 1) else sqrt(2)

            linear_layer = mlpLinear(
                in_dim=h_in,
                out_dim=h_out,
                alpha=gain / sqrt(norm_dim),
                bias=bias,
            )

            mlp.append(linear_layer)

            if layer < len(self.dims) - 2:
                if layer_norm:
                    mlp.append(torch.nn.LayerNorm(h_out))
                elif rms_norm:
                    mlp.append(torch.nn.RMSNorm(h_out))

            del gain, norm_dim

            if (layer != self.num_layers - 1) and (act is not None):
                if isinstance(act, torch.nn.Module):
                    mlp.append(act)
                else:
                    mlp.append(ACTIVATION[act]())
                self.is_nonlinear = True

        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# class SphericalBesselPathMLP(torch.nn.Module):
#     """Generate each CGTP path from its matching spherical-Bessel order."""

#     def __init__(
#         self,
#         tensor_product: O3ScatterTensorProduct,
#         num_shells: int,
#         edge_feats_channel: int,
#         radial_mlp: list[int],
#         bias: bool,
#         layer_norm: bool,
#     ) -> None:
#         super().__init__()
#         self.num_shells = num_shells
#         self.num_radial_basis = (tensor_product.irreps_in2.lmax + 1) * num_shells
#         context_dim = edge_feats_channel - self.num_radial_basis

#         path_indices = [[] for _ in range(tensor_product.irreps_in2.lmax + 1)]
#         offset = 0
#         for instruction in tensor_product.tp.instructions:
#             path_numel = math.prod(instruction.path_shape)
#             degree = tensor_product.irreps_in2[instruction.i_in2].ir.l
#             path_indices[degree].extend(range(offset, offset + path_numel))
#             offset += path_numel
#         if offset != tensor_product.weight_numel:
#             raise RuntimeError("Tensor-product path weights were counted incorrectly.")

#         self.degrees = [
#             degree for degree, indices in enumerate(path_indices) if indices
#         ]
#         grouped_indices = [
#             index for degree in self.degrees for index in path_indices[degree]
#         ]
#         self.register_buffer(
#             "inverse_order",
#             torch.argsort(torch.tensor(grouped_indices, dtype=torch.long)),
#             persistent=False,
#         )
#         self.mlps = torch.nn.ModuleList(
#             MLP(
#                 [num_shells + context_dim] + radial_mlp + [len(path_indices[degree])],
#                 bias=bias,
#                 layer_norm=layer_norm,
#                 act="silu",
#             )
#             for degree in self.degrees
#         )

#     def forward(
#         self,
#         edge_radial_basis: torch.Tensor,
#         edge_feats: torch.Tensor,
#     ) -> torch.Tensor:
#         context = edge_feats[:, self.num_radial_basis :]
#         grouped_weights = []
#         for degree, mlp in zip(self.degrees, self.mlps):
#             radial = edge_radial_basis.narrow(
#                 -1,
#                 degree * self.num_shells,
#                 self.num_shells,
#             )
#             grouped_weights.append(mlp(torch.cat((radial, context), dim=-1)))
#         return torch.cat(grouped_weights, dim=-1).index_select(
#             -1,
#             self.inverse_order,
#         )