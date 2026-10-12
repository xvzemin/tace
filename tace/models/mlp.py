################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from functools import lru_cache
from math import prod, sqrt
from typing import Callable, List, Union

import torch
import torch.nn.functional as F
from e3nn import o3

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


class JnRadialMLP(torch.nn.Module):
    """Independent radial networks per edge degree, in the original TP order.

    Every network uses TACE's standard MLP with widths
    ``[num_basis_by_l[l2], *hidden, weight_numel_by_l[l2]]``. Cutoff placement
    follows ``radial_basis.apply_cutoff`` in the radial basis and interaction.
    """

    def __init__(
        self,
        tensor_product: o3.TensorProduct,
        num_basis_by_l: tuple[int, ...],
        hidden: list[int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        if not num_basis_by_l or any(
            type(count) is not int or count < 1 for count in num_basis_by_l
        ):
            raise ValueError("jn requires at least one radial function per l2.")

        self.num_basis_by_l = tuple(num_basis_by_l)
        self.hidden_widths = tuple(hidden)
        self.use_bias = bias
        self.num_layers = len(hidden) + 1
        self.weight_numel = tensor_product.weight_numel
        indices = [[] for _ in num_basis_by_l]
        offset = 0
        for instruction in tensor_product.instructions:
            if not instruction.has_weight:
                continue
            l2 = tensor_product.irreps_in2[instruction.i_in2].ir.l
            if l2 >= len(num_basis_by_l):
                raise ValueError(f"No jn radial basis supplied for l2={l2}.")
            count = prod(instruction.path_shape)
            indices[l2].extend(range(offset, offset + count))
            offset += count
        if offset != self.weight_numel:
            raise ValueError("jn weight layout does not match the tensor product.")

        self.weight_numel_by_l = tuple(len(group) for group in indices)
        self.radial_mlps = torch.nn.ModuleDict(
            {
                str(l2): MLP([num_basis_by_l[l2], *hidden, width], bias=bias, act="silu")
                for l2, width in enumerate(self.weight_numel_by_l)
                if width
            }
        )
        self.register_buffer(
            "_weight_index",
            torch.tensor(
                [i for group in indices for i in group], dtype=torch.long
            ).argsort(),
            persistent=False,
        )

    def forward(self, radial_basis: torch.Tensor) -> torch.Tensor:
        if self.weight_numel == 0:
            return radial_basis[..., :0]
        basis_by_l = radial_basis.split(self.num_basis_by_l, dim=-1)
        grouped = torch.cat(
            [mlp(basis_by_l[int(l)]) for l, mlp in self.radial_mlps.items()], dim=-1
        )
        return grouped.index_select(-1, self._weight_index)
