################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from functools import lru_cache
from math import sqrt
from typing import Callable, List, Union

import torch
import torch.nn.functional as F

from equivariantx.nn import GroupedRadialMLP
from tace.utils.env import memory_first

from .linear import IndexedFeatures, mlpLinear, radial_features


def convolution_radial(network, features):
    """Prepare radial features and the layers deferred to an EQX convolution.

    Parameters
    ----------
    network : MLPV1 or MLPV2
        Radial network ending in an affine projection.
    features : torch.Tensor or IndexedFeatures
        Dense edge features or indexed edge, target and source partitions.

    Returns
    -------
    radial : torch.Tensor or tuple
        Original input partitions by default, or hidden features when
        ``TACE_MEMORY_FIRST=0``.
    layers : torch.nn.Sequential
        Final affine layer, or the complete radial MLP when streaming it.

    Examples
    --------
    >>> network = MLPV2([4, 8, 16])
    >>> radial, layers = convolution_radial(network, torch.randn(3, 4))
    >>> layers(radial).shape
    torch.Size([3, 16])
    """
    if memory_first():
        return radial_features(features), network.mlp
    if len(network.mlp) > 1:
        features = network.mlp[:-1](features)
    elif isinstance(features, tuple):
        features = torch.cat(
            [value if index is None else value[index] for value, index in features],
            dim=-1,
        )
    return features, network.mlp[-1:]


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


class MLPV1(torch.nn.Module):
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


class MLPV2(torch.nn.Module):
    def __init__(
        self,
        channels: list[int],
        bias: bool = False,
        act: str | None = "silu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain input and output widths")
        if layer_norm and rms_norm:
            raise ValueError("layer_norm and rms_norm are mutually exclusive")

        self.dims = list(channels)
        self.num_layers = len(channels) - 1
        self.is_nonlinear = act is not None and self.num_layers > 1
        layers = []
        for index, (h_in, h_out) in enumerate(zip(channels, channels[1:])):
            layers.append(mlpLinear(h_in, h_out, alpha=1.0 / sqrt(h_in), bias=bias))
            if index == self.num_layers - 1:
                continue
            if layer_norm:
                layers.append(torch.nn.LayerNorm(h_out))
            elif rms_norm:
                layers.append(torch.nn.RMSNorm(h_out))
            if act is not None:
                layers.append(get_scaled_activation(act))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor | IndexedFeatures) -> torch.Tensor:
        """Evaluate dense inputs or indexed input partitions."""
        return self.mlp(x)


MLP: dict[str, type[torch.nn.Module]] = {
    "mlp_v1": MLPV1,
    "mlp_v2": MLPV2,
}


class BatchMLPV1(GroupedRadialMLP):
    """Apply independent MLPs to contiguous input groups.

    Parameters
    ----------
    in_features, out_features : list of int
        Input and output widths per group. Zero output skips an unused group.
    hidden : list of int
        Hidden widths of each MLP.
    context_features : int
        Additional input width shared by all groups (indexed endpoint features).

    Examples
    --------
    >>> mlp = BatchMLPV1([2, 1, 1], [64, 64], [12, 8, 4])
    >>> mlp(torch.randn(5, 4)).shape
    torch.Size([5, 24])
    """

    network = MLPV1

    def __init__(
        self, in_features, hidden, out_features, *, context_features=0, **kwargs
    ):
        if len(in_features) != len(out_features):
            raise ValueError("Input and output group counts must match.")
        paths = [(i, n) for i, n in enumerate(out_features) if n]
        super().__init__(
            in_features,
            [
                self.network([in_features[i] + context_features, *hidden, n], **kwargs)
                for i, n in paths
            ],
            paths,
            context_features=context_features,
        )
        self.groups = []
        if self.branches:
            for layer in self.branches[0].mlp:
                if isinstance(layer, mlpLinear):
                    by_shape = {}
                    step = len(self.groups)
                    for i, branch in enumerate(self.branches):
                        linear = branch.mlp[step]
                        by_shape.setdefault((linear.in_dim, linear.out_dim), []).append(
                            i
                        )
                    self.groups.append(tuple(tuple(v) for v in by_shape.values()))
                else:
                    self.groups.append(())

    @classmethod
    def from_product(cls, product, widths, hidden, **kwargs):
        """Route the output columns to weighted uvu instructions in their native order."""
        out_widths = [0] * len(widths)
        paths = []
        for instruction in product.instructions:
            i, j, _, mode, weighted = instruction[:5]
            if not weighted:
                continue
            if mode != "uvu":
                raise ValueError("Degree-specific radial weights require uvu CGTP paths.")
            mul, ir = product.irreps_in2[j]
            size = product.irreps_in1[i].mul * mul
            paths.append((ir.l, size))
            out_widths[ir.l] += size
        if sum(out_widths) != product.weight_numel:
            raise ValueError("CGTP path widths do not match weight_numel.")
        module = cls(widths, hidden, out_widths, **kwargs)
        module.set_paths(paths)
        return module

    def stream(self, features, *, backend="torch", paired=False):
        """Prepare full networks or only their final projections for EQX."""
        example = features[0][0] if isinstance(features, tuple) else features
        projection = example.new_empty(0, self.out_features)
        if memory_first() or paired or len(self.groups) <= 1:
            return radial_features(features), projection, self
        layers = [b.mlp[-1] for b in self.branches]
        network = self.final_projection([layer.in_dim for layer in layers], layers)
        return (
            self.evaluate_groups(features, hidden_only=True, backend=backend),
            projection,
            network,
        )

    def forward(self, features):
        return self.evaluate_groups(features, hidden_only=False)

    def evaluate_groups(self, features, *, hidden_only, backend="torch"):
        if self.chain_steps and not isinstance(features, tuple):
            return self.packed(features, hidden_only=hidden_only, backend=backend)
        if isinstance(features, tuple):
            radial, context = features[0][0], features[1:]
        else:
            radial, context = features, ()
            if self.context_features:
                radial, shared = radial.split(
                    (sum(self.in_features), self.context_features), -1
                )
                context = ((shared, None),)
        if not self.branches:
            return radial[:, :0]
        parts = radial.split(self.in_features, dim=-1)
        values = [parts[i] for i in self.active]
        for step, groups in enumerate(self.groups[:-1] if hidden_only else self.groups):
            layers = [b.mlp[step] for b in self.branches]
            if not groups:
                values = [layer(x) for layer, x in zip(layers, values)]
                continue
            weights = [layer.get_weight() for layer in layers]
            extras = []
            if step == 0 and context:
                widths = [x.shape[1] for x in values]
                # Project each endpoint table on nodes once for all active degrees.
                for table, index in context:
                    projected = table @ torch.cat(
                        [
                            w[start : start + table.size(1)]
                            for w, start in zip(weights, widths)
                        ],
                        dim=1,
                    )
                    if index is not None:
                        projected = projected[index]
                    extras.append(
                        projected.split([layer.out_dim for layer in layers], dim=1)
                    )
                    widths = [start + table.size(1) for start in widths]
                weights = [w[: x.size(1)] for w, x in zip(weights, values)]
            for ids in groups:
                if len(ids) == 1:
                    i = ids[0]
                    values[i] = values[i] @ weights[i]
                else:
                    result = torch.bmm(
                        torch.stack([values[i] for i in ids]),
                        torch.stack([weights[i] for i in ids]),
                    )
                    for j, i in enumerate(ids):
                        values[i] = result[j]
            for i, layer in enumerate(layers):
                if layer.bias is not None:
                    values[i] = values[i] + layer.bias
                for extra in extras:
                    values[i] = values[i] + extra[i]
        out = torch.cat(values, dim=-1)
        return (
            out
            if hidden_only or self.output_order is None
            else out.index_select(-1, self.output_order)
        )


class BatchMLPV2(BatchMLPV1):
    """Apply independent :class:`MLPV2` networks with the layout of :class:`BatchMLPV1`.

    Examples
    --------
    >>> mlp = BatchMLPV2([3, 2, 1], [64], [16, 8, 8], rms_norm=True)
    >>> mlp(torch.randn(5, 6)).shape
    torch.Size([5, 32])
    """

    network = MLPV2


BATCH_MLP = {"mlp_v1": BatchMLPV1, "mlp_v2": BatchMLPV2}
