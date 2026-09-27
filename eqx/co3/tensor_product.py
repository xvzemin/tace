"""Delta and Levi-Civita couplings with independent channel paths."""

import math
from functools import lru_cache
from typing import NamedTuple

import torch
from e3nn import o3

from .basis import Projector, path_matrix
from .basis import path_normalization as coupling_scale
from .irreps import Irreps


def cartesian_product(x, y, l1, l2, l3):
    """Return an unprojected Cartesian coupling with trailing size 3**l3.

    Inputs must be symmetric traceless tensors with broadcastable leading
    dimensions. Delta and epsilon contractions use factors 3**(-k/2) and
    (2*3**k)**(-1/2), respectively.
    """
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError("Degrees must satisfy the triangle inequality.")
    k, odd = divmod(l1 + l2 - l3, 2)
    if not odd:
        x = x.reshape(*x.shape[:-1], 3 ** (l1 - k), 3**k)
        y = y.reshape(*y.shape[:-1], 3**k, 3 ** (l2 - k))
        return (x @ y).flatten(-2) / math.sqrt(3**k)
    x = x.reshape(*x.shape[:-1], 3 ** (l1 - k - 1), 3, 3**k)
    y = y.reshape(*y.shape[:-1], 3**k, 3, 3 ** (l2 - k - 1))
    values = [
        x[..., a, :] @ y[..., b, :] - x[..., b, :] @ y[..., a, :]
        for a, b in ((1, 2), (2, 0), (0, 1))
    ]
    return torch.stack(values, dim=-2).flatten(-3) / math.sqrt(2 * 3**k)


@lru_cache(maxsize=None)
def coupling_phase(l1, l2, l3):
    """Return the phase of a Cartesian coupling in the supplied CG convention."""
    cg = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")
    indices = tuple(int(i) for i in torch.unravel_index(cg.abs().argmax(), cg.shape))
    x, y, z = [path_matrix(l)[:, m] for l, m in zip((l1, l2, l3), indices)]
    # A single nonzero coefficient determines the phase; the magnitude is analytic.
    value = cartesian_product(x, y, l1, l2, l3) @ z
    return -1.0 if value * cg[indices] < 0 else 1.0


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple


class TensorProduct(torch.nn.Module):
    """Couple Cartesian irreps through delta and Levi-Civita contractions.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        Input and output representations in flattened mul_ir layout.
    instructions : sequence of tuple
        (i_in1, i_in2, i_out, mode, has_weight[, path_weight]) entries.
        Modes are uvu, uvv, uuu, uuw, uvw and uvuv.
    irrep_normalization : {"component", "norm", "none"}, optional
        Normalization of independent irreducible coordinates.
    path_normalization : {"element", "path", "none"}, optional
        Normalization of paths feeding the same output entry.
    in1_var, in2_var, out_var : sequence of float, optional
        Variance for each input or output entry. Defaults to one.
    internal_weights, shared_weights : bool, optional
        Store trainable weights and share weights over leading dimensions.
        Internal weights require shared weights.
    project : bool, optional
        Project outputs to symmetric traceless tensors. Set False to defer
        projection until after aggregation and channel mixing.

    Notes
    -----
    Inputs must be symmetric traceless. Each path includes the analytic
    Cartesian-to-CG scale. Repeated output irreps are not regrouped or merged.
    Unprojected outputs must be projected before another tensor product or
    a nonlinear operation.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        *,
        irrep_normalization="component",
        path_normalization="element",
        in1_var=None,
        in2_var=None,
        out_var=None,
        internal_weights=None,
        shared_weights=None,
        project=True,
    ):
        super().__init__()
        if irrep_normalization not in (
            "component",
            "norm",
            "none",
        ) or path_normalization not in ("element", "path", "none"):
            raise ValueError("Invalid tensor-product normalization.")
        self.irreps_in1, self.irreps_in2, self.irreps_out = map(
            Irreps, (irreps_in1, irreps_in2, irreps_out)
        )
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.project = project
        variances = []
        for value, irreps in zip(
            (in1_var, in2_var, out_var),
            (self.irreps_in1, self.irreps_in2, self.irreps_out),
        ):
            value = [1.0] * len(irreps) if value is None else list(value)
            if len(value) != len(irreps) or any(v < 0 for v in value):
                raise ValueError(
                    "Variances must be non-negative, with one value per irrep entry."
                )
            variances.append(value)
        in1_var, in2_var, out_var = variances
        paths, counts = [], []
        for entry in instructions:
            if len(entry) not in (5, 6):
                raise ValueError("An instruction must contain five or six entries.")
            i, j, k, mode, weighted = entry[:5]
            path_weight = entry[5] if len(entry) == 6 else 1.0
            if min(i, j, k) < 0:
                raise IndexError(
                    "Tensor-product instruction indices must be non-negative."
                )
            u, a = self.irreps_in1[i]
            v, b = self.irreps_in2[j]
            w, c = self.irreps_out[k]
            if c not in a * b:
                raise ValueError(
                    "Tensor-product paths must obey angular and parity selection rules."
                )
            if mode not in ("uvw", "uvu", "uvv", "uuu", "uuw", "uvuv"):
                raise NotImplementedError(f"Unsupported connection mode: {mode}")
            if (
                (mode == "uvu" and w != u)
                or (mode == "uvv" and w != v)
                or (mode in ("uuu", "uuw") and u != v)
                or (mode == "uuu" and w != u)
                or (mode == "uvuv" and w != u * v)
            ):
                raise ValueError(f"Incompatible channel multiplicities for {mode}.")
            if mode == "uvw" and not weighted:
                raise ValueError("uvw requires weights.")
            if mode == "uuw" and not weighted and w != 1:
                raise ValueError("Unweighted uuw requires a single output channel.")
            shape = {
                "uvw": (u, v, w),
                "uvu": (u, v),
                "uvv": (u, v),
                "uuu": (u,),
                "uuw": (u, w),
                "uvuv": (u, v),
            }[mode]
            counts.append(
                {"uvw": u * v, "uvu": v, "uvv": u, "uuu": 1, "uuw": u, "uvuv": 1}[mode]
            )
            paths.append(Instruction(i, j, k, mode, weighted, path_weight, shape))
        normalized, scales = [], []
        for index, ins in enumerate(paths):
            a, b, c = (
                self.irreps_in1[ins.i_in1].ir,
                self.irreps_in2[ins.i_in2].ir,
                self.irreps_out[ins.i_out].ir,
            )
            alpha = {
                "component": c.spherical_dim,
                "norm": a.spherical_dim * b.spherical_dim,
                "none": 1,
            }[irrep_normalization]
            if path_normalization == "element":
                denominator = sum(
                    in1_var[p.i_in1] * in2_var[p.i_in2] * n
                    for p, n in zip(paths, counts)
                    if p.i_out == ins.i_out
                )
            elif path_normalization == "path":
                denominator = (
                    in1_var[ins.i_in1]
                    * in2_var[ins.i_in2]
                    * counts[index]
                    * sum(p.i_out == ins.i_out for p in paths)
                )
            else:
                denominator = 1
            alpha = math.sqrt(
                alpha / (denominator or 1) * out_var[ins.i_out] * ins.path_weight
            )
            normalized.append(ins._replace(path_weight=alpha))
            scales.append(
                alpha * coupling_phase(a.l, b.l, c.l) / coupling_scale(a.l, b.l, c.l)
            )
        self.instructions = normalized
        self.shared_weights = True if shared_weights is None else shared_weights
        self.internal_weights = (
            (self.shared_weights and any(p.has_weight for p in paths))
            if internal_weights is None
            else internal_weights
        )
        if self.internal_weights and not self.shared_weights:
            raise ValueError("Internal weights require shared_weights=True.")
        self.weight_numel = sum(math.prod(p.path_shape) for p in paths if p.has_weight)
        if self.internal_weights and self.weight_numel:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0))
        self.register_buffer("path_scales", torch.tensor(scales), persistent=False)
        self.projection = Projector(self.irreps_out) if project else torch.nn.Identity()
        self.slices_in1, self.slices_in2 = (
            self.irreps_in1.slices(),
            self.irreps_in2.slices(),
        )
        connected = {
            ins.i_out
            for ins in self.instructions
            if ins.path_weight and all(ins.path_shape)
        }
        self.register_buffer(
            "output_mask",
            (
                torch.cat(
                    [
                        torch.full((item.dim,), float(i in connected))
                        for i, item in enumerate(self.irreps_out)
                    ]
                )
                if self.irreps_out
                else torch.empty(0)
            ),
            persistent=False,
        )

    def forward(self, x, y, weight=None):
        """Evaluate the Cartesian tensor product.

        Parameters
        ----------
        x, y : torch.Tensor
            STF inputs with trailing sizes irreps_in1.dim and irreps_in2.dim.
            Leading dimensions must be broadcastable.
        weight : torch.Tensor, optional
            External weights with trailing size weight_numel. Leading
            dimensions broadcast when shared_weights=False.

        Returns
        -------
        torch.Tensor
            Features of shape (..., irreps_out.dim). Outputs are STF only
            when project=True.
        """
        if x.shape[-1] != self.irreps_in1.dim or y.shape[-1] != self.irreps_in2.dim:
            raise ValueError("Input feature dimensions do not match the irreps.")
        weight = self.weight if weight is None else weight
        if weight.shape[-1] != self.weight_numel:
            raise ValueError("Weight size does not match the instructions.")
        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights must be one-dimensional.")
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        zero = x[..., :0].sum() + y[..., :0].sum() + weight[..., :0].sum()
        outputs = [
            x.new_zeros((*shape, mul, ir.dim)) + zero for mul, ir in self.irreps_out
        ]
        inputs = [
            [
                value[..., section].reshape(*value.shape[:-1], mul, ir.dim)
                for (mul, ir), section in zip(irreps, sections)
            ]
            for value, irreps, sections in (
                (x, self.irreps_in1, self.slices_in1),
                (y, self.irreps_in2, self.slices_in2),
            )
        ]
        offset = 0
        for index, ins in enumerate(self.instructions):
            a, b, c = (
                self.irreps_in1[ins.i_in1].ir,
                self.irreps_in2[ins.i_in2].ir,
                self.irreps_out[ins.i_out].ir,
            )
            left, right = inputs[0][ins.i_in1], inputs[1][ins.i_in2]
            mode = ins.connection_mode
            if mode not in ("uuu", "uuw"):
                left, right = left.unsqueeze(-2), right.unsqueeze(-3)
            value = cartesian_product(left, right, a.l, b.l, c.l)
            if ins.has_weight:
                size = math.prod(ins.path_shape)
                w = weight[..., offset : offset + size].reshape(
                    *weight.shape[:-1], *ins.path_shape
                )
                offset += size
                if mode == "uvw":
                    value = torch.einsum("...uvd,...uvw->...wd", value, w)
                elif mode == "uuw":
                    value = torch.einsum("...ud,...uw->...wd", value, w)
                else:
                    value = value * w.unsqueeze(-1)
            elif mode == "uuw":
                value = value.sum(-2, keepdim=True)
            if mode == "uvu":
                value = value.sum(-2)
            elif mode == "uvv":
                value = value.sum(-3)
            elif mode == "uvuv":
                value = value.flatten(-3, -2)
            outputs[ins.i_out] = outputs[ins.i_out] + value * self.path_scales[index]
        result = (
            torch.cat([v.flatten(-2) for v in outputs], -1)
            if outputs
            else x.new_zeros((*shape, 0)) + zero
        )
        return self.projection(result)

    def weight_view_for_instruction(self, instruction, weight=None):
        """Return a view of the weights for one instruction."""
        ins = self.instructions[instruction]
        if not ins.has_weight:
            raise ValueError("The instruction has no weights.")
        weight = self.weight if weight is None else weight
        offset = sum(
            math.prod(p.path_shape)
            for p in self.instructions[:instruction]
            if p.has_weight
        )
        return weight[..., offset : offset + math.prod(ins.path_shape)].reshape(
            *weight.shape[:-1], *ins.path_shape
        )

    def weight_views(self, weight=None, yield_instruction=False):
        """Iterate over weighted paths in instruction order."""
        for i, ins in enumerate(self.instructions):
            if ins.has_weight:
                view = self.weight_view_for_instruction(i, weight)
                yield (i, ins, view) if yield_instruction else view

    def extra_repr(self):
        return f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} | {self.weight_numel} weights, project={self.project}"


class FullyConnectedTensorProduct(TensorProduct):
    """Connect all allowed irrep triples with independent uvw weights.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        Input and output representations.
    **kwargs
        Weight, normalization, and projection options for TensorProduct.
    """

    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        irreps_in1, irreps_in2, irreps_out = map(
            Irreps, (irreps_in1, irreps_in2, irreps_out)
        )
        instructions = [
            (i, j, k, "uvw", True)
            for i, (_, a) in enumerate(irreps_in1)
            for j, (_, b) in enumerate(irreps_in2)
            for k, (_, c) in enumerate(irreps_out)
            if c in a * b
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out, instructions, **kwargs)


class ElementwiseTensorProduct(TensorProduct):
    """Couple corresponding channels without learnable weights.

    Parameters
    ----------
    irreps_in1, irreps_in2 : Irreps or str
        Input representations with equal total multiplicity.
    filter_ir_out : sequence of Irrep or str, optional
        Allowed output irrep labels. Defaults to all allowed couplings.
    **kwargs
        Normalization and projection options for TensorProduct.
    """

    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, **kwargs):
        from .irreps import Irrep

        a, b = [list(Irreps(irreps).simplify()) for irreps in (irreps_in1, irreps_in2)]
        if sum(mul for mul, _ in a) != sum(mul for mul, _ in b):
            raise ValueError("Elementwise products require equal channel counts.")
        i = 0
        while i < len(a):
            n, ir = a[i]
            m, jr = b[i]
            if n < m:
                b[i : i + 1] = [(n, jr), (m - n, jr)]
            if m < n:
                a[i : i + 1] = [(m, ir), (n - m, ir)]
            i += 1
        keep = None if filter_ir_out is None else set(map(Irrep, filter_ir_out))
        output, instructions = [], []
        for i, ((mul, ir), (_, jr)) in enumerate(zip(a, b)):
            for kr in ir * jr:
                if keep is None or kr in keep:
                    instructions.append((i, i, len(output), "uuu", False))
                    output.append((mul, kr))
        super().__init__(a, b, output, instructions, **kwargs)
