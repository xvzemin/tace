"""Shared paths, weights and channel layouts for Cartesian tensor products."""

import math
from collections import defaultdict
from typing import NamedTuple

import torch

from ..ops import config, resolve_backend
from ..ops.common.torch.tensor_product import TensorProduct as GroupedTensorProduct
from ..ops.common.torch.tensor_product import instruction_groups
from .irreps import Irreps


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple


class InstructionGroup(NamedTuple):
    degrees: tuple
    connection_mode: str
    path_shape: tuple
    i_in1: tuple
    i_in2: tuple
    i_out: tuple
    weight_slices: tuple
    weight_slice: slice | None
    scale_slice: slice


class TensorProductBase(torch.nn.Module):
    """Build channel instructions independently of the angular coupling rule."""

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
        backend=None,
    ):
        backend = resolve_backend(backend)
        super().__init__()
        self.backend = backend
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
        self.input_dims = (self.irreps_in1.dim, self.irreps_in2.dim)
        self.output_dim = self.irreps_out.dim
        self.path_normalization = path_normalization
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
            scales.append(alpha * self.coupling_factor(a.l, b.l, c.l))
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
        self.path_factors = tuple(scales)
        groups = defaultdict(list)
        offset = 0
        for index, ins in enumerate(self.instructions):
            degrees = (
                self.irreps_in1[ins.i_in1].ir.l,
                self.irreps_in2[ins.i_in2].ir.l,
                self.irreps_out[ins.i_out].ir.l,
            )
            size = math.prod(ins.path_shape) if ins.has_weight else 0
            key = (degrees, ins.connection_mode, ins.has_weight, ins.path_shape)
            groups[key].append((index, slice(offset, offset + size)))
            offset += size
        self.path_order = tuple(i for entries in groups.values() for i, _ in entries)
        self.groups = []
        offset = 0
        for (degrees, mode, weighted, path_shape), entries in groups.items():
            paths = [self.instructions[i] for i, _ in entries]
            inputs = [tuple(p.i_in1 for p in paths), tuple(p.i_in2 for p in paths)]
            inputs = [ids[:1] if len(set(ids)) == 1 else ids for ids in inputs]
            sections = tuple(section for _, section in entries) if weighted else ()
            contiguous = sections and all(
                a.stop == b.start for a, b in zip(sections, sections[1:])
            )
            self.groups.append(
                InstructionGroup(
                    degrees,
                    mode,
                    path_shape,
                    *inputs,
                    tuple(p.i_out for p in paths),
                    sections,
                    slice(sections[0].start, sections[-1].stop) if contiguous else None,
                    slice(offset, offset + len(paths)),
                )
            )
            offset += len(paths)
        self.groups = tuple(self.groups)
        self.register_buffer(
            "path_scales",
            torch.tensor([scales[i] for i in self.path_order]),
            persistent=False,
        )
        reference_paths = [
            ins._replace(path_weight=scale)
            for ins, scale in zip(self.instructions, self.path_factors)
        ]
        couplings = tuple(
            (
                self.irreps_in1[ins.i_in1].ir.l,
                self.irreps_in2[ins.i_in2].ir.l,
                self.irreps_out[ins.i_out].ir.l,
            )
            for ins in self.instructions
        )
        self.couplings = couplings
        _, reference_groups = instruction_groups(
            reference_paths, couplings, grouped=False
        )
        self.reference = GroupedTensorProduct(
            [
                tuple((mul, ir.dim) for mul, ir in irreps)
                for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
            ],
            reference_groups,
            self.couple,
        )
        self.product_degree = max((max(degrees) for degrees in couplings), default=0)
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

    def mix_channels(self, weight, features):
        """Apply channel weights without changing any angular index."""
        if self.backend == "cuda" and features.is_cuda:
            from ..ops.common.cuda.cartesian import channel_mix
            from ..utils.autograd import forward_ad_active

            if features.shape[-1] <= 9 and not forward_ad_active():
                return channel_mix(weight, features)
            return weight @ features
        from ..ops.torch.co3.channels import mix_channels

        return mix_channels(weight, features)

    def channel_metadata(self, couplings, compact=()):
        """Combine channel-wise paths without expanding intermediate products."""
        if not self.instructions or any(
            ins.connection_mode != "uuu"
            and not (ins.connection_mode == "uvu" and ins.path_shape[1] == 1)
            and not (ins.connection_mode == "uvv" and ins.path_shape[0] == 1)
            for ins in self.instructions
        ):
            return None
        remaining = config.CARTESIAN_LITERAL_TERMS
        rows, offset = [], 0
        sections = [
            irreps.slices()
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        ]
        output_starts, output_sizes, dim = [], [], 0
        for i, (mul, ir) in enumerate(self.irreps_out):
            width = math.comb(ir.l + 2, 2) if i in compact else ir.dim
            output_starts.append(dim)
            output_sizes.append(width)
            dim += mul * width
        for ins, degrees, scale in zip(self.instructions, couplings, self.path_factors):
            entries = self.channel_entries(degrees, remaining, ins.i_out in compact)
            if entries is None:
                return None
            remaining -= len(entries)
            for a, b, c, coefficient in entries:
                rows.append(
                    (
                        (
                            sections[0][ins.i_in1].start + a,
                            sections[1][ins.i_in2].start + b,
                            offset if ins.has_weight else -1,
                            output_starts[ins.i_out] + c,
                        ),
                        coefficient * scale,
                        self.irreps_out[ins.i_out].mul,
                        (
                            0 if ins.connection_mode == "uvv" else 3 ** degrees[0],
                            0 if ins.connection_mode == "uvu" else 3 ** degrees[1],
                            1,
                            output_sizes[ins.i_out],
                        ),
                    )
                )
            offset += math.prod(ins.path_shape) if ins.has_weight else 0
        return repr((0, (*self.input_dims, self.weight_numel, dim), tuple(rows)))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.path_scales = torch.tensor(
            [self.path_factors[i] for i in self.path_order],
            dtype=torch.float64,
            device="cpu",
        ).to(self.path_scales)
        return self

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
