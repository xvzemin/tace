"""Generalized Clebsch–Gordan products with independent coupling paths."""

import math
from collections import defaultdict
from functools import lru_cache
from itertools import accumulate

import torch
from e3nn import o3

from ...ops import config, resolve_backend
from ...ops.common.torch.node_polynomial import AngularPolynomial
from ...ops.precision import validate_uniform
from ...utils.autograd import forward_ad_active


@lru_cache(maxsize=1024)
def spherical_rows(l1, l2, l3):
    """Return the nonzero real Wigner-3j entries."""
    cg = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")
    a, b, c = cg.nonzero(as_tuple=True)
    return tuple(zip(a.tolist(), b.tolist(), c.tolist(), cg[a, b, c].tolist()))


def angular_offsets(irreps):
    return tuple(accumulate((ir.dim for _, ir in irreps), initial=0))


def channel_permutation(irreps):
    return torch.tensor(
        [
            sl.start + c * ir.dim + m
            for (mul, ir), sl in zip(irreps, irreps.slices())
            for m in range(ir.dim)
            for c in range(mul)
        ],
        dtype=torch.long,
        device="cpu",
    )


def gated_columns(columns, irreps, targets):
    groups = {ir: [] for _, ir in targets}
    offset = gates = 0
    for _, ir in irreps:
        if ir in groups:
            groups[ir].append((gates, columns[offset : offset + ir.dim]))
            gates += 1
        offset += ir.dim
    result, slices = [], []
    for _, ir in targets:
        start = len(result)
        paths = groups[ir]
        for first in range(0, len(paths), 8):
            tile = paths[first : first + 8]
            for m in range(ir.dim):
                result.append(
                    {(*key, p): w for p, cols in tile for key, w in cols[m].items()}
                )
        slices.append((start, (len(result) - start) // ir.dim))
    return result, gates, tuple(slices)


class CouplingBlock(torch.nn.Module):
    """Evaluate one sparse generalized coupling tensor and its transposes."""

    def __init__(self, columns, sizes, start, backend, dense):
        super().__init__()
        self.start, self.backend = start, backend
        self.width, self.arity = len(columns), len(sizes)
        self.sizes = (*sizes, self.width)
        rows = [
            ((*indices, dst), weight)
            for dst, column in enumerate(columns)
            for indices, weight in column.items()
            if weight
        ]
        locations = torch.tensor(
            [row for row, _ in rows], dtype=torch.int32, device="cpu"
        ).reshape(-1, self.arity + 1)
        coefficients = torch.tensor(
            [w for _, w in rows], dtype=torch.float64, device="cpu"
        )
        self.masters = []
        for role, size in enumerate((*sizes, self.width)):
            order = locations[:, role].argsort(stable=True)
            dst = locations[order, role]
            ptr = torch.nn.functional.pad(
                dst.long().bincount(minlength=size).cumsum(0), (1, 0)
            ).int()
            other = [i for i in range(self.arity + 1) if i != role]
            indices = locations.index_select(0, order)[:, other].contiguous()
            master = coefficients[order]
            self.masters.append(master)
            for key, value in (
                ("ptr", ptr),
                ("idx", indices),
                ("w", master.to(torch.get_default_dtype())),
            ):
                self.register_buffer(f"{key}_{role}", value, persistent=False)
            self.register_buffer(f"dst_{role}", dst.long(), persistent=False)
        self.nnz = len(rows)
        self.polynomial = None
        if dense and backend == "torch" and self.width and self.nnz:
            offsets = tuple(accumulate(sizes, initial=0))
            terms = {key for column in columns for key in column}
            if len(terms) * self.width <= min(32768, 12 * self.nnz):
                self.polynomial = AngularPolynomial(
                    [
                        {
                            tuple(offsets[k] + a for k, a in enumerate(key)): w
                            for key, w in col.items()
                        }
                        for col in columns
                    ],
                    self.arity,
                )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for i, master in enumerate(self.masters):
            w = getattr(self, f"w_{i}")
            setattr(self, f"w_{i}", master.to(w))
        return self

    def angular_rows(self):
        """Return ordered angular indices and FP64 coupling coefficients."""
        indices = getattr(self, f"idx_{self.arity}").tolist()
        return tuple(
            ((*row, dst), w)
            for row, dst, w in zip(
                indices,
                getattr(self, f"dst_{self.arity}").tolist(),
                self.masters[-1].tolist(),
            )
        )

    def forward(self, inputs):
        x = inputs[0]
        if self.backend == "cuda" and x.is_cuda and not forward_ad_active():
            from ...ops.cuda.mbe.ace.generalized import contract

            tables = [
                getattr(self, f"{key}_{i}")
                for i in range(self.arity + 1)
                for key in ("ptr", "idx", "w")
            ]
            return contract(list(inputs), tables, self.arity)
        if self.polynomial is not None and not config.TORCH_REFERENCE:
            packed = torch.cat(inputs, dim=1).transpose(1, 2)
            return self.polynomial(packed, config.ACE_POLYNOMIAL_WORKSPACE).transpose(
                1, 2
            )
        from ...ops.torch.mbe.ace.generalized import contract

        tables = [
            getattr(self, f"{key}_{i}")
            for i in range(self.arity + 1)
            for key in ("dst", "idx", "w")
        ]
        return contract(inputs, tables, self.sizes)


class GeneralizedContraction(torch.nn.Module):
    """Channel-wise generalized CG products in the supplied path order.

    Parameters
    ----------
    irreps_in : o3.Irreps
        Input blocks with a common multiplicity.
    instructions : sequence of sequences of ContractionInstruction
        Ordered couplings for degrees two and above. Output indices follow list order.
    shared_inputs : bool, optional
        Reuse the same factor at every degree.
    backend : {None, "torch", "cuda"}, optional
        Execution backend.
    gated_output : o3.Irreps, optional
        Last-order targets for a path-weighted reduction in packed layout.

    Examples
    --------
    >>> from equivariantx.mbe.ace import GeneralizedContraction
    >>> op = GeneralizedContraction("2x1o", [[(0, 0, "1e")]], backend="torch")
    >>> op(torch.randn(4, 6), torch.randn(4, 6))[-1].shape
    torch.Size([4, 6])
    """

    max_composed_terms = 65536
    max_codegen_terms = 8192

    def __init__(
        self,
        irreps_in,
        instructions,
        shared_inputs=False,
        backend=None,
        gated_output=None,
    ):
        super().__init__()
        irreps = o3.Irreps(irreps_in)
        if not irreps or len({mul for mul, _ in irreps}) != 1 or not irreps[0].mul:
            raise ValueError(
                "GeneralizedContraction requires a positive common multiplicity."
            )
        self.channels = irreps[0].mul
        self.shared_inputs = shared_inputs
        self.num_factors = 1 if shared_inputs else len(instructions) + 1
        self.input_dim = irreps.dim
        self.register_buffer("to_packed", channel_permutation(irreps), persistent=False)
        self.blocks = torch.nn.ModuleList()
        self.irreps_orders = [irreps]
        self.angular_dims = [irreps.dim // self.channels]
        self.gated = gated_output is not None and bool(instructions)
        self.gate_count, self.gate_slices = 0, ()
        backend = resolve_backend(backend)
        base = angular_offsets(irreps)
        previous = [{(i,): 1.0} for i in range(base[-1])]
        start = 0
        for stage, paths in enumerate(instructions, start=1):
            left = self.irreps_orders[-1]
            offsets = angular_offsets(left)
            out, entries, cost = [], [], 0
            width = 0
            for i, j, ir in paths:
                ir = o3.Irrep(ir)
                if (
                    not 0 <= i < len(left)
                    or not 0 <= j < len(irreps)
                    or ir not in left[i].ir * irreps[j].ir
                ):
                    raise ValueError(
                        f"Invalid degree-{stage + 1} coupling {(i, j, ir)}."
                    )
                scale = math.sqrt(ir.dim)
                for a, b, c, w in spherical_rows(left[i].ir.l, irreps[j].ir.l, ir.l):
                    a += offsets[i]
                    entries.append((a, b + base[j], c + width, w * scale))
                    cost += len(previous[a])
                out.append((self.channels, ir))
                width += ir.dim
            if stage > start + 1 and cost > min(
                self.max_composed_terms, max(2048, 3 * len(entries))
            ):
                start = stage - 1
                previous = [{(i,): 1.0} for i in range(offsets[-1])]
            columns = [defaultdict(float) for _ in range(width)]
            for a, b, c, w in entries:
                for prefix, coefficient in previous[a].items():
                    key = (*prefix, b)
                    if shared_inputs and start == 0:
                        key = tuple(sorted(key))
                    columns[c][key] += w * coefficient
            sizes = (self.angular_dims[start],) + (base[-1],) * (stage - start)
            out = o3.Irreps(out)
            terminal = self.gated and stage == len(instructions)
            if terminal:
                columns, self.gate_count, self.gate_slices = gated_columns(
                    columns, out, o3.Irreps(gated_output)
                )
                sizes += (self.gate_count,)
            self.blocks.append(
                CouplingBlock(columns, sizes, start, backend, dense=not terminal)
            )
            self.irreps_orders.append(out)
            self.angular_dims.append(width)
            if not self.gated or stage < len(instructions):
                self.register_buffer(
                    f"to_native_{stage}",
                    channel_permutation(out).argsort(),
                    persistent=False,
                )
            previous = columns
        self.codegen_supported = (
            sum(block.nnz for block in self.blocks) <= self.max_codegen_terms
        )

    def packed(self, factors, gates=None, *, max_degree=None):
        """Return path blocks in (batch, angular, channel) layout.

        Parameters
        ----------
        factors : sequence of torch.Tensor
            Independent input factors, or a single repeated factor.
        gates : torch.Tensor, optional
            Last-order coefficients of shape (batch, selected paths).
        max_degree : int, optional
            Highest returned correlation order.
        """
        if max_degree is not None and not 1 <= max_degree <= len(self.blocks) + 1:
            raise ValueError(
                "max_degree must be between one and the correlation order."
            )
        if len(factors) != self.num_factors:
            raise ValueError(
                f"Expected {self.num_factors} factors, received {len(factors)}."
            )
        if any(
            v.shape != (factors[0].shape[0], self.angular_dims[0], self.channels)
            for v in factors
        ):
            raise ValueError(
                "Packed factors must have shape (batch, angular, channels)."
            )
        if self.gated != (gates is not None):
            raise ValueError("Supply gates only for a gated last-order reduction.")
        if gates is not None and gates.shape != (factors[0].shape[0], self.gate_count):
            raise ValueError(
                "Gates must have shape (batch, selected last-order paths)."
            )
        validate_uniform((factors, gates, self.blocks[0].w_0 if self.blocks else None))
        results = [factors[0]]
        for stage, block in enumerate(self.blocks, start=1):
            if max_degree is not None and stage >= max_degree:
                break
            operands = [results[block.start]]
            operands.extend(
                factors[0 if self.shared_inputs else j]
                for j in range(block.start + 1, stage + 1)
            )
            if gates is not None and stage == len(self.blocks):
                operands.append(gates.unsqueeze(2).expand(-1, -1, self.channels))
            results.append(block(operands))
        return results

    def forward(self, *factors, max_degree=None):
        """Return one flattened mul_ir feature tensor per correlation order."""
        if self.gated:
            raise ValueError("Use packed(factors, gates) for a gated contraction.")
        if not factors or any(
            v.ndim != 2
            or v.shape[1] != self.input_dim
            or v.shape[0] != factors[0].shape[0]
            for v in factors
        ):
            raise ValueError(
                "Factors must have matching (batch, irreps_in.dim) shapes."
            )
        packed = [
            v.index_select(1, self.to_packed).reshape(
                v.shape[0], self.angular_dims[0], self.channels
            )
            for v in factors
        ]
        values = self.packed(packed, max_degree=max_degree)
        return (
            factors[0],
            *(
                v.flatten(1).index_select(1, getattr(self, f"to_native_{i}"))
                for i, v in enumerate(values[1:], 1)
            ),
        )
