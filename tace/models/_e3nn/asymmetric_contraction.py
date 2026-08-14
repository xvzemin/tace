################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
Not all modules included here are ready for direct use; however,
we have retained them for reference by the community.
"""

import math
from typing import Union

import torch

from ..o2.legacy import satisfy


def so2_dim(m: int) -> int:
    return 1 if int(m) == 0 else 2


def so2_clebsch_gordan(
    m1: int,
    m2: int,
    out_m: int,
    mode: str,
    *,
    dtype: Union[torch.dtype, None] = None,
    device: Union[torch.device, None] = None,
) -> torch.Tensor:
    dtype = torch.get_default_dtype() if dtype is None else dtype
    d1 = so2_dim(m1)
    d2 = so2_dim(m2)
    d3 = so2_dim(out_m)
    C = torch.zeros(d1, d2, d3, dtype=dtype, device=device)

    if d1 == 1 and d2 == 1:
        C[0, 0, 0] = 1.0
    elif d1 == 1:
        C[0, :, :] = torch.eye(d2, d3, dtype=dtype, device=device)
    elif d2 == 1:
        C[:, 0, :] = torch.eye(d1, d3, dtype=dtype, device=device)
    elif mode == "sum":
        # (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        C[0, 0, 0] = 1.0
        C[1, 1, 0] = -1.0
        C[0, 1, 1] = 1.0
        C[1, 0, 1] = 1.0
    elif mode == "diff" and out_m == 0:
        # Even m=0 scalar from z_m conj(z_m). The odd pseudoscalar is not kept.
        C[0, 0, 0] = 1.0
        C[1, 1, 0] = 1.0
    elif mode == "diff" and m1 < m2:
        # y * conj(x), using the positive-frequency convention.
        C[0, 0, 0] = 1.0
        C[1, 1, 0] = 1.0
        C[0, 1, 1] = 1.0
        C[1, 0, 1] = -1.0
    elif mode == "diff":
        # x * conj(y).
        C[0, 0, 0] = 1.0
        C[1, 1, 0] = 1.0
        C[1, 0, 1] = 1.0
        C[0, 1, 1] = -1.0
    else:
        raise ValueError(f"Unknown SO(2) coupling mode={mode!r}")

    norm = torch.sum(C * C, dim=(0, 1), keepdim=True).sqrt()
    return torch.where(norm > 0, C / norm.clamp_min(torch.finfo(dtype).eps), C)


class SO2ASymmetricContraction(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        correlation: int,
        *,
        m1m2: Union[str, None] = None,
        internal_weights: bool = True,
    ) -> None:
        super().__init__()

        self.mmax = int(mmax)
        self.lmax = int(lmax)
        self.num_channels = int(num_channels)
        self.correlation = int(correlation)
        self.m1m2 = m1m2
        self.internal_weights = bool(internal_weights)

        if self.correlation < 1:
            raise ValueError(f"correlation must be positive, got {correlation}")

        offsets = []
        offset = 0
        for m in range(self.mmax + 1):
            offsets.append(offset)
            offset += so2_dim(m)
        self.irrep_offsets = offsets
        self.irrep_dim = offset

        self.contractions: list[list[dict[str, object]]] = []
        self.path_orders = self.contractions
        self.order_weight_numels: list[int] = []
        self.order_num_paths: list[int] = []
        self._setup_paths()
        self.out_m_order_counts = self._setup_out_m_order_counts()
        self.num_paths = sum(self.order_num_paths)
        self.weight_numel = sum(self.order_weight_numels)

        n = self.lmax + 1
        if self.internal_weights:
            self.weights = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.randn(
                            num_paths,
                            n,
                            self.num_channels,
                        )
                    )
                    for num_paths in self.order_num_paths
                ]
            )
        else:
            self.weights = torch.nn.ParameterList([])

    def _setup_paths(self) -> None:
        dtype = torch.get_default_dtype()
        previous: list[dict[str, object]] = []

        for m in range(self.mmax + 1):
            dim = so2_dim(m)
            U = torch.eye(dim, dtype=dtype)
            previous.append(
                {
                    "leaves": (m,),
                    "out_m": m,
                    "steps": (),
                    "U": U,
                }
            )

        order_groups = self._register_order_groups(previous, 1)
        self.contractions.append(order_groups)
        self.order_num_paths.append(len(previous))
        self.order_weight_numels.append(
            len(previous) * (self.lmax + 1) * self.num_channels
        )

        for order in range(2, self.correlation + 1):
            current: list[dict[str, object]] = []
            for path in previous:
                prev_m = int(path["out_m"])
                prev_U = path["U"]
                for next_m in range(self.mmax + 1):
                    for out_m, mode in self._valid_pair_outputs(prev_m, next_m):
                        pair = so2_clebsch_gordan(
                            prev_m,
                            next_m,
                            out_m,
                            mode,
                            dtype=dtype,
                        )
                        U = torch.tensordot(prev_U, pair, dims=([-1], [0])).contiguous()
                        current.append(
                            {
                                "leaves": tuple(path["leaves"]) + (next_m,),
                                "out_m": out_m,
                                "steps": tuple(path["steps"])
                                + ((prev_m, next_m, out_m, mode),),
                                "U": U,
                            }
                        )
            previous = current
            order_groups = self._register_order_groups(current, order)
            self.contractions.append(order_groups)
            self.order_num_paths.append(len(current))
            self.order_weight_numels.append(
                len(current) * (self.lmax + 1) * self.num_channels
            )

    def _register_order_groups(
        self,
        paths: list[dict[str, object]],
        order: int,
        prefix: str = "U_path",
    ) -> list[dict[str, object]]:
        groups: dict[int, list[tuple[int, dict[str, object]]]] = {}
        for path_idx, path in enumerate(paths):
            groups.setdefault(int(path["out_m"]), []).append((path_idx, path))

        order_groups: list[dict[str, object]] = []
        path_offset = 0
        for out_m in range(self.mmax + 1):
            group = groups.get(out_m, [])
            if not group:
                continue

            compact_paths = []
            for local_idx, (_path_idx, path) in enumerate(group):
                path_name = f"{prefix}_{order}_{out_m}_{local_idx}"
                self.register_buffer(
                    path_name, path["U"].contiguous(), persistent=False
                )
                compact_path = dict(path)
                compact_path["buffer"] = path_name
                compact_path.pop("U", None)
                compact_paths.append(compact_path)
            order_groups.append(
                {
                    "order": order,
                    "out_m": out_m,
                    "num_paths": len(group),
                    "path_slice": slice(path_offset, path_offset + len(group)),
                    "paths": tuple(compact_paths),
                }
            )
            path_offset += len(group)
        return order_groups

    def _valid_pair_outputs(self, m1: int, m2: int) -> list[tuple[int, str]]:
        if not satisfy(m1, m2, self.m1m2):
            return []

        outputs: list[tuple[int, str]] = []
        sum_m = int(m1) + int(m2)
        if sum_m <= self.mmax:
            outputs.append((sum_m, "sum"))

        diff_m = abs(int(m1) - int(m2))
        if diff_m <= self.mmax and diff_m != sum_m:
            outputs.append((diff_m, "diff"))
        return outputs

    def _setup_out_m_order_counts(self) -> list[int]:
        counts = [0 for _ in range(self.mmax + 1)]
        for groups in self.contractions:
            for group in groups:
                counts[int(group["out_m"])] += 1
        return [max(count, 1) for count in counts]

    def _to_blocks(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.size(0)
        n = self.lmax + 1
        blocks: list[torch.Tensor] = []
        offset = 0
        blocks.append(x[:, offset : offset + n].unsqueeze(-1))
        offset += n
        for _m in range(1, self.mmax + 1):
            xm = x[:, offset : offset + 2 * n]
            xm = xm.view(B, 2, n, self.num_channels)
            blocks.append(xm.permute(0, 2, 3, 1).contiguous())
            offset += 2 * n
        return blocks

    def _merge_dense(self, dense: torch.Tensor) -> torch.Tensor:
        parts = [dense[..., self.irrep_offsets[0]]]
        for m in range(1, self.mmax + 1):
            start = self.irrep_offsets[m]
            xm = dense[..., start : start + 2]
            parts.append(
                xm.permute(0, 3, 1, 2).reshape(dense.shape[0], -1, self.num_channels)
            )
        return torch.cat(parts, dim=1)

    def _contract_path(
        self,
        blocks_by_key: dict[int, list[torch.Tensor]],
        order_to_key: list[int],
        path: dict[str, object],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        leaves = tuple(int(m) for m in path["leaves"])
        U = getattr(self, str(path["buffer"])).to(
            device=blocks_by_key[order_to_key[0]][0].device,
            dtype=dtype,
        )
        operands = [
            blocks_by_key[order_to_key[order_idx]][m]
            for order_idx, m in enumerate(leaves)
        ]

        letters = "pqrstuvwxyzadefghijklmABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if len(leaves) > len(letters):
            raise RuntimeError(
                "SO2ASymmetricContraction has too many leaves for einsum"
            )
        in_terms = [f"bnc{letters[i]}" for i in range(len(leaves))]
        u_term = "".join(letters[: len(leaves)]) + "o"
        expr = ",".join(in_terms + [u_term]) + "->bnco"
        return torch.einsum(expr, *operands, U)

    def _path_scale(self, group: dict[str, object]) -> float:
        num_paths = max(int(group["num_paths"]), 1)
        out_m = int(group["out_m"])
        num_orders = self.out_m_order_counts[out_m]
        return 1.0 / math.sqrt(num_paths * num_orders)

    def _forward_path(
        self,
        input_tensors: dict[int, torch.Tensor],
        order_to_key: list[int],
        selected_weights: list[torch.Tensor],
    ) -> torch.Tensor:
        x0 = input_tensors[order_to_key[0]]
        B = x0.size(0)
        n = self.lmax + 1
        dtype = x0.dtype
        blocks_by_key = {key: self._to_blocks(x) for key, x in input_tensors.items()}
        output = x0.new_zeros(B, n, self.num_channels, self.irrep_dim)

        for order_idx, groups in enumerate(self.contractions):
            w_order = selected_weights[order_idx]
            for group in groups:
                out_m = int(group["out_m"])
                start = self.irrep_offsets[out_m]
                stop = start + so2_dim(out_m)
                group_weight = w_order[:, group["path_slice"]].to(dtype=dtype)
                scale = self._path_scale(group)
                for local_idx, path in enumerate(group["paths"]):
                    z = (
                        self._contract_path(
                            blocks_by_key,
                            order_to_key,
                            path,
                            dtype=dtype,
                        )
                        * scale
                    )
                    w = group_weight[:, local_idx].view(B, n, self.num_channels, 1)
                    output[..., start:stop] = output[..., start:stop] + z * w

        return self._merge_dense(output)

    def _normalize_inputs(
        self,
        x: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]],
    ) -> tuple[dict[int, torch.Tensor], list[int]]:
        if isinstance(x, torch.Tensor):
            return {0: x}, [0] * self.correlation

        if len(x) == 0:
            raise ValueError("input list must contain at least one tensor")

        input_tensors: dict[int, torch.Tensor] = {}
        id_to_key: dict[int, int] = {}
        order_to_key: list[int] = []
        used_length = min(len(x), self.correlation)

        for order_idx in range(self.correlation):
            source_idx = order_idx if order_idx < used_length else used_length - 1
            x_i = x[source_idx]
            tensor_id = id(x_i)
            if tensor_id not in id_to_key:
                key = len(input_tensors)
                id_to_key[tensor_id] = key
                input_tensors[key] = x_i
            order_to_key.append(id_to_key[tensor_id])

        return input_tensors, order_to_key

    def _select_weights(
        self,
        B: int,
        weight_list: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]],
        order_num_paths: list[int],
        *,
        mode: str,
        internal: bool,
    ) -> list[torch.Tensor]:
        if not internal and isinstance(weight_list, torch.Tensor):
            if weight_list.size(-1) != self.weight_numel:
                raise ValueError(
                    f"{mode} external edge-dependent weight must have last "
                    f"dimension {self.weight_numel}, got {weight_list.size(-1)}"
                )
            if weight_list.dim() != 2 or weight_list.size(0) != B:
                raise ValueError(
                    f"{mode} external edge-dependent weight must have shape "
                    f"({B}, {self.weight_numel}), got {tuple(weight_list.shape)}"
                )

            split_weights = []
            offset = 0
            n = self.lmax + 1
            for order_idx, numel in enumerate(self.order_weight_numels):
                w_order = weight_list[:, offset : offset + numel]
                w_order = w_order.view(
                    B,
                    order_num_paths[order_idx],
                    n,
                    self.num_channels,
                )
                split_weights.append(w_order)
                offset += numel
            weight_list = split_weights

        selected_weights: list[torch.Tensor] = []
        for order_idx, w_order in enumerate(weight_list):
            expected_shape = (
                order_num_paths[order_idx],
                self.lmax + 1,
                self.num_channels,
            )
            if internal:
                if tuple(w_order.shape) != expected_shape:
                    raise ValueError(
                        f"{mode} internal weight for order {order_idx + 1} "
                        f"must have shape {expected_shape}, got {tuple(w_order.shape)}"
                    )
                selected_weights.append(w_order.unsqueeze(0).expand(B, -1, -1, -1))
                continue

            expected_external_shape = (B, *expected_shape)
            if tuple(w_order.shape) != expected_external_shape:
                raise ValueError(
                    f"{mode} external edge-dependent weight for order "
                    f"{order_idx + 1} must have shape {expected_external_shape}, "
                    f"got {tuple(w_order.shape)}"
                )
            selected_weights.append(w_order)
        return selected_weights

    def forward(
        self,
        x: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]],
        weights: Union[torch.Tensor, list[torch.Tensor], None] = None,
    ) -> torch.Tensor:
        input_tensors, order_to_key = self._normalize_inputs(x)

        x0 = input_tensors[order_to_key[0]]
        base_shape = tuple(x0.shape)
        for key, x_i in input_tensors.items():
            if tuple(x_i.shape) != base_shape:
                raise ValueError(
                    "all input tensors must have the same shape, "
                    f"got key 0={base_shape} and key {key}={tuple(x_i.shape)}"
                )

        B = x0.size(0)
        weight_params = list(self.weights) if self.internal_weights else weights

        if self.internal_weights:
            weight_list: list[torch.Tensor] = weight_params
        else:
            if weight_params is None:
                raise ValueError(
                    "edge-dependent weights must be provided when "
                    "internal_weights=False"
                )
            weight_list = weight_params

        selected_weights = self._select_weights(
            B,
            weight_list,
            self.order_num_paths,
            mode="contraction",
            internal=self.internal_weights,
        )

        return self._forward_path(input_tensors, order_to_key, selected_weights)

    def extra_repr(self) -> str:
        return (
            f"mmax={self.mmax}, lmax={self.lmax}, "
            f"channels={self.num_channels}, correlation={self.correlation}, "
            f"num_paths={self.order_num_paths}"
        )


class uuuSO2TensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        m1m2: Union[str, None] = None,
        internal_weights: bool = True,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channels = num_channels
        self.m1m2 = m1m2
        self.instructions = []

        self.num_paths = 0
        weight_numel = 0
        for m3 in range(mmax + 1):
            paths = self.enumerate_paths(m3)
            self.instructions.append(paths)
            weight_numel += num_channels * (lmax + 1) * len(paths)
            self.num_paths += len(paths)
        self.weight_numel = weight_numel

        if internal_weights:
            self.weight = torch.nn.Parameter(torch.randn(1, self.weight_numel))
        else:
            self.register_buffer("weight", None)
        self.internal_weights = internal_weights

        output_scales = []
        n = lmax + 1
        # m = 0
        scale0 = torch.full((n,), 1.0 / math.sqrt(len(self.instructions[0])))
        output_scales.append(scale0)
        # m > 0
        for m3 in range(1, mmax + 1):
            scale = 1.0 / math.sqrt(len(self.instructions[m3]))
            output_scales.append(torch.full((2 * n,), scale))
        output_scales = torch.cat(output_scales)
        self.register_buffer("output_scales", output_scales, persistent=False)

    def enumerate_paths(self, m3: int) -> list[tuple[int, int, str]]:
        paths = []

        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                if satisfy(m1, m2, self.m1m2):
                    # x1 * x2
                    if m1 + m2 == m3:
                        paths.append((m1, m2, "sum"))
                    # x1 * conj(x2)
                    elif abs(m1 - m2) == m3:
                        paths.append((m1, m2, "diff"))

        return paths

    def rmul(self, x, y):
        # [B, n, C] * [B, n, C] =>  [B, n, C]
        z = x * y
        return z

    def cmul(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        # [B, 2, n, C] * [B, 2, n, C] => [B, 2, n, C]
        a = x[:, 0]
        b = x[:, 1]
        c = y[:, 0]
        d = y[:, 1]

        if mode == "sum":
            real = a * c - b * d
            imag = a * d + b * c
        else:
            real = a * c + b * d
            imag = b * c - a * d

        B = real.size(0)
        C = real.size(-1)
        n = self.lmax + 1

        real = real.reshape(B, n, C)
        imag = imag.reshape(B, n, C)

        out = torch.stack([real, imag], dim=1)

        return out

    def to_list(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = []
        offset = 0
        n = self.lmax + 1
        # m = 0
        out.append(x[:, offset : offset + n])
        offset += n
        # m > 0
        for m in range(1, self.mmax + 1):
            xm = x[:, offset : offset + 2 * n]
            xm = xm.view(B, 2, n, self.num_channels)
            out.append(xm)
            offset += 2 * n
        return out

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        xs = self.to_list(x)  #  m = 0 [B, lmax+1, C]
        ys = self.to_list(y)  #  m > 0 [B, 2, lmax+1, C]
        if self.internal_weights:
            ws = self.weight
        else:
            ws = weight

        C = self.num_channels

        outputs = []
        w_offset = 0

        # m = 0
        n = self.lmax + 1
        m0 = 0.0
        w_numel = C * n

        for m1, m2, mode in self.instructions[0]:
            w = ws[:, w_offset : w_offset + w_numel]  # [B, C] or [1, C]
            w = w.view(-1, n, C)
            w_offset += w_numel

            # 0 x 0
            if m1 == 0 and m2 == 0:
                z = self.rmul(xs[0], ys[0])
                out = z * w
                m0 = m0 + out

            # m > 0 and m1 -m2 = 0
            elif m1 > 0 and m2 > 0:
                z = self.cmul(xs[m1], ys[m2], "diff")
                out = z[:, 0] * w  # imag is also invariant, but nod add here
                m0 = m0 + out

        outputs.append(m0)

        # m > 0
        for m3 in range(1, self.mmax + 1):
            real = 0.0
            imag = 0.0
            for m1, m2, mode in self.instructions[m3]:
                w = ws[:, w_offset : w_offset + w_numel]
                w_offset += w_numel
                w = w.view(-1, 1, n, C)

                if m1 == 0:
                    z = xs[m1].unsqueeze(1) * ys[m2]
                elif m2 == 0:
                    z = xs[m1] * ys[m2].unsqueeze(1)
                else:
                    if m1 < m2 and mode == "diff":
                        z = self.cmul(ys[m2], xs[m1], mode)
                    else:
                        z = self.cmul(xs[m1], ys[m2], mode)

                out = z * w
                real = real + out[:, 0]
                imag = imag + out[:, 1]

            outputs.append(real)
            outputs.append(imag)

        out = torch.cat(outputs, dim=1)
        out = out * self.output_scales.view(1, -1, 1)
        return out

    def __repr__(self):
        lines = []
        lines.append(f"{self.__class__.__name__}(")
        # lines.append(
        #     f"  mmax={self.mmax}, "
        #     f"lmax={self.lmax}, "
        #     f"channels={self.num_channels}, "
        #     f"weights={self.weight_numel}"
        # )
        # lines.append("")
        lines.append("  instructions:")
        total_paths = 0
        for m3, paths in enumerate(self.instructions):
            total_paths += len(paths)
            path_strs = []
            for m1, m2, mode in paths:
                if mode == "sum":
                    expr = f"{m1}+{m2}"
                else:
                    expr = f"{m1}-{m2}"
                path_strs.append(expr)
            joined = ", ".join(path_strs)
            lines.append(f"    m={m3:<2} : {len(paths):<2} paths | {joined}")
        # lines.append("")
        lines.append(f"  total_paths={total_paths}")
        lines.append(")")
        return "\n".join(lines)


class ComplexProductBasis(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        m1m2: Union[str, None] = ">=",
    ):
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.m1m2 = m1m2
        self.tp = uuuSO2TensorProduct(
            self.mmax,
            self.lmax,
            self.num_channel,
            m1m2=self.m1m2,
            internal_weights=False,
        )
        self.weight_numel = self.tp.weight_numel

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, ws: torch.Tensor
    ) -> torch.Tensor:
        return self.tp(x, y, ws)


# class _SO2DimExpr:
#     def __init__(self, m: int) -> None:
#         self.m = int(m)

#     def __add__(self, other: int) -> int:
#         if int(other) == 1:
#             return so2_dim(self.m)
#         return 2 * (0 if self.m == 0 else 1) + int(other)

#     __radd__ = __add__


# class _SO2Lmax:
#     def __init__(self, m: int) -> None:
#         self.m = int(m)
#         self.value = 0 if self.m == 0 else 1

#     def __lt__(self, other: object) -> bool:
#         return self.value < int(other)

#     def __gt__(self, other: object) -> bool:
#         return self.value > int(other)

#     def __add__(self, other: object) -> int:
#         return self.value + int(other)

#     def __radd__(self, other: object) -> int:
#         return int(other) + self.value

#     def __sub__(self, other: object) -> int:
#         return self.value - int(other)

#     def __rsub__(self, other: object) -> int:
#         return int(other) - self.value

#     def __rmul__(self, other: object) -> Union[int, _SO2DimExpr]:
#         if int(other) == 2:
#             return _SO2DimExpr(self.m)
#         return int(other) * self.value

#     def __int__(self) -> int:
#         return self.value

#     def __index__(self) -> int:
#         return self.value


# class _SO2Irrep:
#     _mmax = 0
#     _m1m2: Union[str, None] = None

#     def __init__(self, ir: object) -> None:
#         if isinstance(ir, _SO2Irrep):
#             self.l = ir.l
#         elif hasattr(ir, "l"):
#             self.l = int(ir.l)
#         elif isinstance(ir, str):
#             token = ir.strip()
#             if "x" in token:
#                 token = token.split("x", 1)[1]
#             token = token.rstrip("eo")
#             self.l = int(token)
#         elif isinstance(ir, tuple):
#             self.l = int(ir[0])
#         else:
#             self.l = int(ir)
#         self.p = 1
#         self.dim = so2_dim(self.l)

#     @classmethod
#     def configure(cls, mmax: int, m1m2: Union[str, None]) -> None:
#         cls._mmax = int(mmax)
#         cls._m1m2 = m1m2

#     def __mul__(self, other: object) -> list["_SO2Irrep"]:
#         other = _SO2Irrep(other)
#         if not satisfy(self.l, other.l, self._m1m2):
#             return []

#         outputs: list[int] = []
#         sum_m = self.l + other.l
#         if sum_m <= self._mmax:
#             outputs.append(sum_m)

#         diff_m = abs(self.l - other.l)
#         if diff_m <= self._mmax and diff_m != sum_m:
#             outputs.append(diff_m)

#         return [_SO2Irrep(m) for m in outputs]

#     def __eq__(self, other: object) -> bool:
#         try:
#             return self.l == _SO2Irrep(other).l
#         except (TypeError, ValueError):
#             return False

#     def __lt__(self, other: object) -> bool:
#         return self.l < _SO2Irrep(other).l

#     def __hash__(self) -> int:
#         return hash((self.l, self.p))

#     def __repr__(self) -> str:
#         return f"{self.l}e"

#     __str__ = __repr__


# class _SO2MulIr:
#     def __init__(self, mul: int, ir: object) -> None:
#         self.mul = int(mul)
#         self.ir = _SO2Irrep(ir)

#     def __iter__(self):
#         yield self.mul
#         yield self.ir

#     def __repr__(self) -> str:
#         return f"{self.mul}x{self.ir}"


# class _SO2Irreps:
#     def __init__(self, irreps: object) -> None:
#         if isinstance(irreps, _SO2Irreps):
#             self._irreps = list(irreps._irreps)
#         elif isinstance(irreps, _SO2Irrep):
#             self._irreps = [_SO2MulIr(1, irreps)]
#         elif isinstance(irreps, str):
#             self._irreps = self._parse_string(irreps)
#         else:
#             self._irreps = []
#             for item in irreps:
#                 if isinstance(item, _SO2MulIr):
#                     self._irreps.append(_SO2MulIr(item.mul, item.ir))
#                 elif isinstance(item, _SO2Irrep) or hasattr(item, "l"):
#                     self._irreps.append(_SO2MulIr(1, item))
#                 else:
#                     mul, ir = item
#                     self._irreps.append(_SO2MulIr(mul, ir))

#     @staticmethod
#     def _parse_string(irreps: str) -> list[_SO2MulIr]:
#         out = []
#         for token in irreps.replace(" ", "").split("+"):
#             if not token:
#                 continue
#             if "x" in token:
#                 mul_text, ir_text = token.split("x", 1)
#                 out.append(_SO2MulIr(int(mul_text), ir_text))
#             else:
#                 out.append(_SO2MulIr(1, token))
#         return out

#     def __iter__(self):
#         return iter(self._irreps)

#     def __len__(self) -> int:
#         return len(self._irreps)

#     def __getitem__(self, index: int) -> _SO2MulIr:
#         return self._irreps[index]

#     def __contains__(self, ir: object) -> bool:
#         target = _SO2Irrep(ir)
#         return any(mul_ir.ir == target for mul_ir in self._irreps)

#     @property
#     def dim(self) -> int:
#         return sum(mul_ir.mul * mul_ir.ir.dim for mul_ir in self._irreps)

#     @property
#     def lmax(self) -> _SO2Lmax:
#         if not self._irreps:
#             return _SO2Lmax(0)
#         return _SO2Lmax(max(mul_ir.ir.l for mul_ir in self._irreps))

#     def count(self, ir: object) -> int:
#         target = _SO2Irrep(ir)
#         return sum(mul_ir.mul for mul_ir in self._irreps if mul_ir.ir == target)

#     def simplify(self) -> "_SO2Irreps":
#         return self

#     def __repr__(self) -> str:
#         return "+".join(repr(mul_ir) for mul_ir in self._irreps)

#     __str__ = __repr__


# def _so2_wigner_3j(
#     out_m: int,
#     left_m: int,
#     right_m: int,
#     dtype: Union[torch.dtype, None] = None,
#     device: Union[torch.device, None] = None,
# ) -> torch.Tensor:
#     out_m = int(out_m)
#     left_m = int(left_m)
#     right_m = int(right_m)
#     shape = (so2_dim(out_m), so2_dim(left_m), so2_dim(right_m))

#     if left_m + right_m == out_m:
#         mode = "sum"
#     elif abs(left_m - right_m) == out_m:
#         mode = "diff"
#     else:
#         dtype = torch.get_default_dtype() if dtype is None else dtype
#         return torch.zeros(shape, dtype=dtype, device=device)

#     cg = so2_clebsch_gordan(
#         left_m,
#         right_m,
#         out_m,
#         mode,
#         dtype=dtype,
#         device=device,
#         irrep_normalization="none",
#     ) # should be norm
#     return cg.permute(2, 0, 1).contiguous()


# class SO2SymmetricContraction(torch.nn.Module):
#     def __init__(
#         self,
#         mmax: int,
#         lmax: int,
#         num_channels: int,
#         correlation: int,
#         num_elements: int = 1,
#         *,
#         m1m2: Union[str, None] = None,
#     ) -> None:
#         super().__init__()

#         self.mmax = int(mmax)
#         self.lmax = int(lmax)
#         self.num_channels = int(num_channels)
#         self.correlation = int(correlation)
#         self.num_elements = int(num_elements)
#         self.m1m2 = m1m2

#         if self.correlation < 1:
#             raise ValueError(f"correlation must be positive, got {correlation}")
#         if self.num_elements < 1:
#             raise ValueError(f"num_elements must be positive, got {num_elements}")

#         offsets = []
#         offset = 0
#         for m in range(self.mmax + 1):
#             offsets.append(offset)
#             offset += so2_dim(m)
#         self.irrep_offsets = offsets
#         self.irrep_dim = offset
#         self.num_features = (self.lmax + 1) * self.num_channels

#         from . import symmetric_contraction as symmetric_contraction_module

#         self.contractions = torch.nn.ModuleList()
#         self.num_paths_by_m: list[list[int]] = []

#         _SO2Irrep.configure(self.mmax, self.m1m2)
#         original_irrep = symmetric_contraction_module.o3.Irrep
#         original_irreps = symmetric_contraction_module.o3.Irreps
#         original_wigner_3j = symmetric_contraction_module.o3.wigner_3j
#         original_cuet_available = symmetric_contraction_module.CUET_AVAILABLE

#         try:
#             symmetric_contraction_module.o3.Irrep = _SO2Irrep
#             symmetric_contraction_module.o3.Irreps = _SO2Irreps
#             symmetric_contraction_module.o3.wigner_3j = _so2_wigner_3j
#             symmetric_contraction_module.CUET_AVAILABLE = False

#             irreps_in = _SO2Irreps(
#                 [(self.num_features, _SO2Irrep(m)) for m in range(self.mmax + 1)]
#             )
#             for out_m in range(self.mmax + 1):
#                 irrep_out = _SO2Irreps([(1, _SO2Irrep(out_m))])
#                 contraction = symmetric_contraction_module.Contraction(
#                     irreps_in=irreps_in,
#                     irrep_out=irrep_out,
#                     correlation=self.correlation,
#                     internal_weights=True,
#                     num_elements=self.num_elements,
#                 )
#                 self.contractions.append(contraction)
#                 self.num_paths_by_m.append(
#                     [
#                         int(contraction.U_tensors(order).size(-1))
#                         for order in range(1, self.correlation + 1)
#                     ]
#                 )
#         finally:
#             symmetric_contraction_module.o3.Irrep = original_irrep
#             symmetric_contraction_module.o3.Irreps = original_irreps
#             symmetric_contraction_module.o3.wigner_3j = original_wigner_3j
#             symmetric_contraction_module.CUET_AVAILABLE = original_cuet_available

#         self.weight_numel = sum(
#             parameter.numel() for parameter in self.parameters()
#         )

#     def _to_dense_features(self, x: torch.Tensor) -> torch.Tensor:
#         B = x.size(0)
#         n = self.lmax + 1
#         dense = x.new_zeros(B, n, self.num_channels, self.irrep_dim)
#         offset = 0
#         dense[..., self.irrep_offsets[0]] = x[:, offset : offset + n]
#         offset += n
#         for m in range(1, self.mmax + 1):
#             xm = x[:, offset : offset + 2 * n]
#             xm = xm.view(B, 2, n, self.num_channels)
#             start = self.irrep_offsets[m]
#             dense[..., start : start + 2] = xm.permute(0, 2, 3, 1)
#             offset += 2 * n
#         return dense.reshape(B, self.num_features, self.irrep_dim)

#     def _merge_dense_features(self, dense: torch.Tensor) -> torch.Tensor:
#         B = dense.size(0)
#         n = self.lmax + 1
#         dense = dense.reshape(B, n, self.num_channels, self.irrep_dim)
#         parts = [dense[..., self.irrep_offsets[0]]]
#         for m in range(1, self.mmax + 1):
#             start = self.irrep_offsets[m]
#             xm = dense[..., start : start + 2]
#             parts.append(xm.permute(0, 3, 1, 2).reshape(B, -1, self.num_channels))
#         return torch.cat(parts, dim=1)

#     def _one_hot_elements(
#         self,
#         node_attrs: Union[torch.Tensor, None],
#         batch: int,
#         device: torch.device,
#         dtype: torch.dtype,
#     ) -> torch.Tensor:
#         if self.num_elements == 1:
#             return torch.ones(batch, 1, device=device, dtype=dtype)
#         if node_attrs is None:
#             raise ValueError("node_attrs is required when num_elements > 1")
#         if node_attrs.dtype == torch.long and node_attrs.dim() == 1:
#             node_type = node_attrs.clamp(min=0, max=self.num_elements - 1)
#             return torch.nn.functional.one_hot(
#                 node_type,
#                 num_classes=self.num_elements,
#             ).to(device=device, dtype=dtype)
#         return node_attrs.to(device=device, dtype=dtype)

#     def forward(
#         self,
#         x: torch.Tensor,
#         node_attrs: Union[torch.Tensor, None] = None,
#     ) -> torch.Tensor:
#         B = x.size(0)
#         dense = self._to_dense_features(x)
#         y = self._one_hot_elements(node_attrs, B, x.device, x.dtype)
#         output = x.new_zeros(B, self.num_features, self.irrep_dim)

#         for out_m, contraction in enumerate(self.contractions):
#             contracted = contraction(dense, y)
#             dim = so2_dim(out_m)
#             contracted = contracted.view(B, self.num_features, dim)
#             start = self.irrep_offsets[out_m]
#             output[..., start : start + dim] = contracted

#         return self._merge_dense_features(output)

#     def extra_repr(self) -> str:
#         return (
#             f"mmax={self.mmax}, lmax={self.lmax}, "
#             f"channels={self.num_channels}, correlation={self.correlation}, "
#             f"features={self.num_features}, num_paths_by_m={self.num_paths_by_m}"
#         )
