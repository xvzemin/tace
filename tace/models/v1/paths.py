################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import itertools
from typing import Optional, List, NamedTuple

import torch
import opt_einsum as oe

from .utils import satisfy

def generate_combinations(
    max_r_1: int,
    max_r_2: int,
    max_r_o: int,
    restriction: Optional[str] = None,
    allow_nosym: bool = False,
):
    combs = []

    for r_1 in range(max_r_1 + 1):
        for r_2 in range(max_r_2 + 1):
            if satisfy(r_1, r_2, restriction):
                for r_o in range(abs(r_2 - r_1), min(max_r_o, r_2 + r_1) + 1, 2):
                    k = (r_1 + r_2 - r_o) // 2
                    if allow_nosym:
                        combs.append((r_1, r_2, r_o))
                    else:
                        if k == r_1 or k == r_2:
                            combs.append((r_1, r_2, r_o))
    return combs


class TC(NamedTuple):
    r_1: int
    r_2: int
    r_o: int
    k: int
    axes_1: List[int]
    axes_2: List[int]


def parse_einsum_expr(expr: str) -> TC:

    inputs, output = expr.split("->")
    in1, in2 = [x.strip() for x in inputs.split(",")]

    # exclude B and C
    X = in1[2:]
    Y = in2[2:]
    Z = output[2:]

    common = set(X) & set(Y)
    contracted = common - set(Z)

    axes_1 = [X.index(c) for c in contracted]
    axes_2 = [Y.index(c) for c in contracted]

    return TC(axes_1=axes_1, axes_2=axes_2, r_1=len(X), r_2=len(Y), r_o=len(Z), k=len(axes_1))


def return_tcs(exprs: List[str]) -> List[TC]:
    return [parse_einsum_expr(expr) for expr in exprs]


class TensorContractionUtils:

    @staticmethod
    def generate_paths(
        r_1: int,
        r_2: int,
        r_o: int,
        add_batch_and_channel: bool = True,
        allow_nosym: bool = False,
        max_paths: Optional[int] = None,
    ):

        assert isinstance(r_1, int) and r_1 >= 0, "r_1 must be a non-negative integer"
        assert isinstance(r_2, int) and r_2 >= 0, "r_2 must be a non-negative integer"
        assert isinstance(r_o, int) and r_o >= 0, "k must be a non-negative integer"
        assert (r_1 + r_2 - r_o) % 2 == 0, "Incompatible target rank"

        k = (r_1 + r_2 - r_o) // 2

        einsum_str = list("defghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        X_labels = einsum_str[:r_1]
        Y_labels = einsum_str[r_1 : r_1 + r_2]

        # === always first ===
        target_X_idx = list(range(r_1 - k, r_1))
        target_Y_idx = list(range(k))
        target_X_copy = X_labels.copy()
        target_Y_copy = Y_labels.copy()
        for i in range(k):
            target_Y_copy[target_Y_idx[i]] = target_X_copy[target_X_idx[i]]

        X_str = "".join(target_X_copy)
        Y_str = "".join(target_Y_copy)

        Z_labels = [l for i, l in enumerate(target_X_copy) if i not in target_X_idx] + [
            l for i, l in enumerate(target_Y_copy) if l not in target_X_copy
        ]
        Z_str = "".join(Z_labels)

        if add_batch_and_channel:
            X_str = "bc" + X_str
            Y_str = "bc" + Y_str
            Z_str = "bc" + Z_str

        target_expr = f"{X_str},{Y_str}->{Z_str}"
        einsum_exprs = [target_expr]

        if allow_nosym:
            for X_idx in itertools.combinations(range(r_1), k):
                for Y_idx in itertools.combinations(range(r_2), k):
                    for perm in itertools.permutations(range(k)):
                        X_copy = X_labels.copy()
                        Y_copy = Y_labels.copy()

                        for i in range(k):
                            Y_copy[Y_idx[perm[i]]] = X_copy[X_idx[i]]

                        X_str = "".join(X_copy)
                        Y_str = "".join(Y_copy)

                        Z_labels = [
                            l for i, l in enumerate(X_copy) if i not in X_idx
                        ] + [l for i, l in enumerate(Y_copy) if l not in X_copy]
                        Z_str = "".join(Z_labels)

                        if add_batch_and_channel:
                            X_str = "bc" + X_str
                            Y_str = "bc" + Y_str
                            Z_str = "bc" + Z_str

                        einsum_str = f"{X_str},{Y_str}->{Z_str}"

                        if einsum_str == target_expr:
                            continue

                        if max_paths is None:
                            einsum_exprs.append(einsum_str)
                        elif len(einsum_exprs) < max_paths:
                            einsum_exprs.append(einsum_str)
            return einsum_exprs
        else:
            return einsum_exprs

    @staticmethod
    def random_test(r_1, r_2, r_o, add_batch_and_channel=True, backend="oe"):

        X_shape = [3] * r_1
        Y_shape = [3] * r_2
        X = torch.randn(6, 7, *X_shape)
        Y = torch.randn(6, 7, *Y_shape)

        einsum_exprs = TensorContractionUtils.generate_paths(
            r_1, r_2, r_o, add_batch_and_channel, True
        )

        results = []
        for expr in einsum_exprs:
            if backend == "oe":
                result = oe.contract(expr, X, Y, backend="torch")
            else:
                result = torch.einsum(expr, X, Y)
            results.append((expr, result.shape, result))
        return results


def generate_prod_paths(
    max_left, max_right, max_hidden, max_rank_of_in, rank_of_out, correlation, allow_nosym, restriction
):
    paths_list_list = []
    exprs_list_list = []
    current_ranks = set(range(max_rank_of_in + 1))

    for v in range(correlation - 1):
        paths_list = []
        exprs_list = []
        next_ranks = set()

        for r_1 in current_ranks:
            for r_2 in range(max_rank_of_in + 1):
                r_min = abs(r_1 - r_2)
                r_max = min(max_rank_of_in, r_1 + r_2)
                for r_o in range(r_min, r_max + 1, 2):
                        if satisfy(r_1, r_2, restriction, r_o):
                            k = (r_1 + r_2 - r_o) // 2
                            if allow_nosym:
                                paths_list.append((r_1, r_2, r_o))
                                next_ranks.add(r_o)
                                exprs_list.append(
                                    TensorContractionUtils.generate_paths(
                                        r_1=r_1,
                                        r_2=r_2,
                                        r_o=r_o,
                                        add_batch_and_channel=True,
                                        allow_nosym=True,
                                        max_paths=1,
                                    )
                                )
                            else:
                                if k == r_1 or k == r_2:
                                    paths_list.append((r_1, r_2, r_o))
                                    next_ranks.add(r_o)
                                    exprs_list.append(
                                        TensorContractionUtils.generate_paths(
                                            r_1=r_1,
                                            r_2=r_2,
                                            r_o=r_o,
                                            add_batch_and_channel=True,
                                            allow_nosym=False,
                                            max_paths=1,
                                        )
                                )

                                        
        paths_list_list.append(paths_list)
        exprs_list_list.append(exprs_list)
        current_ranks = next_ranks

    valid_ranks = rank_of_out
    for v in reversed(range(correlation - 1)):
        filtered_paths = []
        filtered_exprs = []
        next_valid_ranks = set()

        for (r_1, r_2, r_o), exprs in zip(
            paths_list_list[v], exprs_list_list[v]
        ):
            if r_o in valid_ranks:
                filtered_paths.append((r_1, r_2, r_o))
                filtered_exprs.append(exprs)
                next_valid_ranks.update([r_1, r_2])

        paths_list_list[v] = filtered_paths
        exprs_list_list[v] = filtered_exprs
        valid_ranks = next_valid_ranks


    for v in (range(correlation - 1)):
        filtered_paths = []
        filtered_exprs = []
        for (r_1, r_2, r_o), exprs in zip(
            paths_list_list[v], exprs_list_list[v]
        ):
            if r_1 <= max_left[v+1] and r_2 <= max_right[v+1] and r_o <= max_hidden[v+1]:
                filtered_paths.append((r_1, r_2, r_o))
                filtered_exprs.append(exprs) 
        paths_list_list[v] = filtered_paths
        exprs_list_list[v] = filtered_exprs

    return paths_list_list, exprs_list_list


