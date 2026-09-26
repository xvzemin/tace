################################################################################
# Authors: Ilyes Batatia
# Symmetric contraction algorithm presented in the MACE paper
################################################################################

import collections
import dataclasses
import itertools
from typing import Dict, List, Optional, Union, Iterator, Union

import numpy as np
import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
try:
    import cuequivariance as cue
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False


_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
    use_cueq_cg=True,
    use_nonsymmetric_product=False,
):
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation

    if correlation == 4 and not use_cueq_cg:
        filter_ir_mid = [(i, 1 if i % 2 == 0 else -1) for i in range(12)]
    if use_cueq_cg and CUET_AVAILABLE:
        return compute_U_cueq(  # pylint: disable=possibly-used-before-assignment
            irreps_in, irreps_out=irreps_out, correlation=correlation, dtype=dtype
        )

    try:
        wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    except NotImplementedError as e:
        if CUET_AVAILABLE:
            return compute_U_cueq(  # pylint: disable=possibly-used-before-assignment
                irreps_in,
                irreps_out=irreps_out,
                correlation=correlation,
                use_nonsymmetric_product=use_nonsymmetric_product,
                dtype=dtype,
            )
        raise NotImplementedError(
            "The requested Clebsch-Gordan coefficients are not implemented, please install cuequivariance; pip install cuequivariance"
        ) from e

    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    try:
        out += [last_ir, stack]
    except:  # pylint: disable=bare-except
        first_dim = irreps_out.dim
        if first_dim != 1:
            size = [first_dim] + [o3.Irreps(irreps_in).dim] * correlation + [1]
        else:
            size = [o3.Irreps(irreps_in).dim] * correlation + [1]
        out = [str(irreps_out)[:-2], torch.zeros(size, dtype=dtype)]
    return out


if CUET_AVAILABLE:
    def compute_U_cueq(
        irreps_in, irreps_out, correlation=2, use_nonsymmetric_product=False, dtype=None
    ):
        if dtype is None:
            dtype = torch.get_default_dtype()
        U = []
        irreps_in = cue.Irreps(O3_e3nn, str(irreps_in))
        irreps_out = cue.Irreps(O3_e3nn, str(irreps_out))
        for _, ir in irreps_out:
            try:
                U_matrix_full_symm = cue.reduced_symmetric_tensor_product_basis(
                    irreps_in,
                    correlation,
                    keep_ir=ir,
                    layout=cue.ir_mul,
                )
                U_matrix_full_symm = U_matrix_full_symm.array
                if use_nonsymmetric_product:
                    try:
                        U_matrix_full_antisymmetric = (
                            cue.reduced_antisymmetric_tensor_product_basis(
                                irreps_in,
                                correlation,
                                keep_ir=ir,
                                layout=cue.ir_mul,
                            ).array
                        )
                        U_matrix_full = torch.cat(
                            (U_matrix_full_symm, U_matrix_full_antisymmetric), dim=-1
                        )
                    except ValueError:
                        continue
                else:
                    U_matrix_full = U_matrix_full_symm

            except ValueError:
                if ir.dim == 1:
                    out_shape = (*([irreps_in.dim] * correlation), 1)
                else:
                    out_shape = (ir.dim, *([irreps_in.dim] * correlation), 1)
                return [
                    torch.zeros(
                        out_shape,
                        dtype=torch.get_default_dtype(),
                    )
                ]
            if U_matrix_full.shape[-1] == 0:
                if ir.dim == 1:
                    out_shape = (*([irreps_in.dim] * correlation), 1)
                else:
                    out_shape = (ir.dim, *([irreps_in.dim] * correlation), 1)
                return [
                    torch.zeros(
                        out_shape,
                        dtype=torch.get_default_dtype(),
                    )
                ]
            ir_str = str(ir)
            U.append(ir_str)
            U_matrix_full = torch.tensor(
                U_matrix_full.reshape(*([irreps_in.dim] * correlation), ir.dim, -1),
                dtype=dtype,
            )
            U_matrix_full = torch.moveaxis(U_matrix_full, -2, 0)
            if ir.dim == 1:
                U_matrix_full = U_matrix_full[0]
            U.append(U_matrix_full)
        return U

    class O3_e3nn(cue.O3):
        def __mul__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> Iterator["O3_e3nn"]:
            return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

        @classmethod
        def clebsch_gordan(
            cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                    rep3.dim
                )
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> bool:
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator["O3_e3nn"]:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)

else:
    class O3_e3nn:
        pass
    print(
        "cuequivariance or cuequivariance_torch is not available. Cuequivariance acceleration will be disabled."
    )


BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        use_reduced_cg: bool = False,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                    use_reduced_cg=use_reduced_cg,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        use_reduced_cg: bool = False,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()

        path_weight = []
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                use_cueq_cg=use_reduced_cg,
                dtype=dtype,
            )[-1]
            path_weight.append(not torch.equal(U_matrix, torch.zeros_like(U_matrix)))
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                # Optimizing the contractions
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
            else:
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                # Symbolic tracing of contractions
                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                # Optimizing the contractions
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)

        for idx, keep in enumerate(path_weight):
            zero_flag = not keep
            if idx < correlation - 1:
                if zero_flag:
                    self.weights[idx] = EmptyParam(self.weights[idx])
                self.register_buffer(
                    f"weights_{idx}_zeroed",
                    torch.tensor(zero_flag, dtype=torch.bool),
                )
            else:
                if zero_flag:
                    self.weights_max = EmptyParam(self.weights_max)
                self.register_buffer(
                    "weights_max_zeroed",
                    torch.tensor(zero_flag, dtype=torch.bool),
                )

        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        out = self.graph_opt_main(
            self.U_tensors(self.correlation),
            self.weights_max,
            x,
            y,
        )
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)

        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]


class EmptyParam(torch.nn.Parameter):
    def __new__(cls, data):  # pylint: disable=signature-differs
        zero = torch.zeros_like(data)
        return super().__new__(cls, zero, requires_grad=False)

    def requires_grad_(
        self, mode: bool = True
    ):  # pylint: disable=arguments-differ, arguments-renamed
        del mode
        return self


@dataclasses.dataclass
class CuEquivarianceConfig:
    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    layout_str: str = "mul_ir"
    group: str = "O3"

cueq_config  = CuEquivarianceConfig()


class SymmetricContractionWrapper:
    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        use_reduced_cg: bool = True,
        use_cueq: bool = False,
    ):

        if use_cueq:
            assert use_reduced_cg
            import cuequivariance as cue
            import cuequivariance_torch as cuet
            return cuet.SymmetricContraction(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout_in=cue.ir_mul,
                layout_out=cueq_config.layout,
                contraction_degree=correlation,
                num_elements=num_elements,
                original_mace=(not use_reduced_cg),
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )

        return SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
        )
