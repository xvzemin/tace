###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from itertools import groupby
from math import sqrt
from typing import Dict

import torch
from e3nn import o3

from equivariantx.o3 import (
    ContractionInstruction,
    GeneralizedContraction,
    TransposeIrrepsLayout,
    UuuFullyConnectedSymmetricContraction,
)
from equivariantx.ops import config as eqx_config
from tace.utils.env import acceleration_enabled, eqx_backend

from ..linear import (
    e3nnElementLinear,
    e3nnLinear,
    has_lora,
    torchLinear,
)
from ..mlp import ACTIVATION
from .base import Product
from .dropout import GraphDropPath
from .fused import UuuSymmetricContraction, UuuTensorProduct
from .paths import (
    SymmetricProductPaths,
    symmetric_contraction_paths,
)


class PathProduct(Product):
    def __init__(self, *args, instructions=None, **kwargs):
        self.instructions = (
            None
            if instructions is None
            else tuple(
                tuple(ContractionInstruction(i, j, o3.Irrep(ir)) for i, j, ir in order)
                for order in instructions
            )
        )
        self.use_generalized_cg = bool(self.use_generalized_cg)
        super().__init__(*args, **kwargs)

    def validate_instructions(self):
        if self.instructions is None:
            return
        if len(self.instructions) != self.correlation - 1:
            raise ValueError(
                "Provide one instruction list per correlation order starting at two."
            )
        for d, (paths, targets) in enumerate(
            zip(self.instructions, self.irreps_tp_out_list), 2
        ):
            allowed = {ir for _, ir in targets}
            if any(ir not in allowed for _, _, ir in paths):
                raise ValueError(
                    f"Instruction output is not an allowed degree-{d} target."
                )

    def build_generalized(self, shared):
        orders = [
            [(i, j, tp.irreps_out[k].ir) for i, j, k, _, _ in tp.instructions]
            for tp in self.aces
        ]
        return GeneralizedContraction(
            self.irreps_hidden,
            orders,
            shared_inputs=shared,
            backend=eqx_backend("product"),
        )


class CgtpACE(PathProduct):
    """Channel-wise ACE with element-dependent or element-independent coefficients.

    Parameters
    ----------
    instructions : sequence of sequences, optional
        Ordered coupling paths for degrees two and above.
    **kwargs
        Parameters of :class:`~tace.models.spherical.base.Product`.
    """

    use_generalized_cg = False

    def _setup(self):
        self.validate_instructions()
        self.use_eqx = acceleration_enabled("eqx", kernel="product")
        for_coefs = {
            "irreps_out": self.irreps_coefs_out,
            "bias": self.use_bias,
        }
        coefs_cls = e3nnLinear if self.agnostic else e3nnElementLinear
        if not self.agnostic:
            for_coefs["num_elements"] = self.num_elements

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        self.coefs.append(
            coefs_cls(
                o3.Irreps(
                    [(self.num_hidden_channel, ir) for _, ir in self.irreps_hidden]
                ).simplify(),
                **for_coefs,
            )
        )
        self.irreps_base = self.irreps_hidden
        product_in1 = self.irreps_hidden
        symmetric_paths = (
            SymmetricProductPaths(product_in1) if self.correlation > 2 else None
        )
        for nu in range(2, self.correlation + 1):
            ace = UuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_base,
                irreps_out=self.irreps_tp_out_list[nu - 2],
                l1l2=self.l1l2,
                trainable=False,
                identical_inputs=nu == 2,
                symmetric_paths=symmetric_paths,
                instructions=None
                if self.instructions is None
                else self.instructions[nu - 2],
            )
            self.aces.append(ace)
            self.coefs.append(
                coefs_cls(
                    o3.Irreps(
                        [(self.num_hidden_channel, ir) for _, ir in ace.irreps_out]
                    ).simplify(),
                    **for_coefs,
                )
            )
            product_in1 = ace.irreps_out

        if self.use_generalized_cg:
            self.generalized = self.build_generalized(True)
        if self.use_eqx and self.aces:
            self.eqx_ace = UuuFullyConnectedSymmetricContraction(
                [ace.tp for ace in self.aces],
                [coef.linear for coef in self.coefs],
            )

        self.linear_up = (
            e3nnLinear(self.irreps_in, self.irreps_hidden, bias=self.use_bias)
            if self.num_channel != self.num_hidden_channel
            else torch.nn.Identity()
        )
        self.linear = e3nnLinear(
            o3.Irreps(
                [(self.num_hidden_channel, ir) for _, ir in self.irreps_coefs_out]
            ),
            self.irreps_out,
            bias=self.use_bias,
        )
        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_type: torch.Tensor,
        sc: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.agnostic:
            for_coefs = {}
        else:
            for_coefs = {"node_type": node_type}

        node_feats = self.linear_up(node_feats)
        corr_feats = node_feats
        if (
            hasattr(self, "eqx_ace")
            and (
                not self.use_generalized_cg
                or (self.eqx_ace.backend == "cuda" and node_feats.is_cuda)
            )
            and not any(has_lora(coef) for coef in self.coefs)
        ):
            weights = []
            for coef in self.coefs:
                if coef.use_matrix_weight:
                    rows = 1 if self.agnostic else self.num_elements
                    weight = torch.cat(
                        [w.reshape(rows, -1) for w in coef.weight], dim=-1
                    )
                else:
                    weight = coef.weight.reshape(
                        1 if self.agnostic else self.num_elements, -1
                    )
                weights.append(weight)
            types = (
                torch.zeros(
                    node_feats.shape[0], dtype=torch.long, device=node_feats.device
                )
                if self.agnostic
                else node_type
            )
            if self.use_generalized_cg and len(self.aces) > 1:
                prefix = self.generalized(node_feats, max_degree=self.correlation - 1)[
                    1:
                ]
                outs = self.eqx_ace(node_feats, weights, types, prefix=prefix)
            else:
                outs = self.eqx_ace(node_feats, weights, types)
            for coef in self.coefs:
                if coef.bias is not None:
                    bias = (
                        coef.bias
                        if self.agnostic
                        else coef.bias.reshape(self.num_elements, -1)[node_type]
                    )
                    outs = (
                        outs + bias.index_select(-1, coef._bias_index) * coef._bias_mask
                    )
        elif self.use_generalized_cg:
            features = self.generalized(node_feats)
            outs = self.coefs[0](features[0], **for_coefs)
            for i, coef in enumerate(self.coefs[1:], 1):
                outs = outs + coef(features[i], **for_coefs)
        else:
            outs = self.coefs[0](corr_feats, **for_coefs)
            for index, ace in enumerate(self.aces):
                corr_feats = ace(corr_feats, node_feats)
                outs = outs + self.coefs[index + 1](corr_feats, **for_coefs)

        outs = self.linear(outs)
        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)
        if sc is not None:
            outs = torch.add(sc, outs, alpha=self.residual_scale)
        return outs


class MaceACE(Product):
    """Channel-wise symmetric contraction with optional input projection.

    Parameters
    ----------
    **kwargs
        Parameters of :class:`~tace.models.spherical.base.Product`. ``correlation``
        selects the maximum polynomial degree. ``num_channel_per_expert``
        sets the contraction width, defaulting to ``num_channel``.
        An input Linear is used only when these widths differ.
        ``agnostic=True`` shares contraction coefficients across elements.
        ``l1l2`` must be ``None``; explicit selection of MACE symmetric
        contraction paths is not currently supported.

    Notes
    -----
    Select with ``product_basis.type: mace``. The contraction backend follows
    ``TACE_USE_EQX`` and ``TACE_USE_CUE``, in that priority order.
    Expert mixtures and bilinear gates are not supported.
    """

    def _setup(self):
        from .symmetric_contraction import U_matrix_real

        if self.l1l2 is not None:
            raise NotImplementedError(
                "Explicit selection of MACE symmetric contraction paths is not "
                "currently supported. Set product_basis.l1l2 to null."
            )
        if self.correlation < 1:
            raise ValueError("MACE products require a positive correlation.")
        if self.num_expert != 1 or self.use_shared_expert:
            raise ValueError("MACE products do not support expert mixtures.")
        if self.nonlinear is not None:
            raise ValueError("MACE products require nonlinear=None.")

        self.linear_up = (
            e3nnLinear(self.irreps_in, self.irreps_hidden, bias=self.use_bias)
            if self.num_channel != self.num_hidden_channel
            else torch.nn.Identity()
        )
        self.reshape = TransposeIrrepsLayout(
            self.irreps_hidden,
            source="mul_ir",
            target="mul_ir",
            flat_out=False,
            backend=eqx_backend("product"),
        )
        coupling_irreps = o3.Irreps([(1, ir) for _, ir in self.irreps_hidden])
        tensors, weights = [], []
        num_elements = 1 if self.agnostic else self.num_elements
        for output, (_, ir) in enumerate(self.irreps_coefs_out):
            for degree in range(self.correlation, 0, -1):
                tensor = U_matrix_real(
                    coupling_irreps,
                    o3.Irreps([(1, ir)]),
                    degree,
                    dtype=torch.float64,
                    use_cueq_cg=False,
                )[-1]
                if not tensor.any():
                    continue
                num_paths = tensor.shape[-1]
                tensors.append((output, degree, tensor))
                weights.append(
                    torch.randn(num_elements, num_paths, self.num_hidden_channel)
                    / num_paths
                )
        self.product = UuuSymmetricContraction(
            self.irreps_hidden,
            self.irreps_coefs_out,
            tensors,
        )
        self.coupling_blocks = tuple(
            (self.irreps_coefs_out[output].ir, degree, tensor.shape[-1])
            for output, degree, tensor in tensors
        )
        self.weight = torch.nn.Parameter(
            torch.cat(weights, dim=1)
            if weights
            else torch.empty(num_elements, 0, self.num_hidden_channel)
        )
        self.linear = e3nnLinear(
            self.irreps_coefs_out, self.irreps_out, bias=self.use_bias
        )
        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_type: torch.Tensor,
        sc: torch.Tensor | None,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.agnostic:
            node_type = torch.zeros(
                node_feats.shape[0], dtype=torch.long, device=node_feats.device
            )
        features = self.reshape(self.linear_up(node_feats))
        output = self.linear(self.product(features, self.weight, node_type))
        if hasattr(self, "stochastic_depth"):
            output = self.stochastic_depth(output, batch)
        if sc is not None:
            output = torch.add(sc, output, alpha=self.residual_scale)
        return output


class NonlinearCgtpACE(PathProduct):
    """Channel-wise recursive products with invariant path gates.

    Parameters
    ----------
    **kwargs
        Parameters of :class:`~tace.models.spherical.base.Product`.
    instructions : sequence of sequences, optional
        Ordered coupling paths for degrees two and above.
    """

    asymmetric = True
    use_generalized_cg = False
    _compiled_cg_rows = True

    def _setup(self) -> None:
        self.validate_instructions()
        self.asymmetric = bool(self.asymmetric)
        if self.correlation < 1:
            raise ValueError("Nonlinear products require a positive correlation.")
        if self.num_expert != 1 or self.use_shared_expert:
            raise ValueError("Nonlinear products do not support expert mixtures.")
        if self.nonlinear is not None:
            raise ValueError("Nonlinear products require nonlinear=None.")
        if not any(ir.is_scalar() for _, ir in self.irreps_in):
            raise ValueError("Nonlinear products require invariant scalar inputs.")

        self.num_inputs = self.correlation if self.asymmetric else 1
        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        irreps = self.irreps_hidden
        paths = SymmetricProductPaths(irreps)
        self.gate_slices = []
        self.weight_numel = 0
        for degree in range(1, self.correlation + 1):
            if degree > 1:
                product = UuuTensorProduct(
                    irreps,
                    self.irreps_hidden,
                    self.irreps_tp_out_list[degree - 2],
                    l1l2=self.l1l2,
                    symmetric_paths=paths,
                    instructions=None
                    if self.instructions is None
                    else self.instructions[degree - 2],
                )
                self.aces.append(product)
                irreps = product.irreps_out
            self.coefs.append(
                e3nnLinear(irreps.simplify(), self.irreps_coefs_out, bias=self.use_bias)
            )
            offset = sum(mul for mul, _ in irreps)
            self.gate_slices.append((self.weight_numel, self.weight_numel + offset))
            self.weight_numel += offset

        self.factor_dim = self.irreps_hidden.dim
        irreps_up = (
            o3.Irreps([(self.num_hidden_channel, "0e")])
            + self.irreps_hidden * self.num_inputs
        )
        slices = irreps_up.slices()
        order = torch.cat(
            [
                torch.arange(slices[i].start, slices[i].stop)
                for i in irreps_up.sort().inv
            ]
        )
        self.register_buffer("factor_index", order.argsort(), persistent=False)
        for degree, irreps in enumerate(
            [self.irreps_hidden, *(tp.irreps_out for tp in self.aces)], start=1
        ):
            indices, offset = [], 0
            for mul, ir in irreps:
                indices.extend(
                    offset + channel for channel in range(mul) for _ in range(ir.dim)
                )
                offset += mul
            self.register_buffer(
                f"gate_index_{degree}",
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )

        self.linear_up = e3nnLinear(
            self.irreps_in.simplify(),
            irreps_up.sort().irreps.simplify(),
            bias=self.use_bias,
        )
        self.embedding = torch.nn.Embedding(self.num_elements, self.num_hidden_channel)
        self.gate = torch.nn.Sequential(
            torchLinear(
                2 * self.num_hidden_channel, self.weight_numel, bias=self.use_bias
            ),
            ACTIVATION["scaled_silu"](),
        )
        self.linear = e3nnLinear(
            self.irreps_coefs_out, self.irreps_out, bias=self.use_bias
        )
        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)
        if self.use_generalized_cg:
            self.generalized = self.build_generalized(not self.asymmetric)
        self._configure_eqx()

    def _configure_eqx(self) -> None:
        """Prepare the optional fusion only if selected for this representation."""
        if acceleration_enabled("eqx", kernel="product") and (
            not self.use_generalized_cg or self.generalized.codegen_supported
        ):
            self.build_eqx()
        else:
            self._clear_eqx()

    def _clear_eqx(self):
        for name in ("eqx_metadata", "eqx_bindings", "eqx_products"):
            if hasattr(self, name):
                delattr(self, name)

    def build_eqx(self) -> None:
        """Build the fused product from its operators and parameters."""
        from equivariantx_foundation.models.tace.nonlinear_ace import (
            build,
            build_product,
        )

        self.eqx_metadata, self.eqx_bindings = build(self)
        if not self.use_generalized_cg:
            self.eqx_products = tuple(
                (build_product(product.tp), product.irreps_out.dim)
                for product in self.aces
            )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_type: torch.Tensor,
        sc: torch.Tensor | None,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hasattr(self, "eqx_metadata")
            and node_feats.is_cuda
            and eqx_config.DEFAULT_BACKEND == "cuda"
            and node_feats.dtype in (torch.float32, torch.float64)
            and acceleration_enabled("eqx", kernel="product")
            # Inductor reuses the native factors across force derivatives.
            and not (self.training and acceleration_enabled("compile"))
            and not any(
                has_lora(layer)
                for layer in (self.linear_up, self.gate[0], *self.coefs, self.linear)
            )
        ):
            from equivariantx_foundation.models.tace.nonlinear_ace import run

            output = run(self, node_feats, self.embedding(node_type))
        else:
            projected = self.linear_up(node_feats).index_select(-1, self.factor_index)
            scalars = projected[:, : self.num_hidden_channel]
            weight = self.gate(torch.cat((scalars, self.embedding(node_type)), dim=-1))
            factors = projected[:, self.num_hidden_channel :].reshape(
                node_feats.shape[0], self.num_inputs, self.factor_dim
            )
            corr = factors[:, 0]
            start, stop = self.gate_slices[0]
            output = self.coefs[0](corr * weight[:, start:stop][:, self.gate_index_1])
            fused_products = (
                self._compiled_cg_rows
                and hasattr(self, "eqx_products")
                and not self.use_generalized_cg
                and node_feats.is_cuda
                and node_feats.dtype in (torch.float32, torch.float64)
                and acceleration_enabled("eqx", kernel="product")
                and self.training
                and acceleration_enabled("compile")
            )
            if fused_products:
                from equivariantx.ops.common.channel_product import channel_product
            expanded = (
                self.generalized(*factors.unbind(1)) if self.use_generalized_cg else ()
            )
            for index, (product, linear) in enumerate(zip(self.aces, self.coefs[1:])):
                factor = factors[:, index + 1 if self.asymmetric else 0]
                if self.use_generalized_cg:
                    corr = expanded[index + 1]
                elif fused_products:
                    metadata, output_dim = self.eqx_products[index]
                    corr = channel_product(
                        metadata,
                        repr((((0, 1, 2), 2, 0, 0),)),
                        [corr, factor, corr.new_empty((0, output_dim))],
                    )[0]
                else:
                    corr = product(corr, factor)
                start, stop = self.gate_slices[index + 1]
                gate = weight[:, start:stop].index_select(
                    -1, getattr(self, f"gate_index_{index + 2}")
                )
                output = output + linear(corr * gate)
            output = self.linear(output)
        if hasattr(self, "stochastic_depth"):
            output = self.stochastic_depth(output, batch)
        if sc is not None:
            output = torch.add(sc, output, alpha=self.residual_scale)
        return output


class TaceACE(PathProduct):
    """Contract independently projected factors with invariant path coefficients.

    Parameters
    ----------
    instructions : sequence of sequences, optional
        Couplings per order starting at two, indexed in the previous order's
        output. Defaults to :func:`~tace.models.spherical.paths.symmetric_contraction_paths`.
    **kwargs
        Parameters of :class:`~tace.models.spherical.base.Product`.
    """

    use_generalized_cg = True
    fully_connected = False
    coefficient_workspace = 16 << 20

    def _setup(self):
        self.validate_instructions()
        if self.correlation < 1 or self.num_expert != 1 or self.use_shared_expert:
            raise ValueError(
                "TACE requires a positive correlation and a single expert."
            )
        if self.nonlinear is not None or not any(
            ir.is_scalar() for _, ir in self.irreps_in
        ):
            raise ValueError(
                "TACE requires invariant scalar inputs and nonlinear=None."
            )
        orders = self.instructions
        if orders is None:
            orders = symmetric_contraction_paths(
                self.irreps_hidden, self.irreps_tp_out_list, l1l2=self.l1l2
            )
        c = self.num_hidden_channel
        self.fully_connected = bool(self.fully_connected)
        irreps = [self.irreps_hidden]
        self.aces = torch.nn.ModuleList()
        for stage, paths in enumerate(orders):
            if not self.use_generalized_cg:
                product = UuuTensorProduct(
                    irreps[-1],
                    self.irreps_hidden,
                    self.irreps_tp_out_list[stage],
                    instructions=paths,
                )
                for module in product.modules():
                    module._non_persistent_buffers_set.update(module._buffers.keys())
                self.aces.append(product)
            irreps.append(o3.Irreps([(c, ir) for _, _, ir in paths]))
        if self.use_generalized_cg:
            self.generalized = GeneralizedContraction(
                self.irreps_hidden,
                orders,
                backend=eqx_backend("product"),
                gated_output=None if self.fully_connected else self.irreps_coefs_out,
            )
        else:
            self.layouts = torch.nn.ModuleList(
                [
                    TransposeIrrepsLayout(
                        order,
                        source="mul_ir",
                        target="ir_mul",
                        flat_out=False,
                        backend=eqx_backend("product"),
                    )
                    if order
                    else torch.nn.Identity()
                    for order in irreps
                ]
            )
        self.angular_dim = self.irreps_hidden.dim // c
        groups = {ir: [] for _, ir in self.irreps_coefs_out}
        self.weight_numel = 0
        for stage, order in enumerate(irreps):
            offset = 0
            for ir, run in groupby(order, key=lambda block: block.ir):
                paths = sum(1 for _ in run)
                if ir in groups:
                    width = paths * (c * c if self.fully_connected else 1)
                    groups[ir].append((stage, offset, paths, self.weight_numel, width))
                    self.weight_numel += width
                offset += paths * ir.dim
        self.groups = tuple(
            (
                ir.dim,
                tuple(groups[ir]),
                1
                / sqrt(
                    max(
                        1,
                        sum(b[2] for b in groups[ir])
                        * (c if self.fully_connected else 1),
                    )
                ),
            )
            for _, ir in self.irreps_coefs_out
        )
        up = (
            o3.Irreps([(self.weight_numel, "0e")])
            + self.irreps_hidden * self.correlation
        )
        sort = up.sort()
        starts = [0] * len(up)
        offset = 0
        for index in sort.inv:
            starts[index] = offset
            offset += up[index].dim
        self.gate_start = starts[0]
        indices = []
        for i, (_, ir) in enumerate(up[1:], 1):
            block = torch.arange(c * ir.dim).reshape(c, ir.dim)
            if self.use_generalized_cg:
                block = block.T
            indices.append(starts[i] + block.reshape(-1))
        self.register_buffer("factor_index", torch.cat(indices), persistent=False)
        source = self.irreps_in.simplify()
        sort_in = source.sort()
        grouped = sort_in.irreps.simplify()
        input_index = None
        if len(source) != len(grouped):
            slices = source.slices()
            input_index = torch.cat(
                [torch.arange(slices[i].start, slices[i].stop) for i in sort_in.inv]
            )
            source = grouped
        self.register_buffer("input_index", input_index, persistent=False)
        self.linear_up = e3nnLinear(source, sort.irreps.simplify(), bias=self.use_bias)
        self.activation = ACTIVATION["scaled_silu"]()
        self.linear = e3nnLinear(
            self.irreps_coefs_out, self.irreps_out, bias=self.use_bias
        )
        if (
            self.use_generalized_cg
            and self.generalized.codegen_supported
            and not self.fully_connected
            and acceleration_enabled("eqx", kernel="product")
        ):
            from equivariantx_foundation.models.tace.nonlinear_ace import (
                build_path_gates,
            )

            self.eqx_metadata, self.eqx_bindings = build_path_gates(self)
        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

    def contract(self, x):
        n, c = x.shape[0], self.num_hidden_channel
        projected = self.linear_up(x)
        gates = self.activation(projected.narrow(1, self.gate_start, self.weight_numel))
        raw = projected.index_select(1, self.factor_index)
        if self.use_generalized_cg:
            factors = raw.reshape(n, self.correlation, self.angular_dim, c).unbind(1)
            folded = self.generalized.gated
            count = self.generalized.gate_count
            values = self.generalized.packed(
                factors,
                gates.narrow(1, self.weight_numel - count, count) if folded else None,
            )
        else:
            folded = False
            factors = raw.reshape(n, self.correlation, self.angular_dim * c).unbind(1)
            current = factors[0]
            values = [self.layouts[0](current)]
            for i, (tp, layout) in enumerate(zip(self.aces, self.layouts[1:]), 1):
                current = tp(current, factors[i])
                values.append(layout(current))
        parts = []
        for j, (dim, blocks, scale) in enumerate(self.groups):
            total = None
            if folded:
                offset, tiles = self.generalized.gate_slices[j]
                if tiles:
                    part = values[-1][:, offset : offset + tiles * dim]
                    total = (
                        part if tiles == 1 else part.reshape(n, tiles, dim, c).sum(1)
                    )
            for stage, offset, paths, start, width in blocks:
                if folded and stage == self.correlation - 1:
                    continue
                value = values[stage][:, offset : offset + paths * dim]
                weight = gates.narrow(1, start, width)
                if self.fully_connected:
                    value = (
                        value.reshape(n, paths, dim, c)
                        .transpose(1, 2)
                        .reshape(n, dim, paths * c)
                    )
                    result = torch.bmm(value, weight.reshape(n, paths * c, c))
                else:
                    result = torch.bmm(
                        weight.reshape(n, 1, paths), value.reshape(n, paths, dim * c)
                    ).reshape(n, dim, c)
                total = result if total is None else total + result
            if total is None:
                total = x.new_zeros((n, dim, c))
            parts.append((total * scale).transpose(1, 2).flatten(1))
        return self.linear(torch.cat(parts, 1))

    def forward(self, node_feats, node_type, sc, batch):
        if self.input_index is not None:
            node_feats = node_feats.index_select(1, self.input_index)
        if (
            hasattr(self, "eqx_metadata")
            and node_feats.is_cuda
            and eqx_config.DEFAULT_BACKEND == "cuda"
            and acceleration_enabled("eqx", kernel="product")
            and not (self.training and acceleration_enabled("compile"))
            and not any(has_lora(linear) for linear in (self.linear_up, self.linear))
        ):
            from equivariantx_foundation.models.tace.nonlinear_ace import run

            outs = run(self, node_feats)
        elif self.fully_connected and self.weight_numel:
            from torch.utils.checkpoint import checkpoint

            tile = max(
                1,
                self.coefficient_workspace
                // (node_feats.element_size() * self.weight_numel),
            )
            if node_feats.shape[0] > tile:
                blocks = node_feats.split(tile)
                outs = torch.cat(
                    [
                        checkpoint(self.contract, v, use_reentrant=False)
                        if torch.is_grad_enabled()
                        else self.contract(v)
                        for v in blocks
                    ]
                )
            else:
                outs = self.contract(node_feats)
        else:
            outs = self.contract(node_feats)
        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)
        return (
            torch.add(sc, outs, alpha=self.residual_scale) if sc is not None else outs
        )


PRODUCT: Dict[str, torch.nn.Module] = {
    "cgtp": CgtpACE,
    "mace": MaceACE,
    "nonlinear_cgtp": NonlinearCgtpACE,
    "tace": TaceACE,
}

product_aliases = {
    "coupled": "cgtp",
}

PRODUCT.update({alias: PRODUCT[target] for alias, target in product_aliases.items()})
