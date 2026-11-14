###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Union

import torch
from e3nn import o3

from equivariantx.mbe.ace import UuuFullyConnectedSymmetricContraction
from tace.utils.env import acceleration_enabled, select_acceleration

from ..layout import LayoutTransform
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
from .paths import SymmetricProductPaths


class CgtpACE(Product):
    """Channel-wise ACE with element-dependent or element-independent coefficients."""

    def _setup(self):
        self.use_eqx = select_acceleration("eqx", "eqt", kernel="product") == "eqx"
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
                warning=self.correlation > 2 and self.layer == 0 and not self.use_eqx,
                use_fused=self.correlation > 2
                and not self.use_time_reversal
                and not self.use_eqx,
                symmetric_paths=symmetric_paths,
                use_eqt=False if self.use_eqx else None,
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

        if self.use_eqx and self.aces:
            self.eqx_ace = UuuFullyConnectedSymmetricContraction(
                [ace.tp for ace in self.aces],
                [coef.linear for coef in self.coefs],
                backend="cuda",
            )

        self.linear_up = (
            e3nnLinear(
                self.irreps_in, self.irreps_hidden, bias=self.use_bias
            )
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
            and node_feats.is_cuda
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
        else:
            outs = self.coefs[0](corr_feats, **for_coefs)
            for index, ace in enumerate(self.aces):
                corr_feats = ace(corr_feats, node_feats)
                outs = outs + self.coefs[index + 1](corr_feats, **for_coefs)

        outs = self.linear(outs)
        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)
        if sc is not None:
            outs = outs + sc
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
            e3nnLinear(
                self.irreps_in, self.irreps_hidden, bias=self.use_bias
            )
            if self.num_channel != self.num_hidden_channel
            else torch.nn.Identity()
        )
        self.reshape = LayoutTransform(self.irreps_hidden, layout_out="mul_ir")
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
            output = output + sc
        return output


class NonlinearCgtpACE(Product):
    """Channel-wise recursive products with invariant path gates.

    Parameters
    ----------
    **kwargs
        Parameters of :class:`~tace.models.spherical.base.Product`.

    Notes
    -----
    Select with ``product_basis.type: nonlinear_cgtp``. Linear up produces
    independent factors and invariant scalars. Scalars and element embeddings
    generate one scaled-SiLU coefficient per path and channel. Ordinary Linear
    maps mix the gated products. Only symmetric-product paths are retained.
    Set the class attribute ``asymmetric=False`` before construction to reuse
    one factor at every order. No additional model parameter is required.
    """

    asymmetric = True

    def _setup(self) -> None:
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
            [torch.arange(slices[i].start, slices[i].stop) for i in irreps_up.sort().inv]
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
            self.irreps_in.simplify(), irreps_up.regroup(), bias=self.use_bias
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
        self.build_eqx()

    def build_eqx(self) -> None:
        """Build the fused product from its operators and parameters."""
        from equivariantx.models.tace.nonlinear_ace import build, build_product

        self.eqx_metadata, self.eqx_bindings = build(self)
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
            node_feats.is_cuda
            and node_feats.dtype in (torch.float32, torch.float64)
            and acceleration_enabled("eqx", kernel="product")
            # Inductor reuses the native factors across force derivatives.
            and not (self.training and acceleration_enabled("compile"))
            and not any(
                has_lora(layer)
                for layer in (self.linear_up, self.gate[0], *self.coefs, self.linear)
            )
        ):
            from equivariantx.models.tace.nonlinear_ace import run

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
                node_feats.is_cuda
                and node_feats.dtype in (torch.float32, torch.float64)
                and acceleration_enabled("eqx", kernel="product")
                and self.training
                and acceleration_enabled("compile")
            )
            if fused_products:
                from equivariantx.ops.common.channel_product import channel_product
            for index, (product, linear) in enumerate(zip(self.aces, self.coefs[1:])):
                factor = factors[:, index + 1 if self.asymmetric else 0]
                if fused_products:
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
        return output if sc is None else output + sc


PRODUCT: Dict[str, torch.nn.Module] = {
    "cgtp": CgtpACE,
    "mace": MaceACE,
    "nonlinear_cgtp": NonlinearCgtpACE,
}

product_aliases = {
    "coupled": "cgtp",
}

PRODUCT.update(
    {
        alias: PRODUCT[target]
        for alias, target in product_aliases.items()
    }
)
