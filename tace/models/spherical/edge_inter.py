################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

import equivariantx as eqx
from tace.utils.env import acceleration_enabled

from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import BATCH_MLP, MLP
from ..time_reversal import contains_time_odd_irreps
from .base import EdgeInteraction
from .fused import UvO2TensorProduct, UvuTensorProduct
from .paths import generate_paths


class _RowwisePairProduct(UvuTensorProduct):
    """Couple one directed edge per row, without a node reduction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_rowwise()
        self._certify_cg()
        self._cg_load_hook = self.register_load_state_dict_post_hook(_refresh_pair_cg)

    def _certify_cg(self):
        # The EQX row program owns canonical CG constants; a customized e3nn
        # buffer is an operand of the host product instead. Certify once after
        # construction, a module move, or loading a checkpoint.
        tp = self.tp
        self._cg_certificate = None
        if type(tp) is not o3.TensorProduct or any(True for _ in tp.parameters()):
            return
        tables = []
        for name, value in tp.named_buffers():
            if name in ("weight", "output_mask"):
                continue
            prefix = "_compiled_main_left_right._w3j_"
            if not name.startswith(prefix):
                return
            degrees = name[len(prefix) :].split("_")
            if (
                len(degrees) != 3
                or not all(n.isdecimal() for n in degrees)
                or value.is_meta
                or value.is_inference()
                or is_fake(value)
                or value.dtype not in (torch.float32, torch.float64)
            ):
                return
            if not torch.equal(
                value.detach().cpu(),
                o3.wigner_3j(*map(int, degrees), dtype=value.dtype, device="cpu"),
            ):
                return
            tables.append((name.rsplit(".", 1)[-1], value, value._version))
        self._cg_certificate = (tp, tuple(tables))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if type(self) is _RowwisePairProduct:
            self._certify_cg()
        return self

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_cg_certificate", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if type(self) is _RowwisePairProduct:
            self._certify_cg()

    def _canonical_cg(self):
        cert = getattr(self, "_cg_certificate", None)
        if cert is None or self.tp is not cert[0]:
            return False
        if not cert[1]:
            return True
        if torch.compiler.is_compiling() or torch.jit.is_tracing():
            return False
        mode = get_proxy_mode()
        if mode is not None and mode.tracing_mode == "real":
            # A real-mode trace can see the original tensor object even when
            # it is a live placeholder. TACE's frozen traces use fake tensors.
            return False
        # functional_call can replace these with live graph inputs. Tensor
        # versions alone cannot distinguish that case from the original table.
        compiled = self.tp._compiled_main_left_right
        return all(
            getattr(compiled, name, None) is value
            and not value.requires_grad
            and statically_known_true(value._version == version)
            for name, value, version in cert[1]
        )

    def _setup_rowwise(self):
        """Install row-wise contractions on the existing weighted CG instructions."""
        self.contraction = eqx.conv.O3TensorProductConv(self.tp, batch_paths=True)
        self.to_channels_last = LayoutTransform(
            self.irreps_in1, layout_out="flatten_ir_mul"
        )
        self.to_mul_ir = LayoutTransform(
            self.irreps_out, layout_in="flatten_ir_mul", layout_out="flatten_mul_ir"
        )

    def forward(self, x, y, weights):
        if (
            x.is_cuda
            and x.dtype in (torch.float32, torch.float64)
            and y.dtype == weights.dtype == x.dtype
            and not torch.is_autocast_enabled("cuda")
            and x.ndim == y.ndim == weights.ndim == 2
            and x.size(0) == y.size(0) == weights.size(0)
            and self._canonical_cg()
        ):
            # The angular input has unit multiplicity and needs no permutation.
            out = self.contraction(
                self.to_channels_last(x),
                y,
                weights,
                weights.new_empty(0, self.weight_numel),
                None,
            )
            return self.to_mul_ir(out)
        return super().forward(x, y, weights)


def _refresh_pair_cg(module, incompatible_keys):
    module._certify_cg()


class O3CgtpEdgeInteraction(EdgeInteraction):
    """Couple directed edges with supplied harmonics, keeping periodic images separate.

    The EQX convolution option can evaluate the same weighted CG paths row by
    row on CUDA. Image summation and block transposition belong to the readout.
    """

    def _setup(self) -> None:
        self.irreps_hidden = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.irreps_in.regroup()]
        )
        self._setup_convolution()

        radial_basis = self.radial_basis
        radial_norm = radial_basis["mlp_norm"] or "layer_norm"
        options = dict(
            bias=radial_basis["bias"],
            layer_norm=radial_norm == "layer_norm",
            rms_norm=radial_norm == "rms_norm",
            act="silu",
        )
        widths = radial_basis.get("widths")
        if widths is None:
            self.edge_info = MLP[self.mlp_type](
                [radial_basis["num_radial_basis"] + 2 * self.num_elements]
                + list(radial_basis["hidden"])
                + [self.weight_numel],
                **options,
            )
        else:
            self.edge_info = BATCH_MLP[self.mlp_type].from_product(
                self.rejector,
                widths,
                radial_basis["hidden"],
                context_features=2 * self.num_elements,
                **options,
            )
            self.register_buffer(
                "element_basis", torch.eye(self.num_elements), persistent=False
            )

    def _setup_convolution(self) -> None:
        self.linear_up = e3nnLinear(
            self.irreps_in + self.irreps_in, self.irreps_hidden, bias=False
        )
        instructions, self.irreps_out = generate_paths(
            self.irreps_out, self.irreps_hidden, self.irreps_sh
        )
        product = UvuTensorProduct
        if (
            acceleration_enabled("eqx", kernel="conv")
            and not acceleration_enabled("oeq")
            and instructions
            and all(mul == 1 for mul, _ in self.irreps_sh)
            and not contains_time_odd_irreps(
                self.irreps_hidden, self.irreps_sh, self.irreps_out
            )
        ):
            product = _RowwisePairProduct
        self.rejector = product(
            self.irreps_hidden,
            self.irreps_sh,
            self.irreps_out,
            instructions,
            shared_weights=False,
        )
        self.weight_numel = self.rejector.weight_numel

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_attrs: torch.Tensor | None,
        edge_cutoff: torch.Tensor,
        edge_wigner: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source, target = edge_index
        edge_feats = self.linear_up(
            torch.cat((node_feats[source], node_feats[target]), dim=-1)
        )
        weights = self.radial_weights(edge_radial_basis, node_type, edge_index)
        return self.rejector(edge_feats, edge_attrs, weights * edge_cutoff)

    def radial_weights(self, radial, node_type, edge_index):
        """Project radial features and element-table rows without one-hot inputs."""
        if self.radial_basis.get("widths") is not None:
            source, target = edge_index
            return self.edge_info(
                (
                    (radial, None),
                    (self.element_basis, node_type[source]),
                    (self.element_basis, node_type[target]),
                )
            )
        linear = self.edge_info.mlp[0]
        weight = linear.get_weight()
        radial_width = self.radial_basis["num_radial_basis"]
        source, target = edge_index
        radial_weight, source_weight, target_weight = weight.split(
            (radial_width, self.num_elements, self.num_elements), dim=0
        )
        output = radial @ radial_weight
        output = (
            output + source_weight[node_type[source]] + target_weight[node_type[target]]
        )
        if linear.bias is not None:
            output = output + linear.bias
        return self.edge_info.mlp[1:](output)


class O2EdgeInteraction(O3CgtpEdgeInteraction):
    """Apply radially weighted O(2) Linear--Gate--Linear in an edge-aligned frame."""

    def _setup_convolution(self) -> None:
        self.linear_up = e3nnLinear(self.irreps_in, self.irreps_hidden, bias=False)
        self.irreps_out = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.irreps_out]
        )
        self.rejector = UvO2TensorProduct(
            self.irreps_hidden + self.irreps_hidden, self.irreps_out
        )
        self.weight_numel = self.rejector.weight_numel

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_attrs: torch.Tensor | None,
        edge_cutoff: torch.Tensor,
        edge_wigner: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_feats = self.linear_up(node_feats)
        source, target = edge_index
        edge_feats = torch.cat((node_feats[source], node_feats[target]), dim=-1)
        weights = self.radial_weights(edge_radial_basis, node_type, edge_index)
        return self.rejector(edge_feats, edge_wigner, weights * edge_cutoff)


EDGE_INTERACTION: dict[str, type[EdgeInteraction]] = {
    "cgtp": O3CgtpEdgeInteraction,
    "o2": O2EdgeInteraction,
}
