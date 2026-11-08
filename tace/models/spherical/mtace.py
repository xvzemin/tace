################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Magnetic representations and interactions."""

from collections.abc import Mapping
from numbers import Real
from typing import Optional, Union

import torch
from e3nn import o3

from equivariantx import o2
from tace.utils.env import acceleration_enabled
from tace.utils.torch_scatter import scatter_sum

from ..angular import SolidHarmonics
from ..blocks import format_list
from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import ACTIVATION, MLP
from ..time_reversal import make_irrep, spherical_harmonics_irreps
from ..utils import repr_without
from .fused import UuuTensorProduct, Wigner6jTensorProductConv
from .inter import O3CgtpInteraction, UvO2Interaction
from .o2 import uv_convolution, uv_kernel
from .paths import to_possible_tp_irreps


class MagneticChebyshevBasis(torch.nn.Module):
    """Evaluate Chebyshev polynomials on normalized magnetic magnitudes.

    Parameters
    ----------
    num_basis : int
        Number of polynomials.
    include_constant : bool
        Include the degree-zero polynomial.

    Examples
    --------
    >>> basis = MagneticChebyshevBasis(4)
    >>> basis(torch.zeros(2, 1)).shape
    torch.Size([2, 4])
    """

    def __init__(
        self,
        num_basis: int = 8,
        include_constant: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(include_constant, bool):
            raise TypeError("include_constant must be a bool")
        self.num_basis = num_basis
        self.include_constant = include_constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t0 = torch.ones_like(x)
        t1 = x
        basis = [t0, t1]
        start = 0 if self.include_constant else 1
        stop = start + self.num_basis
        for _ in range(2, stop):
            t0, t1 = t1, 2.0 * x * t1 - t0
            basis.append(t1)
        return torch.cat(basis[start:stop], dim=-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_basis={self.num_basis}, "
            f"include_constant={self.include_constant})"
        )


class MagneticBasis(torch.nn.Module):
    """Construct radial, node, and edge magnetic representations.

    Parameters
    ----------
    magnetic_scale : sequence of float or dict
        Magnetic cutoff per fidelity, optionally indexed by atomic number.
    num_mag_radial_basis : int
        Number of radial polynomials.
    Lmax : int
        Maximum magnetic angular degree.
    atomic_numbers : list of int
        Elements in model order.
    time_reversal : bool
        Include time-reversal labels.
    angular_normalization : {'integral'}
        Solid-harmonic normalization.
    radial_normalization : {'clamp'}
        Clamp normalized squared magnitudes to one.
    use_spin_orbit_coupling : bool
        Retain tensor-valued magnetic edge representations instead of scalars.
    parity : bool or None
        Use full parity labels, natural parity only, or inferred output parity.
    path_irreps : e3nn.o3.Irreps or None
        Allowed magnetic edge representations.

    Examples
    --------
    >>> basis = MagneticBasis([2.0], 4, 1, [26])
    >>> basis.magnetic_scale.shape
    torch.Size([1, 1])
    """

    geometry_precision = True

    def __init__(
        self,
        magnetic_scale,
        num_mag_radial_basis: int,
        Lmax: int,
        atomic_numbers: list[int],
        time_reversal: bool = False,
        angular_normalization: str = "integral",
        radial_normalization: str = "clamp",
        use_spin_orbit_coupling: bool = True,
        parity: bool | None = None,
        path_irreps: o3.Irreps | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(Lmax, int) or Lmax < 0:
            raise ValueError("Lmax must be a non-negative integer.")
        if not isinstance(num_mag_radial_basis, int) or num_mag_radial_basis < 1:
            raise ValueError("num_mag_radial_basis must be a positive integer.")
        if angular_normalization != "integral":
            raise ValueError("angular_normalization currently only supports 'integral'.")
        if radial_normalization != "clamp":
            raise ValueError("radial_normalization currently only supports 'clamp'.")

        self.Lmax = Lmax
        self.num_mag_radial_basis = num_mag_radial_basis
        self.angular_normalization = angular_normalization
        self.radial_normalization = radial_normalization
        self.use_spin_orbit_coupling = use_spin_orbit_coupling
        self.parity = parity
        self.register_buffer(
            "magnetic_scale",
            self._resolve_magnetic_scale(magnetic_scale, atomic_numbers),
        )
        self.radial_basis = MagneticChebyshevBasis(
            num_basis=num_mag_radial_basis,
            include_constant=False,
        )
        self.magnetic_node_irreps_out = spherical_harmonics_irreps(
            Lmax,
            p=-1 if parity is False else 1,
            time_reversal=-1 if time_reversal else 1,
        ).sort().irreps.simplify()
        self.angular_basis = SolidHarmonics(
            self.magnetic_node_irreps_out,
            normalization=angular_normalization,
        )

        if use_spin_orbit_coupling:
            magnetic_edge_irreps = to_possible_tp_irreps(
                self.magnetic_node_irreps_out,
                self.magnetic_node_irreps_out,
                parity,
                Lmax,
                target_irreps=path_irreps,
            )
        else:
            magnetic_edge_irreps = o3.Irreps(
                [(1, make_irrep(0, 1, 1))]
            )
        self.magnetic_edge_tensor_product = UuuTensorProduct(
            self.magnetic_node_irreps_out,
            self.magnetic_node_irreps_out,
            magnetic_edge_irreps,
            trainable=False,
        )
        self.magnetic_edge_irreps_out = (
            self.magnetic_edge_tensor_product.irreps_out.sort().irreps.simplify()
        )

    @staticmethod
    def _resolve_magnetic_scale(
        magnetic_scale, atomic_numbers: list[int]
    ) -> torch.Tensor:
        if not isinstance(magnetic_scale, (list, tuple)) or not magnetic_scale:
            raise TypeError(
                "magnetic_scale must be a non-empty per-fidelity sequence"
            )
        values = []
        for scale in magnetic_scale:
            if isinstance(scale, Mapping):
                row = []
                for atomic_number in atomic_numbers:
                    value = scale.get(atomic_number, scale.get(str(atomic_number)))
                    if value is None:
                        raise ValueError(
                            f"magnetic_scale is missing atomic number {atomic_number}"
                        )
                    row.append(float(value))
            elif isinstance(scale, Real):
                row = [float(scale)] * len(atomic_numbers)
            else:
                raise TypeError(
                    "each magnetic_scale entry must be a scalar or an "
                    "element-dependent mapping"
                )
            values.append(row)

        magnetic_scale = torch.tensor(values, dtype=torch.float64)
        if not torch.isfinite(magnetic_scale).all() or (magnetic_scale <= 0.0).any():
            raise ValueError("all magnetic_scale values must be finite and positive")
        return magnetic_scale

    def forward(
        self,
        initial_noncollinear_magmoms: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        node_fidelity: torch.Tensor,
        compute_edge: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        magnetic_scale = self.magnetic_scale[node_fidelity, node_type].unsqueeze(-1)
        scaled_magmoms = initial_noncollinear_magmoms / magnetic_scale
        squared_magnitude = scaled_magmoms.square().sum(dim=-1, keepdim=True)
        radial_coordinate = 1.0 - 2.0 * torch.clamp(
            squared_magnitude,
            max=1.0,
        )
        magnetic_radial_basis = self.radial_basis(radial_coordinate)
        magnetic_node_attrs = self.angular_basis(initial_noncollinear_magmoms)
        source, target = edge_index
        magnetic_edge_attrs = (
            self.magnetic_edge_tensor_product(
                magnetic_node_attrs[target], magnetic_node_attrs[source]
            )
            if compute_edge else None
        )

        return (
            magnetic_radial_basis,
            magnetic_node_attrs,
            magnetic_edge_attrs,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  parity={self.parity},\n"
            f"  use_spin_orbit_coupling={self.use_spin_orbit_coupling},\n"
            f"  num_mag_radial_basis={self.num_mag_radial_basis},\n"
            f"  magnetic_node_irreps_out={self.magnetic_node_irreps_out},\n"
            f"  magnetic_edge_irreps_out={self.magnetic_edge_irreps_out}\n"
            f"  magnetic_scale={format_list(self.magnetic_scale.tolist())},\n"
            ")"
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        key = f"{prefix}magnetic_scale"
        if key in state_dict and state_dict[key].ndim == 1:
            state_dict[key] = state_dict[key].unsqueeze(0)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MagneticUvO2TensorProductConv(torch.nn.Module):
    """Apply a local O(2) convolution to node and magnetic edge features.

    Parameters
    ----------
    irreps_in, irreps_out, magnetic_edge_irreps : e3nn.o3.Irreps
        Input, output and magnetic edge representations.
    num_channel : int
        Multiplicity of each representation.
    mmax : int
        Maximum local order.
    even_scalar_act, odd_scalar_act, tensor_act : torch.nn.Module
        Even-scalar, odd-scalar and gate activations. An odd-scalar activation
        of ``None`` gates odd scalars instead.

    Examples
    --------
    >>> conv = MagneticUvO2TensorProductConv(
    ...     "2x0e", "2x0e", "2x0e", num_channel=2, mmax=0,
    ...     even_scalar_act=torch.nn.SiLU(), odd_scalar_act=torch.nn.Tanh(),
    ...     tensor_act=torch.nn.Sigmoid(),
    ... )
    >>> conv.weight_numel
    6
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_edge_irreps: o3.Irreps,
        *,
        num_channel: int,
        mmax: int,
        even_scalar_act: torch.nn.Module,
        odd_scalar_act: Optional[torch.nn.Module],
        tensor_act: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.magnetic_edge_irreps = o3.Irreps(magnetic_edge_irreps)
        self.num_channel = num_channel
        self.mmax = min(
            max(self.irreps_in.lmax, self.magnetic_edge_irreps.lmax),
            mmax,
        )
        if any(entry.mul != num_channel for entry in self.irreps_in + self.irreps_out):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")
        if any(entry.mul != num_channel for entry in self.magnetic_edge_irreps):
            raise ValueError(
                "magnetic_edge_irreps multiplicity must equal num_channel."
            )

        self.local_frame_in = o2.LocalFrame(self.irreps_in, mmax=self.mmax)
        self.local_frame_out = o2.LocalFrame(
            self.irreps_out,
            mmax=self.mmax,
            reverse=True,
        )
        self.magnetic_frame = o2.LocalFrame(
            self.magnetic_edge_irreps,
            mmax=self.mmax,
        )
        self.reshape_in = LayoutTransform(
            self.irreps_in,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.reshape_out = LayoutTransform(
            self.irreps_out,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.reshape_magnetic = LayoutTransform(
            self.magnetic_edge_irreps,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.node_irreps = self.local_frame_in.irreps_out
        self.local_magnetic_irreps = self.magnetic_frame.irreps_out
        self.local_irreps_in = (
            self.node_irreps + self.node_irreps + self.local_magnetic_irreps
        ).regroup()
        self.local_irreps_out = self.local_frame_out.irreps_out
        self.use_time_reversal = any(ir.t == -1 for ir, _ in self.local_irreps_in)
        hidden_irreps = self.local_irreps_out.filter(
            keep=lambda ir_mul: self.local_irreps_in.count(ir_mul.ir) > 0
        )
        scalar_entries = []
        scalar_acts = []
        gated_entries = []
        for ir, mul in hidden_irreps:
            if ir.is_invariant_scalar():
                scalar_entries.append((ir, mul))
                scalar_acts.append(even_scalar_act)
            elif ir.m == 0 and odd_scalar_act is not None:
                scalar_entries.append((ir, mul))
                scalar_acts.append(odd_scalar_act)
            else:
                gated_entries.append((ir, mul))

        irreps_gated = o2.Irreps(gated_entries)
        if self.use_time_reversal:
            gate_entries = [(o2.Irrep("0ee"), mul) for _, mul in gated_entries]
            gate_acts = [tensor_act] * len(gate_entries)
            time_odd_scalars = [
                ir for ir, _ in self.local_irreps_in if ir.m == 0 and ir.t == -1
            ]
            time_odd_irreps = [ir for ir, _ in self.local_irreps_in if ir.t == -1]
            for ir_out, mul in self.local_irreps_out:
                path = next(
                    (
                        (ir_gate, ir_gated)
                        for ir_gate in time_odd_scalars
                        for ir_gated in time_odd_irreps
                        if ir_gated * ir_gate == (ir_out,)
                    ),
                    None,
                )
                if path is not None:
                    ir_gate, ir_gated = path
                    gate_entries.append((ir_gate, mul))
                    gate_acts.append(odd_scalar_act)
                    gated_entries.append((ir_gated, mul))
            irreps_gated = o2.Irreps(gated_entries)
            irreps_gates = o2.Irreps(gate_entries)
        else:
            irreps_gates = (
                o2.Irreps([(o2.Irrep("0ee"), irreps_gated.num_irreps)])
                if irreps_gated.num_irreps
                else o2.Irreps()
            )
            gate_acts = [tensor_act] if len(irreps_gates) else []
        self.nonlinearity = o2.Gate(
            o2.Irreps(scalar_entries),
            scalar_acts,
            irreps_gates,
            gate_acts,
            irreps_gated,
        )
        self.linear_up = o2.Linear(
            self.local_irreps_in,
            self.nonlinearity.irreps_in,
            biases=True,
        )
        self.linear_down = o2.Linear(
            self.nonlinearity.irreps_out,
            self.local_irreps_out,
            biases=True,
        )
        self.weight_numel = self.local_irreps_in.num_irreps
        self.attention = None
        self.eqx_tp = uv_kernel(self)

    def __repr__(self) -> str:
        return repr_without(self, "reshape_in", "reshape_out")

    def _to_local(
        self,
        node_features: torch.Tensor,
        magnetic_edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self.reshape_in(node_features)
        magnetic_edge_attrs = self.reshape_magnetic(magnetic_edge_attrs)
        paired_node = self.local_frame_in.to_local(node_features[edge_index.T], wigner)
        magnetic_edge_attrs = self.magnetic_frame.to_local(
            magnetic_edge_attrs,
            wigner,
        )
        return (
            node_features,
            paired_node[:, 0],
            paired_node[:, 1],
            magnetic_edge_attrs,
        )

    def _convolution(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        magnetic_edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: Optional[torch.Tensor],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        weighted = []
        offset = 0
        feature_sets = (
            (target_features, self.node_irreps),
            (source_features, self.node_irreps),
            (magnetic_edge_attrs, self.local_magnetic_irreps),
        )
        for ir, mul in self.local_irreps_in:
            inputs = []
            for features, irreps in feature_sets:
                for (input_ir, input_mul), ir_slice in zip(
                    irreps,
                    irreps.slices(),
                ):
                    if input_ir == ir:
                        inputs.append(
                            features[..., ir_slice].view(
                                features.size(0), ir.dim, input_mul
                            )
                        )
            values = torch.cat(inputs, dim=-1)
            weight = conv_weights[..., offset : offset + mul].unsqueeze(-2)
            weighted.append((values * weight).reshape(values.size(0), ir.dim * mul))
            offset += mul
        if offset != conv_weights.size(-1):
            raise ValueError("Invalid O2 convolution weight size.")
        projected = self.linear_up(torch.cat(weighted, dim=-1))
        hidden = self.nonlinearity(projected)
        message = self.linear_down(hidden)
        return message

    def _to_global(
        self,
        message: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        message = self.local_frame_out.to_global(message, wigner_inv)
        message = message * edge_cutoff
        message = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
        return self.reshape_out.inverse(message)

    def forward(
        self,
        node_feats: torch.Tensor,
        magnetic_edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Optional[torch.Tensor],
        wigner_inv: Optional[torch.Tensor],
        edge_radial_basis: Optional[torch.Tensor] = None,
        edge_cutoff: Optional[torch.Tensor] = None,
        radial_network=None,
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        if wigner is not None and wigner.ndim == 2:
            wigner_inv = wigner
        if wigner is None or wigner_inv is None:
            raise ValueError("O2 convolution requires Wigner matrices.")
        if (
            getattr(self, "eqx_tp", None) is not None
            and node_feats.is_cuda
            and node_feats.dtype in (torch.float32, torch.float64)
            and acceleration_enabled("eqx", kernel="conv")
        ):
            return uv_convolution(
                self,
                node_feats,
                magnetic_edge_attrs,
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_radial_basis,
                edge_cutoff,
                radial_network=radial_network,
            )
        (
            node_features,
            source_features,
            target_features,
            magnetic_edge_attrs,
        ) = self._to_local(node_feats, magnetic_edge_attrs, edge_index, wigner)
        message = self._convolution(
            source_features,
            target_features,
            magnetic_edge_attrs,
            conv_weights,
            edge_index,
            edge_radial_basis,
            edge_cutoff,
            node_features.size(0),
        )
        return self._to_global(
            message,
            edge_index,
            wigner_inv,
            edge_cutoff,
            node_features.size(0),
        )


class O3MagneticWigner6jInteraction(O3CgtpInteraction):
    """Couple node features, spatial harmonics and source magnetic harmonics."""

    def _prepare_setup(self) -> None:
        if self.magnetic_node_irreps is None:
            raise ValueError("w6j_mag requires magnetic node irreps.")
        self.irreps_mid = self.irrreps_tp_out
        self.irrreps_tp_out = to_possible_tp_irreps(
            self.irreps_mid,
            self.magnetic_node_irreps,
            self.parity,
            lmax=self.Lmax if self.correlation == 1 else self.lmax,
            target_irreps=self.path_irreps,
        )
        self.irreps_out = (
            (self.irrreps_tp_out * self.num_channel).sort().irreps.simplify()
        )
        if self.layer != self.num_layers - 1:
            self.irreps_sc = to_possible_tp_irreps(
                self.irreps_out,
                self.irreps_out,
                self.parity,
                lmax=self.Lmax,
                target_irreps=self.path_irreps,
            )
            self.irreps_sc = (
                (self.irreps_sc * self.num_channel).sort().irreps.simplify()
            )

    def _build_rejector(self) -> torch.nn.Module:
        return Wigner6jTensorProductConv(
            self.irreps_in,
            self.irreps_sh,
            self.magnetic_node_irreps,
            self.irreps_mid,
            self.irreps_out,
            l1l2=self.l1l2,
        )

    def _setup_additional_modules(self) -> None:
        radial_norm = self.radial_mlp_norm
        if radial_norm is None and self.mlp_type == "mlp_v1":
            radial_norm = (
                "layer_norm"
                if self.magnetic_edge_feats_channel != self.num_mag_radial_basis
                else None
            )
        self.magnetic_edge_info = MLP[self.mlp_type](
            [self.magnetic_edge_feats_channel]
            + self.radial_mlp
            + [self.rejector.magnetic_weight_numel],
            bias=self.radial_bias,
            layer_norm=radial_norm == "layer_norm",
            rms_norm=radial_norm == "rms_norm",
            act="silu",
        )

    def _compute_messages(
        self,
        node_feats,
        node_types,
        edge_radial_basis,
        edge_feats,
        edge_attrs,
        edge_index,
        edge_cutoff,
        edge_wigner=None,
        edge_wigner_inv=None,
        magnetic_node_info=None,
        magnetic_edge_attrs=None,
        graph=None,
        *,
        magnetic_node_attrs=None,
        magnetic_edge_feats=None,
    ) -> torch.Tensor:
        if magnetic_node_attrs is None or magnetic_edge_feats is None:
            raise ValueError(
                "w6j_mag requires magnetic harmonics and edge features."
            )
        weights = self.edge_info(edge_feats)
        if edge_cutoff is not None:
            weights = weights * edge_cutoff
        magnetic_weights = self.magnetic_edge_info(magnetic_edge_feats)
        return self.rejector(
            node_feats,
            edge_attrs,
            magnetic_node_attrs,
            weights,
            magnetic_weights,
            edge_index,
        )


class O2MagneticInteraction(UvO2Interaction):
    """Magnetic interaction evaluated by local O(2) frame."""

    magnetic_info_type = "node"  # [node]

    def _prepare_setup(self) -> None:
        if self.magnetic_edge_irreps is None:
            raise ValueError("uv_o2_mag requires magnetic_edge_irreps.")
        if not 0 <= self.magnetic_edge_irreps.lmax <= self.Lmax:
            raise ValueError("magnetic edge irreps must satisfy 0 <= lmax <= Lmax.")
        self.magnetic_edge_irreps_out = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.magnetic_edge_irreps]
        ).sort().irreps.simplify()

        # A magnetic SOC edge carries both the spatial bond harmonics and the
        # magnetic source-target tensor product.  We do not truncate their
        # intermediate product: higher intermediate l can couple with the
        # node field back into the retained output range.
        edge_irreps = to_possible_tp_irreps(
            self.magnetic_edge_irreps,
            self.irreps_sh,
            parity=self.parity,
            lmax=self.magnetic_edge_irreps.lmax + self.irreps_sh.lmax,
            target_irreps=self.path_irreps,
        )
        output_lmax = self.Lmax if self.correlation == 1 else self.lmax
        self.irrreps_tp_out = to_possible_tp_irreps(
            self.irreps_in,
            edge_irreps,
            parity=self.parity,
            lmax=output_lmax,
            target_irreps=self.path_irreps,
        )
        self.irreps_out = (
            self.irrreps_tp_out * self.num_channel
        ).sort().irreps.simplify()

        # Match the skip connection to the product-basis output constructed
        # from the corrected interaction irreps.
        if self.layer == self.num_layers - 1:
            irreps_sc = o3.Irreps(self.target_irreps)
        elif self.correlation == 1:
            irreps_sc = o3.Irreps(
                [(1, ir) for _, ir in self.irreps_out if ir.l <= self.Lmax]
            ).sort().irreps.simplify()
        else:
            irreps_sc = to_possible_tp_irreps(
                self.irreps_out,
                self.irreps_out,
                parity=self.parity,
                lmax=self.Lmax,
                target_irreps=self.path_irreps,
            )
        self.irreps_sc = (irreps_sc * self.num_channel).sort().irreps.simplify()
        super()._prepare_setup()

    def _build_rejector(self) -> torch.nn.Module:
        rejector = MagneticUvO2TensorProductConv(
            self.irreps_in,
            self.irreps_out,
            self.magnetic_edge_irreps_out,
            num_channel=self.num_channel,
            mmax=self.mmax,
            even_scalar_act=ACTIVATION[
                (
                    self.scalar_act[0]
                    if isinstance(self.scalar_act, list)
                    else self.scalar_act or "silu"
                ).removeprefix("scaled_")
            ](),
            odd_scalar_act=ACTIVATION[
                (
                    self.scalar_act[1]
                    if isinstance(self.scalar_act, list)
                    else "tanh"
                ).removeprefix("scaled_")
            ](),
            tensor_act=ACTIVATION[
                (self.tensor_act or "sigmoid").removeprefix("scaled_")
            ](),
        )
        return rejector

    def _setup_additional_modules(self) -> None:
        if self.magnetic_info_type != "node":
            raise NotImplementedError(
                "Only magnetic_info_type='node' is currently implemented."
            )
        self.magnetic_linear = e3nnLinear(
            self.magnetic_edge_irreps,
            self.magnetic_edge_irreps_out,
            bias=False,
            internal_weights=False,
        )
        magnetic_info_channels = (
            [self.magnetic_node_info_channel]
            + self.radial_mlp
            + [self.magnetic_linear.weight_numel]
        )
        self.source_magnetic_info = MLP[self.mlp_type](
            magnetic_info_channels,
            bias=self.radial_bias,
            act="silu",
        )
        self.target_magnetic_info = MLP[self.mlp_type](
            magnetic_info_channels,
            bias=self.radial_bias,
            act="silu",
        )

    def _magnetic_weights(
        self,
        magnetic_node_info: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source_info, target_info = magnetic_node_info
        source, target = edge_index
        return self.source_magnetic_info(source_info)[source] * (
            self.target_magnetic_info(target_info)[target]
        )

    def _apply_rejector(
        self,
        node_feats: torch.Tensor,
        magnetic_node_info: Union[
            tuple[torch.Tensor, torch.Tensor], None
        ],
        magnetic_edge_attrs: Union[torch.Tensor, None],
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        radial_network=None,
    ) -> torch.Tensor:
        if magnetic_node_info is None or magnetic_edge_attrs is None:
            raise ValueError(
                "O2MagneticInteraction requires magnetic node information and "
                "edge attrs"
            )
        magnetic_edge_attrs = self.magnetic_linear(
            magnetic_edge_attrs,
            self._magnetic_weights(magnetic_node_info, edge_index),
        )
        return self.rejector(
            node_feats,
            magnetic_edge_attrs,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_radial_basis,
            edge_cutoff,
            radial_network=radial_network,
        )
