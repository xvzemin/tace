"""SPACE physical radial bases and per-degree CGTP weight networks.

Basis, spline, cutoff and linear initialization follow metatrain's SPACE modules
(commit 911683a32eaccc1d99af9a3683b1bd016facc544). The eigenfunctions are
provided by the optional ``physical-basis`` package, not regenerated here.

The adapted SPACE code is distributed under the following license:

BSD 3-Clause License
Copyright (c) 2023, Laboratory of Computational Science and Modeling

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import math
from functools import lru_cache

import numpy as np
import torch
from ase.data import covalent_radii
from e3nn import o3

from .linear import torchLinear


@lru_cache(maxsize=8)
def _physical_basis_table(lmax, max_eigenvalue, cutoff, accuracy):
    """Build the same adaptive cubic Hermite table as SPACE, in float64."""
    try:
        from physical_basis import PhysicalBasis
    except ImportError as exc:
        raise ImportError(
            "SPACE radial weights require physical-basis. Install TACE[space]."
        ) from exc

    basis = PhysicalBasis()
    if not isinstance(lmax, int) or not 0 <= lmax <= basis.l_max:
        raise ValueError(f"lmax must be between 0 and {basis.l_max}.")
    n_max_l = tuple(
        int(np.count_nonzero(basis.E_ln[l] <= max_eigenvalue)) for l in range(lmax + 1)
    )
    for l, count in enumerate(n_max_l):
        if count == 0:
            raise ValueError(
                f"SPACE max_eigenvalue={max_eigenvalue} supplies no basis for "
                f"l2={l}; use at least {basis.E_ln[l, 0]:.10g} to retain "
                "all TACE CGTP paths."
            )
    if n_max_l[0] > 50:
        raise ValueError("SPACE supports at most 50 radial functions per degree.")

    normalization = math.sqrt(cutoff**3 / 3.0)

    def evaluate(x, derivative=False):
        compute = basis.compute_derivative if derivative else basis.compute
        return normalization * np.stack(
            [compute(n, l, x) for l, count in enumerate(n_max_l) for n in range(count)],
            axis=-1,
        )

    positions = np.linspace(0.0, 10.0, 11)
    values = evaluate(positions)
    derivatives = evaluate(positions, derivative=True)
    while len(positions) - 1 < 50000:
        step = positions[1] - positions[0]
        midpoints = (positions[:-1] + positions[1:]) / 2.0
        actual = evaluate(midpoints)
        estimated = (values[:-1] + values[1:]) / 2.0 + step / 8.0 * (
            derivatives[:-1] - derivatives[1:]
        )
        error = np.abs(estimated - actual)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_error = np.mean(error / np.abs(actual))
        if np.mean(error) < accuracy or relative_error < accuracy:
            return n_max_l, step, values, derivatives
        midpoint_derivatives = evaluate(midpoints, derivative=True)
        refined_values = np.empty((2 * len(values) - 1, values.shape[1]))
        refined_derivatives = np.empty_like(refined_values)
        refined_values[::2], refined_values[1::2] = values, actual
        refined_derivatives[::2], refined_derivatives[1::2] = (
            derivatives,
            midpoint_derivatives,
        )
        values, derivatives = refined_values, refined_derivatives
        positions = np.linspace(0.0, 10.0, len(values))
    raise ValueError("SPACE radial spline exceeded 50000 intervals.")


class SpaceRadialBasis(torch.nn.Module):
    """Evaluate SPACE's physical basis separately for each angular degree.

    Parameters
    ----------
    lmax : int
        Largest TACE edge degree. No CGTP paths are removed by the eigenvalue
        cutoff; an insufficient ``max_eigenvalue`` raises an error.
    atomic_numbers : sequence of int
        Atomic numbers in TACE's node-type order.
    cutoff : float
        Neighbor cutoff in angstrom. Also sets the SPACE volume normalization.
    max_eigenvalue : float
        Retain eigenfunctions with eigenvalue at most this value for each l.
    element_scale : float
        Covalent-radius scale in ``x = r / (0.1 + scale * (R_i + R_j))``.
    cutoff_width : float
        Width of the SPACE bump cutoff, applied before the radial MLP.
    spline_accuracy : float
        Mean absolute or relative midpoint error used to refine the Hermite
        table, as in SPACE. This is not a maximum-error bound.
    """

    def __init__(
        self,
        lmax: int,
        atomic_numbers,
        cutoff: float,
        max_eigenvalue: float = 100.0,
        element_scale: float = 0.7,
        cutoff_width: float = 1.0,
        spline_accuracy: float = 1e-8,
    ) -> None:
        super().__init__()
        for name, value in (
            ("cutoff", cutoff),
            ("max_eigenvalue", max_eigenvalue),
            ("element_scale", element_scale),
            ("cutoff_width", cutoff_width),
            ("spline_accuracy", spline_accuracy),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"SPACE {name} must be finite and positive.")
        if cutoff_width > cutoff:
            raise ValueError("SPACE cutoff_width must not exceed cutoff.")
        if not atomic_numbers or any(
            not 0 < z < len(covalent_radii) for z in atomic_numbers
        ):
            raise ValueError("SPACE requires valid, nonempty atomic_numbers.")

        self.lmax = lmax
        self.cutoff = float(cutoff)
        self.cutoff_width = float(cutoff_width)
        self.max_eigenvalue = float(max_eigenvalue)
        self.n_max_l, self.spline_step, values, derivatives = _physical_basis_table(
            lmax, float(max_eigenvalue), float(cutoff), float(spline_accuracy)
        )
        dtype = torch.get_default_dtype()
        self.register_buffer("spline_values", torch.tensor(values, dtype=dtype))
        self.register_buffer(
            "spline_derivatives", torch.tensor(derivatives, dtype=dtype)
        )
        self.register_buffer(
            "lengthscales",
            torch.tensor(
                np.log(element_scale * covalent_radii[list(atomic_numbers)]),
                dtype=dtype,
            ),
        )

    def forward(
        self,
        edge_length: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return one ``(num_edges, n_max_l[l])`` tensor per degree."""
        radial, _ = self.forward_with_cutoff(edge_length, node_type, edge_index)
        return radial

    def forward_with_cutoff(
        self,
        edge_length: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return the radial functions and their SPACE envelope, computed once."""
        r = edge_length.reshape(-1)
        scales = self.lengthscales.exp()
        x = r / (
            0.1 + scales[node_type[edge_index[0]]] + scales[node_type[edge_index[1]]]
        )
        # Evaluate a safe interior point outside the tabulated domain, then mask.
        capped_x = torch.where(x < 10.0, x, 5.0)
        index = torch.floor(capped_x / self.spline_step).to(torch.long)
        index = index.clamp(0, self.spline_values.shape[0] - 2)
        t = ((capped_x - index * self.spline_step) / self.spline_step).unsqueeze(-1)
        t2, t3 = t.square(), t.pow(3)
        radial = (
            (2 * t3 - 3 * t2 + 1) * self.spline_values[index]
            + (t3 - 2 * t2 + t) * self.spline_step * self.spline_derivatives[index]
            + (-2 * t3 + 3 * t2) * self.spline_values[index + 1]
            + (t3 - t2) * self.spline_step * self.spline_derivatives[index + 1]
        )
        radial = torch.where(x.unsqueeze(-1) < 10.0, radial, 0.0)

        s = (r - (self.cutoff - self.cutoff_width)) / self.cutoff_width
        active = (s > 0.0) & (s < 1.0)
        safe_s = torch.where(active, s, 0.5)
        bump = 0.5 * (1 + torch.tanh(1 / torch.tan(torch.pi * safe_s)))
        envelope = torch.where(s <= 0.0, 1.0, torch.where(active, bump, 0.0))
        envelope = envelope.unsqueeze(-1)
        return (radial * envelope).split(self.n_max_l, dim=-1), envelope

    def extra_repr(self) -> str:
        return f"lmax={self.lmax}, n_max_l={self.n_max_l}, cutoff={self.cutoff}"


class SpaceRadialMLP(torch.nn.Module):
    """Generate every TACE CGTP weight using independent per-l2 networks.

    ``hidden`` and ``bias`` come directly from ``radial_basis``. Each l2 has
    an independent network with dimensions ``[n_max_l[l2], *hidden, W_l2]``;
    ``W_l2`` is the original number of TP weights for that degree. An empty
    hidden list gives a single linear map. SPACE's NTK linear initialization
    and SiLU activations are retained. Flat weights follow e3nn's instruction
    order. With bias, the output also receives the SPACE cutoff envelope so
    weights vanish smoothly when an edge leaves the neighbor list.
    """

    def __init__(
        self,
        tensor_product: o3.TensorProduct,
        n_max_l,
        hidden: list[int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(hidden, (list, tuple)) or any(
            type(width) is not int or width < 1 for width in hidden
        ):
            raise ValueError("radial_basis.hidden must be a list of positive integers.")
        if not isinstance(bias, bool):
            raise TypeError("radial_basis.bias must be a boolean.")
        if not n_max_l or any(n < 1 for n in n_max_l):
            raise ValueError("SPACE requires at least one radial function per l2.")

        self.n_max_l = tuple(n_max_l)
        self.num_layers = len(hidden) + 1
        self.use_bias = bias
        self.weight_numel = tensor_product.weight_numel
        self.hidden_widths = tuple(hidden)
        indices = [[] for _ in n_max_l]
        offset = 0
        for instruction in tensor_product.instructions:
            if not instruction.has_weight:
                continue
            l2 = tensor_product.irreps_in2[instruction.i_in2].ir.l
            if l2 >= len(n_max_l):
                raise ValueError(f"No SPACE radial basis supplied for l2={l2}.")
            count = math.prod(instruction.path_shape)
            indices[l2].extend(range(offset, offset + count))
            offset += count
        if offset != self.weight_numel:
            raise ValueError("SPACE weight layout does not match the tensor product.")

        self.weight_numel_by_l = tuple(len(group) for group in indices)
        self.radial_mlps = torch.nn.ModuleDict()
        for l2, output_width in enumerate(self.weight_numel_by_l):
            if not output_width:
                continue
            widths = [n_max_l[l2], *hidden, output_width]
            layers = []
            for index, (width_in, width_out) in enumerate(zip(widths, widths[1:])):
                layers.append(torchLinear(width_in, width_out, bias=bias))
                if index < self.num_layers - 1:
                    layers.append(torch.nn.SiLU())
            self.radial_mlps[str(l2)] = torch.nn.Sequential(*layers)
        self.register_buffer(
            "weight_order",
            torch.tensor(
                [i for group in indices for i in group], dtype=torch.long
            ).argsort(),
        )

    def forward(
        self,
        radial_basis: tuple[torch.Tensor, ...],
        cutoff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``(num_edges, weight_numel)`` weights for the existing CGTP."""
        if self.use_bias and cutoff is None:
            raise ValueError("A biased SPACE radial MLP requires its cutoff envelope.")
        if self.weight_numel == 0:
            return radial_basis[0][..., :0]
        grouped = torch.cat(
            [mlp(radial_basis[int(l)]) for l, mlp in self.radial_mlps.items()], dim=-1
        )
        weights = grouped.index_select(-1, self.weight_order)
        return weights * cutoff if self.use_bias else weights
