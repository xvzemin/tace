"""Unprojected tensor contractions in Cartesian or fully coupled coordinates."""

import math
from functools import lru_cache

import torch

from eqx import co3
from eqx.co3.ictd.tensor_product import cartesian_product
from eqx.co3.symmetric import symmetric_basis, symmetric_powers
from eqx.o3 import ICTD
from tace.utils.torch_scatter import scatter_sum


@lru_cache(maxsize=None)
def coupled_basis(rank):
    """Return the complete orthogonal rank decomposition in CPU float64."""
    return ICTD(rank, dtype=torch.float64, device="cpu").change_of_basis


@lru_cache(maxsize=None)
def coupled_product(rank1, rank2, rank_out):
    """Express an unprojected Cartesian contraction in fully coupled bases."""
    left = coupled_basis(rank1).T[:, None, :]
    right = coupled_basis(rank2).T[None, :, :]
    return cartesian_product(left, right, rank1, rank2, rank_out) @ coupled_basis(
        rank_out
    )


def reachable_basis(rank, values):
    """Return an orthonormal basis for an invariant Cartesian row space.

    Parameters
    ----------
    rank : int
        Cartesian tensor rank.
    values : torch.Tensor
        CPU float64 tensor with trailing dimension ``3**rank``. Its rows
        must span a rotation-invariant subspace.

    Returns
    -------
    torch.Tensor
        Cartesian basis columns. Only multiplicity indices are mixed;
        all magnetic components of each retained irrep are kept together.
    """
    decomposition = ICTD(rank, dtype=torch.float64, device="cpu")
    basis = decomposition.change_of_basis
    rows = values.reshape(-1, 3**rank)
    columns = []
    for degree in range(rank, -1, -1):
        paths = [
            basis[:, section]
            for path, section in zip(decomposition.paths, decomposition.slices)
            if path[-1].l == degree
        ]
        if not paths:
            continue
        block = torch.stack(paths, dim=1)
        coefficients = (
            (rows @ block.flatten(1))
            .reshape(rows.shape[0], len(paths), 2 * degree + 1)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if not coefficients.numel():
            continue
        vectors, singular, _ = torch.linalg.svd(coefficients, full_matrices=False)
        # Discard roundoff in exact null spaces, not small learned features.
        tolerance = 32 * torch.finfo(torch.float64).eps * max(rows.shape) * rows.norm()
        count = int((singular > tolerance).sum())
        if count == len(paths):
            columns.extend(paths)
        elif count:
            vectors = vectors[:, :count]
            signs = vectors.gather(0, vectors.abs().argmax(0, keepdim=True)).sign()
            vectors = vectors * signs
            columns.extend(torch.einsum("apm,pq->aqm", block, vectors).unbind(1))
    result = torch.cat(columns, dim=1) if columns else basis[:, :0]
    if result.shape[1] == basis.shape[1]:
        return basis
    # Retain the full space if numerical rank detection is ambiguous.
    residual = rows - (rows @ result) @ result.T
    bound = 128 * torch.finfo(torch.float64).eps * max(rows.shape) * rows.norm()
    return basis if residual.norm() > bound else result


@lru_cache(maxsize=None)
def moment_basis(rank):
    """Return Cartesian basis columns for the symmetric rank-r subspace."""
    data = symmetric_basis(rank)
    values = torch.zeros(
        len(symmetric_powers(rank)), 3**rank, dtype=torch.float64, device="cpu"
    )
    values[data["indices"], torch.arange(3**rank, device="cpu")] = (
        1 / data["norm"][data["indices"]]
    )
    return reachable_basis(rank, values)


def merge_spaces(*spaces):
    """Form the invariant span of rank-indexed Cartesian subspaces."""
    ranks = sorted({rank for space in spaces for rank in space})
    return {
        rank: reachable_basis(
            rank, torch.cat([space[rank].T for space in spaces if rank in space])
        )
        for rank in ranks
    }


def rank_sections(irreps, bases):
    """Return flattened block slices for rank-dependent storage widths."""
    sections, offset = [], 0
    for mul, ir in irreps:
        width = mul * bases[ir.l].shape[1]
        sections.append(slice(offset, offset + width))
        offset += width
    return tuple(sections), offset


def select_spaces(irreps, spaces=None):
    """Select rank spaces, retaining invariant scalars for biases and gates."""
    return {
        ir.l: coupled_basis(ir.l)
        if spaces is None or ir.l == 0
        else spaces.get(ir.l, coupled_basis(ir.l)[:, :0])
        for _, ir in irreps
    }


def optimize_representation(representation):
    """Plan reachable contractions once for a reducible spherical network.

    Angular moments and edge messages use compact storage. Node block
    boundaries retain complete coordinates so existing channel linears,
    gates, residuals and normalization keep their parameter conventions.
    """
    from ._e3nn.node import SphericalTensorNodeEmbedding
    from ._e3nn.prod import CgtpACE

    angular = representation.o3_angular_basis
    if not isinstance(angular, TensorMoments) or not angular.spherical:
        return
    if not all(
        isinstance(inter.rejector, TensorProductConv)
        for inter in representation.interactions
    ):
        return
    angular_spaces = {ir.l: moment_basis(ir.l) for _, ir in angular.irreps_out}
    tensor_embedding = isinstance(
        representation.node_embedding, SphericalTensorNodeEmbedding
    )
    if not tensor_embedding:
        angular.set_compact()
    node_spaces = select_spaces(
        representation.node_embedding.irreps_out,
        angular_spaces if tensor_embedding else None,
    )
    for inter, product in zip(representation.interactions, representation.products):
        inputs = select_spaces(inter.rejector.tp.irreps_in1, node_spaces)
        messages = inter.rejector.tp.optimize(
            inputs, angular_spaces, compact_in2=angular.compact
        )
        messages = select_spaces(inter.irreps_out, messages)
        if hasattr(inter, "resnetBA"):
            messages = select_spaces(
                inter.irreps_out, merge_spaces(messages, node_spaces)
            )
        residual = {}
        if hasattr(inter, "resnetBB"):
            residual = select_spaces(inter.irreps_sc, node_spaces)
        elif hasattr(inter, "resnetAB"):
            residual = select_spaces(inter.irreps_sc, messages)
        if representation.uee_embeddings is not None or representation.use_dens:
            messages = select_spaces(product.irreps_in)
        if not isinstance(product, CgtpACE):
            node_spaces = select_spaces(product.irreps_out)
            continue
        base = select_spaces(product.irreps_hidden, messages)
        current = base
        outputs = [select_spaces(product.irreps_coefs_out, base), residual]
        for index, ace in enumerate(product.aces):
            if not isinstance(ace.tp, TensorProduct):
                current = select_spaces(ace.irreps_out)
            else:
                current = ace.tp.optimize(
                    select_spaces(ace.tp.irreps_in1, current),
                    select_spaces(ace.tp.irreps_in2, base),
                    identical_inputs=index == 0,
                )
            outputs.append(select_spaces(product.irreps_coefs_out, current))
        node_spaces = select_spaces(product.irreps_out, merge_spaces(*outputs))


class TensorProduct(co3.TensorProduct):
    """Contract rank blocks without discarding irreducible components.

    Parameters
    ----------
    spherical : bool, optional
        Use complete coupled coordinates instead of Cartesian indices.
        Each rank-r block has width 3**r in either basis.
    **kwargs
        Instructions, weights and normalization of co3.TensorProduct.
        Rank labels select contraction routes, not irreducible subspaces.
    """

    def __init__(self, *args, spherical=False, **kwargs):
        kwargs["project"] = False
        super().__init__(*args, **kwargs)
        self.spherical = spherical
        self.optimized = False
        self.degrees = tuple(dict.fromkeys(group.degrees for group in self.groups))
        if spherical:
            for degrees in self.degrees:
                self.register_buffer(
                    "coupling_" + "_".join(map(str, degrees)),
                    coupled_product(*degrees).to(torch.get_default_dtype()).clone(),
                    persistent=False,
                )

    def forward(self, x, y, weight=None):
        """Couple (..., dim) features with optional (..., weight_numel) weights.

        Inputs need not be symmetric or traceless. Each output rank block
        retains all 3**rank entries in the selected basis. The second input
        may use the compact coordinates configured by ``optimize``.
        """
        if self.optimized:
            return self.forward_compact(x, y, weight, expand_output=True)
        return super().forward(x, y, weight)

    def optimize(self, input1, input2, *, compact_in2=False, identical_inputs=False):
        """Restrict contractions to the supplied invariant input subspaces.

        Parameters
        ----------
        input1, input2 : dict of int to torch.Tensor
            Orthonormal Cartesian basis columns for each input rank.
        compact_in2 : bool, optional
            The second input is already expressed in its reduced basis.
        identical_inputs : bool, optional
            Both operands are the same features. Same-block channel-wise
            paths then use the symmetric bilinear map.
        """
        if not self.spherical:
            raise ValueError("Path restriction requires coupled spherical coordinates.")
        self.input_bases = (input1, input2)
        self.compact_in2 = compact_in2
        self.compact_sections1, self.compact_dim1 = rank_sections(
            self.irreps_in1, input1
        )
        self.compact_sections2, self.compact_dim2 = rank_sections(
            self.irreps_in2, input2
        )
        images, products = {}, []
        for group in self.groups:
            r1, r2, r3 = group.degrees
            value = cartesian_product(
                input1[r1].T[:, None], input2[r2].T[None, :], r1, r2, r3
            )
            symmetric = (
                identical_inputs
                and group.connection_mode == "uuu"
                and group.i_in1 == group.i_in2
            )
            if symmetric:
                value = (value + value.transpose(0, 1)) * 0.5
            products.append(value)
            images.setdefault(r3, []).append(value.reshape(-1, 3**r3))
        self.output_bases = {
            ir.l: reachable_basis(
                ir.l,
                torch.cat(images[ir.l])
                if ir.l in images
                else coupled_basis(ir.l).T[:0],
            )
            for _, ir in self.irreps_out
        }
        self.compact_sections_out, self.compact_dim_out = rank_sections(
            self.irreps_out, self.output_bases
        )
        for name in tuple(self._buffers):
            if name.startswith(("coupling_", "analysis1_", "analysis2_", "synthesis_")):
                delattr(self, name)
        self.constants = {}
        self.identity1, self.identity2, self.identity_out = {}, {}, {}
        for index, bases in enumerate(self.input_bases, 1):
            identities = self.identity1 if index == 1 else self.identity2
            for rank, basis in bases.items():
                identities[rank] = torch.equal(basis, coupled_basis(rank))
                self.constants[f"analysis{index}_{rank}"] = (
                    coupled_basis(rank).T @ basis
                )
        for rank, basis in self.output_bases.items():
            self.identity_out[rank] = torch.equal(basis, coupled_basis(rank))
            self.constants[f"synthesis_{rank}"] = basis.T @ coupled_basis(rank)
        self.scalar_groups = []
        for index, (group, value) in enumerate(zip(self.groups, products)):
            coefficient = value @ self.output_bases[group.degrees[2]]
            scalar = 0
            if (
                coefficient.shape[0] == 1
                and coefficient.shape[1] == coefficient.shape[2]
            ):
                identity = torch.eye(
                    coefficient.shape[1], dtype=torch.float64, device="cpu"
                )
                if torch.allclose(coefficient[0], identity, atol=1e-14, rtol=1e-14):
                    scalar = 1
            elif (
                coefficient.shape[1] == 1
                and coefficient.shape[0] == coefficient.shape[2]
            ):
                identity = torch.eye(
                    coefficient.shape[0], dtype=torch.float64, device="cpu"
                )
                if torch.allclose(coefficient[:, 0], identity, atol=1e-14, rtol=1e-14):
                    scalar = 2
            self.scalar_groups.append(scalar)
            self.constants[f"coupling_{index}"] = coefficient
        for name, value in self.constants.items():
            self.register_buffer(
                name, value.to(self.path_scales).clone(), persistent=False
            )
        self.full_inputs = (all(self.identity1.values()), all(self.identity2.values()))
        self.full_output = all(self.identity_out.values())
        self.optimized = True
        return self.output_bases

    def input_blocks(self, features, index, *, packed=False):
        """View input blocks, changing only nontrivial multiplicity bases."""
        irreps = self.irreps_in1 if index == 1 else self.irreps_in2
        bases = self.input_bases[index - 1]
        if packed:
            sections = self.compact_sections1 if index == 1 else self.compact_sections2
            size = self.compact_dim1 if index == 1 else self.compact_dim2
        else:
            sections = self.slices_in1 if index == 1 else self.slices_in2
            size = irreps.dim
        if features.shape[-1] != size:
            raise ValueError("Input feature dimensions do not match the rank layout.")
        identities = self.identity1 if index == 1 else self.identity2
        outputs = []
        for (mul, ir), section in zip(irreps, sections):
            width = bases[ir.l].shape[1] if packed else ir.dim
            block = features[..., section].reshape(*features.shape[:-1], mul, width)
            if not packed and not identities[ir.l]:
                block = block @ getattr(self, f"analysis{index}_{ir.l}")
            outputs.append(block)
        return outputs

    def pack_input(self, features, index):
        """Pack complete coupled blocks into their reachable coordinates."""
        if self.full_inputs[index - 1]:
            return features
        outputs = [block.flatten(-2) for block in self.input_blocks(features, index)]
        return torch.cat(outputs, dim=-1) if outputs else features[..., :0]

    def expand(self, features):
        """Restore complete coupled coordinates at a block boundary."""
        if self.full_output:
            return features
        outputs = []
        for (mul, ir), section in zip(self.irreps_out, self.compact_sections_out):
            width = self.output_bases[ir.l].shape[1]
            value = features[..., section].reshape(*features.shape[:-1], mul, width)
            if not self.identity_out[ir.l]:
                value = value @ getattr(self, f"synthesis_{ir.l}")
            outputs.append(value.flatten(-2))
        return torch.cat(outputs, dim=-1) if outputs else features[..., :0]

    def forward_compact(
        self, x, y, weight=None, *, packed_in1=False, expand_output=False
    ):
        """Contract reachable blocks without materializing discarded coordinates."""
        weight = self.weight if weight is None else weight
        if weight.shape[-1] != self.weight_numel:
            raise ValueError("Weight size does not match the instructions.")
        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights must be one-dimensional.")
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        zero = x[..., :0].sum() + y[..., :0].sum() + weight[..., :0].sum()
        inputs = [
            self.input_blocks(x, 1, packed=packed_in1),
            self.input_blocks(y, 2, packed=self.compact_in2),
        ]
        outputs = [None] * len(self.irreps_out)
        for index, group in enumerate(self.groups):
            left, right = [
                blocks[indices[0]].unsqueeze(-3)
                if len(indices) == 1
                else torch.stack([blocks[i] for i in indices], dim=-3)
                for blocks, indices in zip(inputs, (group.i_in1, group.i_in2))
            ]
            mode = group.connection_mode
            if mode not in ("uuu", "uuw"):
                left, right = left.unsqueeze(-2), right.unsqueeze(-3)
            coefficients = getattr(self, f"coupling_{index}")
            if self.scalar_groups[index]:
                value = left * right
            elif coefficients.shape[0] <= coefficients.shape[1]:
                value = torch.einsum(
                    "...i,...ik->...k",
                    left,
                    torch.einsum("...j,ijk->...ik", right, coefficients),
                )
            else:
                value = torch.einsum(
                    "...j,...jk->...k",
                    right,
                    torch.einsum("...i,ijk->...jk", left, coefficients),
                )
            if group.weight_slices:
                if group.weight_slice is not None:
                    w = weight[..., group.weight_slice].reshape(
                        *weight.shape[:-1], len(group.i_out), *group.path_shape
                    )
                else:
                    w = torch.stack(
                        [
                            weight[..., section].reshape(
                                *weight.shape[:-1], *group.path_shape
                            )
                            for section in group.weight_slices
                        ],
                        dim=-len(group.path_shape) - 1,
                    )
                if mode == "uvw":
                    value = torch.einsum("...puvd,...puvw->...pwd", value, w)
                elif mode == "uuw":
                    value = torch.einsum("...pud,...puw->...pwd", value, w)
                else:
                    value = value * w.unsqueeze(-1)
            elif mode == "uuw":
                value = value.sum(-2, keepdim=True)
            if mode == "uvu":
                value = value.sum(-2)
            elif mode == "uvv":
                value = value.sum(-3)
            elif mode == "uvuv":
                value = value.flatten(-3, -2)
            value = value * self.path_scales[group.scale_slice, None, None]
            for destination, block in zip(group.i_out, value.unbind(-3)):
                previous = outputs[destination]
                outputs[destination] = block if previous is None else previous + block
        outputs = [
            x.new_zeros((*shape, mul, self.output_bases[ir.l].shape[1]))
            if block is None
            else block.expand(*shape, mul, self.output_bases[ir.l].shape[1])
            for block, (mul, ir) in zip(outputs, self.irreps_out)
        ]
        if expand_output and not self.full_output:
            outputs = [
                block
                if self.identity_out[ir.l]
                else block @ getattr(self, f"synthesis_{ir.l}")
                for block, (_, ir) in zip(outputs, self.irreps_out)
            ]
        return (
            torch.cat([v.flatten(-2) for v in outputs], -1) if outputs else x[..., :0]
        ) + zero

    def couple(self, x, y, l1, l2, l3):
        """Evaluate a contraction route in the selected complete basis."""
        if not self.spherical:
            return cartesian_product(x, y, l1, l2, l3)
        coefficients = getattr(self, f"coupling_{l1}_{l2}_{l3}")
        # Binary contractions preserve symbolic batch dimensions during tracing.
        if l1 <= l2:
            intermediate = torch.einsum("...j,ijk->...ik", y, coefficients)
            return torch.einsum("...i,...ik->...k", x, intermediate)
        intermediate = torch.einsum("...i,ijk->...jk", x, coefficients)
        return torch.einsum("...j,...jk->...k", y, intermediate)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if self.optimized:
            for name, value in self.constants.items():
                self._buffers[name] = value.to(self._buffers[name]).clone()
        elif self.spherical:
            for degrees in self.degrees:
                name = "coupling_" + "_".join(map(str, degrees))
                self._buffers[name] = (
                    coupled_product(*degrees).to(self._buffers[name]).clone()
                )
        return self


class TensorProductConv(torch.nn.Module):
    """Gather, contract and sum unprojected rank blocks at destination nodes.

    Parameters
    ----------
    tensor_product : TensorProduct
        Unprojected product with external per-edge weights.
    """

    def __init__(self, tensor_product):
        super().__init__()
        self.tp = tensor_product
        for name in ("irreps_in1", "irreps_in2", "irreps_out", "weight_numel"):
            setattr(self, name, getattr(tensor_product, name))

    def forward(self, features, edge_attrs, weights, edge_index):
        if self.tp.optimized:
            features = self.tp.pack_input(features, 1)
            messages = self.tp.forward_compact(
                features[edge_index[0]], edge_attrs, weights, packed_in1=True
            )
            output = scatter_sum(
                messages, edge_index[1], dim=0, dim_size=features.shape[0]
            )
            return self.tp.expand(output)
        messages = self.tp(features[edge_index[0]], edge_attrs, weights)
        return scatter_sum(messages, edge_index[1], dim=0, dim_size=features.shape[0])


class TensorMoments(torch.nn.Module):
    """Evaluate tensor powers without symmetric traceless projection.

    Parameters
    ----------
    irreps_out : e3nn.o3.Irreps
        Rank and parity labels of the original angular basis.
    normalize : bool
        Normalize input vectors before forming powers.
    normalization : {"integral", "component", "norm"}
        Harmonic normalization recovered by the final highest-degree projection.
    spherical : bool, optional
        Evaluate polynomials in the complete coupled basis.
    """

    def __init__(self, irreps_out, normalize, normalization, *, spherical=False):
        super().__init__()
        self.irreps_out = co3.Irreps(irreps_out)
        self.normalize = normalize
        self.normalization = normalization
        self.spherical = spherical
        self.compact = False
        self.lmax = self.irreps_out.lmax
        self.scales = []
        self.powers = {}
        for _, ir in self.irreps_out:
            rank = ir.l
            scale = math.sqrt(math.comb(2 * rank, rank) / 2**rank)
            if normalization == "component":
                scale *= math.sqrt(2 * rank + 1)
            elif normalization == "integral":
                scale *= math.sqrt((2 * rank + 1) / (4 * math.pi))
            self.scales.append(scale)
            if spherical and rank not in self.powers:
                # Combine identical monomials once, not Cartesian tensors at runtime.
                self.powers[rank] = symmetric_powers(rank)
                self.register_buffer(
                    f"coefficients_{rank}",
                    self.polynomial_coefficients(rank).to(torch.get_default_dtype()),
                    persistent=False,
                )

    def polynomial_coefficients(self, rank):
        """Return coefficients of distinct Cartesian monomials in float64."""
        basis = moment_basis(rank) if self.compact else coupled_basis(rank)
        coefficients = torch.zeros(
            len(self.powers[rank]), basis.shape[1], dtype=torch.float64, device="cpu"
        )
        return coefficients.index_add_(0, symmetric_basis(rank)["indices"], basis)

    def set_compact(self):
        """Store only the symmetric irreducible components of each tensor power."""
        if not self.spherical:
            raise ValueError("Compact moments require spherical coordinates.")
        self.compact = True
        for rank in self.powers:
            name = f"coefficients_{rank}"
            self._buffers[name] = self.polynomial_coefficients(rank).to(
                self._buffers[name]
            )

    def forward(self, vectors):
        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        if self.spherical:
            powers = [torch.ones_like(vectors), vectors]
            for _ in range(2, self.lmax + 1):
                powers.append(powers[-1] * vectors)
            values = {}
            for rank, exponents in self.powers.items():
                monomials = torch.stack(
                    [
                        powers[a][..., 0] * powers[b][..., 1] * powers[c][..., 2]
                        for a, b, c in exponents
                    ],
                    dim=-1,
                )
                values[rank] = monomials @ getattr(self, f"coefficients_{rank}")
        else:
            values = [torch.ones_like(vectors[..., :1])]
            for _ in range(self.lmax):
                values.append(
                    (values[-1].unsqueeze(-1) * vectors.unsqueeze(-2)).flatten(-2)
                )
        return torch.cat(
            [
                values[ir.l] * scale
                for (mul, ir), scale in zip(self.irreps_out, self.scales)
                for _ in range(mul)
            ],
            dim=-1,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for rank in self.powers:
            name = f"coefficients_{rank}"
            self._buffers[name] = self.polynomial_coefficients(rank).to(
                self._buffers[name]
            )
        return self


class ReadoutProjection(torch.nn.Module):
    """Extract the original readout irreps from complete rank blocks.

    Parameters
    ----------
    irreps : e3nn.o3.Irreps
        Original descriptor irreps, one highest-degree path per rank block.
    spherical : bool, optional
        Input uses complete coupled coordinates rather than Cartesian indices.
    """

    def __init__(self, irreps, *, spherical=False):
        super().__init__()
        self.irreps_in = co3.Irreps(irreps)
        self.spherical = spherical
        self.sections = tuple(self.irreps_in.slices())
        for rank in sorted(set(self.irreps_in.ls)):
            self.register_buffer(
                f"projection_{rank}",
                self.matrix(rank).to(torch.get_default_dtype()),
                persistent=False,
            )

    def matrix(self, rank):
        """Return the final highest-degree extraction matrix in float64."""
        matrix = co3.path_matrix(rank)
        return coupled_basis(rank).T @ matrix if self.spherical else matrix.clone()

    def forward(self, features):
        return torch.cat(
            [
                (
                    features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
                    @ getattr(self, f"projection_{ir.l}")
                ).flatten(-2)
                for (mul, ir), section in zip(self.irreps_in, self.sections)
            ],
            dim=-1,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for rank in set(self.irreps_in.ls):
            name = f"projection_{rank}"
            self._buffers[name] = self.matrix(rank).to(self._buffers[name])
        return self
