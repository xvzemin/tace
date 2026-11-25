"""Lower the five-layer OAM-L graph without replacing learned parameters.

Only structural metadata is prepared at conversion. Angular and radial geometry
are recomputed from the caller's graph; no coordinate- or weight-dependent
tensor is cached across forwards or optimizer updates.
"""

from __future__ import annotations

import copy

import torch
from e3nn import o3

from equivariantx.models.tace.nonlinear_ace import product_metadata
from equivariantx.ops.common.channel_product import channel_product
from equivariantx.utils.layout import permute
from equivariantx.utils.metadata import numeric_rows
from tace.models.linear import e3nnElementLinear, has_lora
from tace.models.spherical.prod import CgtpACE
from tace.models.spherical.representation import NodeRepresentation


def _column_order(irreps):
    return torch.cat(
        [
            torch.arange(mul * ir.dim).reshape(mul, ir.dim).T.reshape(-1) + sl.start
            for (mul, ir), sl in zip(irreps, irreps.slices())
        ]
    )


class _QuadraticACE(CgtpACE):
    """One sparse angular contraction per channel, then indexed matrix maps."""

    native_square = None

    def prepare_paths(self):
        tp = self.aces[0].tp
        device = next(self.parameters()).device
        self.scalar_norms = ()
        if all(ir.is_scalar() for _, ir in tp.irreps_out):
            norms = []
            for i, ins in enumerate(tp.instructions):
                if ins.i_in1 != ins.i_in2 or i != ins.i_out:
                    raise ValueError(
                        "OAM-L invariant products require diagonal self paths"
                    )
                mul, ir = tp.irreps_in1[ins.i_in1]
                cg = o3.wigner_3j(ir.l, ir.l, 0, dtype=torch.float64)[:, :, 0]
                if not torch.allclose(cg, torch.eye(ir.dim, dtype=cg.dtype) * cg[0, 0]):
                    raise ValueError(
                        "Scalar coupling does not reduce to the irrep norm"
                    )
                sl = tp.irreps_in1.slices()[ins.i_in1]
                norms.append(
                    (sl.start, sl.stop, mul, ir.dim, float(cg[0, 0]) * ins.path_weight)
                )
            self.scalar_norms = tuple(norms)
        else:
            width, dims, paths = product_metadata(tp)
            self.cg_metadata = numeric_rows(width, dims, paths)
            self.cg_width = width * dims[2]
            for name, irreps in (("input", tp.irreps_in1), ("output", tp.irreps_out)):
                order = _column_order(irreps).to(device)
                self.register_buffer(f"{name}_order", order, persistent=False)
                self.register_buffer(
                    f"{name}_inverse", order.argsort(), persistent=False
                )

    def forward(self, node_feats, node_type, sc, batch, groups=None):
        features = self.linear_up(node_feats)
        tiled = (
            self.tiled_quadratic
            and not self.training
            and features.is_cuda
            and not self.scalar_norms
            and not self.coefs[1].weight.requires_grad
        )
        if tiled:
            from .product import frozen_quadratic

            quadratic = frozen_quadratic(self, features, node_type)
        elif self.scalar_norms:
            quadratic = torch.cat(
                [
                    features[:, start:stop]
                    .reshape(features.shape[0], mul, dim)
                    .square()
                    .sum(-1)
                    * scale
                    for start, stop, mul, dim, scale in self.scalar_norms
                ],
                dim=-1,
            )
        elif self.native_square is not None and features.is_cuda:
            from .product import square

            quadratic = square(self.native_square, features)
        elif features.is_cuda:
            x = permute(features, self.input_order, self.input_inverse)
            y = channel_product(
                self.cg_metadata, "(((0, 0, 1), 2, 0, 0),)",
                [x, x.new_empty((0, self.cg_width))],
            )[0]
            quadratic = permute(y, self.output_inverse, self.output_order)
        else:
            quadratic = self.aces[0](features, features)
        output = self.coefs[0](features, node_type, groups) + (
            quadratic if tiled else self.coefs[1](quadratic, node_type, groups)
        )
        output = self.linear(output)
        if hasattr(self, "stochastic_depth"):
            output = self.stochastic_depth(output, batch)
        return (
            output if sc is None else torch.add(sc, output, alpha=self.residual_scale)
        )


class _OAMLRepresentation(NodeRepresentation):
    """Use the static coupling/conditioning schedule of the released OAM-L."""

    def forward(self, data, graph):
        from .radial import affine_heads, prepare_radial, project, radial_heads

        types = graph.node_type
        index = data["edge_index"]
        local = graph.lmp_natoms[0] if graph.lmp else None
        local_types = types[:local]
        groups = None
        if (
            self.grouped_coefficients
            and not self.training
            and types.is_cuda
            and torch.version.hip is None
            and self.model_dtype == torch.float32
        ):
            from .element_grouped import grouping

            groups = grouping(local_types, self.num_elements)
        source_types, target_types = types[index[0]], types[index[1]]
        basis, cutoff = self.radial_basis(
            graph.edge_length, types, index, self.atomic_numbers
        )
        radial = self.edge_embedding(
            None, types, basis.to(self.model_dtype), index, None
        )
        if self.fused_geometry or (self.fused_geometry is None and self.training):
            from .angular import edge_attributes

            attrs = edge_attributes(
                graph.edge_vector, graph.edge_length, cutoff, self.model_dtype
            )
        else:
            attrs = self.o3_angular_basis(
                graph.edge_vector / graph.edge_length.clamp_min(1e-12)
            )
            attrs = attrs.to(self.model_dtype)
            if cutoff is not None:
                attrs = attrs * cutoff.to(self.model_dtype)

        # Node embedding and the first lift depend on the element only.
        # A ghost's first message uses its type, not communicated features.
        table = self.interactions[0].linear_up(self.node_embedding.embedding.weight)
        x = torch.nn.functional.embedding(types, table)
        descriptors = []
        for i, (inter, update, product) in enumerate(
            zip(
                self.interactions,
                self.edge_updates,
                self.products,
            )
        ):
            residual = None
            if i:
                residual = inter.resnetBB(x, local_types, groups)
                x = inter.linear_up(x)
                x = inter.handle_lammps(x, graph.lmp_data, graph.lmp_natoms, i)
            params = prepare_radial(inter, update)
            network = None
            total = None
            conv = inter.rejector
            inputs = conv.reshape_in(x)
            if not self.training and self.memory_mode == "recompute" and radial.is_cuda:
                from .recompute import streamed_heads

                weights, projection, network, total = streamed_heads(
                    params,
                    radial,
                    source_types,
                    target_types,
                    index,
                    local_types.shape[0],
                )
            elif not self.training and self.fused_projection:
                from .convolution import bias_convolution

                hidden, density = radial_heads(
                    params, radial, source_types, target_types
                )
                message = bias_convolution(
                    conv.eqx_tp,
                    inputs,
                    attrs,
                    hidden,
                    params.projection,
                    params.projection_bias,
                    index,
                    local_types.shape[0],
                )
            elif self.training and self.memory_mode == "stream":
                hidden, density = radial_heads(
                    params, radial, source_types, target_types
                )
                weights = project(hidden, params.projection, params.projection_bias)
                projection = weights.new_empty((0, weights.shape[-1]))
            else:
                # EQX tiles the affine path projection inside both convolution
                # and VJP. Its scratch is bounded independently of edge count;
                # four full E-by-2688 matrices must never survive an MD step.
                weights, projection, density = affine_heads(
                    params, radial, source_types, target_types
                )
            if self.training or not self.fused_projection:
                message = conv.eqx_tp(
                    inputs,
                    attrs,
                    weights,
                    projection,
                    index,
                    num_nodes=local_types.shape[0],
                    radial_network=network,
                )
            message = conv.reshape_out(message)
            norm = inter.scatter_norm
            if total is None:
                total = density.new_zeros((local_types.shape[0], 1)).index_add(
                    0, index[1], density
                )
            x = inter.linear_down(message) / (total[:local] * norm.beta + norm.alpha)
            x = inter.linear_nonlinearity(inter.nonlinearity(x))
            x = product(x, local_types, residual, data["batch"], groups)
            descriptors.append(x)

        return {
            "descriptors": descriptors,
            "uie_feats": None,
            "noise_mask_tensor": None,
            "dens_batch_mask_tensor": None,
            "magnetic_radial_basis": None,
        }


def _validate(rep):
    def require(condition, message):
        if not condition:
            raise ValueError(f"Not a supported TACE-OAM-L architecture: {message}")

    require(
        isinstance(rep, NodeRepresentation), "expected spherical node representation"
    )
    require(
        (rep.num_channel, rep.num_layers, rep.num_elements) == (64, 5, 89),
        "expected C=64, five layers and 89 species",
    )
    require(
        len(rep.interactions) == len(rep.edge_updates) == len(rep.products) == 5,
        "expected five matching interaction/update/product blocks",
    )
    require(
        type(rep.node_embedding).__name__ == "LinearNodeEmbedding"
        and type(rep.edge_embedding).__name__ == "NonLinearEdgeEmbedding",
        "unexpected node/edge embedding",
    )
    require(
        not rep.use_dens
        and not rep.use_magnetic_interaction
        and not hasattr(rep, "uie_embedding")
        and rep.uee_embeddings is None
        and not hasattr(rep, "final_norm"),
        "extra embeddings/normalization",
    )
    require(not any(has_lora(m) for m in rep.modules()), "LoRA adapters")
    for i, (inter, update, product) in enumerate(
        zip(rep.interactions, rep.edge_updates, rep.products)
    ):
        require(
            type(inter).__name__ == "O3CgtpInteraction"
            and type(product) is CgtpACE
            and product.correlation == 2
            and len(product.aces) == 1
            and not product.agnostic
            and not product.use_generalized_cg
            and len(product.coefs) == 2
            and all(type(coef) is e3nnElementLinear for coef in product.coefs),
            "non-OAM-L interaction/product",
        )
        require(
            type(update).__name__ == "Element2EdgeUpdate"
            and inter.edge_info.dims == [192, 128, 384 if i == 0 else 2688]
            and type(inter.scatter_norm).__name__ == "NoCutoffDensityScatterNorm",
            "unexpected radial network or normalization",
        )
        require(
            tuple((m, ir.l) for m, ir in inter.rejector.irreps_in2)
            == tuple((1, l) for l in range(6)),
            "expected degrees zero through five",
        )
        require(
            not any(
                hasattr(inter, n) for n in ("norm1", "norm2", "resnetBA", "resnetAB")
            )
            and hasattr(inter, "resnetBB") == bool(i)
            and (
                not i
                or (
                    inter.resnet_linear_type == "aware"
                    and type(inter.resnetBB) is e3nnElementLinear
                )
            ),
            "unexpected residual path",
        )


def optimize_oam_l(
    model,
    *,
    memory_mode="stream",
    fused_geometry=None,
    fused_projection=False,
    grouped_coefficients=False,
):
    """Return an independent adapter sharing the input model's parameter objects.

    Supports the released TACE-OAM-L graph in its native model precision. The
    eager/compile adapter, target properties, fidelity, and LAMMPS flags are
    retained. Conversion rejects incompatible architectures and changes no
    process-wide acceleration, precision or optimizer settings. For inference,
    `stream` retains the narrow radial head and projects paths in tiles;
    `recompute` also reevaluates that head inside each convolution/VJP tile.
    Both are shape-polymorphic and evaluate the same physical model.
    In training, `stream` materializes the last projection for speed and
    `recompute` tiles this projection for the forward and mixed derivatives.
    `fused_geometry=True` fuses all six FP64 harmonic degrees and their
    cutoff. By default this is enabled in training and disabled in evaluation.
    `fused_projection=True` uses a K=128 affine path projection on bounded
    edge tiles and is supported in `stream` mode.
    `grouped_coefficients=True` shares one GPU-side species permutation across
    frozen atom maps, without reordering atoms, neighbor lists or features.
    """
    if memory_mode not in ("stream", "recompute"):
        raise ValueError("memory_mode must be 'stream' or 'recompute'")
    if fused_projection and memory_mode != "stream":
        raise ValueError("fused_projection requires memory_mode='stream'")
    rep = getattr(getattr(model, "readout_fn", None), "representation", None)
    if isinstance(rep, _OAMLRepresentation):
        if (
            rep.memory_mode == memory_mode
            and rep.fused_geometry == fused_geometry
            and rep.fused_projection == fused_projection
            and rep.grouped_coefficients == grouped_coefficients
        ):
            return model
    else:
        _validate(rep)
    # FX-generated modules may register their constants during deepcopy. Match
    # the source's persistent state so canonical state_dicts remain loadable.
    persistent = set(model.state_dict())
    result = copy.deepcopy(model, {id(p): p for p in model.parameters()})
    for path, child in result.named_modules():
        for name in child._buffers:
            if (f"{path}.{name}" if path else name) not in persistent:
                child._non_persistent_buffers_set.add(name)
    rep = result.readout_fn.representation
    rep.memory_mode = memory_mode
    for product in rep.products:
        product.tiled_quadratic = memory_mode == "recompute"
    rep.fused_geometry = fused_geometry
    rep.fused_projection = bool(fused_projection)
    rep.grouped_coefficients = bool(grouped_coefficients)
    if rep.fused_geometry is not False and (
        rep.o3_angular_basis.normalize
        or rep.o3_angular_basis.normalization != "component"
        or list(rep.o3_angular_basis._ls_list) != list(range(6))
    ):
        raise ValueError(
            "Fused geometry requires unnormalized component harmonics l=0..5"
        )
    if isinstance(rep, _OAMLRepresentation):
        return result
    rep.__class__ = _OAMLRepresentation
    for product in rep.products:
        product.__class__ = _QuadraticACE
        product.prepare_paths()
    from .element import OAMLElementLinear

    for module in rep.modules():
        if type(module) is e3nnElementLinear:
            module.__class__ = OAMLElementLinear
            module.prepare_plan()
    return result
