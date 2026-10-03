"""Receiver-tiled TACE execution with layer-boundary storage."""

import torch

from eqx.utils import copy_model


class ReceiverPlan:
    """Partition receivers while retaining every incident directed edge."""

    def __init__(self, edge_index, num_nodes, tile_size):
        order = edge_index[1].argsort(stable=True)
        indices = edge_index[:, order]
        boundaries = torch.arange(0, num_nodes, tile_size, device=indices.device)
        boundaries = torch.cat((boundaries, boundaries.new_tensor([num_nodes])))
        offsets = torch.searchsorted(indices[1].contiguous(), boundaries).tolist()
        self.tiles = []
        for begin, start, stop in zip(
            range(0, num_nodes, tile_size), offsets, offsets[1:]
        ):
            end = min(begin + tile_size, num_nodes)
            sources, inverse = indices[0, start:stop].unique(return_inverse=True)
            local = torch.stack((inverse, indices[1, start:stop] - begin))
            self.tiles.append((begin, end, order[start:stop], sources, local))


def gradients(outputs, inputs, seed, create_graph=False):
    """Evaluate a VJP, including identically unused inputs."""
    active = [i for i, value in enumerate(inputs) if value.requires_grad]
    result = [None] * len(inputs)
    if active:
        values = torch.autograd.grad(
            outputs,
            [inputs[i] for i in active],
            seed,
            allow_unused=True,
            create_graph=create_graph,
        )
        for i, value in zip(active, values):
            result[i] = value
    return result


class CompressedMessage(torch.autograd.Function):
    """Replay a frozen convolution from its compressed node output."""

    @staticmethod
    def forward(
        ctx, inter, index, message, features, radial, projection, cutoff, vectors
    ):
        conv = inter.rejector.eqx_tp
        if conv.normalize:
            raise ValueError("The receiver plan expects unnormalized solid harmonics.")
        ctx.inter = inter
        ctx.count = message.shape[0]
        ctx.save_for_backward(index, features, radial, projection, cutoff, vectors)
        return message.view_as(message)

    @staticmethod
    def backward(ctx, seed):
        from eqx.conv.o3.convolution import contraction_adjoint

        index, features, radial, projection, cutoff, vectors = ctx.saved_tensors
        inter = ctx.inter
        rejector = inter.rejector
        conv = rejector.eqx_tp
        # The input adjoint of a frozen linear map is independent of its input.
        # Only a receiver tile of path features is allocated, never edge paths.
        if hasattr(inter.linear_down, "transpose"):
            path_seed = rejector.reshape_out.inverse(inter.linear_down.transpose(seed))
            paths = seed.new_empty(1).expand(ctx.count, conv.output_dim)
        else:
            with torch.enable_grad():
                paths = seed.new_zeros((ctx.count, conv.output_dim), requires_grad=True)
                message = inter.linear_down(rejector.reshape_out(paths))
                path_seed = torch.autograd.grad(message, paths, seed)[0]
        operands = [features, radial, projection, cutoff, paths.detach(), vectors]
        result = contraction_adjoint(
            conv.harmonic_metadata,
            ((tuple(range(6)), False, ((4, 0),)),),
            index[0],
            index[1],
            operands,
            [path_seed],
            [
                features.requires_grad,
                radial.requires_grad,
                False,
                cutoff.requires_grad,
                False,
                vectors.requires_grad,
            ],
        )
        return None, None, None, result[0], result[1], None, result[3], result[5]


class LayerSweep(torch.autograd.Function):
    """Retain node boundaries and replay receiver workspaces during reverse."""

    @staticmethod
    def forward(ctx, module, plan, attrs, types, batch, vectors, *parameters):
        states, messages = module.evaluate(
            plan,
            attrs,
            types,
            batch,
            vectors,
            retain_messages=module.cache_messages
            and not any(p.requires_grad for p in parameters),
        )
        ctx.module, ctx.plan = module, plan
        ctx.num_parameters = len(parameters)
        ctx.num_states = len(states)
        ctx.save_for_backward(
            attrs, types, batch, vectors, *parameters, *states, *messages
        )
        ctx.set_materialize_grads(False)
        return tuple(states[1:])

    @staticmethod
    def backward(ctx, *seeds):
        attrs, types, batch, vectors, *saved = ctx.saved_tensors
        parameters = saved[: ctx.num_parameters]
        states = saved[ctx.num_parameters : ctx.num_parameters + ctx.num_states]
        messages = saved[ctx.num_parameters + ctx.num_states :]
        module, plan = ctx.module, ctx.plan
        if torch.is_grad_enabled():
            # A differentiable replay preserves mixed parameter/geometry and
            # higher derivatives. Its graph is larger than the inference sweep.
            values = module.evaluate(plan, attrs, types, batch, vectors)[0][1:]
            active = [
                (value, seed) for value, seed in zip(values, seeds) if seed is not None
            ]
            inputs = [vectors, *parameters]
            if active:
                result = gradients(
                    [value for value, _ in active],
                    inputs,
                    [seed for _, seed in active],
                    create_graph=True,
                )
            else:
                result = [None] * len(inputs)
            return (None, None, None, None, None, *result)

        dv = torch.zeros_like(vectors)
        parameter_grads = [None] * len(parameters)
        parameter_slots = {id(value): i for i, value in enumerate(parameters)}

        def accumulate(parameters, values):
            for parameter, value in zip(parameters, values):
                if value is None:
                    continue
                slot = parameter_slots[id(parameter)]
                if parameter_grads[slot] is None:
                    parameter_grads[slot] = value
                else:
                    parameter_grads[slot].add_(value)

        adjoint = None
        for layer in reversed(range(len(module.reference.interactions))):
            if seeds[layer] is not None:
                adjoint = seeds[layer] if adjoint is None else adjoint + seeds[layer]
            if adjoint is None:
                continue
            previous = states[layer]
            source = module.project(layer, previous)
            ds = torch.zeros_like(source)
            dh = torch.zeros_like(previous)
            tile_parameters = module.tile_parameters(layer)
            for tile in plan.tiles:
                begin, end, edges, sources, _ = tile
                x = source[sources].detach().requires_grad_(True)
                h = previous[begin:end].detach().requires_grad_(True)
                v = vectors[edges].detach().requires_grad_(True)
                with torch.enable_grad():
                    value = module.receiver(
                        layer,
                        tile,
                        attrs,
                        types,
                        batch,
                        x,
                        h,
                        v,
                        saved_message=messages[layer][begin:end] if messages else None,
                    )
                    gx, gh, gv, *gp = gradients(
                        value, [x, h, v, *tile_parameters], adjoint[begin:end]
                    )
                if gx is not None:
                    ds.index_add_(0, sources, gx)
                if gh is not None:
                    dh[begin:end].add_(gh)
                if gv is not None:
                    dv.index_add_(0, edges, gv)
                accumulate(tile_parameters, gp)
            inter = module.reference.interactions[layer]
            project_parameters = list(inter.linear_up.parameters())
            h = previous.detach().requires_grad_(True)
            with torch.enable_grad():
                x = module.project(layer, h)
                gh, *gp = gradients(x, [h, *project_parameters], ds)
            if gh is not None:
                dh.add_(gh)
            accumulate(project_parameters, gp)
            adjoint = dh
        if adjoint is not None:
            embedding = module.reference.node_embedding
            embedding_parameters = list(embedding.parameters())
            if any(p.requires_grad for p in embedding_parameters):
                with torch.enable_grad():
                    value = embedding.elem_emb1(attrs)
                    gp = gradients(value, embedding_parameters, adjoint)
                accumulate(embedding_parameters, gp)
        return (None, None, None, None, None, dv, *parameter_grads)


class Representation(torch.nn.Module):
    """Evaluate TACE layers using bounded receiver workspaces.

    Parameters
    ----------
    reference : torch.nn.Module
        TACE representation with scalar element embedding, O(3) CGTP
        interactions, and channel-wise ACE products.
    tile_size : int
        Maximum number of receivers in each workspace.
    cache_messages : bool
        Retain compressed node messages for frozen models, avoiding a second
        convolution forward during the force sweep.

    Notes
    -----
    Inter-layer features remain graph-wide. Radial and density MLPs, path
    outputs, gates, and ACE intermediates are evaluated per receiver tile.
    No interpolation or parameter approximation is used. First derivatives
    replay one tile at a time. Higher derivatives use a differentiable tiled
    replay and may retain more memory. The ordinary TACE readout is unchanged.
    """

    def __init__(self, reference, tile_size, cache_messages):
        super().__init__()
        self.reference = reference
        self.tile_size = tile_size
        self.cache_messages = cache_messages

    def project(self, layer, features):
        """Apply the source map once per node, before gathering."""
        return self.reference.interactions[layer].linear_up(features)

    def tile_parameters(self, layer):
        """Return unique parameters used by a receiver tile."""
        rep = self.reference
        modules = (
            rep.radial_basis,
            rep.edge_embedding,
            rep.edge_updates[layer],
            rep.interactions[layer],
            rep.products[layer],
        )
        excluded = {id(p) for p in rep.interactions[layer].linear_up.parameters()}
        return list(
            {
                id(p): p
                for m in modules
                for p in m.parameters()
                if id(p) not in excluded
            }.values()
        )

    def receiver(
        self,
        layer,
        tile,
        attrs,
        types,
        batch,
        source,
        previous,
        vectors,
        saved_message=None,
        messages=None,
    ):
        """Complete a receiver sum before its linear map, gate, and ACE."""
        rep = self.reference
        inter, product, update = (
            rep.interactions[layer],
            rep.products[layer],
            rep.edge_updates[layer],
        )
        begin, end, _, sources, index = tile
        count = end - begin
        length = vectors.square().sum(-1, keepdim=True).sqrt() + 1e-9
        # The admitted radial functions and edge embeddings do not mix nodes.
        radial, cutoff = rep.radial_basis(
            length, attrs, index, rep.atomic_numbers, node_type=types
        )
        features = rep.edge_embedding(None, attrs, radial, index, cutoff)
        if hasattr(update, "source_embedding"):
            features = (
                (features, None),
                (update.target_embedding(attrs[begin:end]), index[1]),
                (update.source_embedding(attrs[sources]), index[0]),
            )
        weights = features
        for linear in inter.edge_info.mlp[:-1]:
            weights = linear(weights)
        if isinstance(weights, tuple):
            weights = torch.cat(
                [value if index is None else value[index] for value, index in weights],
                -1,
            )
        last = inter.edge_info.mlp[-1]
        projection = last.get_weight()
        if last.bias is not None:
            weights = torch.cat((weights, torch.ones_like(weights[:, :1])), -1)
            projection = torch.cat((projection, last.bias[None]), 0)
        rejector = inter.rejector
        if saved_message is None:
            message = rejector.eqx_tp(
                rejector.reshape_in(source),
                None,
                weights,
                projection,
                index,
                count,
                vectors=vectors / length,
                amplitudes=cutoff,
            )
            message = inter.linear_down(rejector.reshape_out(message))
        else:
            amplitudes = (
                source.new_ones((1, rejector.eqx_tp.amplitude_dim))
                if cutoff is None
                else cutoff.expand(-1, rejector.eqx_tp.amplitude_dim)
            )
            message = CompressedMessage.apply(
                inter,
                index,
                saved_message,
                rejector.reshape_in(source),
                weights,
                projection,
                amplitudes,
                vectors / length,
            )
        if messages is not None:
            messages.append(message)
        density = None
        if hasattr(inter, "edge_density"):
            value = torch.tanh(inter.edge_density(features).square())
            if cutoff is not None and inter.apply_density_cutoff:
                value = value * cutoff
            density = value.new_zeros((count, value.shape[-1])).index_add(
                0, index[1], value
            )
            density = density * inter.beta + inter.alpha
            density = density.masked_fill(density == 0, 1e-9)
        message = inter._normalize_messages(message, density)
        message = inter.linear_nonlinearity(inter.nonlinearity(message))
        residual = None
        if hasattr(inter, "resnetBB"):
            residual = (
                inter.resnetBB(previous, attrs[begin:end], types[begin:end])
                if inter.resnet_linear_type == "aware"
                else inter.resnetBB(previous)
            )
        return product(
            message,
            attrs[begin:end],
            residual,
            batch[begin:end],
            node_type=types[begin:end],
        )

    def evaluate(self, plan, attrs, types, batch, vectors, retain_messages=False):
        """Evaluate all layers, retaining only their node boundaries."""
        states = [self.reference.node_embedding.elem_emb1(attrs)]
        messages = []
        for layer in range(len(self.reference.interactions)):
            previous = states[-1]
            source = self.project(layer, previous)
            compressed = [] if retain_messages else None
            values = [
                self.receiver(
                    layer,
                    tile,
                    attrs,
                    types,
                    batch,
                    source[tile[3]],
                    previous[tile[0] : tile[1]],
                    vectors[tile[2]],
                    messages=compressed,
                )
                for tile in plan.tiles
            ]
            states.append(torch.cat(values, 0))
            if retain_messages:
                messages.append(torch.cat(compressed, 0))
        return states, messages

    def forward(self, data, graph):
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                "Receiver-tiled model execution currently requires eager mode. "
                "Use the unconverted model for torch.compile or AOTI."
            )
        if graph.lmp:
            raise NotImplementedError(
                "Receiver-tiled execution requires a serial graph."
            )
        attrs = data["node_attrs"]
        if not attrs.is_cuda:
            raise ValueError("Receiver-tiled TACE execution requires CUDA.")
        if attrs.requires_grad:
            raise NotImplementedError(
                "Differentiable element attributes are not supported."
            )
        plan = ReceiverPlan(data["edge_index"], attrs.shape[0], self.tile_size)
        if not attrs.shape[0]:
            descriptors = [
                attrs.new_empty((0, p.irreps_out.dim)) for p in self.reference.products
            ]
        else:
            descriptors = LayerSweep.apply(
                self,
                plan,
                attrs,
                graph.node_type if graph.node_type is not None else attrs.argmax(-1),
                data["batch"],
                graph.edge_vector,
                *self.reference.parameters(),
            )
        return dict(
            descriptors=list(descriptors),
            uie_feats=None,
            noise_mask_tensor=None,
            dens_batch_mask_tensor=None,
            magnetic_radial_basis=None,
        )


def convert_tace_to_eqx(model, *, tile_size=2048, cache_messages=True, inplace=False):
    """Use receiver-tiled execution for a loaded TACE model.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded TACE tensor model with EQX convolution and ACE enabled.
        TACE-OAM-L is the initial supported architecture.
    tile_size : int, optional
        Maximum number of receivers per tile. Defaults to 2048. The temporary
        edge workspace additionally depends on these receivers' degrees.
    cache_messages : bool, optional
        Save compressed node messages when parameters are frozen. Defaults
        to true. False reduces storage and recomputes these messages in reverse.
    inplace : bool, optional
        Modify the supplied model. Otherwise return an independent copy.

    Returns
    -------
    torch.nn.Module
        Model with the same energy, forces, stress, and ASE interface.

    Notes
    -----
    Load parameters before conversion. Converted state dictionaries require
    an identically converted model. Magnetic, long-range, tensor-embedding,
    stochastic-depth, distributed, and compiled variants are not supported.
    """
    from tace.models._e3nn.edge import (
        Element2EdgeUpdate,
        IdentityEdgeEmbedding,
        IdentityEdgeUpdate,
        LinearEdgeEmbedding,
        NonLinearEdgeEmbedding,
    )
    from tace.models._e3nn.inter import O3CgtpInteraction
    from tace.models._e3nn.node import LinearNodeEmbedding
    from tace.models._e3nn.prod import CgtpACE

    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer.")
    core = getattr(model, "readout_fn", None)
    if not isinstance(core, torch.nn.Module) or not hasattr(core, "representation"):
        raise TypeError("Expected a loaded TACE TensorModel.")
    rep = core.representation
    if isinstance(rep, Representation):
        if rep.tile_size != tile_size or rep.cache_messages != cache_messages:
            raise ValueError("Convert the original model to change the execution plan.")
        return model if inplace else copy_model(model)
    if set(core.target_property) - {"energy", "forces", "stress", "virials"}:
        raise NotImplementedError(
            "This execution plan supports energy, forces, stress, and virials."
        )
    if not isinstance(rep.node_embedding, LinearNodeEmbedding):
        raise NotImplementedError("A scalar element node embedding is required.")
    if type(rep.edge_embedding) not in (
        IdentityEdgeEmbedding,
        LinearEdgeEmbedding,
        NonLinearEdgeEmbedding,
    ) or any(
        type(update) not in (IdentityEdgeUpdate, Element2EdgeUpdate)
        for update in rep.edge_updates
    ):
        raise NotImplementedError(
            "This plan requires scalar radial edge embeddings and identity or element2 updates."
        )
    if (
        rep.use_magnetic_interaction
        or rep.use_dens
        or hasattr(rep, "uie_embedding")
        or rep.uee_embeddings is not None
    ):
        raise NotImplementedError(
            "Magnetic, field, and denoising embeddings are not supported."
        )
    if hasattr(rep, "final_norm") or hasattr(core, "les"):
        raise NotImplementedError(
            "Final normalization and long-range terms are not supported."
        )
    if rep.radial_basis.use_distance_transform:
        raise NotImplementedError(
            "Distance transforms require a separate execution plan."
        )
    for inter, product in zip(rep.interactions, rep.products):
        if type(inter) is not O3CgtpInteraction or not inter.use_eqx:
            raise NotImplementedError(
                "Enable EQX O3 CGTP for every interaction before conversion."
            )
        if type(product) is not CgtpACE or not hasattr(product, "eqx_ace"):
            raise NotImplementedError("Enable EQX channel-wise ACE before conversion.")
        if any(
            hasattr(inter, name)
            for name in ("norm1", "norm2", "resnetBA", "resnetAB", "stochastic_depth")
        ) or hasattr(product, "stochastic_depth"):
            raise NotImplementedError(
                "This execution plan supports deterministic BB residuals without pre-normalization."
            )
    if not inplace:
        model = copy_model(model)
        core = model.readout_fn
    core.representation = Representation(
        core.representation, tile_size, cache_messages
    ).train(model.training)
    return model
