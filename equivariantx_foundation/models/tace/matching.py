"""Structural checks shared by TACE execution schedules."""

import torch
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.nn.modules import module as module_hooks


def spherical_release_representation(model):
    """Return the native spherical representation for a spherical release matcher."""
    from tace.models.spherical.tace import SphericalTACE

    host = getattr(model, "readout_fn", None)
    return host.representation if type(host) is SphericalTACE else None


def tracing_rows(edge_index):
    """Whether edge row maps need to be preserved as tensor operations."""
    return (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or is_fake(edge_index)
    )


def edge_rows(features):
    """Gather each radial partition using its own row map."""
    return tuple(value if rows is None else value[rows] for value, rows in features)


def mapped_radial_inputs(features, network, edge_index, prototype):
    """Bind explicit radial row maps for a directed edge program."""
    from equivariantx.conv import indexed_radial

    (edge, first_rows), *tables = features
    if (
        first_rows is None
        and edge.dtype == prototype.dtype
        and edge.device == prototype.device
        and edge.ndim == 2
        and edge.shape[0] == edge_index.size(1)
        and all(
            rows is not None
            and rows.dtype == torch.int64
            and rows.device == prototype.device
            and rows.shape == (edge.shape[0],)
            and table.ndim == 2
            and table.dtype == edge.dtype
            and table.device == edge.device
            for table, rows in tables
        )
    ):
        return edge, indexed_radial(edge, network, tables) if tables else network
    return torch.cat(edge_rows(features), dim=-1), network


def directed_radial(features, edge_index):
    """Whether the radial partitions use this graph's target and source rows."""
    if isinstance(features, torch.Tensor):
        return True
    if tracing_rows(edge_index):
        return False
    return (
        isinstance(features, tuple)
        and len(features) == 3
        and features[0][1] is None
        and all(
            isinstance(index, torch.Tensor)
            and index.dtype == edge_index.dtype
            and index.is_set_to(edge_index[row])
            for (_, index), row in zip(features[1:], (1, 0))
        )
    )


def streamable_activation(module):
    """Whether a scalar activation has an exact radial program instruction."""
    from tace.models.mlp import ScaledActivation, ScaledSigmoid, ScaledSiLU, ScaledTanh

    if type(module) is ScaledActivation:
        return streamable_activation(module.activation)
    return type(module) in (
        torch.nn.Identity,
        torch.nn.SiLU,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        ScaledSigmoid,
        ScaledSiLU,
        ScaledTanh,
    )


def streamable_radial_mlp(radial):
    """Whether the radial MLP can be evaluated from its live affine weights."""
    from tace.models.linear import has_lora, mlpLinear
    from tace.models.mlp import MLPV1, MLPV2

    return (
        type(radial) in (MLPV1, MLPV2)
        and type(radial.mlp) is torch.nn.Sequential
        and len(radial.mlp) > 0
        and type(radial.mlp[-1]) is mlpLinear
        and all(
            type(m) is mlpLinear
            or (
                type(m) in (torch.nn.LayerNorm, torch.nn.RMSNorm)
                and len(m.normalized_shape) == 1
            )
            or streamable_activation(m)
            for m in radial.mlp
        )
        and not any(
            has_lora(m) or "get_weight" in vars(m) or "weight_and_scale" in vars(m)
            for m in radial.mlp
        )
    )


def _local_custom_call(module):
    if (
        module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
        or module._compiled_call_impl is not None
    ):
        return True
    if "forward" not in vars(module):
        return False
    forward = vars(module)["forward"]
    # RecursiveScriptModule.__getattr__ memoizes _c._get_method("forward")
    # on first use. A Python override or another ScriptModule's same-named
    # method is not that cache. Checking the owner does not invoke forward.
    return not (
        isinstance(module, torch.jit.RecursiveScriptModule)
        and isinstance(forward, torch.ScriptMethod)
        and forward.name == "forward"
        and forward.owner == module._c
    )


def custom_forward(module, *, recursive=False):
    """Return whether changing the call graph would bypass customized execution."""
    if (
        module_hooks._global_forward_hooks
        or module_hooks._global_forward_pre_hooks
        or module_hooks._global_backward_hooks
        or module_hooks._global_backward_pre_hooks
    ):
        return True
    if recursive:
        return any(_local_custom_call(child) for child in module.modules())
    return _local_custom_call(module)


def plain_element_lookups(rep):
    """Can the node and directed endpoint embedding calls be replaced by their weights?"""
    from tace.models.spherical.edge import Element2EdgeUpdate
    from tace.models.spherical.node import LinearNodeEmbedding

    def plain(table):
        # padding/frequency/sparse change the VJP; max_norm also mutates weights.
        return (
            type(table) is torch.nn.Embedding
            and table.padding_idx is None
            and table.max_norm is None
            and not table.scale_grad_by_freq
            and not table.sparse
            and not custom_forward(table)
        )

    node = rep.node_embedding
    return (
        type(node) is LinearNodeEmbedding
        and not custom_forward(node)
        and plain(node.embedding)
        and all(
            type(update) is Element2EdgeUpdate
            and not custom_forward(update)
            and plain(update.source_embedding)
            and plain(update.target_embedding)
            for update in rep.edge_updates
        )
    )


def custom_message_call(layer):
    """A plan must not inline a message helper replaced on a host instance."""
    return bool(
        {
            "_compute_messages",
            "_radial_projection",
            "_conv_weights",
            "_apply_rejector",
            "_convolution_radial",
            "_rotary_radial",
            "handle_lammps",
        }.intersection(vars(layer))
        or "forward_stream" in vars(layer.rejector)
    )


def native_spherical_message(layer, *, host_type=None):
    """Match the actual host call boundaries, not a subclass name or tensor shape."""
    from tace.models.spherical import fused, inter, mtace, o2, tece_oam_rra

    convolutions = {
        inter.O3CgtpInteraction: fused.O3TensorProductConv,
        inter.O2CgtpInteraction: fused.O2TensorProductConv,
        inter.UuO2Interaction: fused.UuO2TensorProductConv,
        inter.UvO2Interaction: o2.UvO2TensorProductConv,
        mtace.O2MagneticInteraction: mtace.MagneticUvO2TensorProductConv,
        mtace.O3MagneticWigner6jInteraction: fused.Wigner6jTensorProductConv,
        tece_oam_rra.UvSO2Interaction: tece_oam_rra.Convolution,
    }
    return convolutions.get(host_type or type(layer)) is type(layer.rejector)


def require_native_messages(rep, *, installed=()):
    """Whole schedules inline message/radial execution; reject opaque boundaries."""
    from tace.models.mlp import BATCH_MLP, MLP

    equivalents = dict(installed)
    radial_types = (*MLP.values(), *BATCH_MLP.values())
    for i, layer in enumerate(rep.interactions):
        if (
            not native_spherical_message(layer, host_type=equivalents.get(type(layer)))
            or type(layer.edge_info) not in radial_types
            or custom_message_call(layer)
            or custom_forward(layer, recursive=True)
        ):
            raise ValueError(
                f"Cannot inline customized message execution at interactions.{i}."
            )


def customized_codegen(model):
    """Whether e3nn's serialization during deepcopy may lose a Python customization."""
    return any(
        _local_custom_call(child)
        for module in model.modules()
        if isinstance(module, CodeGenMixin)
        for name in getattr(module, "__codegen__", ())
        for child in getattr(module, name).modules()
    )


def customized_message_codegen(rep):
    """A host callback can mutate its descendant e3nn generated program's buffers.

    e3nn serializes generated modules during deepcopy. The resulting program
    need not observe Python-side buffer mutations like the original instance,
    even when its enclosing interaction is not lowered at all.
    """
    return any(
        _local_custom_call(owner)
        and any(isinstance(child, CodeGenMixin) for child in owner.modules())
        for layer in rep.interactions
        for owner in layer.modules()
    )


def canonical_component_harmonics(basis, lmax):
    """Check the exact e3nn degree order, conventions and polynomial evaluator."""
    from e3nn.o3._spherical_harmonics import _spherical_harmonics

    if (
        type(basis) is not o3.SphericalHarmonics
        or basis.normalize
        or basis.normalization != "component"
        or basis.irreps_in != o3.Irreps("1o")
        or basis.irreps_out != o3.Irreps.spherical_harmonics(lmax)
        or basis._lmax != lmax
        or not basis._is_range_lmax
        or list(basis._ls_list) != list(range(lmax + 1))
        or custom_forward(basis, recursive=True)
    ):
        return False
    if not hasattr(basis, "sph_func"):
        # Older e3nn called its module-level polynomial directly.
        return "sph_func" not in o3.SphericalHarmonics.forward.__code__.co_names
    evaluator = basis.sph_func
    from .interactions import ComponentPolynomial

    if type(evaluator) is ComponentPolynomial:
        evaluator = evaluator.reference
    if evaluator is _spherical_harmonics:
        return True
    if isinstance(evaluator, torch.jit.ScriptFunction):
        # Compare the existing compiled graph; probing a plan must not script
        # new code or accept an unrelated script function with the same name.
        cached = torch.jit._state._try_get_jit_cached_function(_spherical_harmonics)
        return cached is not None and evaluator.graph is cached.graph
    # torch.compile installs both markers. functools.wraps alone is not proof
    # that a user wrapper actually computes the original polynomial.
    return (
        getattr(evaluator, "_torchdynamo_orig_callable", None) is _spherical_harmonics
        and getattr(evaluator, "__wrapped__", None) is _spherical_harmonics
    )


def canonical_radial(rep):
    """Check the radial modules before sharing work between reciprocal edges."""
    from tace.models.radial import RadialBasis
    from tace.models.spherical.edge import EDGE_EMBEDDING, EDGE_UPDATE

    return (
        type(rep.radial_basis) is RadialBasis
        and type(rep.edge_embedding) in EDGE_EMBEDDING.values()
        and all(type(m) in EDGE_UPDATE.values() for m in rep.edge_updates)
        and not any(
            custom_forward(root, recursive=True)
            for root in (rep.radial_basis, rep.edge_embedding, rep.edge_updates)
        )
    )
