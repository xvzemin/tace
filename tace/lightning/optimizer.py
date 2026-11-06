################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from collections.abc import Callable, Mapping, Sequence
from fnmatch import fnmatchcase

import torch
from e3nn.nn import BatchNorm

from tace.models.blocks import ScaleShift
from tace.models.linear import (
    e3nnElementLinear,
    e3nnLinear,
    e3nnMoEElementLinear,
    mlpLinear,
    torchLinear,
)
from tace.models.radial import (
    AgnesiTransform,
    RadialBasis,
    SoftTransform,
    ZBLBasis,
    j0SphericalBesselBasis,
    jnSphericalBesselBasis,
)
from tace.models.spherical.base import ReadOut
from tace.models.spherical.edge import Element2EdgeUpdate
from tace.models.spherical.layer_norm import EquivariantMergeLayerNorm
from tace.models.spherical.mtace import MagneticBasis
from tace.models.spherical.node import SphericalTensorNodeEmbedding
from tace.models.spherical.prod import MaceACE
from tace.models.spherical.readout import DirectHessianReadout, EdgeReadOut
from tace.models.spherical.scatter_norm import ScatterNorm
from tace.utils.loss.uncertainty import UncertaintyLoss

# Exclude all parameters recursively, including those of child modules.
NO_WEIGHT_DECAY_MODULES = (
    torch.nn.LayerNorm,
    torch.nn.RMSNorm,
    torch.nn.GroupNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.Embedding,
    torch.nn.EmbeddingBag,
    torch.nn.PReLU,
    BatchNorm,
    EquivariantMergeLayerNorm,
    ScatterNorm,
    ScaleShift,
    AgnesiTransform,
    SoftTransform,
    RadialBasis,
    j0SphericalBesselBasis,
    jnSphericalBesselBasis,
    MagneticBasis,
    ZBLBasis,
    UncertaintyLoss,
)

# Glob patterns match full parameter names at any nesting depth.
NO_WEIGHT_DECAY_NAMES = (
    "*bias",
    "*bias.*",
    "biases",
    "biases.*",
    "*.biases",
    "*.biases.*",
    "temperature_logit",
    "*.temperature_logit",
    "*.router.*",
)

# Relative names select output maps, one-hot lookups and physical parameters.
# Qualified names keep external model packages optional.
NO_WEIGHT_DECAY_PARAMETERS = {
    ReadOut: lambda module: (
        f"linear2.{len(module.linear2) - 1}.weight*"
        if module.last_layer
        else "linear1.*.weight*",
    ),
    EdgeReadOut: ("linear_nonlinearity.weight*",),
    DirectHessianReadout: ("node_readout.weight*",),
    SphericalTensorNodeEmbedding: ("node_embedding.weight*",),
    Element2EdgeUpdate: ("source_embedding.weight*", "target_embedding.weight*"),
    "equivariantx.models.sevennet.conversion.IrrepsConvolution": ("denominator",),
    "equivariantx.models.equflash.conversion.FullConv": ("denominator",),
    "sevenn.nn.convolution.IrrepsConvolution": ("denominator",),
    "sevenn.nn.convolution.IrrepsScatterGatterFusedConvolution": ("denominator",),
    "GGNN.model.EquFlashV2.nn.convolution.FullConv": ("denominator",),
    "GGNN.model.EquFlashV2.nn.normalization.EquivariantRMSNormArraySphericalHarmonicsV2": (
        "*",
    ),
    "GGNN.model.EquFlashV2.nn.scale.SpeciesWiseRescale": ("*",),
    "GGNN.model.EquFlashV2.nn.edge_embedding.BesselBasis": ("*",),
    "prophet.layer_norm.RMSLayerNorm": ("*",),
    "nequip.nn.atomwise.PerTypeScaleShift": ("*",),
    "nequip.nn.embedding._edge.BesselEdgeLengthEncoding": ("*",),
    "nequip.nn.pair_potential.LennardJones": ("epsilon", "sigma", "delta"),
    "mace.modules.blocks.RealAgnosticResidualNonLinearInteractionBlock": (
        "alpha",
        "beta",
    ),
    "mace.modules.radial.BesselBasis": ("*",),
    "mace.modules.radial.GaussianBasis": ("*",),
    "mace.modules.radial.ZBLBasis": ("*",),
    "mace.modules.radial.AgnesiTransform": ("*",),
    "mace.modules.radial.SoftTransform": ("*",),
}

# These subtrees use Adam independently of their weight-decay setting.
NO_MUON_MODULES = (
    torch.nn.LayerNorm,
    torch.nn.RMSNorm,
    torch.nn.GroupNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.Embedding,
    torch.nn.EmbeddingBag,
    torch.nn.PReLU,
    BatchNorm,
    EquivariantMergeLayerNorm,
    ScaleShift,
    AgnesiTransform,
    SoftTransform,
    RadialBasis,
    j0SphericalBesselBasis,
    jnSphericalBesselBasis,
    MagneticBasis,
    ZBLBasis,
    UncertaintyLoss,
)

NO_MUON_NAMES = (
    "*bias",
    "*bias.*",
    "*.biases",
    "*.biases.*",
    "*.temperature_logit",
    "*.router.*",
    "*lora_A*",
    "*lora_B*",
)


def linear_parameter_roles(module: torch.nn.Module) -> dict[str, str]:
    """Describe independent channel matrices without changing their storage."""
    roles = {"bias": "adam", "lora_A.*": "adam", "lora_B.*": "adam"}
    weight = module.weight
    if not isinstance(weight, torch.nn.ParameterList):
        roles["weight"] = "packed"
        return roles
    prefix = ()
    if isinstance(module, (e3nnElementLinear, e3nnMoEElementLinear)):
        prefix = (module.num_elements,)
    if isinstance(module, e3nnMoEElementLinear):
        prefix += (module.num_experts,)
    shapes = [tuple(ins.path_shape) for ins in module.linear.instructions]
    if len(weight) != len(shapes):
        raise ValueError("Linear weights must match their instruction paths.")
    for index, (param, shape) in enumerate(zip(weight, shapes)):
        if len(shape) != 2 or tuple(param.shape) != prefix + shape:
            raise ValueError("Linear weights must retain their declared channel axes.")
        roles[f"weight.{index}"] = "batched_matrix" if prefix else "matrix"
    return roles


def readout_parameter_roles(module: ReadOut) -> dict[str, str]:
    """Exclude physical output maps, retaining hidden channel maps."""
    if module.last_layer:
        return {f"linear2.{len(module.linear2) - 1}.*": "adam"}
    return {"linear1.*": "adam"}


# Select by parameter role; dimensions only validate the declared matrix axes.
# Qualified names allow optional libraries to remain unimported.
PARAMETER_RULES = {
    torch.nn.Linear: {"weight": "matrix", "bias": "adam"},
    torchLinear: {"weight": "matrix", "bias": "adam"},
    mlpLinear: {"weight": "matrix", "bias": "adam"},
    e3nnLinear: linear_parameter_roles,
    e3nnElementLinear: linear_parameter_roles,
    e3nnMoEElementLinear: linear_parameter_roles,
    "e3nn.nn._fc._Layer": {"weight": "matrix"},
    "e3nn.o3._linear.Linear": {"weight": "packed", "bias": "adam"},
    "e3nn.o3._tensor_product._tensor_product.TensorProduct": {"weight": "packed"},
    SphericalTensorNodeEmbedding: {"node_embedding.*": "adam"},
    Element2EdgeUpdate: {"source_embedding.*": "adam", "target_embedding.*": "adam"},
    ReadOut: readout_parameter_roles,
    EdgeReadOut: {"linear_nonlinearity.*": "adam"},
    DirectHessianReadout: {"node_readout.*": "adam"},
    "tace.models.spherical.scatter_norm.DensityScatterNorm": {
        "alpha": "adam",
        "beta": "adam",
    },
    "tace.models.radial.j0SphericalBesselBasis": {"bessel_weights": "adam"},
    "tace.models.radial.jnSphericalBesselBasis": {"zeros": "adam"},
    MaceACE: {"weight": "batched_matrix"},
    "tace.models.spherical.symmetric_contraction.Contraction": {
        "weights_max": "batched_matrix",
        "weights.*": "batched_matrix",
    },
    "equivariantx.o2.linear.Linear": {"weight": "packed", "bias": "adam"},
    "equivariantx.co2.linear.Linear": {"weight": "packed", "bias": "adam"},
    "equivariantx.co3.linear.Linear": {"weight": "packed", "bias": "adam"},
    "equivariantx.o2.tensor_product.TensorProduct": {"weight": "packed"},
    "equivariantx.co2.tensor_product.TensorProduct": {"weight": "packed"},
    "equivariantx.co3.ictd.tensor_product.TensorProduct": {"weight": "packed"},
    "equivariantx.o2.o3_tensor_product.O3TensorProduct": {"weight": "packed"},
    "equivariantx.co2.o3_tensor_product.O3TensorProduct": {"weight": "packed"},
    "equivariantx.o3.tensor_product.FactorizedTensorProduct": {"weight": "adam"},
    "equivariantx.o3.wigner6j.tensor_product.Wigner6jTensorProduct": {
        "weight1": "packed",
        "weight2": "packed",
    },
    "equivariantx.models.mace.product.SymmetricContraction": {
        "weight": "batched_matrix",
        "contractions.*.weights_max": "batched_matrix",
        "contractions.*.weights.*": "batched_matrix",
    },
    "mace.modules.symmetric_contraction.Contraction": {
        "weights_max": "batched_matrix",
        "weights.*": "batched_matrix",
    },
}

ParameterRules = Mapping[
    type | str,
    Mapping[str, str] | Callable[[torch.nn.Module], Mapping[str, str]],
]

WeightDecayRules = Mapping[
    type | str,
    Sequence[str] | Callable[[torch.nn.Module], Sequence[str]],
]


def get_weight_decay_exclusions(
    model: torch.nn.Module,
    parameter_rules: WeightDecayRules | None = None,
) -> set[str]:
    """Return trainable parameter names excluded from weight decay.

    Parameters
    ----------
    model : torch.nn.Module
        Model and trainable loss modules.
    parameter_rules : mapping, optional
        Module types or qualified class names mapped to relative parameter-name
        globs, or callables returning those globs. Entries override matching
        keys in :data:`NO_WEIGHT_DECAY_PARAMETERS`.

    Returns
    -------
    set of str
        Exempt names, including all aliases of shared parameters. Exclusions
        from :data:`NO_WEIGHT_DECAY_MODULES` apply recursively; name exclusions
        from :data:`NO_WEIGHT_DECAY_NAMES` apply at any nesting depth.

    Examples
    --------
    >>> get_weight_decay_exclusions(torch.nn.Linear(4, 8))
    {'bias'}
    >>> rules = {torch.nn.Linear: ('weight',)}
    >>> sorted(get_weight_decay_exclusions(torch.nn.Linear(4, 8), rules))
    ['bias', 'weight']
    """
    rules = {**NO_WEIGHT_DECAY_PARAMETERS, **(parameter_rules or {})}
    aliases = [
        (name, param)
        for name, param in model.named_parameters(remove_duplicate=False)
        if param.requires_grad
    ]
    excluded_ids = set()
    for module in model.modules():
        if isinstance(module, NO_WEIGHT_DECAY_MODULES):
            excluded_ids.update(id(param) for param in module.parameters())
            continue
        for cls in type(module).__mro__:
            patterns = rules.get(cls, rules.get(f"{cls.__module__}.{cls.__qualname__}"))
            if patterns is None:
                continue
            patterns = patterns(module) if callable(patterns) else patterns
            if (
                isinstance(patterns, str)
                or not isinstance(patterns, Sequence)
                or any(not isinstance(pattern, str) for pattern in patterns)
            ):
                raise TypeError("Weight-decay rules must return a sequence of globs.")
            excluded_ids.update(
                id(param)
                for name, param in module.named_parameters(remove_duplicate=False)
                if any(fnmatchcase(name, pattern) for pattern in patterns)
            )
            break
    excluded_ids.update(
        id(param)
        for name, param in aliases
        if any(
            fnmatchcase(name, pattern) or fnmatchcase(f".{name}", pattern)
            for pattern in NO_WEIGHT_DECAY_NAMES
        )
    )
    return {name for name, param in aliases if id(param) in excluded_ids}


def get_parameter_roles(
    model: torch.nn.Module,
    parameter_rules: ParameterRules | None = None,
) -> dict[str, str]:
    """Assign roles to trainable parameters, including shared aliases.

    Parameters
    ----------
    model : torch.nn.Module
        Model to inspect without modifying parameters or buffers.
    parameter_rules : mapping, optional
        Module types or qualified class names mapped to relative parameter-name
        globs and roles. A callable may construct the rules from a module.
        ``matrix`` denotes a two-dimensional map; ``batched_matrix`` keeps
        leading element or expert axes independent. ``packed`` and ``adam``
        select Adam without inferring matrix axes from storage dimensions.
        Additional rules replace matching entries in :data:`PARAMETER_RULES`.

    Returns
    -------
    dict
        Full parameter names mapped to roles. Unrecognized parameters use
        ``unknown``. Semantic Adam exclusions take precedence.

    Examples
    --------
    >>> get_parameter_roles(torch.nn.Linear(2, 3))
    {'weight': 'matrix', 'bias': 'adam'}
    """
    rules = {**PARAMETER_RULES, **(parameter_rules or {})}
    roles = {
        name: set()
        for name, param in model.named_parameters(remove_duplicate=False)
        if param.requires_grad
    }
    for prefix, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, NO_MUON_MODULES):
            selected = {"*": "adam"}
        else:
            selected = {}
            for cls in type(module).__mro__:
                selected = rules.get(
                    cls, rules.get(f"{cls.__module__}.{cls.__qualname__}")
                )
                if selected is not None:
                    selected = selected(module) if callable(selected) else selected
                    break
            if selected is None:
                continue
        if any(
            role not in {"matrix", "batched_matrix", "packed", "adam"}
            for role in selected.values()
        ):
            raise ValueError("Unknown optimizer parameter role.")
        for name, param in module.named_parameters(remove_duplicate=False):
            if not param.requires_grad:
                continue
            full_name = f"{prefix}.{name}" if prefix else name
            roles[full_name].update(
                role for pattern, role in selected.items() if fnmatchcase(name, pattern)
            )
    resolved = {}
    for name, matches in roles.items():
        if any(fnmatchcase(f".{name}", pattern) for pattern in NO_MUON_NAMES):
            matches.add("adam")
        if "adam" in matches:
            resolved[name] = "adam"
        elif len(matches) > 1:
            raise ValueError(
                f"Conflicting optimizer roles for {name}: {sorted(matches)}"
            )
        else:
            resolved[name] = next(iter(matches), "unknown")
    return resolved


def get_optimizer_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    *,
    use_muon: bool = False,
    muon_mode: str = "slice",
    parameter_rules: ParameterRules | None = None,
    weight_decay_rules: WeightDecayRules | None = None,
) -> list[dict]:
    """Group trainable parameters by weight decay and update rule.

    Parameters
    ----------
    model : torch.nn.Module
        Model and trainable loss modules.
    weight_decay : float
        Decay for parameters outside the exclusion rules.
    use_muon : bool
        Automatically select semantic matrices for HybridMuon.
    muon_mode : {'2d', 'flat', 'slice'}
        Validate the optimizer's requested default. Automatic Muon groups use
        ``'slice'``: ordinary matrices are unchanged, while leading element,
        expert and coefficient batch axes remain independent.
    parameter_rules : mapping, optional
        Additional module rules passed to :func:`get_parameter_roles`.
    weight_decay_rules : mapping, optional
        Additional exclusion rules passed to :func:`get_weight_decay_exclusions`,
        independently of Muon parameter roles.

    Returns
    -------
    list of dict
        Optimizer parameter groups. Shared parameters occur once; any exempt
        alias takes precedence. Module exclusions apply recursively.

    Examples
    --------
    >>> model = torch.nn.Linear(4, 8)
    >>> groups = get_optimizer_param_groups(model, 0.01)
    >>> optimizer = torch.optim.AdamW(groups, lr=1e-3)
    >>> groups = get_optimizer_param_groups(model, 0.01, use_muon=True)
    >>> rules = {torch.nn.Linear: {"weight": "matrix", "bias": "adam"}}
    >>> groups = get_optimizer_param_groups(
    ...     model, 0.01, use_muon=True, parameter_rules=rules
    ... )
    """
    named_params = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    aliases = [
        (name, param)
        for name, param in model.named_parameters(remove_duplicate=False)
        if param.requires_grad
    ]
    no_decay_names = get_weight_decay_exclusions(model, weight_decay_rules)
    no_decay_ids = {id(param) for name, param in aliases if name in no_decay_names}

    muon_ids = set()
    parameter_roles = {}
    if use_muon:
        if muon_mode not in {"2d", "flat", "slice"}:
            raise ValueError("muon_mode must be '2d', 'flat', or 'slice'.")
        parameter_roles = get_parameter_roles(model, parameter_rules)
        excluded_ids = set()
        for name, param in aliases:
            role = parameter_roles[name]
            if role in {"matrix", "batched_matrix"}:
                if (role == "matrix" and param.ndim != 2) or (
                    role == "batched_matrix" and param.ndim < 3
                ):
                    raise ValueError(
                        f"Parameter {name} has shape {tuple(param.shape)}, "
                        f"incompatible with its {role} optimizer role."
                    )
                if param.numel() and min(param.shape[-2:]) > 1:
                    muon_ids.add(id(param))
            else:
                excluded_ids.add(id(param))
        # Shared aliases must agree on a matrix role; exemptions take precedence.
        muon_ids.difference_update(excluded_ids)
        packed = [name for name, role in parameter_roles.items() if role == "packed"]
        unknown = [
            name
            for name, param in aliases
            if parameter_roles[name] == "unknown" and param.ndim >= 2
        ]
        if packed:
            logging.warning(
                "HybridMuon keeps %d packed weights on Adam. For TACE Linear "
                "wrappers, set TACE_USE_MATRIX_WEIGHT=1 before construction. "
                "Packed EQX/e3nn weights remain on Adam; their element, expert, "
                "channel and instruction axes must not be inferred from shape.",
                len(packed),
            )
        if unknown:
            logging.warning(
                "HybridMuon uses Adam for parameters without a declared role: %s. "
                "Define their module rules in parameter_rules before using Muon.",
                ", ".join(unknown),
            )

    groups = []
    for decay in (True, False):
        for muon in (True, False) if use_muon else (False,):
            params = [
                param
                for _, param in named_params
                if (id(param) not in no_decay_ids) == decay
                and (id(param) in muon_ids) == muon
            ]
            group = {"params": params, "weight_decay": weight_decay if decay else 0.0}
            if use_muon:
                group["use_muon"] = muon
                if muon:
                    group["muon_mode"] = "slice"
            groups.append(group)
    if use_muon:
        muon_count = sum(
            param.numel() for _, param in named_params if id(param) in muon_ids
        )
        total_count = sum(param.numel() for _, param in named_params)
        logging.info(
            "HybridMuon parameters: Muon=%d, Adam=%d",
            muon_count,
            total_count - muon_count,
        )
    for name, param in named_params:
        logging.debug(
            "Optimizer parameter %s: weight_decay=%g%s",
            name,
            0.0 if id(param) in no_decay_ids else weight_decay,
            f", role={parameter_roles[name]}, use_muon={id(param) in muon_ids}"
            if use_muon
            else "",
        )
    return groups
