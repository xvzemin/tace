"""Execution plans for TACE representations."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable

import torch

from equivariantx.ops import resolve_backend

from .foundation import shared_parameter_copy
from .matching import custom_forward, customized_codegen, customized_message_codegen


@dataclass(frozen=True)
class Region:
    """Static dispatch decision for a region, not a runtime hit or speed claim.

    `implementation` retains the historical values native/torch/cuda.
    `provenance` distinguishes a new or retargeted regional schedule, an
    already-installed one, a recognizable host selection, and delegation
    to an unclassified host. `provider` names the selected operation's
    provider on the model's current device when known. A regional program
    may still delegate individual input signatures to the host. Node paths
    remain relative to the representation; host paths are relative to
    readout_fn. Use module_path with the original adapter's get_submodule.
    """

    path: str
    implementation: str
    reason: str
    lower: Callable | None = field(default=None, repr=False, compare=False)
    provenance: str = "host_fallback"
    provider: str | None = None
    scope: str = "representation"

    @property
    def module_path(self) -> str:
        """Return the qualified submodule path from the TACE property adapter."""
        root = (
            "readout_fn.representation"
            if self.scope == "representation"
            else self.scope
        )
        return f"{root}.{self.path}"


@dataclass(frozen=True)
class ExecutionPlan:
    """Task, backend and selected whole-model or regional schedule.

    In an eager release plan, `schedule` names the node program; the listed
    host regions are retained property modules, not installs by that release.
    Joint E/F/S and ML-IAP schedules own their output graphs and do not list
    the eager source's heads.
    """

    task: str
    backend: str
    schedule: str
    regions: tuple[Region, ...] = ()
    reason: str = ""


WHOLE_MODEL_SCHEDULES = (
    "tace_oam_l",
    "tace_oam_7m",
    "tece_oam_rra",
    "tace_oam_xl_v2",
)


_KERNEL_MODULES = frozenset(
    f"{__package__}.{name}"
    for name in (
        "interactions",
        "products",
        "cartesian",
        "local",
        "magnetic",
        "wigner6j",
        "tece_oam_rra.regional",
    )
)


def _regional_choice(path, backend, reason, lower=None):
    provenance = "already_regional" if lower is None else "regional"
    return Region(path, backend, reason, lower, provenance, f"eqx_{backend}")


def _select_region(path, module, rule, *, backend, task, host_provider=None):
    old = getattr(module, "_foundation_regional_provider", None)
    if old is not None:
        if old[0] == backend:
            return _regional_choice(path, backend, old[1])
        if custom_forward(module, recursive=True):
            raise ValueError(
                f"Cannot retarget {path}: customized forward or hooks may depend on its current schedule. Use the original host model."
            )
    elif any(type(m).__module__ in _KERNEL_MODULES for m in module.modules()):
        # Manually lowered blocks and pre-receipt pickles lack a reliable
        # record of which overrides belong to the planner.
        raise ValueError(
            f"Cannot identify the installed provider at {path}; re-plan from the original host model."
        )
    lower, detail = rule(module, backend=backend, task=task)
    if old is not None and lower is None:
        raise ValueError(
            f"Cannot retarget {path} from {old[0]} to {backend}: {detail}. Use the original host model."
        )
    if lower is not None:
        return _regional_choice(path, backend, detail, lower)
    provider = host_provider(module) if host_provider is not None else None
    return Region(
        path,
        "native",
        detail,
        provenance="host_selected" if provider else "host_fallback",
        provider=provider,
    )


def release_schedule(model, *, task="inference"):
    """Return the first whole-model schedule supporting the task and architecture."""
    host = getattr(model, "readout_fn", None)
    rep = getattr(host, "representation", None)
    if not isinstance(rep, torch.nn.Module):
        return None, "Whole-model schedules require a TACE property adapter."
    if getattr(rep, "_foundation_regional", False):
        return None, "Keep the existing regional execution plan."
    reason = f"No matching whole-model schedule for {task}."
    for name in WHOLE_MODEL_SCHEDULES:
        package = import_module(f".{name}", __package__)
        if task not in package.TASKS:
            continue
        try:
            if not package.matches(model, task=task):
                continue
        except (ValueError, AttributeError) as error:
            reason = str(error)
            continue
        if custom_forward(rep, recursive=True):
            return None, "Preserve the host's customized node representation."
        return name, "Matched the whole-model architecture and task."
    return None, reason


def plan(model, *, task=None, backend=None, regional_only=False):
    """Inspect a loaded TACE model without changing its execution.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded TACE property adapter, placed on its execution device.
    task : {None, "inference", "training", "joint_training", "lammps"}
        By default use the module's current training mode. Joint training
        selects the specialized E/F/S adapter. LAMMPS selects the frozen
        single-rank ML-IAP export.
    backend : {None, "torch", "cuda"}
        EQX implementation. CPU models use Torch by default.
    regional_only : bool, optional
        Match individual regions instead of replacing the whole architecture.
        Available for inference and ordinary training on an eager host model.

    Examples
    --------
    >>> report = plan(model, task="training")  # doctest: +SKIP
    >>> print(report.schedule, report.regions)  # doctest: +SKIP
    """
    rep = getattr(getattr(model, "readout_fn", None), "representation", None)
    if not isinstance(rep, torch.nn.Module) or not all(
        hasattr(rep, name) for name in ("interactions", "products")
    ):
        raise ValueError("No Foundation schedule: expected a TACE property adapter.")
    task = ("training" if model.training else "inference") if task is None else task
    if task not in {"training", "inference", "joint_training", "lammps"}:
        raise ValueError(f"Unknown execution task: {task!r}")
    if regional_only and task not in {"training", "inference"}:
        raise ValueError(f"{task} requires a whole-model schedule")
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    automatic = backend is None
    backend = resolve_backend(backend, device)
    if backend == "cuda" and device.type != "cuda":
        raise ValueError("Move the model to CUDA before selecting the CUDA backend.")
    if task == "joint_training" and backend != "cuda":
        raise ValueError("The specialized joint-training schedule requires CUDA.")
    from .heads import pair_interaction_region, periodic_les_region, property_regions

    # The ordinary eager recipes replace node execution, not the host heads.
    # Frozen ML-IAP exports and joint E/F/S programs own their output graph:
    # their deployed heads cannot be inferred from modules in the eager source.
    heads = (
        tuple(property_regions(model.readout_fn))
        if task in {"inference", "training"}
        else ()
    )
    if customized_codegen(model) or customized_message_codegen(rep):
        reason = "Preserve customized execution on an e3nn generated module or its message owner."
        if task in {"joint_training", "lammps"} or getattr(
            rep, "_foundation_regional", False
        ):
            raise ValueError(f"Cannot rebuild this schedule. {reason}")
        return ExecutionPlan(task, backend, "regional", heads, reason)
    release, reason = (
        (None, "Requested regional execution.")
        if regional_only
        else release_schedule(model, task=task)
    )
    if task == "joint_training" and release is None:
        raise ValueError(f"No specialized joint-training schedule: {reason}")
    if task == "lammps":
        if backend != "cuda" or model.get_model_dtype() != torch.float32:
            raise ValueError("Specialized ML-IAP requires FP32 on CUDA.")
        if release is None:
            raise ValueError(
                f"No specialized ML-IAP schedule: {reason} Use mliap or aoti."
            )
    if release is not None:
        # Released GPU programs use native CUDA kernels. Honor an explicit
        # request for PyTorch on a CUDA device via the regional implementation.
        if backend == "cuda" or device.type == "cpu":
            return ExecutionPlan(task, backend, release, heads, reason)
        reason = "The released GPU schedule uses native CUDA kernels."
    specialized = tuple(f"{__package__}.{name}." for name in WHOLE_MODEL_SCHEDULES)
    if any(
        type(module).__module__.startswith(specialized)
        and type(module).__module__ not in _KERNEL_MODULES
        for module in (rep, *rep.interactions, *rep.products)
    ):
        raise ValueError(
            f"Cannot apply a regional plan to a whole-model schedule. {reason} "
            "Select the backend on the original TACE model."
        )
    from .interactions import harmonic_region
    from .linear import element_regions
    from .normalization import normalization_regions
    from .regional import block_rules, retain

    interactions, products = block_rules(
        model, backend=backend, task=task, automatic=automatic
    )
    regions = []
    opaque = set()
    for i, (interaction, product) in enumerate(
        zip(rep.interactions, rep.products, strict=True)
    ):
        for kind, module, match in (
            ("interactions", interaction, interactions),
            ("products", product, products),
        ):
            select, provider = match(module)
            if select is retain:
                opaque.add(f"{kind}.{i}")
            regions.append(
                _select_region(
                    f"{kind}.{i}",
                    module,
                    select,
                    backend=backend,
                    task=task,
                    host_provider=provider,
                )
            )
    angular = harmonic_region(rep, backend, automatic=automatic)
    if angular is not None:
        regions.append(angular)
    regions.extend(
        _regional_choice(path, backend, detail, lower)
        for path, lower, detail in element_regions(rep, backend=backend)
        if ".".join(path.split(".", 2)[:2]) not in opaque
    )
    regions.extend(
        _regional_choice(path, provider, detail, lower)
        for path, provider, lower, detail in normalization_regions(
            rep, backend=backend, automatic=automatic
        )
        if ".".join(path.split(".", 2)[:2]) not in opaque
    )
    if any(head.path == "les" for head in heads):
        ewald = periodic_les_region(
            model.readout_fn.les,
            automatic=automatic,
            cuda_model=model if backend == "cuda" else None,
        )
        if ewald is not None:
            regions.append(ewald)
    if any(head.path == "pair_representation.interactions.0" for head in heads):
        pair = pair_interaction_region(
            model.readout_fn, backend=backend, task=task, automatic=automatic
        )
        if pair is not None:
            heads = tuple(pair if head.path == pair.path else head for head in heads)
    return ExecutionPlan(task, backend, "regional", (*regions, *heads), reason)


def optimize_model(model, *, task=None, backend=None, regional_only=False, **options):
    """Apply a whole-model or regional plan to a CUDA model."""
    from ...registry import require_cuda

    require_cuda(model)
    report, result = _optimize_with_report(
        model, task=task, backend=backend, regional_only=regional_only, **options
    )
    if (
        report.schedule == "regional"
        and result is model
        and not any(
            region.provenance == "already_regional" for region in report.regions
        )
    ):
        raise NotImplementedError(f"No Foundation schedule applies. {report.reason}")
    return result


def _optimize_with_report(
    model, *, task=None, backend=None, regional_only=False, **options
):
    """Match and lower the same live instance once; never accept a cached plan."""
    report = plan(model, task=task, backend=backend, regional_only=regional_only)
    if report.task == "lammps":
        raise ValueError(
            "Use equivariantx_foundation.targets.lammps.export for ML-IAP."
        )
    if report.schedule != "regional":
        package = import_module(f".{report.schedule}", __package__)
        return report, package.optimize(model, **options)
    if options:
        raise TypeError(f"Unsupported regional schedule options: {', '.join(options)}")
    if not any(region.lower is not None for region in report.regions):
        return report, model
    result = shared_parameter_copy(model)
    rep = result.readout_fn.representation
    for region in report.regions:
        if region.lower is not None:
            original = result.get_submodule(region.module_path)
            updated = region.lower(original)
            if updated is not None and updated is not original:
                group, _, name = region.module_path.rpartition(".")
                setattr(result.get_submodule(group), name, updated)
            installed = original if updated is None else updated
            installed._foundation_regional_provider = (report.backend, region.reason)
    from .local import configure_local_geometry

    configure_local_geometry(
        rep, explicit_cuda_training=(backend == "cuda" and report.task == "training")
    )
    if (
        backend == "cuda"
        and len(rep.interactions)
        and getattr(rep.interactions[0], "_foundation_paired_ictd", False)
    ):
        from .cartesian import configure_paired_cartesian

        configure_paired_cartesian(result)
    rep._foundation_regional = True
    return report, result
