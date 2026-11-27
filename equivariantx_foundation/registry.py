"""Model adapters and AOTInductor operator registration."""

from importlib import import_module

ADAPTERS = {
    "tace": ".models.tace.planner:optimize_model",
    "mace": ".adapters.mace:convert_mace_to_eqx",
    "nequip": ".adapters.nequip:convert_nequip_to_eqx",
    "sevennet": ".adapters.sevennet:convert_sevennet_to_eqx",
    "equflash": ".adapters.equflash:convert_equflash_to_eqx",
    "prophet": ".adapters.prophet:convert_prophet_to_eqx",
}

REGISTRATION_MODULES = {
    "eqx_foundation": "ops.tece",
    "eqx_oam_l": "models.tace.tace_oam_l.radial",
    "eqx_oam_l_geometry": "models.tace.tace_oam_l.angular",
    "eqx_oam_l_product": "models.tace.tace_oam_l.product",
    "eqx_oam_l_conv": "models.tace.tace_oam_l.convolution",
    "eqx_oam_l_tangent": "models.tace.tace_oam_l.tangent",
    "eqx_oam_l_element": "models.tace.tace_oam_l.element_grouped",
    "eqx_tace_7m": "ops.oam_7m",
    "eqx_tece_rra": "models.tace.tece_oam_rra.force",
    "eqx_oam_xl": "ops.oam_xl",
}


def require_cuda(device=None):
    """Require an NVIDIA CUDA runtime and, if supplied, a CUDA device or model."""
    import torch

    if isinstance(device, torch.nn.Module):
        from itertools import chain

        tensor = next(chain(device.parameters(), device.buffers()), None)
        device = tensor.device if tensor is not None else None
    if (
        (device is not None and torch.device(device).type != "cuda")
        or not torch.version.cuda
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "EquivariantX Foundation requires an NVIDIA CUDA device for execution. "
            "For CPU execution use EquivariantX or ordinary TACE."
        )


def optimize(model, *, family, **options):
    """Apply a model-specific execution schedule.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded CUDA model.
    family : str
        One of ``tace``, ``mace``, ``nequip``, ``sevennet``, ``equflash``, or ``prophet``.
    **options
        Keyword arguments accepted by the selected adapter.

    Examples
    --------
    >>> from equivariantx_foundation import optimize
    >>> model = optimize(model, family="tace")  # doctest: +SKIP
    """
    if family not in ADAPTERS:
        raise ValueError(
            f"Unknown family {family!r}. Expected one of {tuple(ADAPTERS)}."
        )
    module, name = ADAPTERS[family].split(":")
    return getattr(import_module(module, __package__), name)(model, **options)


def registration_module(operator):
    """Return the module that registers an exported Foundation operator."""
    return f"{__package__}.{REGISTRATION_MODULES[operator.namespace]}"
