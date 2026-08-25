################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import importlib
import inspect
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Iterable, Union

from omegaconf import ListConfig

from .mse_fn import LOSS_FN

LOSS_MODULES = (
    "mse_fn",
    "mae_fn",
    "huber_fn",
    "l2mae_fn",
    "dens",
    "special_fn",
)
LOSS_NAME_PREFIXES = (
    "l2mae_",
    "huber_",
    "mse_",
    "mae_",
)


def ensure_loss_functions_registered() -> None:
    for module_name in LOSS_MODULES:
        importlib.import_module(f".{module_name}", package=__package__)


def _natural_sort_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _loss_property_name(loss_name: str) -> str:
    for prefix in LOSS_NAME_PREFIXES:
        if loss_name.startswith(prefix):
            return loss_name[len(prefix) :]
    return loss_name


def _is_special_loss(loss_fn) -> bool:
    return loss_fn.__module__.endswith(".special_fn")


def available_losses_by_property(
    *,
    include_special: bool = False,
) -> dict[str, list[str]]:
    ensure_loss_functions_registered()
    losses_by_property: dict[str, list[str]] = defaultdict(list)
    for loss_name, loss_fn in LOSS_FN.items():
        if not include_special and _is_special_loss(loss_fn):
            continue
        losses_by_property[_loss_property_name(loss_name)].append(loss_name)

    return {
        property_name: sorted(loss_names, key=_natural_sort_key)
        for property_name, loss_names in sorted(
            losses_by_property.items(),
            key=lambda item: _natural_sort_key(item[0]),
        )
    }


def format_available_losses_by_property() -> str:
    lines = ["Available loss functions by property:"]
    for property_name, loss_names in available_losses_by_property().items():
        lines.append(f"{property_name}:")
        lines.extend(f"  - {loss_name}" for loss_name in loss_names)
    return "\n".join(lines)


def format_unknown_loss_error(unknown_loss_names: Iterable[str]) -> str:
    unknown = sorted(set(unknown_loss_names), key=_natural_sort_key)
    unknown_lines = "\n".join(f"  - {loss_name}" for loss_name in unknown)
    return (
        "Unknown loss function(s):\n"
        f"{unknown_lines}\n\n"
        f"{format_available_losses_by_property()}"
    )


def validate_loss_function_names(loss_function_names: Iterable[str]) -> None:
    ensure_loss_functions_registered()
    unknown = [
        loss_name for loss_name in loss_function_names if loss_name not in LOSS_FN
    ]
    if unknown:
        raise ValueError(format_unknown_loss_error(unknown))


def prepare_loss_function_kwargs(
    loss_function_names: Iterable[str],
    loss_function_kwargs: Union[list[dict[str, Any]], ListConfig, None],
) -> list[dict[str, Any]]:
    loss_function_names = list(loss_function_names)
    validate_loss_function_names(loss_function_names)

    if loss_function_kwargs is None:
        loss_function_kwargs = [{} for _ in loss_function_names]
    if not isinstance(loss_function_kwargs, (list, ListConfig)):
        raise TypeError("loss_function_kwargs must be a list of mappings.")
    if len(loss_function_kwargs) != len(loss_function_names):
        raise ValueError(
            "loss_function_kwargs and loss_function_name must have the same length."
        )

    prepared = []
    for loss_name, kwargs in zip(loss_function_names, loss_function_kwargs):
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"Parameters for {loss_name!r} must be a mapping.")
        kwargs = dict(kwargs)
        try:
            inspect.signature(LOSS_FN[loss_name]).bind(None, None, **kwargs)
        except TypeError as error:
            raise ValueError(
                f"Invalid parameters for loss function {loss_name!r}: {error}"
            ) from error
        prepared.append(kwargs)
    return prepared
