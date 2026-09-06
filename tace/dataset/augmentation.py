################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from copy import copy
from typing import Iterable, Optional, Union

import torch
from torch.utils.data import Dataset

from .quantity import TIME_ODD_PROPERTIES

SUPPORTED_AUGMENTATIONS = ("spin_rotation", "time_reversal")

PROPERTIES = tuple(
    name
    for name in TIME_ODD_PROPERTIES
    if name
    in (
        "initial_noncollinear_magmoms",
        "noncollinear_magnetic_forces",
    )
)


def validate_augmentations(
    augmentations: Optional[Union[str, Iterable[str]]],
) -> tuple[str, ...]:
    """Validate and normalize the configured training augmentations."""
    if augmentations is None:
        return ()
    if isinstance(augmentations, str):
        augmentations = (augmentations,)
    else:
        try:
            augmentations = tuple(augmentations)
        except TypeError as exc:
            raise TypeError("dataset.augmentation must be a string or a list") from exc

    unknown = set(augmentations) - set(SUPPORTED_AUGMENTATIONS)
    if unknown:
        raise ValueError(
            f"Unknown dataset augmentation {sorted(unknown)}; "
            f"choose from {list(SUPPORTED_AUGMENTATIONS)}"
        )
    if len(augmentations) != len(set(augmentations)):
        raise ValueError("dataset.augmentation must not contain duplicates")
    return augmentations


def _first_tensor(data, names: tuple[str, ...]):
    for name in names:
        value = getattr(data, name, None)
        if torch.is_tensor(value):
            return value
    return None


def _random_rotation(reference: torch.Tensor) -> torch.Tensor:
    sample_dtype = (
        torch.float32
        if reference.dtype in (torch.float16, torch.bfloat16)
        else reference.dtype
    )
    u1, u2, u3 = torch.rand(3, device=reference.device, dtype=sample_dtype)
    two_pi = 2 * torch.pi
    qx = torch.sqrt(1 - u1) * torch.sin(two_pi * u2)
    qy = torch.sqrt(1 - u1) * torch.cos(two_pi * u2)
    qz = torch.sqrt(u1) * torch.sin(two_pi * u3)
    qw = torch.sqrt(u1) * torch.cos(two_pi * u3)
    rotation = torch.stack(
        (
            torch.stack(
                (
                    1 - 2 * (qy.square() + qz.square()),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                )
            ),
            torch.stack(
                (
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx.square() + qz.square()),
                    2 * (qy * qz - qx * qw),
                )
            ),
            torch.stack(
                (
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx.square() + qy.square()),
                )
            ),
        )
    )
    return rotation.to(reference.dtype)


class AugmentedDataset(Dataset):
    """Data augmentation when a training sample is read."""

    def __init__(self, dataset: Dataset, augmentations: Iterable[str]):
        self.dataset = dataset
        self.augmentations = validate_augmentations(augmentations)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = copy(self.dataset[index])

        if "spin_rotation" in self.augmentations:
            reference = _first_tensor(data, PROPERTIES)
            if reference is not None:
                rotation = _random_rotation(reference)
                for name in PROPERTIES:
                    value = getattr(data, name, None)
                    if torch.is_tensor(value):
                        setattr(data, name, value @ rotation.mT)

        if "time_reversal" in self.augmentations:
            reference = _first_tensor(data, PROPERTIES)
            if reference is not None and torch.rand((), device=reference.device) < 0.5:
                for name in PROPERTIES:
                    value = getattr(data, name, None)
                    if torch.is_tensor(value):
                        setattr(data, name, -value)

        return data
