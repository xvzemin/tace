################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""General ASE calculator with explicitly mapped embedding inputs."""

from typing import Dict, Optional

import numpy as np

from tace.dataset.quantity import PROPERTY, SUPPORT_EMBEDDING_PROPERTY

from .calculator import TACEAseCalc


def _same_value(previous, current, tol: float) -> bool:
    if previous is None or current is None:
        return previous is current
    previous = np.asarray(previous)
    current = np.asarray(current)
    if previous.shape != current.shape:
        return False
    if np.issubdtype(previous.dtype, np.number) and np.issubdtype(
        current.dtype, np.number
    ):
        return bool(np.allclose(previous, current, atol=tol, rtol=0.0))
    return bool(np.array_equal(previous, current))


class GeneralTACEAseCalc(TACEAseCalc):
    """TACE ASE calculator with additional named embedding inputs.

    ``keys`` maps a TACE embedding property to its key in ``Atoms.info`` or
    ``Atoms.arrays``. For example,
    ``{"initial_noncollinear_magmoms": "spin"}`` reads the noncollinear
    magnetic input from ``atoms.arrays["spin"]``.
    """

    def __init__(
        self,
        model: str,
        *,
        keys: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        keys = {} if keys is None else dict(keys)
        for name, atoms_key in keys.items():
            if name not in SUPPORT_EMBEDDING_PROPERTY:
                raise ValueError(
                    f"{name!r} is not an embedding property; allowed properties are "
                    f"{SUPPORT_EMBEDDING_PROPERTY}"
                )
            if not isinstance(atoms_key, str) or not atoms_key:
                raise ValueError(
                    f"The Atoms key for {name!r} must be a nonempty string"
                )

        super().__init__(model=model, **kwargs)
        self.keys = keys
        self.embedding_property = list(self.embedding_property)
        for name, atoms_key in keys.items():
            if name not in self.embedding_property:
                self.embedding_property.append(name)
            if PROPERTY[name]["scope"] == "per-atom":
                self.keySpecification.arrays_keys[name] = atoms_key
            else:
                self.keySpecification.info_keys[name] = atoms_key

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if self.atoms is None:
            return changes

        for name, atoms_key in self.keys.items():
            storage = "arrays" if PROPERTY[name]["scope"] == "per-atom" else "info"
            previous = getattr(self.atoms, storage).get(atoms_key)
            current = getattr(atoms, storage).get(atoms_key)
            if atoms_key not in changes and not _same_value(previous, current, tol):
                changes.append(atoms_key)
        return changes
