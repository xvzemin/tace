################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from tace.dataset.element import atomic_numbers
from tace.lightning import export_tace, load_tace
from tace.models.adapter import TensorModel


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model file or registered foundation model name",
    )
    parser.add_argument(
        "-u",
        "--update",
        type=str,
        nargs="+",
        choices=["atomic_energy", "scale", "shift"],
        default=["atomic_energy"],
        help="specify statistics info to update",
    )
    return parser.parse_args()


@torch.no_grad()
def update_statistics(
    model: TensorModel,
    statistics: Mapping[int, Mapping],
    fields: Sequence[str] = ("atomic_energy",),
) -> None:
    """Update element tables and their construction metadata in place.

    Parameters
    ----------
    model : TensorModel
        Uncompiled TACE model.
    statistics : dict
        Fidelity indices mapped to ``atomic_energy``, ``scale`` or ``shift``
        dictionaries keyed by chemical symbols or atomic numbers. Unspecified
        elements and fidelities retain their current values.
    fields : sequence of str, optional
        Tables to update. Defaults to atomic reference energies.
    """
    if hasattr(model, "compiled_model") or not isinstance(model, TensorModel):
        raise ValueError(
            "Statistics updates require an uncompiled .pt, .pth or .ckpt model."
        )
    readout = model.readout_fn
    numbers = model.get_atomic_numbers()
    indices = {z: i for i, z in enumerate(numbers)}
    metadata = deepcopy(readout.statistics)
    for fidelity, stats in statistics.items():
        if not isinstance(fidelity, int) or not 0 <= fidelity < len(metadata):
            raise ValueError(f"Unknown fidelity index: {fidelity!r}.")
        if not isinstance(stats, Mapping):
            raise ValueError(f"Statistics for fidelity {fidelity} must be a mapping.")

    updates = {}
    for field in dict.fromkeys(fields):
        if field not in ("atomic_energy", "scale", "shift"):
            raise ValueError(f"Unknown statistics field: {field!r}.")
        field_updates = {
            i: stats[field] for i, stats in statistics.items() if field in stats
        }
        if not field_updates:
            continue
        layer_name = (
            "atomic_energy_layer" if field == "atomic_energy" else "scale_shift"
        )
        table = getattr(getattr(readout, layer_name, None), field, None)
        if table is None:
            raise ValueError(f"The model has no enabled {field} table.")
        values = table.detach().clone()
        for fidelity, entries in field_updates.items():
            if not isinstance(entries, Mapping) or not entries:
                raise ValueError(
                    f"{field} for fidelity {fidelity} must be a non-empty mapping."
                )
            seen = set()
            for element, value in entries.items():
                number = atomic_numbers.get(element, element)
                if isinstance(number, str) and number.isdecimal():
                    number = int(number)
                if number not in indices:
                    raise ValueError(f"Element {element!r} is not in the model.")
                if number in seen:
                    raise ValueError(f"Duplicate element {element!r} in {field}.")
                seen.add(number)
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{field} for element {element!r} must be finite.")
                values[fidelity, indices[number]] = value
        if not torch.isfinite(values).all():
            raise ValueError(f"{field} values exceed the table's dtype range.")
        updates[field] = table, values

    if not updates:
        raise ValueError(
            "No requested statistics found in statistics_<fidelity_idx>.yaml files."
        )
    for field, (table, values) in updates.items():
        table.copy_(values)
        for stats, row in zip(metadata, values.cpu().tolist()):
            stats[field] = dict(zip(numbers, row))
        if field == "atomic_energy":
            readout.model_config["atomic_energies"] = [s[field] for s in metadata]
        else:
            readout.model_config["scale_shift"][field + "_type"] = field
    readout.statistics = metadata
    readout.model_config["statistics"] = metadata


def main():
    args = parse_args()
    if Path(args.model).suffix.lower() == ".pt2":
        raise ValueError(
            "Update the original .pt model, then re-export the .pt2 package."
        )
    model = load_tace(args.model, device="cpu", strict=True, use_ema=True)

    pattern = re.compile(r"statistics_(\d+)\.yaml")
    statistics_dict = {}
    for path in Path(".").iterdir():
        if path.is_file():
            m = pattern.fullmatch(path.name)
            if m:
                fidelity_idx = int(m.group(1))
                statistics_dict[fidelity_idx] = (
                    yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                )
    update_statistics(model, statistics_dict, args.update)
    export_tace(model, "new_statistics.pt")
    print("Updated model saved to new_statistics.pt")


if __name__ == "__main__":
    main()
