################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import re
from pathlib import Path

import torch
import yaml

from tace.dataset.element import atomic_numbers
from tace.lightning import load_tace


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
        # choices=["atomic_energy", 'avg_num_neighbors', 'scale', 'shift']
        default="atomic_energy",
        help="specify statistics info to update",
    )
    return parser.parse_args()


def safe_dict(info: dict):
    new_dict = {}
    for k, v in info.items():
        if k in atomic_numbers:
            k = atomic_numbers[k]
        new_dict[k] = v
    return dict(sorted(new_dict.items()))


def main():
    args = parse_args()
    model = load_tace(args.model, device="cpu", strict=True, use_ema=1)

    pattern = re.compile(r"statistics_(\d+)\.yaml")
    statistics_dict = {}
    for path in Path(".").iterdir():
        if path.is_file():
            m = pattern.fullmatch(path.name)
            if m:
                fidelity_idx = int(m.group(1))
                statistics_dict[fidelity_idx] = yaml.safe_load(path.read_text())
    statistics_dict: dict = dict(sorted(statistics_dict.items()))
    with torch.no_grad():
        if "atomic_energy" in args.update:
            atomic_energies = (
                model.readout_fn.atomic_energy_layer.atomic_energy.detach().clone()
            )
            for fidelity_idx, stats in statistics_dict.items():
                if "atomic_energy" not in stats:
                    continue
                new_row = torch.tensor(
                    [v for _, v in safe_dict(stats["atomic_energy"]).items()],
                    dtype=atomic_energies.dtype,
                    device=atomic_energies.device,
                )
                atomic_energies[fidelity_idx, :] = new_row
            model.readout_fn.atomic_energy_layer.atomic_energy.copy_(atomic_energies)

        if "scale" in args.update:
            scales = model.readout_fn.scale_shift.scale.detach().clone()
            for fidelity_idx, stats in statistics_dict.items():
                if "scale" not in stats:
                    continue
                new_row = torch.tensor(
                    [v for _, v in safe_dict(stats["scale"]).items()],
                    dtype=scales.dtype,
                    device=scales.device,
                )
                scales[fidelity_idx, :] = new_row
            model.readout_fn.scale_shift.scale.copy_(scales)

        if "shift" in args.update:
            shifts = model.readout_fn.scale_shift.shift.detach().clone()
            for fidelity_idx, stats in statistics_dict.items():
                if "shift" not in stats:
                    continue
                new_row = torch.tensor(
                    [v for _, v in safe_dict(stats["shift"]).items()],
                    dtype=shifts.dtype,
                    device=shifts.device,
                )
                shifts[fidelity_idx, :] = new_row
            model.readout_fn.scale_shift.shift.copy_(shifts)
    print(model)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": model.readout_fn.model_config,
            "target_property": model.readout_fn.target_property,
            "embedding_property": model.readout_fn.embedding_property,
            "statistics": model.readout_fn.statistics,
        },
        "new_statistics.pt",
    )
