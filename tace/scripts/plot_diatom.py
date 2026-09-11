################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
from itertools import combinations_with_replacement

import ase.io
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from torch_geometric.loader import DataLoader

from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KEYS, KeySpecification, update_keyspec_from_kwargs
from tace.lightning import load_tace

NUM_POINTS = 256
BATCH_SIZE = 128
OUTPUT_PATH = "diatom_{label}.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot diatomic energy and radial-force curves for a TACE model."
    )
    parser.add_argument("-m", "--model", required=True, help="Path to a TACE model")
    parser.add_argument("-i", "--input", help="Optional ASE-readable DFT dataset")
    parser.add_argument(
        "--energy_key", default="energy", help="DFT energy key in atoms.info"
    )
    parser.add_argument(
        "--forces_key", default="forces", help="DFT forces key in atoms.arrays"
    )
    parser.add_argument(
        "--heteronuclear",
        action="store_true",
        help="Include heteronuclear diatomic pairs",
    )
    return parser.parse_args()


def _key_spec(energy_key="energy", forces_key="forces"):
    key_spec = KeySpecification()
    update_keyspec_from_kwargs(
        key_spec,
        {**KEYS, "energy_key": energy_key, "forces_key": forces_key},
    )
    return key_spec


def distance_and_direction(atoms):
    vector = atoms.get_distance(0, 1, mic=bool(np.any(atoms.pbc)), vector=True)
    distance = np.linalg.norm(vector)
    if distance == 0.0:
        raise ValueError("Diatomic structures must have a nonzero separation.")
    return distance, vector / distance


def read_reference(
    path, atomic_numbers, energy_key, forces_key, include_heteronuclear=False
):
    reference = {}
    for index, atoms in enumerate(ase.io.read(path, index=":")):
        numbers = atoms.get_atomic_numbers()
        if len(numbers) != 2 or any(z not in atomic_numbers for z in numbers):
            continue
        if not include_heteronuclear and numbers[0] != numbers[1]:
            continue
        if energy_key not in atoms.info:
            raise KeyError(f"Structure {index} has no atoms.info[{energy_key!r}].")
        if forces_key not in atoms.arrays:
            raise KeyError(f"Structure {index} has no atoms.arrays[{forces_key!r}].")

        distance, direction = distance_and_direction(atoms)
        forces = np.asarray(atoms.arrays[forces_key])
        if forces.shape != (2, 3):
            raise ValueError(
                f"Structure {index} forces have shape {forces.shape}, expected (2, 3)."
            )
        element_pair = tuple(sorted(int(z) for z in numbers))
        reference.setdefault(element_pair, []).append(
            (
                distance,
                atoms,
                float(np.asarray(atoms.info[energy_key]).reshape(-1)[0]),
                float(forces[1] @ direction),
                direction,
            )
        )

    if not reference:
        kind = "diatomic" if include_heteronuclear else "homonuclear diatomic"
        raise ValueError(f"The input contains no supported {kind} structures.")
    for values in reference.values():
        values.sort(key=lambda value: value[0])
    return reference


def generated_atoms(z1, z2, distances, fidelity_idx):
    atoms_list = []
    for distance in distances:
        atoms = Atoms(
            numbers=(z1, z2),
            positions=((0.0, 0.0, 0.0), (distance, 0.0, 0.0)),
            pbc=False,
        )
        atoms.info["fidelity_idx"] = fidelity_idx
        atoms_list.append(atoms)
    return atoms_list


def build_dataset(model, atoms_list, key_spec):
    element = model.get_torch_element()
    embedding_property = model.get_embedding_property()
    fidelity_key = key_spec.info_keys["fidelity_idx"]
    dataset = []
    for atoms in atoms_list:
        atoms = atoms.copy()
        atoms.info.setdefault(fidelity_key, model.get_fidelity_idx())
        dataset.append(
            from_atoms(
                element,
                atoms,
                model.get_cutoff(),
                max_neighbors=model.get_max_neighbors(),
                keyspec=key_spec,
                target_property=["energy", "forces"],
                embedding_property=embedding_property,
                training=False,
                neighborlist_backend="matscipy",
            )
        )
    return dataset


def predict(model, dataset, directions, device):
    energies = []
    radial_forces = []
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    offset = 0

    for batch in dataloader:
        batch = batch.to(device)
        batch.positions.requires_grad_(True)
        with torch.enable_grad():
            output = model(batch)
        batch_size = batch.num_graphs
        direction = torch.as_tensor(
            directions[offset : offset + batch_size],
            dtype=output["forces"].dtype,
            device=output["forces"].device,
        )
        forces = output["forces"].reshape(-1, 2, 3)
        energies.append(output["energy"].detach().reshape(-1).cpu())
        radial_forces.append(
            (forces[:, 1] * direction).sum(dim=-1).detach().cpu()
        )
        offset += batch_size

    return torch.cat(energies).numpy(), torch.cat(radial_forces).numpy()


def plot(curves):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, curve in enumerate(curves):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        color = colors[index % len(colors)]
        label = curve["label"]
        figure.suptitle(label)
        axes[0].plot(
            curve["distances"], curve["energies"], color=color, label=label
        )
        axes[1].plot(
            curve["distances"], curve["forces"], color=color, label=label
        )
        if curve["reference_energies"] is not None:
            axes[0].scatter(
                curve["distances"],
                curve["reference_energies"],
                color=color,
                marker="x",
                label=f"{label} DFT",
                zorder=3,
            )
            axes[1].scatter(
                curve["distances"],
                curve["reference_forces"],
                color=color,
                marker="x",
                label=f"{label} DFT",
                zorder=3,
            )

        axes[0].set_title("Diatomic energy")
        axes[0].set_ylabel("Energy (eV)")
        axes[1].set_title("Diatomic forces")
        axes[1].set_ylabel(r"$F_{2,r}$ (eV/$\AA$)")
        for axis in axes:
            axis.set_xlabel(r"Distance ($\AA$)")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)

        output_path = OUTPUT_PATH.format(label=label)
        figure.savefig(output_path, dpi=300)
        plt.close(figure)
        print(f"Diatomic curves written to {output_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_tace(
        args.model,
        device=device,
        strict=True,
        use_ema=True,
        target_property=["energy", "forces"],
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    torch.set_default_dtype(model.get_model_dtype())
    atomic_numbers = model.get_atomic_numbers()
    key_spec = _key_spec(args.energy_key, args.forces_key)

    if args.input:
        reference = read_reference(
            args.input,
            atomic_numbers,
            args.energy_key,
            args.forces_key,
            include_heteronuclear=args.heteronuclear,
        )
        element_pairs = sorted(reference)
    else:
        reference = None
        element_pairs = (
            list(combinations_with_replacement(atomic_numbers, 2))
            if args.heteronuclear
            else [(z, z) for z in atomic_numbers]
        )

    curves = []
    cutoff = model.get_cutoff()
    for index, (z1, z2) in enumerate(element_pairs, start=1):
        label = f"{chemical_symbols[z1]}-{chemical_symbols[z2]}"
        print(f"[{index}/{len(element_pairs)}] {label}")
        if reference is None:
            distances = np.linspace(
                0.5, cutoff, NUM_POINTS
            )
            atoms_list = generated_atoms(
                z1, z2, distances, model.get_fidelity_idx()
            )
            directions = np.tile((1.0, 0.0, 0.0), (len(distances), 1))
            reference_energies = None
            reference_forces = None
        else:
            values = reference[(z1, z2)]
            distances = np.asarray([value[0] for value in values])
            atoms_list = [value[1] for value in values]
            reference_energies = np.asarray([value[2] for value in values])
            reference_forces = np.asarray([value[3] for value in values])
            directions = np.asarray([value[4] for value in values])

        energies, radial_forces = predict(
            model,
            build_dataset(model, atoms_list, key_spec),
            directions,
            device,
        )
        curves.append(
            {
                "label": label,
                "distances": distances,
                "energies": energies,
                "forces": radial_forces,
                "reference_energies": reference_energies,
                "reference_forces": reference_forces,
            }
        )

    plot(curves)


if __name__ == "__main__":
    main()
