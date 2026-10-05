"""Export a model with a selected set of chemical elements."""

import argparse
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import torch

from tace.dataset.element import atomic_numbers, chemical_symbols
from tace.lightning import create_model, export_tace, load_tace
from tace.models._e3nn.base import NodeEmbedding
from tace.models._e3nn.edge import Element2EdgeUpdate
from tace.models._e3nn.magnetic import MagneticBasis
from tace.models._e3nn.symmetric_contraction import Contraction
from tace.models.adapter import TensorModel
from tace.models.blocks import OneHotToAtomicEnergy, ScaleShift
from tace.models.linear import (
    e3nnElementLinear,
    e3nnLinear,
    e3nnMoEElementLinear,
    has_lora,
    switch_e3nn_weight_layout,
)
from tace.utils.env import enable_acceleration
from tace.utils.utils import torch_default_dtype


@torch.no_grad()
def select_elements(model: TensorModel, elements: Sequence[str | int]) -> TensorModel:
    """Copy a model with a different element vocabulary.

    Parameters
    ----------
    model : TensorModel
        Uncompiled model with construction metadata. Merge active LoRA weights
        before selecting elements.
    elements : sequence of str or int
        Chemical symbols or atomic numbers. The exported order is increasing
        atomic number.

    Returns
    -------
    TensorModel
        Model with retained element parameters and unchanged shared parameters.
        New element parameters retain their constructor initialization. Their
        atomic energies and shifts default to zero, and scales to one.

    Notes
    -----
    Element embeddings are rescaled to compensate for the changed input
    multiplicity. Predictions on retained elements agree up to floating-point
    rounding. Predictions involving new elements require training.
    """
    numbers = []
    for element in elements:
        number = atomic_numbers.get(element) if isinstance(element, str) else element
        if number is None and isinstance(element, str) and element.isdecimal():
            number = int(element)
        if not isinstance(number, int) or not 1 <= number < len(chemical_symbols):
            raise ValueError(f"Unknown chemical element: {element!r}")
        numbers.append(number)
    if not numbers or len(set(numbers)) != len(numbers):
        raise ValueError("Provide a non-empty list of distinct elements.")
    numbers.sort()
    if hasattr(model, "compiled_model") or not isinstance(model, TensorModel):
        raise ValueError(
            "Element export requires an uncompiled .pt, .pth or .ckpt model."
        )
    if any(has_lora(module) for module in model.modules()):
        raise ValueError(
            "Merge LoRA weights with tace-convert-lora before element export."
        )

    old_numbers = model.get_atomic_numbers()
    retained = [z for z in numbers if z in old_numbers]
    device = next(model.parameters()).device
    source_index = torch.tensor(
        [old_numbers.index(z) for z in retained], dtype=torch.long, device=device
    )
    target_index = torch.tensor(
        [numbers.index(z) for z in retained], dtype=torch.long, device=device
    )
    config = deepcopy(model.readout_fn.model_config)
    statistics = deepcopy(model.readout_fn.statistics)
    for stats in statistics:
        stats["atomic_numbers"] = numbers.copy()
        for key, value in stats.items():
            if isinstance(value, dict) and all(str(z).isdecimal() for z in value):
                stats[key] = {int(z): v for z, v in value.items() if int(z) in numbers}
    config["statistics"] = statistics
    config["atomic_numbers"] = numbers
    fidelities = config["fidelity"]
    if isinstance(fidelities, dict):
        fidelities = [fidelities]
    for fidelity in fidelities:
        for key in ("atomic_energy", "magnetic_scale"):
            values = fidelity.get(key)
            if isinstance(values, dict):
                fidelity[key] = {
                    z: values.get(
                        z, values.get(str(z), 1.0 if key == "magnetic_scale" else 0.0)
                    )
                    for z in numbers
                }

    with torch_default_dtype(model.get_model_dtype()):
        selected = create_model(
            config,
            statistics,
            model.get_target_property(),
            model.get_embedding_property(),
            prune_removed_keys=True,
        ).to(device=device, dtype=model.get_model_dtype())

    # Identify element axes by the owning layer, not by coincidental tensor sizes.
    element_axes = {}
    modules = dict(model.named_modules())
    for name, module in selected.named_modules():
        original = modules[name]
        if isinstance(module, (e3nnLinear, e3nnElementLinear, e3nnMoEElementLinear)):
            switch_e3nn_weight_layout(
                module, "matrix" if original.use_matrix_weight else "flat"
            )
        if isinstance(module, (e3nnElementLinear, e3nnMoEElementLinear, Contraction)):
            for key, _ in module.named_parameters():
                element_axes[f"{name}.{key}"] = (0, 1.0)
        elif isinstance(module, OneHotToAtomicEnergy):
            element_axes[f"{name}.atomic_energy"] = (1, 1.0)
        elif isinstance(module, ScaleShift):
            for key in ("scale", "shift"):
                if hasattr(module, key):
                    element_axes[f"{name}.{key}"] = (1, 1.0)
        elif isinstance(module, MagneticBasis):
            element_axes[f"{name}.magnetic_scale"] = (1, 1.0)
        elif isinstance(module, e3nnLinear):
            parent_name, _, field = name.rpartition(".")
            parent = modules[parent_name]
            if isinstance(parent, (NodeEmbedding, Element2EdgeUpdate)) and field in {
                "elem_emb1",
                "element_embedding",
                "node_embedding",
                "source_embedding",
                "target_embedding",
            }:
                old_paths = original.linear.instructions
                new_paths = module.linear.instructions
                if len(old_paths) != 1 or len(new_paths) != 1:
                    raise ValueError(f"Unsupported element embedding: {name}")
                factor = old_paths[0].path_weight / new_paths[0].path_weight
                for key, _ in module.named_parameters():
                    if key == "weight" or key.startswith("weight."):
                        element_axes[f"{name}.{key}"] = (0, factor)

    source_state = model.state_dict()
    target_state = selected.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError(
            "Model reconstruction changed state keys; export an eager model first."
        )
    for name, target in target_state.items():
        if name.rsplit(".", 1)[-1] == "atomic_numbers":
            continue
        source = source_state[name]
        if name in element_axes:
            axis, factor = element_axes[name]
            if axis == 0:
                source = source.reshape(len(old_numbers), -1)
                target = target.reshape(len(numbers), -1)
            target.index_copy_(
                axis, target_index, source.index_select(axis, source_index) * factor
            )
        elif source.shape == target.shape:
            target.copy_(source)
        else:
            raise ValueError(f"Unrecognized element-dependent tensor: {name}")
    selected.load_state_dict(target_state, strict=True)
    selected.train(model.training)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", required=True, help="Model file or foundation model name"
    )
    parser.add_argument(
        "-e",
        "--elements",
        nargs="+",
        required=True,
        help="Chemical symbols or atomic numbers",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output .pt file"
    )
    args = parser.parse_args()
    if Path(args.model).suffix.lower() == ".pt2":
        parser.error(
            "Use the original .pt/.pth/.ckpt model, not a compiled .pt2 package."
        )
    if args.output.suffix != ".pt":
        parser.error("The output filename must end in .pt.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    enable_acceleration(force=True)
    model = load_tace(args.model, device="cpu")
    try:
        selected = select_elements(model, args.elements)
    except ValueError as error:
        parser.error(str(error))
    export_tace(selected, str(args.output))
    numbers = selected.get_atomic_numbers()
    added = [
        chemical_symbols[z] for z in numbers if z not in model.get_atomic_numbers()
    ]
    print(f"Elements: {', '.join(chemical_symbols[z] for z in numbers)}")
    if added:
        print(f"New elements initialized (training required): {', '.join(added)}")
    print(f"Exported model: {args.output.resolve()}")


if __name__ == "__main__":
    main()
