################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
from pathlib import Path
from typing import Dict, Union

import ase.io
import torch
from torch_geometric.loader import DataLoader

from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KEYS, KeySpecification, update_keyspec_from_kwargs
from tace.lightning import export_tace, load_tace
from tace.models.compile import export_aotinductor
from tace.utils.env import enable_acceleration

ALLOWED_BACKEND = ["state_dict", "full_model", "aoti"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a TACE model for eval or native torch deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model path")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "--backend",
        type=str,
        default="state_dict",
        choices=ALLOWED_BACKEND,
        help="Export format",
    )
    parser.add_argument(
        "-f",
        "--fidelity_idx",
        type=int,
        default=None,
        help="Which fidelity to compile",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default=None,
        help="Model dtype",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Load or compile device"
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Optional ASE-readable structures used to pick realistic compile shapes",
    )
    parser.add_argument(
        "--sample-index",
        type=str,
        default=":",
        help="ASE index expression for --sample",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size used when building sample graph inputs",
    )
    parser.add_argument(
        "--nl-backend",
        type=str,
        default="matscipy",
        choices=["ase", "matscipy", "vesin", "alchemiops"],
        help="Neighbor-list backend for --sample",
    )
    return parser.parse_args()


def _default_aoti_output_path(model_path: str) -> str:
    return str(Path(model_path).with_suffix(".pt2"))


def _default_state_dict_output_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_name(path.name + "-state_dict.pt"))


def _default_full_model_output_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_name(path.name + "-full_model.pt"))


def _build_sample_data(
    model: torch.nn.Module,
    sample_path: str,
    sample_index: str,
    batch_size: int,
    nl_backend: str,
    device: Union[str, torch.device],
) -> Dict[str, torch.Tensor]:
    atoms_list = ase.io.read(sample_path, index=sample_index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f"No structures were read from {sample_path!r}.")

    key_spec = KeySpecification()
    update_keyspec_from_kwargs(key_spec, KEYS)
    embedding_property = model.get_embedding_property()
    if (
        "noncollinear_magnetic_forces" in model.get_target_property()
        and "initial_noncollinear_magmoms" not in embedding_property
    ):
        embedding_property.append("initial_noncollinear_magmoms")
    dataset = [
        from_atoms(
            model.get_torch_element(),
            atoms,
            model.get_cutoff(),
            max_neighbors=model.get_max_neighbors(),
            keyspec=key_spec,
            target_property=model.get_target_property(),
            embedding_property=embedding_property,
            training=False,
            neighborlist_backend=nl_backend,
        )
        for atoms in atoms_list[: max(1, batch_size)]
    ]
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(dataloader)).to(device)
    return {key: batch[key] for key in batch.keys()}


def main():
    args = parse_args()
    if args.backend == "aoti":
        enable_acceleration(enable_compile=True)
    model = load_tace(
        args.model,
        args.device,
        strict=True,
        use_ema=True,
        dtype=args.dtype,
    )
    model.reset_fidelity_idx(args.fidelity_idx)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    sample_data = None
    if args.backend == "aoti" and args.sample:
        sample_data = _build_sample_data(
            model,
            args.sample,
            args.sample_index,
            args.batch_size,
            args.nl_backend,
            args.device,
        )

    if args.backend == "state_dict":
        output_path = args.output or _default_state_dict_output_path(args.model)
        export_tace(model, output_path)
        print(f"[Done] state_dict model saved to: {output_path}")
    elif args.backend == "full_model":
        output_path = args.output or _default_full_model_output_path(args.model)
        torch.save(model, output_path)
        print(f"[Done] full_model saved to: {output_path}")
    elif args.backend == "aoti":
        output_path = args.output or _default_aoti_output_path(args.model)
        output_path = export_aotinductor(model, output_path, sample_data=sample_data)
        print(f"[Done] AOTInductor graph model saved to: {output_path}")
    else:
        raise ValueError(
            f"Unsupported backend '{args.backend}'. One of {ALLOWED_BACKEND} is available."
        )


if __name__ == "__main__":
    main()
