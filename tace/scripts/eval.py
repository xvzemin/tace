################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import copy
import time
from typing import Dict

import ase.io
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from tace.dataset.graph import from_atoms
from tace.dataset.quantity import PROPERTY, KeySpecification, update_keyspec_from_kwargs
from tace.dataset.read import check_keys
from tace.lightning import load_tace
from tace.utils.metrics import build_metrics, update_metrics
from tace.utils.utils import num_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict properties using a trained model. "
        "You could also print test metrics."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to ASE-readable input file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predict.xyz",
        help="Path to ASE-writeable output file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt or .pt or .pth)",
    )
    parser.add_argument(
        "-t", "--test", type=int, default=0, help="print test set metric"
    )
    parser.add_argument(
        "-e", "--ema", type=int, default=1, help="Use EMA parameters during evaluation"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "-sd",
        "--scalar_descriptor",
        type=int,
        default=0,
        help="if a scalar descriptor exists, output to xyz",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Tensor precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--nl_backend",
        type=str,
        default="matscipy",
        choices=["ase", "matscipy", "vesin", "alchemiops"],
        help="nl_backend",
    )

    # Keys for properties, if need print test metrics
    for k, v in PROPERTY.items():
        if v["enable_prediction"] or v["enable_embedding"]:
            parser.add_argument(f"--{k}_key", type=str, default=f"{k}")
    return parser.parse_args()


def main():
    args = parse_args()
    atomsList = ase.io.read(args.input, index=":")
    atoms_list_copy = copy.deepcopy(atomsList)
    key_spec = KeySpecification()
    update_keyspec_from_kwargs(key_spec, vars(args))
    model = load_tace(
        args.model,
        args.device,
        strict=True,
        use_ema=args.ema,
        dtype=args.dtype,
    )
    print(f"Number of parameters: {num_params(model)}")
    max_neighbors = model.get_max_neighbors()
    cutoff = model.get_cutoff()
    element = model.get_torch_element()
    target_property = model.get_target_property()
    embedding_property = model.get_embedding_property()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Build metrics
    if args.test == 1:
        metrics: Dict = build_metrics("test", target_property)
        for m in metrics.values():
            m.to(args.device)

    # Dataset
    training = (
        True if args.test else False
    )  # test which means your should extra provide label
    dataset = [
        from_atoms(
            element,
            atoms,
            cutoff,
            max_neighbors="inf" if max_neighbors is None else max_neighbors,
            keyspec=key_spec,
            target_property=target_property,
            embedding_property=embedding_property,
            neighborlist_backend=args.nl_backend,
        )
        for atoms in check_keys(
            atomsList, target_property, key_spec, embedding_property, training
        )
    ]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    preds, ptrs = [], []
    start_time = time.time()

    # Run inference
    for batch in tqdm(dataloader):
        batch.to(args.device)
        for p in target_property:
            for requires_grad_p in PROPERTY[p]["requires_grad_with"]:
                batch[requires_grad_p].requires_grad_(True)

        with torch.enable_grad():
            outputs = model(batch)
            if args.test == 1:
                update_metrics(metrics, "test", outputs, batch, target_property)

            preds.append(
                {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else None
                    for k, v in outputs.items()
                }
            )
            ptrs.append(batch.ptr.cpu())

    print(
        f"Inference finished in {time.time() - start_time:.2f} s "
        f"(batch_size={args.batch_size})"
    )

    if args.device == "cuda" and torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print(f"Device name: {args.device.upper()}")

    if args.test == 1:
        # Print metrics
        print("\n====== Evaluation Metrics ======")
        for k, v in metrics.items():
            print(f"{k}: {v.compute().item():.4f}")

    # Assign predictions back to atoms in the same order as atomsList
    pred_atoms_list = []
    structure_index = 0  # global counter across all batches
    for pred, ptr in zip(preds, ptrs):
        n_structures = len(ptr) - 1
        for i in range(n_structures):
            start, end = ptr[i], ptr[i + 1]
            atoms_i = atoms_list_copy[structure_index]
            for p in target_property:
                p_scope = PROPERTY[p]["scope"]
                p_rank = PROPERTY[p]["rank"]
                if p_scope == "per-system":
                    if p_rank == 0:
                        atoms_i.info[f"TACE_{p}"] = pred[p][i].item()
                    else:
                        atoms_i.info[f"TACE_{p}"] = (
                            pred[p][i].detach().cpu().numpy().reshape(-1)
                        )
                elif p_scope == "per-atom":
                    atoms_i.arrays[f"TACE_{p}"] = (
                        pred[p][start:end].detach().cpu().numpy().reshape(-1, 3**p_rank)
                    )
                elif p_scope == "per-edge":
                    atoms_i.info[f"TACE_{p}"] = (
                        pred[p][i].detach().cpu().numpy().reshape(-1)
                    )
                else:
                    atoms_i.info[f"TACE_{p}"] = (
                        pred[p][i].detach().cpu().numpy().reshape(-1)
                    )
            if args.scalar_descriptor == 1 and "scalar_descriptor" in pred:
                atoms_i.arrays["TACE_scalar_descriptor"] = (
                    pred["scalar_descriptor"][start:end].detach().cpu().numpy()
                )

            pred_atoms_list.append(atoms_i)
            structure_index += 1  # move to next structure

    ase.io.write(args.output, pred_atoms_list, format="extxyz")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
