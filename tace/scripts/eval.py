################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import copy
import argparse
import time
from typing import Dict

import ase.io
from tqdm import tqdm
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
from e3nn.util.jit import compile

from ..lightning.lit_model import LightningWrapperModel
from ..dataset.element import TorchElement
from ..dataset.graph import from_atoms
from ..utils.metrics import build_metrics, update_metrics
from ..utils._global import DTYPE
from ..utils.utils import num_params
from ..dataset.quantity import KeySpecification, update_keyspec_from_kwargs, PROPERTY
from ..dataset.read import check_keys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict properties for structures using a trained model."
        "You could also print test metrics or predict property not in your training data "
        "(such as training with energy only and wishing to predict forces)"
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to ASE-readable input file")
    parser.add_argument("-o", "--output", type=str, default="predict.xyz",help="Path to ASE-writeable output file")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model checkpoint (.ckpt or .pt or .pth)")
    parser.add_argument("-t", "--test", type=int, default=0, help="print test set metric")
    parser.add_argument("-e", "--ema", type=int, default=1, help="Use EMA parameters during evaluation")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Tensor precision")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for inference")
    # parser.add_argument("-c", "--compile", type=int, default=0, help="Compile to jit-model, not support know")

    # Keys for properties, if need print test metrics
    for k, v in PROPERTY.items():
        if v['enable_prediction'] or v['enable_embedding']:
            parser.add_argument(f"--{k}_key", type=str, default=f"{k}")

    # Compute flags for predict property that not exist in training data
    for k, v in PROPERTY.items():
        if v['enable_prediction']:
            parser.add_argument(
                f"--compute_{k}", 
                type=int, 
                choices=[0, 1], 
                default=0, 
                help='Allow predict property beyond training'
            )
    return parser.parse_args()

def main():
    args = parse_args()
    atomsList = ase.io.read(args.input, index=":")
    atoms_list_copy = copy.deepcopy(atomsList)
    key_spec = KeySpecification()
    update_keyspec_from_kwargs(key_spec, vars(args))
    # avoid conflict with ase 
    check = False
    if args.test == 1:
        check = True

    # Load model
    if args.model.endswith(".ckpt"):
        model = LightningWrapperModel.load_from_checkpoint(
            args.model,
            map_location=args.device,
            strict=True,
            use_ema=args.ema,
        )
    elif args.model.endswith(".pt") or args.model.endswith(".pth"):
        model = torch.load(args.model, weights_only=False, map_location=args.device)
    else:
        raise ValueError("❌ Model path must end with '.ckpt', '.pt', or '.pth'")
    max_neighbors = model.max_neighbors.item() if hasattr(model, "max_neighbors") else None
    cutoff = model.readout_fn.cutoff.item()
    atomic_numbers = model.readout_fn.atomic_numbers.cpu().tolist()
    device = args.device
    model_dtype = model.readout_fn.cutoff.dtype
    args_dtype = DTYPE[args.dtype]

    # Enable requested properties
    target_property = list(set(model.target_property))
    embedding_property = getattr(model, "embedding_property", [])

    # Compute flag
    compute_flags = {k[8:]: v for k, v in vars(args).items() if k.startswith("compute_")}
    for prop, flag in compute_flags.items():
        if flag:
            setattr(model.flags, f"compute_{prop}", True)
            target_property.append(prop)
    target_property = list(set(target_property))
    model.eval().to(device)
    print(f"Number of parameters: {num_params(model)}")
    if args_dtype != model_dtype:
        print(f"[Warning] Model dtype {(model_dtype)} does not match args.dtype {(args_dtype)}. Forcing dtype to {args_dtype}")
    torch.set_default_dtype(args_dtype)
    model.to(args_dtype)

    # Build metrics
    if args.test == 1:
        metrics: Dict = build_metrics("test", target_property)
        for m in metrics.values():
            m.to(device)

    # Dataset
    element = TorchElement([int(z) for z in atomic_numbers])
    dataset = [
        from_atoms(
            element,
            atoms,
            cutoff,
            max_neighbors="inf" if max_neighbors is None else max_neighbors,
            keyspec=key_spec,
            target_property=target_property,
            embedding_property=embedding_property,
            universal_embedding=getattr(model, 'universal_embedding', {}),
        )
        for atoms in check_keys(atomsList, target_property, key_spec, embedding_property, check)
    ]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    preds, ptrs = [], []
    start_time = time.time()

    # Run inference
    for batch in tqdm(dataloader):
        batch.to(device)
        for p in target_property:
            for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                batch[requires_grad_p].requires_grad_(True)

        with torch.enable_grad():
            outputs = model(batch)
            if args.test == 1:
                update_metrics(metrics, "test", outputs, batch, target_property)

            preds.append(
                {
                    k: v.detach().cpu() if isinstance(v, Tensor) else None
                    for k, v in outputs.items()
                }
            )
            ptrs.append(batch.ptr.cpu())

    print(f"Inference finished in {time.time() - start_time:.2f} s "
          f"(batch_size={args.batch_size})")

    if args.device == "cuda" and torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print(f"Device name: {device.upper()}")

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
                p_type = PROPERTY[p]['type']
                p_rank = PROPERTY[p]['rank']
                if p_type == 'graph':
                    if p_rank == 0:
                        atoms_i.info[f"TACE_{p}"] = pred[p][i].item()
                    else:
                        atoms_i.info[f"TACE_{p}"] = pred[p][i].numpy().reshape(-1)
                elif p_type == 'atom':
                    atoms_i.arrays[f"TACE_{p}"] = pred[p][start:end].numpy().reshape(-1, 3**p_rank)
                else:
                    raise TypeError(
                        f"Property '{p}' has unsupported type '{p_type}'. "
                        "Only 'graph' and 'atom' types are supported for writing to ASE files now."
                    )
            pred_atoms_list.append(atoms_i)
            structure_index += 1  # move to next structure

    ase.io.write(args.output, pred_atoms_list, format="extxyz")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
