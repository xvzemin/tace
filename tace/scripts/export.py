################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import argparse

import torch
from e3nn.util.jit import compile

from ..lightning.lit_model import LightningWrapperModel
from ..interface.lammps.mliap import LAMMPS_MLIAP_TACE


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Model dtype",
        choices=['float32', 'float64'],
        default=None,
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cpu", "cuda"], 
        help="Device for inference"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        default="lammps",
        choices=["lammps", "ase", "torch"], 
        help="Specify the backend to export"
    )
    return parser.parse_args()

DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    None: None
}

def main():
    args = parse_args()
    model_path = args.model
    if model_path.endswith(".ckpt"):
        model = LightningWrapperModel.load_from_checkpoint(
            model_path,
            map_location=args.device,
            strict=True,
            use_ema=1,
        )
    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        model = torch.load(model_path, weights_only=False, map_location=args.device)
    else:
        raise ValueError("❌ Model path must end with '.ckpt', '.pt', or '.pth'")

    model_dtype = model.readout_fn.cutoff.dtype
    args_dtype = DTYPE[args.dtype] or model_dtype
    if args_dtype != model_dtype:
        print(f"[Warning] Model dtype does not match args.dtype. Forcing dtype from {model_dtype} to {args_dtype}")
    torch.set_default_dtype(args_dtype)
    model.to(args_dtype)
    model.to(args.device)
    if args.backend == "lammps":
        model.lmp = True
        lammps_model = LAMMPS_MLIAP_TACE(model)
        torch.save(lammps_model, model_path + "-lmp_mliap.pt")
    elif args.backend == "ase":
        torch.save(model, model_path + "-ase.pt") 
    elif args.backend == "torch":
        torch.save(model, model_path + "-torch.pt") 
    else:
        raise ValueError(f"Unsupported backend '{args.backend}'. Currently only 'lammps' and 'ase(python)' is available.")

if __name__ == "__main__":
    main()