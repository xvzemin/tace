################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import argparse

import torch

from tace.lightning import load_tace
from tace.lightning.lora import from_lora_to_merged_model

ALLOWED_TYPE = ["merge_lora", "merged_lora"]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Model dtype",
        choices=["float32", "float64"],
        default=None,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="merge_lora",
        choices=ALLOWED_TYPE,
        help="Specify convert type",
    )
    parser.add_argument(
        "--debug", type=int, default=0, help="print some extra information for debug"
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()
    model = load_tace(
        args.model,
        "cpu",
        strict=True,
        use_ema=True,
        dtype=args.dtype,
    )
    if bool(args.debug):
        print(model)
    if args.type in {"merge_lora", "merged_lora"}:
        total_before = count_parameters(model)
        model = from_lora_to_merged_model(model)
        total_after = count_parameters(model)
        if bool(args.debug):
            print(model)
        print("The number of parameters: ")
        print(f"  Your LoRA:     {total_before - total_after}")
        print(f"  Before merged: {total_before}")
        print(f"  After merged:  {total_after}")
        torch.save(model, args.model + "-merged_lora.pt")
    else:
        raise ValueError(
            f"Unsupported convert type '{args.type}'. One of {ALLOWED_TYPE} is available."
        )


if __name__ == "__main__":
    main()
