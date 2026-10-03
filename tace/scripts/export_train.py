################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
from pathlib import Path

from tace.lightning import export_tace, load_tace


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a TACE model for training, fine-tuning, or transfer.",
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
        "-o", "--output", type=str, default=None, help="Output .pt path"
    )
    parser.add_argument(
        "-f",
        "--fidelity_idx",
        type=int,
        default=None,
        help="Which fidelity to export",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default=None,
        help="Model dtype",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Load device")
    return parser.parse_args()


def _default_output_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_name(path.name + "-state.pt"))


def main():
    args = parse_args()
    model = load_tace(
        args.model,
        args.device,
        strict=True,
        use_ema=True,
        dtype=args.dtype,
    )
    model.reset_fidelity_idx(args.fidelity_idx)

    output_path = args.output or _default_output_path(args.model)
    export_tace(model, output_path)
    print(f"[Done] training model saved to: {output_path}")


if __name__ == "__main__":
    main()
