################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Switch between equivalent global and aligned-frame CGTP implementations."""

import argparse
from pathlib import Path

from tace.foundations import tace_foundations
from tace.lightning import convert_cgtp, export_tace, load_tace


def main():
    parser = argparse.ArgumentParser(
        description="Automatically switch cgtp and o2_cgtp interactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model file or registered foundation model name",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="Model dtype; omitted to preserve the stored precision",
    )
    parser.add_argument("--device", default="cpu", help="Conversion device")
    args = parser.parse_args()

    path = Path(args.model)
    name = path.name if args.model in tace_foundations else path.stem
    output = path.with_name(f"{name}-converted.pt")
    if output.exists():
        parser.error(f"Output already exists: {output}")

    model = load_tace(args.model, device=args.device, dtype=args.dtype)
    converted = convert_cgtp(model)
    export_tace(converted, str(output))
    print(f"Converted model saved to: {output}")


if __name__ == "__main__":
    main()
