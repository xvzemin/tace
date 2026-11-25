################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import logging
from pathlib import Path

import torch

from tace.lightning import load_tace
from tace.models.compile import export_lammps_aotinductor
from tace.utils.env import enable_acceleration

FOUNDATION_BACKENDS = ("eqx-foundation", "eqx-foundation-throughput")
ALLOWED_BACKEND = ["mliap", "aoti", "eqx-oam-l", "eqx-oam-l-throughput", *FOUNDATION_BACKENDS]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a TACE model for LAMMPS MLIAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model file or registered foundation model name",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "--backend",
        type=str,
        default="mliap",
        choices=ALLOWED_BACKEND,
        help="LAMMPS model backend",
    )
    parser.add_argument(
        "--aoti-package",
        type=str,
        default=None,
        help="Output .pt2 path used by the aoti backend",
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
    parser.add_argument("--device", type=str, default="cuda", help="Load device")
    return parser.parse_args()


def _default_mliap_output_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_name(path.name + "-lammps_mliap.pt"))


def _default_aoti_output_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_name(path.name + "-lammps_aoti.pt"))


def _default_aoti_package_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_name(path.name + "-lammps_aoti.pt2"))


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if args.backend in ("eqx-oam-l", "eqx-oam-l-throughput", *FOUNDATION_BACKENDS):
        if torch.device(args.device).type != "cuda" or args.dtype == "float64":
            raise ValueError("The single-rank EQX foundation package requires FP32 on CUDA.")
        args.dtype = "float32"
        enable_acceleration(enable_eqx=True, enable_compile=True, force=True)
    elif args.backend == "aoti":
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

    try:
        from tace.interface.lammps import TACEAOTILammpsCalc, TACELammpsCalc
    except ImportError as exc:
        raise RuntimeError("LAMMPS export requires the LAMMPS Python package.") from exc

    if args.backend in FOUNDATION_BACKENDS:
        from equivariantx.models.tace.foundation import foundation_architecture

        capacity = args.backend == "eqx-foundation"
        arch = foundation_architecture(model)
        args.output = args.output or f"{args.model}-{args.backend}.pt"
        if arch == "l":
            args.backend = "eqx-oam-l" if capacity else "eqx-oam-l-throughput"
        else:
            if arch == "7m":
                from equivariantx.models.tace.tace_oam_7m.lammps import export_lammps
            else:
                from equivariantx.models.tace.tece_oam_rra.lammps import export_lammps
            path = export_lammps(
                model, args.output, capacity=capacity, aoti_package=args.aoti_package
            )
            print(f"[Done] Single-rank EQX ML-IAP loader saved to: {path}")
            return
    if args.backend == "mliap":
        model.lmp = True
        lammps_model = TACELammpsCalc(model)
        output_path = args.output or _default_mliap_output_path(args.model)
        torch.save(lammps_model, output_path)
        print(f"[Done] LAMMPS MLIAP model saved to: {output_path}")
    elif args.backend == "eqx-oam-l":
        from equivariantx.models.tace.tace_oam_l import optimize_oam_l
        from equivariantx.models.tace.tace_oam_l.receiver import CapacityOAMLammpsCalc
        from equivariantx.models.tace.tace_oam_l.tangent import frozen_tangent_model

        output = Path(args.output or f"{args.model}-eqx-oam-l.pt")
        streamed = Path(args.aoti_package or output.with_suffix(".pt2"))
        if streamed.suffix != ".pt2":
            streamed = Path(str(streamed) + ".pt2")
        compact = streamed.with_name(streamed.stem + "-compact.pt2")
        if output.resolve() in (streamed.resolve(), compact.resolve()):
            raise ValueError("Use distinct paths for the ML-IAP loader and its .pt2 packages.")
        lowered = frozen_tangent_model(model)
        export_lammps_aotinductor(lowered, streamed, local_owners=True)
        del lowered
        lowered = optimize_oam_l(
            model, memory_mode="recompute", fused_geometry=True,
            fused_projection=False, grouped_coefficients=False,
        )
        export_lammps_aotinductor(lowered, compact, local_owners=True)
        del lowered
        loader = CapacityOAMLammpsCalc.from_packages(
            streamed,
            compact,
            model.get_atomic_numbers(),
            model.get_cutoff(),
            model.get_model_dtype(),
        ).attach_receiver(model)
        loader.receiver_tangent = True
        loader.domain_receiver_tangent = True
        torch.save(loader, output, pickle_protocol=5)
        print(f"[Done] Single-rank OAM-L ML-IAP loader saved to: {output}")
    elif args.backend == "eqx-oam-l-throughput":
        from equivariantx.models.tace.tace_oam_l.tangent import export_tangent_lammps

        output = args.output or f"{args.model}-eqx-oam-l-throughput.pt"
        path = export_tangent_lammps(model, output, aoti_package=args.aoti_package)
        print(f"[Done] Single-rank OAM-L ML-IAP loader saved to: {path}")
    elif args.backend == "aoti":
        package_path = args.aoti_package or _default_aoti_package_path(args.model)
        package_path = export_lammps_aotinductor(model, package_path)
        lammps_model = TACEAOTILammpsCalc(
            package_path=package_path,
            atomic_numbers=model.get_atomic_numbers(),
            cutoff=model.get_cutoff(),
            dtype=model.get_model_dtype(),
        )
        output_path = args.output or _default_aoti_output_path(args.model)
        torch.save(lammps_model, output_path)
        print(f"[Done] LAMMPS AOTInductor package saved to: {package_path}")
        print(f"[Done] LAMMPS MLIAP loader saved to: {output_path}")
    else:
        raise ValueError(
            f"Unsupported backend '{args.backend}'. One of {ALLOWED_BACKEND} is available."
        )


if __name__ == "__main__":
    main()
