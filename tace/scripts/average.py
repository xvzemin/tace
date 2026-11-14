################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse

import torch

from tace.lightning import export_tace, load_tace


def average_models(model_paths, ema=False):
    if not model_paths:
        raise ValueError("No model paths provided.")
    model_avg = load_tace(
        model_paths[0],
        device="cpu",
        strict=True,
        use_ema=ema,
    )
    model_avg.eval()
    averaged = dict(model_avg.named_parameters())
    with torch.no_grad():
        for p in model_avg.parameters():
            p.zero_()

    for path in model_paths:
        model = load_tace(
            path,
            device="cpu",
            strict=True,
            use_ema=ema,
            dtype=model_avg.get_model_dtype(),
        )
        model.eval()
        parameters = dict(model.named_parameters())
        if (
            parameters.keys() != averaged.keys()
            or any(
                parameters[name].shape != value.shape
                for name, value in averaged.items()
            )
            or model.get_atomic_numbers() != model_avg.get_atomic_numbers()
        ):
            raise ValueError(
                "Models must have matching parameter names, shapes and elements."
            )
        with torch.no_grad():
            for name, parameter in averaged.items():
                parameter.add_(parameters[name])
    num_models = len(model_paths)
    with torch.no_grad():
        for p in model_avg.parameters():
            p.div_(num_models)
    return model_avg


def main():
    parser = argparse.ArgumentParser(description="SWA average TACE models")
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="Model files or registered foundation model names",
    )
    parser.add_argument(
        "-e",
        "--ema",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether use ema params, It is recommended not to use the ema parameter for averaging",
    )
    args = parser.parse_args()
    print(f"A total of {len(args.models)} models will be averaged")
    model_avg = average_models(args.models, args.ema)
    output_path = "average_model-state.pt"
    export_tace(model_avg, output_path)
    print(f"Averaged model saved to {output_path}")


if __name__ == "__main__":
    main()
