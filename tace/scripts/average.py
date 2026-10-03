################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse

import torch

from tace.lightning import load_tace


def average_models(model_paths, ema=False):
    assert len(model_paths) > 0, "No model paths provided."
    model_avg = load_tace(
        model_paths[0],
        device="cpu",
        strict=True,
        use_ema=ema,
    )
    model_avg.eval()
    with torch.no_grad():
        for p in model_avg.parameters():
            p.data.zero_()

    for path in model_paths:
        model = load_tace(
            path,
            device="cpu",
            strict=True,
            use_ema=ema,
            dtype=model_avg.get_model_dtype(),
        )
        model.eval()
        with torch.no_grad():
            for p_avg, p in zip(model_avg.parameters(), model.parameters()):
                p_avg.data.add_(p.data)
    num_models = len(model_paths)
    with torch.no_grad():
        for p in model_avg.parameters():
            p.data.div_(num_models)
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
    torch.save(
        {
            "state_dict": model_avg.state_dict(),
            "cfg": model_avg.readout_fn.model_config,
            "target_property": model_avg.readout_fn.target_property,
            "embedding_property": model_avg.readout_fn.embedding_property,
            "statistics": model_avg.readout_fn.statistics,
        },
        output_path,
    )
    print(f"Averaged model saved to {output_path}")


if __name__ == "__main__":
    main()
