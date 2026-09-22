################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse

import yaml

from tace.lightning import load_tace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt or .pt or .pth)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_tace(args.model, "cpu", strict=True, use_ema=True)
    finetune_cfg = {}

    # === LoRA ===
    finetune_cfg["lora"] = {}

    def set_lora_config(name: str, r: int, alpha: float):
        finetune_cfg["lora"][name] = {
            "r": int(r),
            "alpha": float(alpha),
        }

    set_lora_config("torchLinear", 4, 8)
    set_lora_config("mlpLinear", 4, 8)
    set_lora_config("e3nnLinear", 4, 8)
    set_lora_config("e3nnElementLinear", 4, 8)
    set_lora_config("e3nnMoEElementLinear", 4, 8)

    # === Freeze ===
    finetune_cfg["freeze"] = {}
    for name, _ in model.named_parameters():
        finetune_cfg["freeze"][name] = True

    with open("finetune_config.yaml", "w") as f:
        yaml.dump(finetune_cfg, f, default_flow_style=False, sort_keys=False)

    print(
        "The finetune_config.yaml file has been created. tace-train will automatically read this file. "
        "You can modify the parameters in it to control the degree of freedom for fine-tuning. "
        "By default, all parameters are frozen and LoRA is used for fine-tuning. "
        "After training is completed, it is recommended to use the tace-convert-lora command "
        "to merge the LoRA parameters with the base model's parameters."
    )


if __name__ == "__main__":
    main()
