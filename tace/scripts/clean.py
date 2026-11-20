################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import shutil
from pathlib import Path

GB = 1
MAX_SIZE_BYTES = GB * 1024**3  # 1 GB


def get_size(path: Path) -> int:
    """Return the file or directory size in bytes."""
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def all_under_limit(paths) -> bool:
    """Check the per-target size limit before removing any outputs."""
    for path in paths:
        if not path.exists():
            continue
        size = get_size(path)
        if size > MAX_SIZE_BYTES:
            size_gb = size / 1024**3
            print(f"⛔ Skipped: {path} size {size_gb:.2f} GB exceeds {GB} GB limit.")
            return False
    return True


def delete_path(path: Path):
    """Remove a training output without following a directory symlink."""
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()


def main(argv=None):
    argparse.ArgumentParser(
        description="Remove training outputs from the current directory (1 GiB limit per target)."
    ).parse_args(argv)
    targets = [
        "train.index",
        "valid.index",
        "wandb_logs",
        "lightning_logs",
        "outputs",
        "checkpoints_epoch",
        "checkpoints_step",
        "_full_config.yaml",
        "out.txt",
        "finetune_config.yaml",
    ]
    current_dir = Path.cwd()
    target_paths = [current_dir / t for t in targets]

    if all_under_limit(target_paths):
        for path in target_paths:
            if path.exists():
                delete_path(path)
    else:
        print("❌ Deletion aborted: At least one target exceeds size limit.")


if __name__ == "__main__":
    main()
