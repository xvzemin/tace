################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import shutil
from pathlib import Path

MAX_SIZE_BYTES = 5 * 1024**3  # 5 GB


def get_size(path: Path) -> int:
    """
    Return size of file or directory in bytes.
    - For files: directly use stat().st_size
    - For directories: sum sizes of all contained files
    """
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def all_under_limit(paths) -> bool:
    """
    Check if all given paths are under MAX_SIZE_BYTES.
    Returns True if all pass the size limit, otherwise False.
    """
    for path in paths:
        if not path.exists():
            continue
        size = get_size(path)
        if size > MAX_SIZE_BYTES:
            size_gb = size / 1024**3
            print(f"⛔ Skipped: {path} size {size_gb:.2f} GB exceeds 5GB limit.")
            return False
    return True


def delete_path(path: Path):
    """
    Delete a file or directory without checking size.
    """
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()


def main():
    targets = [
        "train.index",
        "valid.index",
        "wandb_logs",
        "lightning_logs",
        "outputs",
        "checkpoints",
        "_tace.yaml",
        "out.txt",
        "graphCache",
    ]
    current_dir = Path.cwd()
    target_paths = [current_dir / t for t in targets]
    extra_stats = list(current_dir.glob("statistics_*.yaml"))
    target_paths.extend(extra_stats)
    target_paths.extend(extra_stats)

    if all_under_limit(target_paths):
        for path in target_paths:
            if path.exists():
                delete_path(path)
    else:
        print("❌ Deletion aborted: At least one target exceeds size limit.")


if __name__ == "__main__":
    main()
