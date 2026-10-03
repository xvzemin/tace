################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download

from ..utils._global import CACHE_DIR

OAM_SERIES = {
    "TACE-OMat24-L": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-OMat24-L.pt",
    "TACE-OAM-L": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-OAM-L.pt",
    "TACE-OMat24-RRA-1.0": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-OMat24-RRA-1.0.pt",
    "TACE-OMat24-RRA-Preview": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-OMat24-RRA-Preview.pt",
    "TECE-OMat24-RRA-1.0": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TECE-OMat24-RRA-1.0.pt",
    "TECE-OAM-RRA-1.0": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TECE-OAM-RRA-1.0.pt",
    "TACE-OMat24-7M": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-OMat24-7M.pt",
    "TACE-OAM-7M": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-OAM-7M.pt",
}

REICO_SERIES = {}

LEGACY = {
    "TACE-v1-OMat24-M": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-v1-OMat24-M.pt",
    "TACE-v1-OAM-M": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-v1-OAM-M.pt",
    "TACE-v1-LES-REICO-5-PdAgCHO.pt": "https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-v1-LES-REICO-5-PdAgCHO.pt",
}


def _parse_hf_resolve_url(url: str) -> tuple[str, str, str]:
    """
    Parse Hugging Face resolve URL into (repo_id, revision, filename)

    Example:
    https://huggingface.co/xvzemin/tace-oam/resolve/main/foo.pt
    -> ("xvzemin/tace-oam", "main", "foo.pt")
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")

    try:
        resolve_idx = parts.index("resolve")
    except ValueError:
        raise ValueError(f"Not a valid Hugging Face resolve URL: {url}")

    repo_id = "/".join(parts[:resolve_idx])
    revision = parts[resolve_idx + 1]
    filename = "/".join(parts[resolve_idx + 2 :])

    if not repo_id or not revision or not filename:
        raise ValueError(f"Malformed Hugging Face resolve URL: {url}")

    return repo_id, revision, filename


class CachedModelRegistry(Mapping):
    def __init__(self, registry: dict[str, str], legacy: dict[str, str]):
        self._registry = registry
        self._legacy = legacy

    def __getitem__(self, key: str) -> Path:

        if key in self._legacy:
            print(f"[ERROR], Legacy pretrained model: {key}")
            self.print_models()
            raise KeyError(key)

        if key not in self._registry:
            print(f"[ERROR], Unknown pretrained model: {key}")
            self.print_models()
            raise KeyError(key)

        url = self._registry[key]
        target = Path(CACHE_DIR) / Path(urlparse(url).path).name

        if target.exists():
            return Path(target)

        # download
        try:
            repo_id, revision, filename = _parse_hf_resolve_url(url)
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=CACHE_DIR,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download pretrained model.\n"
                f"URL:\n"
                f"  {url}\n"
                f"Please manually download and place it at:\n"
                f"  {target}"
            ) from e

        # copy
        shutil.copy2(path, target)

        return Path(target)

    def __iter__(self):
        return iter(self._registry)

    def __contains__(self, key):
        return key in self._registry

    def __len__(self):
        return len(self._registry)

    def list_models(self) -> list[str]:
        return sorted(self._registry.keys())

    def print_models(self):
        print("Available pretrained models:")
        for name in self.list_models():
            print(f"  - {name}")


tace_foundations = CachedModelRegistry(
    registry=OAM_SERIES | REICO_SERIES, legacy=LEGACY
)


def resolve_model_path(model: Union[str, Path]) -> Path:
    """Resolve a model file or download a registered foundation model.

    Parameters
    ----------
    model : str or pathlib.Path
        Model file ending in ``.pt``, ``.pth``, ``.ckpt``, or ``.pt2``,
        or an exact name from ``tace_foundations.list_models()``.

    Returns
    -------
    pathlib.Path
        Local model path. Registered models are cached under ``~/.cache/tace``.
    """
    path = Path(model).expanduser()
    if path.suffix.lower() in (".pt", ".pth", ".ckpt", ".pt2"):
        return path
    if str(model) in tace_foundations:
        return tace_foundations[str(model)]
    raise ValueError(
        f"Unknown model {str(model)!r}. Provide a .pt, .pth, .ckpt, or .pt2 file "
        "or a registered foundation model name. Available models: "
        + ", ".join(tace_foundations.list_models())
    )
