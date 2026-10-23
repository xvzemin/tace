"""Download and extract the complete 3BPA dataset to ~/dataset/3BPA."""

import shutil
import urllib.request
from pathlib import Path
from tempfile import TemporaryFile
from zipfile import ZipFile

URL = "https://api.figshare.com/v2/file/download/31325990"
DATA_DIR = Path.home() / "dataset" / "3BPA"


def main():
    print(f"Downloading {URL}", flush=True)
    with TemporaryFile() as temporary:
        with urllib.request.urlopen(URL, timeout=60) as response:
            shutil.copyfileobj(response, temporary)
        temporary.seek(0)
        with ZipFile(temporary) as archive:
            for member in archive.infolist():
                relative = Path(member.filename).relative_to("dataset_3BPA")
                destination = DATA_DIR / relative
                if not destination.resolve().is_relative_to(DATA_DIR.resolve()):
                    raise ValueError(f"Invalid archive path: {member.filename}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if destination.exists():
                    print(f"Already exists: {destination}", flush=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
                print(f"Extracted: {destination}", flush=True)
    print(f"3BPA is available in {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
