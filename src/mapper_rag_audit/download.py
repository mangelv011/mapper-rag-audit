"""Download and unpack SciFact (BEIR), refusing a changed or hostile archive."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"

Fetch = Callable[[str, Path], None]


def _fetch_url(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (dest / name).resolve().is_relative_to(dest):
                raise ValueError(f"archive member {name!r} points outside {dest}")
        z.extractall(dest)


def ensure_dataset(
    data_dir: Path,
    url: str = SCIFACT_URL,
    sha256: str = SCIFACT_SHA256,
    fetch: Fetch = _fetch_url,
) -> Path:
    """Return the unpacked dataset folder, downloading the zip only if needed."""
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "scifact.zip"
    if not (archive.exists() and _sha256(archive) == sha256):
        partial = archive.with_suffix(".part")
        fetch(url, partial)
        if _sha256(partial) != sha256:
            partial.unlink()
            raise ValueError(f"SHA-256 of the download from {url} does not match the pinned value")
        partial.replace(archive)
    _extract(archive, raw)
    return raw / "scifact"
