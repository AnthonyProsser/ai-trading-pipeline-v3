"""SHA256 integrity manifest over deployment artifacts.

Scope (agent-config.md): predictor weights (.pt), scaler (.pkl), and constants.py.
The execution engine verifies these hashes at startup and on every weight reload; a
mismatch means refuse to start / refuse the reload.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ManifestError(Exception):
    """An artifact is missing or its on-disk hash does not match the manifest."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(artifacts: dict[str, str | Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in artifacts.items()}


def write_manifest(manifest: dict[str, str], out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, str]:
    loaded: dict[str, str] = json.loads(Path(path).read_text(encoding="utf-8"))
    return loaded


def verify_manifest(manifest: dict[str, str], artifacts: dict[str, str | Path]) -> None:
    uncovered = set(artifacts) - set(manifest)
    if uncovered:
        raise ManifestError(f"artifacts not covered by the manifest: {sorted(uncovered)}")
    for name, expected in manifest.items():
        if name not in artifacts:
            raise ManifestError(f"manifest artifact '{name}' was not provided for verification")
        path = Path(artifacts[name])
        if not path.exists():
            raise ManifestError(f"artifact '{name}' missing at {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ManifestError(f"artifact '{name}' hash mismatch: {actual} != {expected}")
