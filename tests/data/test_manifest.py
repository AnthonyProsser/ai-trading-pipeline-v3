"""Exit criteria for the SHA256 manifest (agent-config.md "SHA256 manifest")."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.data.manifest import (
    ManifestError,
    build_manifest,
    load_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)


def _write(p: Path, content: bytes) -> Path:
    p.write_bytes(content)
    return p


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    f = _write(tmp_path / "a.bin", b"hello world")
    assert sha256_file(f) == hashlib.sha256(b"hello world").hexdigest()


def test_build_manifest_maps_names_to_hashes(tmp_path: Path) -> None:
    f1 = _write(tmp_path / "weights.pt", b"weights")
    f2 = _write(tmp_path / "scaler.pkl", b"scaler")
    m = build_manifest({"weights": f1, "scaler": f2})
    assert m["weights"] == hashlib.sha256(b"weights").hexdigest()
    assert set(m) == {"weights", "scaler"}


def test_write_load_roundtrip(tmp_path: Path) -> None:
    m = build_manifest({"constants": _write(tmp_path / "constants.py", b"X = 1")})
    out = tmp_path / "manifest.json"
    write_manifest(m, out)
    assert load_manifest(out) == m


def test_verify_passes_when_unchanged(tmp_path: Path) -> None:
    arts = {"weights": _write(tmp_path / "w.pt", b"data")}
    verify_manifest(build_manifest(arts), arts)  # must not raise


def test_verify_raises_on_modification(tmp_path: Path) -> None:
    f = _write(tmp_path / "w.pt", b"data")
    arts = {"weights": f}
    m = build_manifest(arts)
    _write(f, b"tampered")
    with pytest.raises(ManifestError):
        verify_manifest(m, arts)


def test_verify_raises_when_artifact_absent_from_manifest(tmp_path: Path) -> None:
    # A stale manifest covering only 'weights' must not silently pass a fuller artifact set.
    f1 = _write(tmp_path / "w.pt", b"weights")
    f2 = _write(tmp_path / "s.pkl", b"scaler")
    manifest = build_manifest({"weights": f1})
    with pytest.raises(ManifestError):
        verify_manifest(manifest, {"weights": f1, "scaler": f2})


def test_build_manifest_raises_on_missing_path(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        build_manifest({"weights": tmp_path / "does_not_exist.pt"})


def test_verify_raises_on_missing_artifact(tmp_path: Path) -> None:
    f = _write(tmp_path / "w.pt", b"data")
    arts = {"weights": f}
    m = build_manifest(arts)
    f.unlink()
    with pytest.raises(ManifestError):
        verify_manifest(m, arts)
