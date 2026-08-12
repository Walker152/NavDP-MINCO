from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 of a regular file without loading it into memory."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_regular_file(path: Path, root: Path) -> tuple[Path, str]:
    root = root.resolve()
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact is outside receipt root: {path}") from error
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"artifact must be a regular non-symlink file: {path}")
    return path, relative.as_posix()


def file_receipt(path: Path | str, root: Path | str) -> dict[str, object]:
    path, relative = _relative_regular_file(Path(path), Path(root))
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory_receipts(
    root: Path | str,
    *,
    exclude: Iterable[Path | str] = (),
) -> list[dict[str, object]]:
    root = Path(root).resolve()
    excluded = {Path(path).resolve() for path in exclude}
    return [
        file_receipt(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.resolve() not in excluded
    ]


def validate_file_receipt(
    root: Path | str, receipt: Mapping[str, object]
) -> list[str]:
    root = Path(root).resolve()
    relative = str(receipt.get("path", ""))
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return [f"artifact path escapes receipt root: {relative}"]
    if not candidate.is_file() or candidate.is_symlink():
        return [f"missing artifact {relative}"]
    errors: list[str] = []
    try:
        expected_size = int(receipt["size_bytes"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"invalid size receipt {relative}")
    else:
        if candidate.stat().st_size != expected_size:
            errors.append(f"size mismatch {relative}")
    expected_hash = receipt.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        errors.append(f"invalid hash receipt {relative}")
    elif sha256_file(candidate) != expected_hash:
        errors.append(f"hash mismatch {relative}")
    return errors
