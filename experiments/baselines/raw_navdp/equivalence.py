import hashlib
import json
from pathlib import Path


def _hash(path):
    digest = hashlib.sha256(); digest.update(Path(path).read_bytes()); return digest.hexdigest()


def verify_provenance(repo_root):
    root = Path(repo_root); provenance = json.loads((Path(__file__).parent / "provenance.json").read_text()); errors = []
    for source in provenance["sources"]:
        path = root / source["source_relative_path"]
        if not path.exists(): errors.append(f"missing source: {path}")
        elif _hash(path) != source["source_sha256"]: errors.append(f"hash mismatch: {path}")
    return errors
