"""Small explicit I/O helpers with atomic finalization."""

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"}
