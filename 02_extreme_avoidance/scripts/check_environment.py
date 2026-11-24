"""Fail fast when the project environment differs from the validated baseline."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    expected = {}
    for line in LOCK.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            expected[name.lower()] = version

    import matplotlib
    import numpy
    actual = {"numpy": numpy.__version__, "matplotlib": matplotlib.__version__}
    errors = [f"{name}: expected {version}, found {actual.get(name)}"
              for name, version in expected.items() if actual.get(name) != version]
    in_project_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    if not in_project_venv:
        errors.append(f"interpreter is outside project venv: {sys.executable}")
    result = {
        "ok": not errors,
        "python": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "packages": actual,
        "system_site_packages": True,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        if args.output.exists():
            parser.error("Output file already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
