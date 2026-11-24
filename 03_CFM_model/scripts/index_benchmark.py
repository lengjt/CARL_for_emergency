#!/usr/bin/env python3
"""Build or rebuild the standalone replay index for an existing benchmark."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfm_model.benchmark_index import write_benchmark_index
from cfm_model.io import manifest, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    write_benchmark_index(args.benchmark, read_jsonl(args.benchmark / "episodes.jsonl"))
    write_json(args.benchmark / "manifest.json", manifest(args.benchmark))


if __name__ == "__main__":
    main()
