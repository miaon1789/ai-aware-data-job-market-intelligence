#!/usr/bin/env python3
"""Download the pinned MIT-licensed SkillSpan benchmark files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b"
BASE_URL = f"https://raw.githubusercontent.com/kris927b/SkillSpan/{COMMIT}"
FILES = ("data/json/test.json", "LICENSE", "README.md")


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "job-market-intelligence/0.1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/external/skillspan"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    written = []
    for relative in FILES:
        destination = args.output / Path(relative).name
        destination.write_bytes(download(f"{BASE_URL}/{relative}"))
        written.append(destination.name)
    provenance = {
        "dataset": "SkillSpan",
        "repository": "https://github.com/kris927b/SkillSpan",
        "commit": COMMIT,
        "license": "MIT",
        "files": written,
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
