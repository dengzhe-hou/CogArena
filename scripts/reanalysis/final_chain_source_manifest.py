#!/usr/bin/env python3
"""Build or verify the Git-independent source closure for the final chain.

Batch node c01 does not provide Git.  The committed JSON manifest therefore
pins every Python/shell source that can affect the paper recomputation.  The
runtime verifies the exact enumerated set and every SHA-256 before analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path


ROOT = Path(os.environ.get("COGARENA_ROOT", Path(__file__).resolve().parents[2])).resolve()
MANIFEST = ROOT / "results/reanalysis/aplus_20260718/FINAL_CHAIN_SOURCE_MANIFEST.json"
SOURCE_ROOTS = (
    "cogarena",
    "results/recompute_20260703",
    "results/construct_native_20260711",
    "results/twolevel_bootstrap_20260712",
    "results/pc1_validation_20260711",
    "scripts/reanalysis",
    "paper/figures",
)
EXPLICIT_FILES = (
    "scripts/reanalysis/run_final_chain.sh",
    "scripts/reanalysis/run_final_chain.sbatch",
    "scripts/reanalysis/prepare_final_chain_source.sbatch",
)


def req(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FINAL-SOURCE GATE FAILED: {message}")


def enforce_c01() -> None:
    req(bool(os.environ.get("SLURM_JOB_ID")), "must run inside Slurm")
    req(socket.gethostname().split(".", 1)[0].startswith("c01"), "must run on c01")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths() -> list[Path]:
    paths: set[Path] = set()
    for relative in SOURCE_ROOTS:
        base = ROOT / relative
        req(base.is_dir() and not base.is_symlink(), f"source root missing or unsafe: {relative}")
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            req(path.is_file() and not path.is_symlink(), f"unsafe source entry: {path}")
            paths.add(path.resolve())
    for relative in EXPLICIT_FILES:
        path = (ROOT / relative).resolve()
        req(path.is_file() and not path.is_symlink(), f"explicit source missing or unsafe: {relative}")
        paths.add(path)
    ordered = sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())
    req(ordered, "source closure is empty")
    return ordered


def build() -> dict:
    files = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in source_paths()
    }
    return {
        "schema_version": "cogarena.final_chain_source.v1",
        "hash_algorithm": "sha256",
        "selection": {
            "recursive_python_roots": list(SOURCE_ROOTS),
            "explicit_files": list(EXPLICIT_FILES),
            "exact_set_required": True,
        },
        "n_source_files": len(files),
        "source_files": files,
    }


def atomic_write(payload: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_name(MANIFEST.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, MANIFEST)


def verify() -> None:
    req(MANIFEST.is_file() and not MANIFEST.is_symlink(), "source manifest missing or unsafe")
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = build()
    req(frozen.get("schema_version") == current["schema_version"], "schema mismatch")
    req(frozen.get("source_files") == current["source_files"], "source set or hash drift")
    req(frozen.get("n_source_files") == len(current["source_files"]), "source count drift")
    print(f"FINAL SOURCE PASS: {current['n_source_files']} files")


def main() -> None:
    enforce_c01()
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        atomic_write(build())
        print(f"WROTE {MANIFEST.relative_to(ROOT)}")
    else:
        verify()


if __name__ == "__main__":
    main()
