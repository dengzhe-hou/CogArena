#!/usr/bin/env python3
"""Verify the six cached Ollama tags without pulling mutable registry tags."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import MODELS, RUN_ROOT, atomic_write_json, model_safe


EXPECTED_MANIFEST_SHA256 = {
    "openai/qwen2.5vl:7b": "5ced39dfa4bac325dc183dd1e4febaa1c46b3ea28bce48896c8e69c1e79611cc",
    "openai/llava:7b": "8dd30f6b0cb19f555f2c7a7ebda861449ea2cc76bf1f44e262931f45fc81d081",
    "openai/gemma3:4b": "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
    "openai/moondream:1.8b": "55fc3abd386771e5b5d1bbcc732f3c3f4df6e9f9f08f1131f9cc27ba2d1eec5b",
    "openai/llama3.2-vision:11b": "6f2f9757ae97e8a3f8ea33d6adb2b11d93d9a35bef277cd2c0b1b5af8e8d0b1e",
    "openai/minicpm-v:8b": "c92bfad0120556eda311984f1ac2f0d0a589b8d68c4053c13486b526276aa205",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_id: str) -> dict[str, Any]:
    api_model = model_id.split("/", 1)[-1]
    name, tag = api_model.rsplit(":", 1)
    store = Path.home() / ".ollama" / "models"
    manifest_path = store / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest_path.is_file():
        raise RuntimeError(f"cached manifest is missing: {manifest_path}")
    actual_manifest_sha = _sha256(manifest_path)
    expected_manifest_sha = EXPECTED_MANIFEST_SHA256[model_id]
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            f"cached manifest drift for {model_id}: {actual_manifest_sha} "
            f"!= {expected_manifest_sha}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptors = [manifest["config"], *manifest["layers"]]
    blobs: list[dict[str, Any]] = []
    for descriptor in descriptors:
        digest = descriptor["digest"]
        algorithm, hex_digest = digest.split(":", 1)
        if algorithm != "sha256" or len(hex_digest) != 64:
            raise RuntimeError(f"unsupported blob digest: {digest}")
        path = store / "blobs" / f"{algorithm}-{hex_digest}"
        if not path.is_file():
            raise RuntimeError(f"cached blob is missing: {path}")
        expected_size = descriptor["size"]
        if path.stat().st_size != expected_size:
            raise RuntimeError(
                f"cached blob size drift for {digest}: {path.stat().st_size} "
                f"!= {expected_size}"
            )
        blobs.append(
            {
                "digest": digest,
                "size": expected_size,
                "media_type": descriptor.get("mediaType"),
            }
        )
    result = {
        "model_id": model_id,
        "api_model": api_model,
        "manifest_sha256": actual_manifest_sha,
        "manifest_path_tail": f"registry.ollama.ai/library/{name}/{tag}",
        "blobs": blobs,
        "total_blob_bytes": sum(blob["size"] for blob in blobs),
    }
    atomic_write_json(RUN_ROOT / "raw" / model_safe(model_id) / "cache.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    result = verify(args.model)
    print(
        f"CACHE VERIFIED {result['model_id']} "
        f"manifest={result['manifest_sha256']} bytes={result['total_blob_bytes']}"
    )


if __name__ == "__main__":
    main()
