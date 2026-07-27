#!/usr/bin/env python3
"""Run one frozen VLM tag with atomic per-item response capture."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai

from .common import (
    MAX_TOKENS,
    MODELS,
    RUN_ROOT,
    TEMPERATURE,
    atomic_write_json,
    load_image_manifest,
    load_scoring_contract,
    model_safe,
    read_json,
    request_fingerprint,
    sha256_file,
    sha256_json,
)
from .scoring import SCORER_ID, parse_response


SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).with_name("common.py"),
    Path(__file__).with_name("scoring.py"),
    Path(__file__).with_name("run_array.sbatch"),
    Path(__file__).with_name("models.txt"),
    Path(__file__).with_name("verify_cache.py"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _image_content(paths: list[Path], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in paths:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
    return content


def _git_head() -> str:
    declared = os.environ.get("COGARENA_GIT_HEAD")
    if declared:
        return declared
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _call(
    client: openai.OpenAI,
    model_id: str,
    prompt: str,
    image_paths: list[Path],
    *,
    attempts: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retry transport faults only; a valid blank completion is scientific data."""

    messages = [{"role": "user", "content": _image_content(image_paths, prompt)}]
    api_model = model_id.split("/", 1)[-1]
    attempt_log: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        started_at = _utc_now()
        try:
            response = client.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.monotonic() - started
            raw = response.model_dump(mode="json")
            if not response.choices:
                raise RuntimeError("completion has no choices")
            choice = response.choices[0]
            content = choice.message.content
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise RuntimeError(f"completion content is not text: {type(content)}")
            if choice.finish_reason is None:
                raise RuntimeError("completion has no finish_reason")
            event = {
                "attempt": attempt,
                "started_at": started_at,
                "elapsed_seconds": elapsed,
                "transport_status": "ok",
                "finish_reason": choice.finish_reason,
                "content": content,
                "blank": not content.strip(),
                "raw_response": raw,
            }
            attempt_log.append(event)
            return event, attempt_log
        except Exception as exc:
            elapsed = time.monotonic() - started
            attempt_log.append(
                {
                    "attempt": attempt,
                    "started_at": started_at,
                    "elapsed_seconds": elapsed,
                    "transport_status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if attempt == attempts:
                raise RuntimeError(
                    f"VLM request failed after {attempts} transport attempts: {exc}"
                ) from exc
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def _api_json(path: str) -> Any:
    base = os.environ["OPENAI_BASE_URL"].rsplit("/v1", 1)[0]
    with urllib.request.urlopen(f"{base}{path}", timeout=30) as response:
        return json.load(response)


def capture_serving(model_id: str, cache: dict[str, Any]) -> dict[str, Any]:
    tags = _api_json("/api/tags")
    api_model = model_id.split("/", 1)[-1]
    matches = [
        entry
        for entry in tags.get("models", [])
        if entry.get("name") == api_model or entry.get("model") == api_model
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one /api/tags entry for {api_model}, got {len(matches)}")
    if matches[0].get("digest") != cache.get("manifest_sha256"):
        raise RuntimeError(
            f"runtime model digest differs from pinned cache manifest for {model_id}"
        )
    version = subprocess.run(
        ["ollama", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    ps = subprocess.run(
        ["ollama", "ps"], check=True, capture_output=True, text=True
    ).stdout
    matching_ps_rows = [
        line
        for line in ps.splitlines()
        if api_model in line and "100% GPU" in line
    ]
    if len(matching_ps_rows) != 1:
        raise RuntimeError(
            f"expected one 100% GPU ollama ps row for {api_model}, got:\n{ps}"
        )
    return {
        "captured_at": _utc_now(),
        "model_id": model_id,
        "api_model": api_model,
        "tag": matches[0],
        "ollama_version": version,
        "ollama_ps": ps,
        "gpu_row": matching_ps_rows[0],
        "node": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "serving_environment": {
            key: os.environ.get(key)
            for key in (
                "OLLAMA_CONTEXT_LENGTH",
                "OLLAMA_FLASH_ATTENTION",
                "OLLAMA_NUM_PARALLEL",
                "OLLAMA_MAX_LOADED_MODELS",
                "OLLAMA_KEEP_ALIVE",
            )
        },
    }


def _validate_existing(
    path: Path,
    model_id: str,
    task: dict[str, Any],
    request_fingerprint: str,
    manifest_sha256: str,
    scoring_contract_sha256: str,
    scoring: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid existing VLM record: {path}") from exc
    response = value.get("response")
    if not isinstance(response, str):
        raise RuntimeError(f"existing response is not text: {path}")
    expected_parse = parse_response(response=response, **scoring)
    attempts = value.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError(f"existing record lacks attempts: {path}")
    last = attempts[-1]
    raw_choices = (last.get("raw_response") or {}).get("choices") or []
    raw_choice = raw_choices[0] if len(raw_choices) == 1 else {}
    raw_content = (
        raw_choice.get("message", {}).get("content")
        if len(raw_choices) == 1
        else None
    )
    if raw_content is None:
        raw_content = ""
    session_id = value.get("session_id")
    session_sha256 = value.get("session_sha256")
    session_path = path.parents[1] / "sessions" / f"{session_id}.json"
    valid = (
        value.get("schema_version") == 2
        and value.get("model_id") == model_id
        and value.get("api_model") == model_id.split("/", 1)[-1]
        and value.get("task_id") == task["task_id"]
        and value.get("request_fingerprint") == request_fingerprint
        and value.get("image_manifest_sha256") == manifest_sha256
        and value.get("scoring_contract_sha256") == scoring_contract_sha256
        and value.get("scorer_id") == SCORER_ID
        and value.get("prompt_sha256") == sha256_json(task["prompt"])
        and value.get("image_sha256")
        == [entry["sha256"] for entry in task["images"]]
        and value.get("parse") == expected_parse
        and value.get("correct") is expected_parse["correct"]
        and value.get("blank") is (not response.strip())
        and last.get("transport_status") == "ok"
        and last.get("content") == response
        and last.get("blank") is (not response.strip())
        and raw_content == response
        and value.get("finish_reason") == last.get("finish_reason")
        and raw_choice.get("finish_reason") == value.get("finish_reason")
        and (last.get("raw_response") or {}).get("model")
        == value.get("api_model")
        and isinstance(session_id, str)
        and bool(session_id)
        and isinstance(session_sha256, str)
        and session_path.is_file()
        and sha256_file(session_path) == session_sha256
    )
    if not valid:
        raise RuntimeError(f"existing record failed replay validation: {path}")
    return True


def readiness(
    client: openai.OpenAI,
    model_id: str,
    probe: dict[str, Any],
) -> dict[str, Any]:
    """Exercise the image endpoint without conditioning on scientific outputs."""

    image_paths = [RUN_ROOT / entry["path"] for entry in probe["images"]]
    selected, attempts = _call(
        client, model_id, probe["prompt"], image_paths, attempts=3
    )
    return {
        "status": "pass",
        "model_id": model_id,
        "probe_id": probe["probe_id"],
        "output_is_not_scored": True,
        "response": selected["content"],
        "blank": not selected["content"].strip(),
        "finish_reason": selected["finish_reason"],
        "attempts": attempts,
    }


def run(model_id: str) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("VLM inference is permitted only inside a Slurm job")
    if os.uname().nodename.startswith(("login", "head")):
        raise RuntimeError("refusing to run VLM inference on a login node")

    manifest, manifest_sha256 = load_image_manifest(verify_images=True)
    scoring_contract, scoring_contract_sha256 = load_scoring_contract(
        manifest, manifest_sha256
    )
    safe = model_safe(model_id)
    model_root = RUN_ROOT / "raw" / safe
    record_root = model_root / "records"
    cache = read_json(model_root / "cache.json")
    if cache.get("model_id") != model_id:
        raise RuntimeError(f"cache identity mismatch for {model_id}")
    client = openai.OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        timeout=600,
    )

    readiness_result = readiness(client, model_id, manifest["readiness_probe"])
    serving = capture_serving(model_id, cache)
    created_at = _utc_now()
    session_id = sha256_json(
        {
            "model_id": model_id,
            "created_at": created_at,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "pid": os.getpid(),
        }
    )[:24]
    session = {
        "schema_version": 1,
        "session_id": session_id,
        "model_id": model_id,
        "created_at": created_at,
        "source_revision": _git_head(),
        "image_manifest_sha256": manifest_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "scorer_id": SCORER_ID,
        "cache_manifest_sha256": cache["manifest_sha256"],
        "source_files": {
            path.relative_to(Path(__file__).resolve().parents[3]).as_posix():
            sha256_file(path)
            for path in SOURCE_PATHS
        },
        "readiness": readiness_result,
        "serving": serving,
    }
    session_path = model_root / "sessions" / f"{session_id}.json"
    atomic_write_json(session_path, session)
    session_sha256 = sha256_file(session_path)

    for index, task in enumerate(manifest["tasks"], start=1):
        scoring = scoring_contract["tasks"][task["task_id"]]
        fingerprint = request_fingerprint(
            model_id, task, manifest_sha256, scoring_contract_sha256
        )
        path = record_root / f"{task['task_id']}.json"
        if _validate_existing(
            path,
            model_id,
            task,
            fingerprint,
            manifest_sha256,
            scoring_contract_sha256,
            scoring,
        ):
            print(f"[{index:03d}/250] {task['task_id']} resume")
            continue
        image_paths = [RUN_ROOT / entry["path"] for entry in task["images"]]
        selected, attempts = _call(
            client, model_id, task["prompt"], image_paths
        )
        response = selected["content"]
        parsed = parse_response(response=response, **scoring)
        correct = parsed["correct"]
        record = {
            "schema_version": 2,
            "task_id": task["task_id"],
            "model_id": model_id,
            "api_model": model_id.split("/", 1)[-1],
            "paradigm": task["paradigm"],
            "dimension": task["dimension"],
            "congruent": task["congruent"],
            "expected": task["expected"],
            "response": response,
            "blank": not response.strip(),
            "correct": correct,
            "parse": parsed,
            "scorer_id": SCORER_ID,
            "finish_reason": selected["finish_reason"],
            "request_fingerprint": fingerprint,
            "image_manifest_sha256": manifest_sha256,
            "scoring_contract_sha256": scoring_contract_sha256,
            "prompt_sha256": sha256_json(task["prompt"]),
            "image_sha256": [entry["sha256"] for entry in task["images"]],
            "attempts": attempts,
            "session_id": session_id,
            "session_sha256": session_sha256,
            "stored_at": _utc_now(),
        }
        atomic_write_json(path, record)
        state = "blank" if record["blank"] else ("correct" if correct else "wrong")
        print(f"[{index:03d}/250] {task['task_id']} {state}")

    records = sorted(record_root.glob("*.json"))
    if len(records) != len(manifest["tasks"]):
        raise RuntimeError(f"{model_id} wrote only {len(records)} records")
    print(f"MODEL DONE {model_id} records={len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
