"""Checkpointing and result persistence for CogArena.

Provides :class:`CheckpointManager` for saving, loading, and querying
evaluation results.  Results are stored as JSON files on disk, organised
by ``model_id / dimension / paradigm /``.

All file operations are thread-safe (uses ``threading.Lock`` and atomic
writes via temp files).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class CheckpointManager:
    """Manages persistence of evaluation results on disk.

    Directory layout::

        <root>/
          <model_id>/
            <dimension>/
              <paradigm>/
                <task_id>.json

    Usage::

        cm = CheckpointManager("results/checkpoints")
        cm.save_result("nback_abc123", "gpt-4o", result_dict)
        assert cm.has_result("nback_abc123", "gpt-4o")
        data = cm.load_result("nback_abc123", "gpt-4o")
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- Public API ---------------------------------------------------------

    def save_result(
        self,
        task_id: str,
        model_id: str,
        result: Dict[str, Any],
        dimension: Optional[str] = None,
        paradigm: Optional[str] = None,
    ) -> Path:
        """Persist a single result dict to disk (atomic write).

        If ``dimension`` and ``paradigm`` are not provided, they are inferred
        from the result dict (keys ``"dimension"`` / ``"paradigm"``) or
        defaulted to ``"_unknown"``.

        Returns:
            The path to the written JSON file.
        """
        dim = dimension or result.get("dimension", "_unknown")
        para = paradigm or result.get("paradigm", _infer_paradigm(task_id))
        safe_model = _safe_filename(model_id)
        safe_dim = _safe_filename(dim)
        safe_para = _safe_filename(para)
        safe_task = _safe_filename(task_id)

        dir_path = self.root / safe_model / safe_dim / safe_para
        file_path = dir_path / f"{safe_task}.json"

        payload = {
            "task_id": task_id,
            "model_id": model_id,
            "dimension": dim,
            "paradigm": para,
            "saved_at": time.time(),
            "data": result,
        }

        with self._lock:
            dir_path.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(file_path, payload)

        return file_path

    def load_result(
        self,
        task_id: str,
        model_id: str,
        dimension: Optional[str] = None,
        paradigm: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load a previously saved result.

        Returns the ``data`` value from the checkpoint, or ``None`` if not
        found.
        """
        path = self._find_result_path(task_id, model_id, dimension, paradigm)
        if path is None or not path.exists():
            return None
        with self._lock:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload.get("data")
            except (json.JSONDecodeError, OSError):
                return None

    def has_result(
        self,
        task_id: str,
        model_id: str,
        dimension: Optional[str] = None,
        paradigm: Optional[str] = None,
    ) -> bool:
        """Check whether a result exists without loading it."""
        path = self._find_result_path(task_id, model_id, dimension, paradigm)
        return path is not None and path.exists()

    def list_results(
        self,
        model_id: Optional[str] = None,
        dimension: Optional[str] = None,
        paradigm: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List saved results matching the given filters.

        Returns a list of checkpoint metadata dicts (including ``data``).
        """
        results: List[Dict[str, Any]] = []
        search_root = self.root

        if model_id:
            search_root = search_root / _safe_filename(model_id)
            if not search_root.exists():
                return []
        if dimension:
            search_root = search_root / _safe_filename(dimension)
            if not search_root.exists():
                return []
        if paradigm:
            search_root = search_root / _safe_filename(paradigm)
            if not search_root.exists():
                return []

        for json_file in search_root.rglob("*.json"):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
                results.append(payload)
            except (json.JSONDecodeError, OSError):
                continue

        return results

    def count_results(self, model_id: str) -> int:
        """Count how many results exist for a model."""
        model_dir = self.root / _safe_filename(model_id)
        if not model_dir.exists():
            return 0
        return sum(1 for _ in model_dir.rglob("*.json"))

    def get_completed_task_ids(self, model_id: str) -> set[str]:
        """Return the set of task_ids already completed for a model."""
        results = self.list_results(model_id=model_id)
        return {r.get("task_id", "") for r in results if r.get("task_id")}

    def delete_result(
        self,
        task_id: str,
        model_id: str,
        dimension: Optional[str] = None,
        paradigm: Optional[str] = None,
    ) -> bool:
        """Delete a specific result. Returns True if deleted."""
        path = self._find_result_path(task_id, model_id, dimension, paradigm)
        if path is None or not path.exists():
            return False
        with self._lock:
            try:
                path.unlink()
                return True
            except OSError:
                return False

    # -- Resume support -----------------------------------------------------

    def get_remaining_tasks(
        self,
        model_id: str,
        all_task_ids: List[str],
    ) -> List[str]:
        """Return task_ids from *all_task_ids* that have NOT been completed.

        Useful for resuming interrupted runs.
        """
        completed = self.get_completed_task_ids(model_id)
        return [tid for tid in all_task_ids if tid not in completed]

    # -- Internal -----------------------------------------------------------

    def _find_result_path(
        self,
        task_id: str,
        model_id: str,
        dimension: Optional[str] = None,
        paradigm: Optional[str] = None,
    ) -> Optional[Path]:
        """Locate the JSON file for a given result.

        If dimension/paradigm are unknown, search recursively under the
        model directory.
        """
        safe_model = _safe_filename(model_id)
        safe_task = _safe_filename(task_id)
        model_dir = self.root / safe_model

        if dimension and paradigm:
            return (
                model_dir
                / _safe_filename(dimension)
                / _safe_filename(paradigm)
                / f"{safe_task}.json"
            )

        # Search recursively
        if not model_dir.exists():
            return None
        for candidate in model_dir.rglob(f"{safe_task}.json"):
            return candidate
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Sanitise a string for use as a directory or file name."""
    # Replace slashes and other problematic characters
    safe = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace(" ", "_").replace("..", "_")
    # Truncate and add hash suffix if very long
    if len(safe) > 128:
        h = hashlib.sha256(name.encode()).hexdigest()[:8]
        safe = safe[:120] + "_" + h
    return safe


def _infer_paradigm(task_id: str) -> str:
    """Heuristic: extract paradigm from task_id (text before last ``_``)."""
    if "_" in task_id:
        return task_id.rsplit("_", 1)[0]
    return "_unknown"


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically: write to a temp file then rename.

    This prevents partial writes from corrupting checkpoints.
    """
    dir_path = path.parent
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(path))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError:
        # Fallback: direct write (less safe but works on all filesystems)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
