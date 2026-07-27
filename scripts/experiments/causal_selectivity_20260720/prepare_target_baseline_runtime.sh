#!/bin/bash
set -euo pipefail

FORMAL_REVISION="${1:?usage: prepare_target_baseline_runtime.sh FORMAL_REVISION ANALYSIS_REVISION}"
ANALYSIS_REVISION="${2:?usage: prepare_target_baseline_runtime.sh FORMAL_REVISION ANALYSIS_REVISION}"
ROOT="$(git rev-parse --show-toplevel)"
[[ "${FORMAL_REVISION}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "FATAL: formal revision is not 40-hex"
  exit 1
}
[[ "${ANALYSIS_REVISION}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "FATAL: analysis revision is not 40-hex"
  exit 1
}
[[ "$(git rev-parse "${FORMAL_REVISION}^{commit}")" == "${FORMAL_REVISION}" ]] || {
  echo "FATAL: formal revision is unavailable"
  exit 1
}
[[ "$(git rev-parse "${ANALYSIS_REVISION}^{commit}")" == "${ANALYSIS_REVISION}" ]] || {
  echo "FATAL: analysis revision is unavailable"
  exit 1
}

RUNTIME_PARENT="${ROOT}/results/causal_selectivity_20260720/runtime"
RUNTIME_ROOT="${RUNTIME_PARENT}/formal_${FORMAL_REVISION}__analysis_${ANALYSIS_REVISION}"
OVERLAY_PATHS=(
  "scripts/experiments/causal_selectivity_20260720/TARGET_BASELINE_SENSITIVITY_SPEC.json"
  "scripts/experiments/causal_selectivity_20260720/analyze_target_baseline.py"
  "scripts/experiments/causal_selectivity_20260720/analyze_target_baseline.sbatch"
)
mkdir -p "${RUNTIME_PARENT}"
if [[ -e "${RUNTIME_ROOT}" ]]; then
  echo "FATAL: runtime already exists; refuse in-place mutation: ${RUNTIME_ROOT}"
  exit 1
fi
TMP_ROOT="$(mktemp -d "${RUNTIME_PARENT}/.target-baseline-runtime.XXXXXX")"
cleanup() {
  rm -rf "${TMP_ROOT}"
}
trap cleanup EXIT

# Batch nodes need no Git: assemble both immutable layers before submission.
git archive "${FORMAL_REVISION}" | tar --exclude='results' -x -C "${TMP_ROOT}"
git archive "${ANALYSIS_REVISION}" "${OVERLAY_PATHS[@]}" | tar -x -C "${TMP_ROOT}"
ln -s ../../.. "${TMP_ROOT}/results"

FORMAL_TREE="$(git rev-parse "${FORMAL_REVISION}^{tree}")"
export TMP_ROOT FORMAL_REVISION ANALYSIS_REVISION FORMAL_TREE
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["TMP_ROOT"])
paths = [
    "scripts/experiments/causal_selectivity_20260720/"
    "TARGET_BASELINE_SENSITIVITY_SPEC.json",
    "scripts/experiments/causal_selectivity_20260720/"
    "analyze_target_baseline.py",
    "scripts/experiments/causal_selectivity_20260720/"
    "analyze_target_baseline.sbatch",
]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = {
    "schema_version": "cogarena.causal_selectivity.target_baseline_runtime.v1",
    "formal_source_revision": os.environ["FORMAL_REVISION"],
    "formal_source_git_tree": os.environ["FORMAL_TREE"],
    "analysis_overlay_revision": os.environ["ANALYSIS_REVISION"],
    "overlay_paths": paths,
    "overlay_sha256": {relative: sha256(root / relative) for relative in paths},
    "results_symlink_relative_target": "../../..",
    "assembly": "git archive of formal source, then exactly three paths from analysis revision",
}
(root / "TARGET_BASELINE_RUNTIME_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

mv "${TMP_ROOT}" "${RUNTIME_ROOT}"
trap - EXIT
RUNTIME_MANIFEST="${RUNTIME_ROOT}/TARGET_BASELINE_RUNTIME_MANIFEST.json"
echo "COGARENA_FROZEN_RUNTIME_ROOT=${RUNTIME_ROOT}"
echo "COGARENA_TARGET_BASELINE_RUNTIME_MANIFEST_SHA256=$(sha256sum "${RUNTIME_MANIFEST}" | awk '{print $1}')"
