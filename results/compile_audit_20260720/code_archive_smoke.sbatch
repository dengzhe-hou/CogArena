#!/bin/bash
#SBATCH --job-name=cog-code-smoke
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/compile_audit_20260720/code_archive_smoke_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
[[ "$(hostname -s)" == c01* ]] || { echo "FATAL: smoke test must run on c01"; exit 1; }
command -v python >/dev/null || { echo "FATAL: python is not on PATH"; exit 1; }
: "${COGARENA_CODE_ARCHIVE:?inject the code_data.zip path}"
: "${COGARENA_CODE_ARCHIVE_SHA256:?inject the archive SHA-256}"
: "${COGARENA_CODE_ARCHIVE_HEAD:?inject the 40-hex source revision}"
[[ "${COGARENA_CODE_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "FATAL: archive SHA-256 is not 64-hex"; exit 1; }
[[ "${COGARENA_CODE_ARCHIVE_HEAD}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "FATAL: archive source revision is not 40-hex"; exit 1; }
ARCHIVE="$(realpath "${COGARENA_CODE_ARCHIVE}")"
[[ -f "${ARCHIVE}" ]] || { echo "FATAL: archive missing: ${ARCHIVE}"; exit 1; }
[[ "$(sha256sum "${ARCHIVE}" | awk '{print $1}')" == "${COGARENA_CODE_ARCHIVE_SHA256}" ]] \
  || { echo "FATAL: archive SHA-256 mismatch"; exit 1; }
python - "${ARCHIVE}" "${COGARENA_CODE_ARCHIVE_HEAD}" <<'PY'
import pathlib
import stat
import sys
import zipfile

archive, expected_revision = sys.argv[1:]
with zipfile.ZipFile(archive) as source:
    actual_revision = source.comment.decode("ascii")
    if actual_revision != expected_revision:
        raise SystemExit(
            f"FATAL: git-archive revision mismatch: "
            f"{actual_revision} != {expected_revision}"
        )
    if source.testzip() is not None:
        raise SystemExit("FATAL: ZIP CRC/integrity failure")
    infos = source.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise SystemExit("FATAL: duplicate ZIP member names")
    for info in infos:
        member = pathlib.PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise SystemExit(f"FATAL: unsafe ZIP path: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise SystemExit(f"FATAL: ZIP symlink not allowed: {info.filename}")
        if info.flag_bits & 1:
            raise SystemExit(f"FATAL: encrypted ZIP member: {info.filename}")
    required = {
        "RELEASE_REDACTIONS.json",
        "scripts/reanalysis/apply_corrected_results.py",
        "results/reanalysis/scaling_mixedeffects.json",
        "results/reanalysis/scaling_mixedeffects_table.csv",
        "results/reanalysis/scaling_mixedeffects_table.tex",
        "tests/test_scaling_mixedlm_publication.py",
        "scripts/experiments/causal_selectivity_20260720/"
        "TARGET_BASELINE_SENSITIVITY_SPEC.json",
        "scripts/experiments/causal_selectivity_20260720/"
        "analyze_target_baseline.py",
        "scripts/experiments/causal_selectivity_20260720/"
        "analyze_target_baseline.sbatch",
        "scripts/experiments/causal_selectivity_20260720/"
        "prepare_target_baseline_runtime.sh",
        "scripts/experiments/causal_selectivity_20260720/tests/"
        "test_target_baseline.py",
        "results/causal_selectivity_20260720/analysis/"
        "target_baseline_sensitivity/target_baseline_results.json",
        "results/causal_selectivity_20260720/analysis/"
        "target_baseline_sensitivity/TARGET_BASELINE_MANIFEST.json",
        "tests/test_target_baseline_artifact.py",
        "scripts/posthoc/signature_family_sensitivity.py",
        "results/reanalysis/signature_family_sensitivity.json",
        "results/vlm_rerun_20260724_authority/VLM_FIGURE_ADDENDUM.json",
        "results/vlm_rerun_20260724_authority/VLM_RERUN_MANIFEST.json",
        "results/vlm_rerun_20260724_authority/VLM_RERUN_SUMMARY.json",
    }
    missing = sorted(required - set(names))
    if missing:
        raise SystemExit(f"FATAL: release analysis closure missing: {missing}")
    release_excluded = {
        "paper/HUMAN_BASELINE_REMEDIATION_20260720.md",
        "paper/ReproducibilityChecklist_cogarena.tex",
    }
    leaked_release_internal = sorted(release_excluded & set(names))
    if leaked_release_internal:
        raise SystemExit(
            f"FATAL: internal paper file leaked into code archive: "
            f"{leaked_release_internal}"
        )
    forbidden = [
        name
        for name in names
        if name.startswith(("results/full_eval", "results/multiturn"))
        or "/runtime/" in name
        or "/slurm/" in name
        or name.endswith((".log", ".out"))
    ]
    if forbidden:
        raise SystemExit(f"FATAL: private/runtime output leaked: {forbidden[:5]}")
PY
DEST="$SLURM_SUBMIT_DIR/results/compile_audit_20260720/code_archive_smoke_$SLURM_JOB_ID"
[ ! -e "$DEST" ] || { echo "FATAL: smoke-test directory already exists"; exit 1; }
python - "$ARCHIVE" "$DEST" <<'PY'
import pathlib, sys, zipfile
archive, dest = map(pathlib.Path, sys.argv[1:])
dest.mkdir()
with zipfile.ZipFile(archive) as z:
    z.extractall(dest)
print(f'extracted {archive} to {dest}')
PY
cd "$DEST"
python -m venv .venv
.venv/bin/python -m pip install -e ".[analysis,dev,image,openai]"
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze \
  > "$SLURM_SUBMIT_DIR/results/compile_audit_20260720/code_archive_smoke_${SLURM_JOB_ID}_requirements.txt"
export PYTHONPATH="$DEST"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTEST_REPORT="$SLURM_SUBMIT_DIR/results/compile_audit_20260720/code_archive_smoke_${SLURM_JOB_ID}_pytest.xml"
.venv/bin/python -m pytest -q -rs --junitxml="$PYTEST_REPORT"
.venv/bin/python - "$PYTEST_REPORT" <<'PY'
import sys
import xml.etree.ElementTree as ET

report = ET.parse(sys.argv[1]).getroot()
skips = []
for case in report.iter("testcase"):
    skipped = case.find("skipped")
    if skipped is not None:
        skips.append((case.get("classname"), case.get("name"), skipped.get("message", "")))
allowed_fragments = (
    "full results tree not present",
    "local annotator bundle not built",
)
unexpected = [
    row for row in skips
    if not any(fragment in row[2] for fragment in allowed_fragments)
]
if len(skips) != 4 or unexpected:
    raise SystemExit(
        f"FATAL: expected exactly four documented raw-data skips; "
        f"observed={skips}, unexpected={unexpected}"
    )
print(f"pytest skip gate OK: {len(skips)} documented raw-data skips")
PY
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
manifest_path = root / "results/reanalysis/aplus_20260718/FINAL_CHAIN_SOURCE_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
addendum = json.loads(
    (
        root
        / "results/vlm_rerun_20260724_authority/VLM_FIGURE_ADDENDUM.json"
    ).read_text(encoding="utf-8")
)
replacements = {
    row["path"]: row for row in addendum.get("source_replacements", [])
}
if len(replacements) != len(addendum.get("source_replacements", [])):
    raise SystemExit("FATAL: duplicate VLM source-replacement path")
files = manifest.get("source_files", {})
if manifest.get("n_source_files") != len(files):
    raise SystemExit("FATAL: frozen source-manifest count mismatch")
for relative, expected in files.items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"FATAL: frozen chain source missing: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        replacement = replacements.get(relative)
        if (
            not replacement
            or replacement["frozen_sha256"] != expected
            or replacement["replacement_sha256"] != actual
        ):
            raise SystemExit(
                f"FATAL: frozen chain source drift: {relative}: "
                f"{actual} != {expected}"
            )
unused = sorted(set(replacements) - set(files))
if unused:
    raise SystemExit(f"FATAL: VLM replacement lacks a frozen source: {unused}")
print(
    f"frozen source subset OK: {len(files)} files, "
    f"{len(replacements)} VLM replacements bound; post-chain additions permitted"
)
PY
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
from scripts.experiments.causal_selectivity_20260720 import preflight

root = Path.cwd().resolve()
preflight.validate_formal_freeze_gates(preflight.load_spec())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


chain_path = root / "results/reanalysis/aplus_20260718/CHAIN_MANIFEST.json"
chain = json.loads(chain_path.read_text(encoding="utf-8"))
vlm_addendum_path = (
    root
    / "results/vlm_rerun_20260724_authority/VLM_FIGURE_ADDENDUM.json"
)
vlm_addendum = json.loads(vlm_addendum_path.read_text(encoding="utf-8"))
vlm_figure = vlm_addendum["figure"]["path"]
for relative, expected in chain["artifacts"].items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"FATAL: chain artifact missing: {relative}")
    actual = sha256(path)
    if relative == vlm_figure and actual != expected:
        reference = vlm_addendum["frozen_chain_reference"]
        if (
            reference["manifest_sha256"] != sha256(chain_path)
            or reference["prior_figure_sha256"] != expected
            or vlm_addendum["figure"]["sha256"] != actual
        ):
            raise SystemExit("FATAL: VLM figure addendum does not bind replacement")
        continue
    if actual != expected:
        raise SystemExit(
            f"FATAL: chain artifact hash mismatch: {relative}: "
            f"{actual} != {expected}"
        )

for section, path_key, hash_key in (
    ("figure", "path", "sha256"),
    ("generator", "path", "sha256"),
):
    relative = vlm_addendum[section][path_key]
    if sha256(root / relative) != vlm_addendum[section][hash_key]:
        raise SystemExit(f"FATAL: VLM addendum {section} hash mismatch")
vlm_inputs = vlm_addendum["authority_inputs"]
for path_key, hash_key in (
    ("manifest_path", "manifest_sha256"),
    ("summary_path", "summary_sha256"),
):
    if sha256(root / vlm_inputs[path_key]) != vlm_inputs[hash_key]:
        raise SystemExit(f"FATAL: VLM addendum input mismatch: {path_key}")

wager_path = (
    root
    / "results/reanalysis/profile_validity_20260720/"
    "wager_replay/WAGER_REPLAY_MANIFEST.json"
)
wager = json.loads(wager_path.read_text(encoding="utf-8"))
if wager.get("status") != "final" or wager.get("all_gates_passed") is not True:
    raise SystemExit("FATAL: wagering replay manifest is not final")
for relative, expected in wager["code"]["files"].items():
    path = root / relative
    if not path.is_file() or sha256(path) != expected:
        raise SystemExit(f"FATAL: wagering code closure failed: {relative}")
if canonical_hash(sorted(wager["code"]["files"].items())) != wager["code"]["tree_sha256"]:
    raise SystemExit("FATAL: wagering code-tree hash mismatch")
for relative, expected in wager["outputs"].items():
    path = root / relative
    if not path.is_file() or sha256(path) != expected:
        raise SystemExit(f"FATAL: wagering artifact closure failed: {relative}")

sm_path = root / "results/sm_rerun_20260718/SM_MANIFEST.json"
sm = json.loads(sm_path.read_text(encoding="utf-8"))
if sm.get("status") != "final":
    raise SystemExit("FATAL: source-monitoring manifest is not final")
for relative, expected in sm["scripts"].items():
    path = root / relative
    if not path.is_file() or sha256(path) != expected:
        raise SystemExit(f"FATAL: source-monitoring script closure failed: {relative}")
sm_provenance = root / "results/reanalysis/sm_20260718"
for name, expected in sm["manifests"].items():
    path = sm_provenance / name
    if not path.is_file() or sha256(path) != expected:
        raise SystemExit(f"FATAL: source-monitoring manifest closure failed: {name}")
for name, expected in sm["outputs"].items():
    path = sm_path.parent / name
    if not path.is_file() or sha256(path) != expected:
        raise SystemExit(f"FATAL: source-monitoring artifact closure failed: {name}")

target_path = (
    root
    / "results/causal_selectivity_20260720/analysis/"
    "target_baseline_sensitivity/TARGET_BASELINE_MANIFEST.json"
)
target = json.loads(target_path.read_text(encoding="utf-8"))
if target.get("status") != "complete_post_hoc_sensitivity":
    raise SystemExit("FATAL: target-versus-baseline manifest is not complete")
target_code = {
    "analyze_target_baseline.py": (
        "scripts/experiments/causal_selectivity_20260720/"
        "analyze_target_baseline.py"
    ),
    "analyze_target_baseline.sbatch": (
        "scripts/experiments/causal_selectivity_20260720/"
        "analyze_target_baseline.sbatch"
    ),
    "frozen_loader_analyze.py": (
        "scripts/experiments/causal_selectivity_20260720/analyze.py"
    ),
}
for name, relative in target_code.items():
    path = root / relative
    if not path.is_file() or sha256(path) != target["code_sha256"][name]:
        raise SystemExit(f"FATAL: target-versus-baseline code closure failed: {name}")
target_inputs = {
    "ANALYSIS_MANIFEST.json": (
        "results/causal_selectivity_20260720/analysis/ANALYSIS_MANIFEST.json"
    ),
    "EXPLORATORY_MANIFEST.json": (
        "results/causal_selectivity_20260720/analysis/exploratory/"
        "EXPLORATORY_MANIFEST.json"
    ),
    "PREPILOT_SPEC.json": (
        "scripts/experiments/causal_selectivity_20260720/PREPILOT_SPEC.json"
    ),
    "RUN_MANIFEST_formal.json": (
        "results/causal_selectivity_20260720/RUN_MANIFEST_formal.json"
    ),
    "TARGET_BASELINE_SENSITIVITY_SPEC.json": (
        "scripts/experiments/causal_selectivity_20260720/"
        "TARGET_BASELINE_SENSITIVITY_SPEC.json"
    ),
    "analysis_results.json": (
        "results/causal_selectivity_20260720/analysis/analysis_results.json"
    ),
    "exploratory_results.json": (
        "results/causal_selectivity_20260720/analysis/exploratory/"
        "exploratory_results.json"
    ),
    "formal_item_manifest.json": (
        "results/causal_selectivity_20260720/item_manifest_formal.json"
    ),
}
for name, relative in target_inputs.items():
    path = root / relative
    if not path.is_file() or sha256(path) != target["inputs_sha256"][name]:
        raise SystemExit(f"FATAL: target-versus-baseline input closure failed: {name}")
for name, expected in target["outputs_sha256"].items():
    path = target_path.parent / name
    if not path.is_file() or sha256(path) != expected:
        raise SystemExit(f"FATAL: target-versus-baseline output closure failed: {name}")

print(
    f"manifest closure OK: {len(chain['artifacts'])} chain artifacts, "
    f"{len(wager['code']['files'])} wagering code files, "
    f"{len(wager['outputs'])} wagering artifacts, "
    f"{len(sm['scripts'])} source-monitoring scripts, "
    f"{len(sm['manifests'])} source-monitoring manifests, "
    f"{len(sm['outputs'])} source-monitoring artifacts, "
    f"{len(target_code)} target-versus-baseline code files, "
    f"{len(target_inputs)} public target-versus-baseline inputs, "
    f"{len(target['outputs_sha256'])} target-versus-baseline artifacts"
)
PY
.venv/bin/cogarena eval --dry-run --model test
.venv/bin/python - <<'PY'
import cogarena
from pathlib import Path

root = Path.cwd().resolve()
module = Path(cogarena.__file__).resolve()
if not module.is_relative_to(root):
    raise SystemExit(f"FATAL: cogarena imported outside extracted archive: {module}")
print("cold import OK:", module)
PY
echo CODE_ARCHIVE_SMOKE_DONE
