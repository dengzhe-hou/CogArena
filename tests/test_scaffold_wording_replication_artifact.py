import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "scaffold_wording_replication_20260725"
SPEC = (
    ROOT
    / "scripts"
    / "experiments"
    / "scaffold_wording_replication_20260725"
    / "SPEC.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_paper_reported_wording_replication_is_manifest_bound() -> None:
    run_path = RESULT_ROOT / "RUN_MANIFEST_formal.json"
    item_path = RESULT_ROOT / "item_manifest_formal.json"
    result_path = RESULT_ROOT / "analysis" / "analysis_results.json"
    analysis_manifest_path = (
        RESULT_ROOT / "analysis" / "ANALYSIS_MANIFEST.json"
    )

    run = _read(run_path)
    items = _read(item_path)
    result = _read(result_path)
    manifest = _read(analysis_manifest_path)

    assert run["status"] == "formal_raw_complete"
    assert run["record_count"] == 19_656
    assert run["model_count"] == 12
    assert run["fully_gpu_served_model_count"] == 12
    assert run["record_reuse_allowed"] is False
    assert items["item_count"] == 234
    assert items["condition_count"] == 7
    assert items["task_record_count_per_model"] == 1_638

    assert manifest["status"] == "complete"
    assert manifest["confirmatory_gate_pass"] is False
    assert manifest["spec_sha256"] == _sha256(SPEC)
    assert manifest["formal_item_manifest_sha256"] == _sha256(item_path)
    assert manifest["formal_run_manifest_sha256"] == _sha256(run_path)
    assert (
        manifest["outputs_sha256"]["analysis_results.json"]
        == _sha256(result_path)
    )

    primary = result["primary"]
    lofo = result["predictive_family_lofo"]
    assert math.isclose(primary["gamma"], 0.01340753596814708)
    assert primary["bootstrap"]["gamma_ci95"] == [
        -0.002970219393672172,
        0.029803585023515575,
    ]
    assert math.isclose(
        primary["exact_mapping_permutation"]["gamma_p_two"],
        0.03333333333333333,
    )
    assert primary["families_positive"] == 4
    assert math.isclose(lofo["delta_log_likelihood"], 0.7714330452849936)
    assert lofo["families_improved"] == 3
    assert result["confirmatory_gate"]["pass"] is False
