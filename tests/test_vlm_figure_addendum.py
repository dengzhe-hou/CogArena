import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADDENDUM = (
    ROOT
    / "results"
    / "vlm_rerun_20260724_authority"
    / "VLM_FIGURE_ADDENDUM.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vlm_figure_addendum_binds_authority_and_replacement() -> None:
    record = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    assert record["schema_version"] == "cogarena.vlm_figure_addendum.v1"
    assert record["status"] == "final_post_chain_figure_replacement"
    for section, path_key, hash_key in (
        ("figure", "path", "sha256"),
        ("generator", "path", "sha256"),
    ):
        relative = record[section][path_key]
        assert sha256(ROOT / relative) == record[section][hash_key]
    inputs = record["authority_inputs"]
    assert sha256(ROOT / inputs["manifest_path"]) == inputs["manifest_sha256"]
    assert sha256(ROOT / inputs["summary_path"]) == inputs["summary_sha256"]
    chain = record["frozen_chain_reference"]
    assert sha256(ROOT / chain["manifest_path"]) == chain["manifest_sha256"]
    frozen = json.loads((ROOT / chain["manifest_path"]).read_text(encoding="utf-8"))
    assert (
        frozen["artifacts"][record["figure"]["path"]]
        == chain["prior_figure_sha256"]
    )
    source_manifest = json.loads(
        (
            ROOT
            / "results"
            / "reanalysis"
            / "aplus_20260718"
            / "FINAL_CHAIN_SOURCE_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    replacements = record["source_replacements"]
    assert len({row["path"] for row in replacements}) == len(replacements)
    for row in replacements:
        assert source_manifest["source_files"][row["path"]] == row["frozen_sha256"]
        assert sha256(ROOT / row["path"]) == row["replacement_sha256"]
