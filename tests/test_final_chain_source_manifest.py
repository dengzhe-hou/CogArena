from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/reanalysis/final_chain_source_manifest.py"
SPEC = importlib.util.spec_from_file_location("final_chain_source_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


def test_source_closure_is_exact_relative_and_contains_runtime_entrypoints() -> None:
    payload = SOURCE.build()
    files = payload["source_files"]
    assert payload["n_source_files"] == len(files) == len(set(files))
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in files)
    for required in (
        "scripts/reanalysis/final_chain_source_manifest.py",
        "scripts/reanalysis/run_final_chain.sh",
        "scripts/reanalysis/run_final_chain.sbatch",
        "scripts/reanalysis/aplus_rescore_20260718.py",
        "results/recompute_20260703/build_and_recompute.py",
        "results/construct_native_20260711/build_construct_matrix.py",
        "paper/figures/generate_all.py",
    ):
        assert required in files


def test_runtime_driver_verifies_source_manifest_without_git() -> None:
    driver = (ROOT / "scripts/reanalysis/run_final_chain.sh").read_text(encoding="utf-8")
    assert "final_chain_source_manifest.py --verify" in driver
    executable = "\n".join(
        line for line in driver.splitlines() if not line.lstrip().startswith("#")
    )
    assert "git rev-parse" not in executable
    assert "command -v git" not in executable
