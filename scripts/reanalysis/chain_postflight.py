#!/usr/bin/env python3
"""Post-chain coherence gates and CHAIN_MANIFEST writer (single source).

Called only by run_final_chain.sh as its postflight stage. Manifest-only
rebuilds are intentionally refused: every generated artifact must have an
mtime at or after the driver's frozen COGARENA_CHAIN_STARTED_NS. This prevents
a standalone hash pass from blessing stale outputs.

Fail-closed: explicit raises, not assert (python -O must not weaken gates).
"""
import hashlib
import json
import os


def req(cond, msg):
    if not cond:
        raise SystemExit(f"POSTFLIGHT FAILED: {msg}")


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


EXPECTED = [
    "results/recompute_20260703/corrected_matrix.csv",
    "results/recompute_20260703/final_inference.json",
    "results/recompute_20260703/sensitivity.json",
    "results/recompute_20260703/difficulty_robustness.json",
    "results/recompute_20260703/step4_artifacts.json",
    "results/recompute_20260703/step4b_artifacts.json",
    "results/construct_native_20260711/construct_matrix.csv",
    "results/twolevel_bootstrap_20260712/twolevel_bootstrap.json",
    "results/pc1_validation_20260711/pc1_validation.json",
    "results/pc1_validation_20260711/joint_exclusion.json",
    "results/reanalysis/aplus_20260718/MANIFEST.json",
    "results/reanalysis/aplus_20260718/FINAL_CHAIN_SOURCE_MANIFEST.json",
    "results/reanalysis/aplus_20260718/matrix_aplus_strict.csv",
    "results/reanalysis/aplus_20260718/matrix_aplus_canonical.csv",
    "results/reanalysis/aplus_20260718/matrix_construct_aplus_strict.csv",
    "results/reanalysis/aplus_20260718/matrix_construct_aplus_canonical.csv",
    "results/reanalysis/b2_expanded.json",
    "results/reanalysis/pca_partialcorr.json",
    "results/reanalysis/split_half_reliability.json",
    "results/reanalysis/restricted_range_robustness.json",
    "results/reanalysis/scaling_mixedeffects.json",
    "results/reanalysis/scaling_mixedeffects_table.csv",
    "results/reanalysis/scaling_mixedeffects_table.tex",
    "results/reanalysis/signature_significance.json",
    "results/reanalysis/fp16_deconfound.json",
    "results/reanalysis/scorer_robustness.json",
    "results/reanalysis/profile_validity_20260720/wager_replay/WAGER_REPLAY_MANIFEST.json",
    "results/reanalysis/profile_validity_20260720/wager_replay/wager_accuracy_overlay.json",
    "results/reanalysis/profile_validity_20260720/wager_replay/wager_replay_results.json",
    "results/reanalysis/profile_validity_20260720/wager_replay/wager_replay_items.csv",
    "results/predictive_validity.json",
    "paper/figures/fig_manifold.pdf",
    "paper/figures/fig2_signatures.pdf",
    "paper/figures/fig2_compact.pdf",
    "paper/figures/fig3_scaling.pdf",
    "paper/figures/fig4_cross_system.pdf",
    "paper/figures/fig5_profiles.pdf",
    "paper/figures/fig_scaling_bars.pdf",
]

TRUSTED_PRECHAIN_INPUTS = {
    "results/reanalysis/aplus_20260718/FINAL_CHAIN_SOURCE_MANIFEST.json",
    "results/reanalysis/profile_validity_20260720/wager_replay/WAGER_REPLAY_MANIFEST.json",
    "results/reanalysis/profile_validity_20260720/wager_replay/wager_accuracy_overlay.json",
    "results/reanalysis/profile_validity_20260720/wager_replay/wager_replay_results.json",
    "results/reanalysis/profile_validity_20260720/wager_replay/wager_replay_items.csv",
}


def main():
    root = os.environ["COGARENA_ROOT"]

    # 1. input-state coherence: the construct matrix the A+ suite consumed
    #    must be byte-identical to the final on-disk construct matrix
    aplus = json.load(open(f"{root}/results/reanalysis/aplus_20260718/MANIFEST.json"))
    cm_path = f"{root}/results/construct_native_20260711/construct_matrix.csv"
    got = sha(cm_path)
    want = aplus["inputs"]["construct_matrix_sha256"]
    req(got == want, f"A+ consumed construct {want[:12]} but disk has {got[:12]}: "
                     "the chain mixed two construct input states")
    corr_path = f"{root}/results/recompute_20260703/corrected_matrix.csv"
    req(sha(corr_path) == aplus["inputs"]["corrected_matrix_sha256"],
        "A+ consumed a different corrected matrix than the one on disk")
    req(aplus.get("primary_config") == "aplus_strict",
        "A+ MANIFEST does not record aplus_strict as primary")
    req(aplus.get("adjudication_status") == "not_performed",
        "A+ MANIFEST does not record adjudication_status=not_performed")
    req(aplus.get("sensitivity_configs") == ["aplus_canonical"],
        "A+ MANIFEST does not freeze canonical as the sole scoring sensitivity")
    req(aplus["inputs"].get("source_revision") == os.environ["COGARENA_GIT_HEAD"],
        "A+ MANIFEST does not bind the active source revision")
    req(aplus["inputs"].get("wager_overlay", {}).get("sha256") ==
        sha(os.environ["COGARENA_WAGER_OVERLAY"]),
        "A+ MANIFEST does not bind the active wagering overlay")
    req(aplus["inputs"].get("sm_overlay", {}).get("sha256") ==
        sha(os.environ["COGARENA_SM_OVERLAY"]),
        "A+ MANIFEST does not bind the active SM overlay")

    # Every A+ output is content-addressed by its own manifest. This prevents
    # a standalone postflight from blessing a primary/canonical/construct
    # matrix left over from a different run.
    aplus_dir = os.path.join(root, "results/reanalysis/aplus_20260718")
    aplus_outputs = aplus.get("outputs_sha256", {})
    required_aplus = {
        "matrix_aplus_strict.csv",
        "matrix_aplus_canonical.csv",
        "matrix_construct_aplus_strict.csv",
        "matrix_construct_aplus_canonical.csv",
    }
    req(required_aplus.issubset(aplus_outputs),
        "A+ MANIFEST lacks one or more frozen primary/sensitivity matrices")
    for name, expected in aplus_outputs.items():
        req(os.path.basename(name) == name, f"unsafe A+ output name: {name}")
        path = os.path.join(aplus_dir, name)
        req(os.path.isfile(path), f"A+ output missing: {name}")
        req(sha(path) == expected, f"A+ output hash mismatch: {name}")

    # Re-verify the formal wagering trust root, not merely the active overlay.
    wager_manifest_path = os.path.join(
        root,
        "results/reanalysis/profile_validity_20260720/wager_replay/WAGER_REPLAY_MANIFEST.json",
    )
    wager_manifest = json.load(open(wager_manifest_path))
    req(wager_manifest.get("status") == "final"
        and wager_manifest.get("all_gates_passed") is True,
        "wagering replay manifest is not final")
    req(wager_manifest.get("execution", {}).get("git_head") ==
        os.environ["COGARENA_GIT_HEAD"],
        "wagering replay manifest source revision mismatch")
    req(wager_manifest.get("checks", {}).get("wager_construct_overlay_representable") is True,
        "wagering replay manifest does not establish construct representability")
    for relative, expected in wager_manifest.get("outputs", {}).items():
        path = os.path.realpath(os.path.join(root, relative))
        req(os.path.commonpath([os.path.realpath(root), path]) == os.path.realpath(root),
            f"wager output escapes repository root: {relative}")
        req(os.path.isfile(path), f"wager output missing: {relative}")
        req(sha(path) == expected, f"wager output hash mismatch: {relative}")

    # 2. explicit expected outputs - every one must exist (no glob-and-hope)
    missing = [p for p in EXPECTED if not os.path.exists(os.path.join(root, p))]
    req(not missing, f"expected outputs missing: {missing}")
    started_ns = int(os.environ["COGARENA_CHAIN_STARTED_NS"])
    stale = [
        p for p in EXPECTED
        if p not in TRUSTED_PRECHAIN_INPUTS
        and os.stat(os.path.join(root, p)).st_mtime_ns < started_ns
    ]
    req(not stale, f"expected outputs were not regenerated by this chain: {stale}")

    man = {
        "spec": "post-chain artifact hashes (primary-estimand run)",
        "primary_config": "aplus_strict",
        "sensitivity_configs": ["aplus_canonical"],
        "adjudication_status": "not_performed",
        "source_revision": os.environ["COGARENA_GIT_HEAD"],
        "source_manifest_sha256": sha(os.path.join(
            root, "results/reanalysis/aplus_20260718/FINAL_CHAIN_SOURCE_MANIFEST.json")),
        "chain_started_ns": started_ns,
        "primary_matrix": os.environ["COGARENA_PRIMARY_MATRIX"],
        "primary_matrix_sha256": sha(os.path.join(root, os.environ["COGARENA_PRIMARY_MATRIX"])),
        "primary_construct_matrix": os.environ["COGARENA_PRIMARY_CONSTRUCT_MATRIX"],
        "primary_construct_matrix_sha256": sha(os.path.join(
            root, os.environ["COGARENA_PRIMARY_CONSTRUCT_MATRIX"])),
        "canonical_sensitivity_matrix":
            "results/reanalysis/aplus_20260718/matrix_aplus_canonical.csv",
        "canonical_sensitivity_matrix_sha256": sha(os.path.join(
            root, "results/reanalysis/aplus_20260718/matrix_aplus_canonical.csv")),
        "canonical_construct_sensitivity_matrix":
            "results/reanalysis/aplus_20260718/matrix_construct_aplus_canonical.csv",
        "canonical_construct_sensitivity_matrix_sha256": sha(os.path.join(
            root, "results/reanalysis/aplus_20260718/matrix_construct_aplus_canonical.csv")),
        "sm_overlay_sha256": sha(os.environ["COGARENA_SM_OVERLAY"]),
        "wager_overlay_sha256": sha(os.environ["COGARENA_WAGER_OVERLAY"]),
        "construct_matrix_sha256": got,
        "artifacts": {p: sha(os.path.join(root, p)) for p in EXPECTED},
    }
    out = os.path.join(root, "results", "reanalysis", "aplus_20260718", "CHAIN_MANIFEST.json")
    tmp = out + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, out)
    print(f"postflight OK: construct state coherent, {len(EXPECTED)} expected outputs present")


if __name__ == "__main__":
    main()
