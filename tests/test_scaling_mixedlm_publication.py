"""Public-artifact checks for the family-random-intercept scaling table."""
import csv
import json
import math
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = os.path.join(ROOT, "results", "reanalysis",
                         "scaling_mixedeffects.json")
CSV_PATH = os.path.join(ROOT, "results", "reanalysis",
                        "scaling_mixedeffects_table.csv")
TEX_PATH = os.path.join(ROOT, "results", "reanalysis",
                        "scaling_mixedeffects_table.tex")

EXPECTED_ORDER = [
    "n_back", "digit_span", "operation_span",
    "stroop", "flanker", "go_nogo",
    "cvlt_word_list", "drm_false_memory", "source_monitoring",
    "false_belief", "epitome_tom",
    "confidence_calibration", "post_decision_wagering",
]


def test_mixedlm_publication_schema_and_convergence_are_explicit():
    artifact = json.load(open(JSON_PATH))
    assert artifact["schema_version"] == "cogarena-scaling-mixedlm-v2"
    assert artifact["fit_population"] == {
        "n_checkpoints": 20,
        "n_families": 11,
        "n_singleton_families": 7,
        "family_counts": {
            "command-r": 1,
            "deepseek-r1": 2,
            "gemma2": 3,
            "llama3.1": 1,
            "llama3.2": 2,
            "mistral": 1,
            "mixtral": 1,
            "phi3": 1,
            "qwen2.5": 6,
            "tinyllama": 1,
            "yi": 1,
        },
    }
    rows = artifact["publication_table"]
    assert [row["paradigm_key"] for row in rows] == EXPECTED_ORDER
    assert artifact["nonconverged_paradigms"] == [
        "drm_false_memory", "false_belief"]
    assert {row["paradigm_key"] for row in rows if not row["converged"]} == {
        "drm_false_memory", "false_belief"}
    for row in rows:
        assert row["n_checkpoints"] == 20
        assert row["n_families"] == 11
        for key in (
            "fixed_slope_per_log10_parameter", "fixed_slope_se", "wald_p",
            "random_intercept_variance", "residual_variance",
            "random_intercept_icc",
        ):
            assert math.isfinite(row[key]), (row["paradigm_key"], key)
        assert row["fixed_slope_se"] > 0
        assert 0 <= row["wald_p"] <= 1
        assert row["random_intercept_variance"] >= 0
        assert row["residual_variance"] > 0
        assert 0 <= row["random_intercept_icc"] <= 1


def test_mixedlm_csv_is_an_exact_projection_of_json():
    artifact = json.load(open(JSON_PATH))
    csv_rows = list(csv.DictReader(open(CSV_PATH)))
    assert len(csv_rows) == len(artifact["publication_table"]) == 13
    for source, exported in zip(artifact["publication_table"], csv_rows):
        assert exported["paradigm_key"] == source["paradigm_key"]
        assert exported["converged"] == str(source["converged"])
        for key in (
            "fixed_slope_per_log10_parameter", "fixed_slope_se", "wald_p",
            "random_intercept_variance", "residual_variance",
            "random_intercept_icc",
        ):
            assert float(exported[key]) == source[key]


def test_mixedlm_latex_discloses_nonconvergence_without_overclaiming():
    tex = open(TEX_PATH).read()
    assert tex.count(r"No$^\dagger$") == 2
    assert "DRM and false-belief optimizers did not converge" in tex
    assert "diagnostics, not as evidence that scaling ranks are preserved" in tex
    for label in (
        "N-back", "Digit span", "Operation span", "Stroop", "Flanker",
        "Go/No-Go", "CVLT", "DRM", "Source monitoring", "False belief",
        "EPITOME", "Calibration", "Wagering",
    ):
        assert label in tex
