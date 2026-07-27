"""Negative tests for the SM overlay trust-root verifier.

verify_overlay_content must reject the four forgery classes the eleventh
re-verification round named: a JSON false standing in for 0.0, NaN values,
a duplicated model in the list, and drift in any summary field beyond the
grand means.
"""
import copy
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'results', 'sm_rerun_20260718'))

from build_sm_manifest import verify_overlay_content  # noqa: E402

MODELS = [f"m{i:02d}" for i in range(55)]
TASKS = [f"t{i:02d}" for i in range(50)]


def _fixture():
    overlay = {m: {t: 0.5 for t in TASKS} for m in MODELS}
    summary = {"spec": "s", "serving_source_map": None, "task_ids_rerun": ["t00"],
               "n_models": 55, "grand_sm_mean_old_corrected": 0.5,
               "grand_sm_mean_new": 0.5,
               "models": {m: {"new_paradigm_mean": 0.5} for m in MODELS}}
    return overlay, summary


def _run(overlay_re, summary_re, models=None, n=2750,
         disk_ov=None, disk_sum=None):
    return verify_overlay_content(
        overlay_re, summary_re, models if models is not None else list(MODELS),
        n, disk_ov if disk_ov is not None else copy.deepcopy(overlay_re),
        disk_sum if disk_sum is not None else copy.deepcopy(summary_re))


def test_clean_fixture_passes():
    ov, su = _fixture()
    ok, msg = _run(ov, su)
    assert ok, msg


def test_false_for_zero_rejected():
    # False == 0.0 under dict equality; the type gate must catch it
    ov, su = _fixture()
    disk = copy.deepcopy(ov)
    disk[MODELS[0]][TASKS[0]] = False
    ov[MODELS[0]][TASKS[0]] = 0.0
    ok, msg = _run(ov, su, disk_ov=disk)
    assert not ok and 'not float' in msg


def test_nan_rejected():
    ov, su = _fixture()
    ov[MODELS[0]][TASKS[0]] = float('nan')
    ok, msg = _run(ov, su)
    assert not ok


def test_duplicated_model_rejected():
    ov, su = _fixture()
    models = list(MODELS)
    models[1] = models[0]  # 55 entries, 54 unique
    ok, msg = _run(ov, su, models=models)
    assert not ok and 'unique' in msg


def test_rescore_count_gate():
    ov, su = _fixture()
    ok, msg = _run(ov, su, n=2800)
    assert not ok and '2750' in msg


def test_summary_top_level_drift_rejected():
    ov, su = _fixture()
    disk_sum = copy.deepcopy(su)
    disk_sum['task_ids_rerun'] = ["t01"]  # grand means untouched
    ok, msg = _run(ov, su, disk_sum=disk_sum)
    assert not ok and 'summary' in msg


def test_out_of_range_rejected():
    ov, su = _fixture()
    ov[MODELS[0]][TASKS[0]] = 1.5
    ok, msg = _run(ov, su)
    assert not ok and 'range' in msg
