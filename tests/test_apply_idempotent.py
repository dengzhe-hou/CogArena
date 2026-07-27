"""Persistent regression test: apply_corrected_results must be idempotent.

Two consecutive runs under the primary environment must leave every
regenerated JSON byte-identical (append-style description/note drift and
inherited-field mutation are the failure modes this guards against).
Local-only: needs the full results tree; skipped on checkouts without it.
"""
import glob
import hashlib
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, 'results', 'recompute_20260703',
                                    'corrected_matrix.csv'))
    or not os.path.exists(os.path.join(ROOT, 'results', 'full_eval_20260526_2208')),
    reason="full results tree not present (LOCAL verification only)")


def _sha_all():
    out = {}
    scaling_tables = [
        os.path.join(ROOT, 'results', 'reanalysis', 'scaling_mixedeffects_table.csv'),
        os.path.join(ROOT, 'results', 'reanalysis', 'scaling_mixedeffects_table.tex'),
    ]
    for p in sorted(glob.glob(os.path.join(ROOT, 'results', 'reanalysis', '*.json'))
                    + [p for p in scaling_tables if os.path.exists(p)]
                    + [os.path.join(ROOT, 'results', 'predictive_validity.json')]):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for c in iter(lambda: f.read(1 << 20), b''):
                h.update(c)
        out[p] = h.hexdigest()
    return out


def _snapshot_all():
    """Capture every file this integration test may regenerate."""
    return {path: open(path, 'rb').read() for path in _sha_all()}


def _restore_all(snapshot):
    """Leave the shared checkout byte-identical even when an assertion fails."""
    current = set(_sha_all())
    for path in current - set(snapshot):
        os.unlink(path)
    for path, data in snapshot.items():
        with open(path, 'wb') as f:
            f.write(data)


@needs_data
@pytest.mark.slow
def test_apply_twice_is_byte_identical():
    env = dict(os.environ,
               COGARENA_ROOT=ROOT,
               COGARENA_SM_OVERLAY=os.path.join(
                   ROOT, 'results', 'sm_rerun_20260718', 'sm_scores_overlay.json'),
               COGARENA_PRIMARY_MATRIX=os.path.join(
                   ROOT, 'results', 'reanalysis', 'aplus_20260718', 'matrix_aplus_strict.csv'),
               COGARENA_PRIMARY_CONSTRUCT_MATRIX=os.path.join(
                   ROOT, 'results', 'reanalysis', 'aplus_20260718',
                   'matrix_construct_aplus_strict.csv'),
               COGARENA_PRIMARY_CONFIG='aplus_strict')
    script = os.path.join(ROOT, 'scripts', 'reanalysis', 'apply_corrected_results.py')
    before = _snapshot_all()
    try:
        r1 = subprocess.run([sys.executable, script], env=env, capture_output=True,
                            text=True, timeout=900)
        assert r1.returncode == 0, r1.stdout[-800:] + r1.stderr[-800:]
        h1 = _sha_all()
        r2 = subprocess.run([sys.executable, script], env=env, capture_output=True,
                            text=True, timeout=900)
        assert r2.returncode == 0, r2.stdout[-800:] + r2.stderr[-800:]
        h2 = _sha_all()
        assert h1 == h2, ("non-idempotent outputs: "
                          f"{sorted(os.path.basename(p) for p in set(h1) ^ set(h2) | {p for p in h1 if h1.get(p) != h2.get(p)})}")
    finally:
        _restore_all(before)
