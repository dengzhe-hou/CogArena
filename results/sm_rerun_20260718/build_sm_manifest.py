#!/usr/bin/env python3
"""Pin the SM rerun chain's full provenance into a tracked SM_MANIFEST.json.

Pins: the two published outputs, every script in the chain, the rerun and
exclusions manifests, the rescore_20260702 overlay tree, the raw per-item
trees of both serving arms (ROOT-relative tree hashes - raw responses stay
untracked, so the zero-mismatch rescore gate G4 is LOCAL verification; a
clean checkout re-verifies outputs and scripts only), generator/scorer
provenance, per-arm serving configuration with job IDs, and ollama model
digests where a job captured /api/tags.

Rerun after the default-context arm completes to fold in its digests and
the adopted serving_source_map.
"""
import glob
import hashlib
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("COGARENA_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
ARM4096 = os.path.join(ROOT, "results", "sm_rerun_20260718")
ARMDEF = os.path.join(ROOT, "results", "sm_rerun_default_ctx_20260718")


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def tree_sha(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, ROOT).encode())
        h.update(bytes.fromhex(sha(f)))
    return {'n_files': len(files), 'sha256': h.hexdigest()}


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def digests_from_capture(arm_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(arm_dir, 'api_tags_*.json'))):
        try:
            for mo in json.load(open(f)).get('models', []):
                out[mo.get('name')] = mo.get('digest')
        except Exception:
            pass
    return out or None


def verify_overlay_content(overlay_re, summary_doc_re, models, n_rescore,
                           disk_ov, disk_sum):
    """Strict trust-root comparison (pure function; unit-tested with
    tampered inputs).  Returns (ok, detail).

    Guards closed here: bool-for-float forgery (False == 0.0 passes dict
    equality, so every value is type-checked BEFORE comparison), NaN
    (serialization with allow_nan=False raises), duplicated models in the
    list (unique-55 gate plus the 2,750 rescore-count gate), and drift in
    ANY summary field (whole-document canonical-JSON comparison, not just
    the grand means)."""
    if not (len(models) == len(set(models)) == 55):
        return False, f"model list not 55 unique models ({len(models)}/{len(set(models))})"
    if n_rescore != 2750:
        return False, f"rescore checks {n_rescore} != 2750"
    for src_name, src in (('recomputed', overlay_re), ('on-disk', disk_ov)):
        if set(src) != set(models):
            return False, f"{src_name} overlay model set mismatch"
        for m, cells in src.items():
            if len(cells) != 50:
                return False, f"{src_name} overlay {m} has {len(cells)} cells"
            for t, v in cells.items():
                if type(v) is not float:
                    return False, f"{src_name} overlay {m}/{t} is {type(v).__name__}, not float"
                if not (v == v and 0.0 <= v <= 1.0):
                    return False, f"{src_name} overlay {m}/{t} out of range: {v!r}"
    task_sets = {frozenset(c) for c in overlay_re.values()}
    if len(task_sets) != 1 or {frozenset(c) for c in disk_ov.values()} != task_sets:
        return False, "overlay task sets differ across models or arms"
    try:
        can = lambda x: json.dumps(x, sort_keys=True, allow_nan=False)
        if can(disk_ov) != can(overlay_re):
            return False, "on-disk overlay differs from gated recomputation"
        if can(disk_sum) != can(summary_doc_re):
            return False, "on-disk summary differs from gated recomputation (any field)"
    except ValueError as e:
        return False, f"NaN/Inf in overlay or summary: {e}"
    return True, "overlay and summary match the gated recomputation exactly (strict types)"


def _overlay_content_verified():
    """Recompute through the summarizer's own gated build_overlay()
    (single source) and apply verify_overlay_content."""
    import sys
    sys.path.insert(0, ARM4096)
    try:
        from summarize_sm_rerun import build_overlay, summary_doc
        overlay_re, summary_re, failures, n_rescore, rerun_ids, models = build_overlay()
        if failures:
            return False, f"{len(failures)} gate failures on recompute"
        disk_ov = json.load(open(os.path.join(ARM4096, 'sm_scores_overlay.json')))
        disk_sum = json.load(open(os.path.join(ARM4096, 'sm_rerun_summary.json')))
        doc = summary_doc(overlay_re, summary_re, rerun_ids, models)
        return verify_overlay_content(overlay_re, doc, models, n_rescore,
                                      disk_ov, disk_sum)
    except Exception as e:
        return False, f"recompute failed: {e}"


def _adoption_block():
    """Hash and schema-validate the serving adoption artifacts; None until
    the source map exists.  'final' status requires schema_valid=True."""
    sm_path = os.path.join(ARM4096, 'serving_source_map.json')
    if not os.path.exists(sm_path):
        return None
    smap = json.load(open(sm_path))
    models = {l.strip() for l in open(os.path.join(ARM4096, 'modellist.txt')) if l.strip()}
    dctx = {l.strip() for l in open(os.path.join(ARMDEF, 'modellist_default_ctx.txt')) if l.strip()}
    arm4096_rel = os.path.relpath(ARM4096, ROOT)
    armdef_rel = os.path.relpath(ARMDEF, ROOT)
    cmp_path = os.path.join(ARMDEF, 'serving_arm_comparison.json')
    cmp_valid = False
    if os.path.exists(cmp_path):
        try:
            cm = json.load(open(cmp_path))

            def _row_ok(r):
                return (isinstance(r.get('response_identical'), bool)
                        and all(isinstance(r.get(k), float) and 0.0 <= r[k] <= 1.0
                                for k in ('acc_default', 'acc_4096')))
            cmp_valid = (cm.get('n_models') == len(dctx)
                         and set(cm.get('models', {})) == dctx
                         and all(len(v.get('rows', [])) == 11
                                 and all(_row_ok(r) for r in v['rows'])
                                 for v in cm['models'].values()))
        except Exception:
            cmp_valid = False
    schema_valid = (
        set(smap) == models
        and set(smap.values()) <= {arm4096_rel, armdef_rel}
        and {m for m, d in smap.items() if d == armdef_rel} == dctx
        and cmp_valid
    )
    return {
        'schema_valid': schema_valid,
        'serving_source_map_sha256': sha(sm_path),
        'serving_arm_comparison_sha256': sha(cmp_path) if os.path.exists(cmp_path) else None,
        'modellist_sha256': sha(os.path.join(ARM4096, 'modellist.txt')),
        'modellist_resume_sha256': sha(os.path.join(ARM4096, 'modellist_resume.txt')),
        'modellist_default_ctx_sha256': sha(os.path.join(ARMDEF, 'modellist_default_ctx.txt')),
        'n_default_arm_models': len(dctx),
        # directory-source count, NOT a context count: the primary root
        # holds the 32 default-context models from job 6803 plus the two
        # explicit-4096 models
        'n_primary_root_models': len(models) - len(dctx),
        'actual_context_counts': {'default': len(models) - 2, 'explicit_4096': 2},
        'explicit_4096_models': ['llama3.1:70b', 'mixtral:8x22b'],
        'context_capture': {
            'slurm_6807_log_sha256': (sha(os.path.join(ARMDEF, 'slurm_6807.log'))
                                      if os.path.exists(os.path.join(ARMDEF, 'slurm_6807.log'))
                                      else None),
            'note': 'the 6807 log records per-model `ollama ps` (actual n_ctx / VRAM '
                    'split) and env; api_tags_*.json files carry model digests',
        },
    }


def main():
    man = {
        'spec': 'SM rerun chain provenance (dedup-fixed 11 episodes x 55 models, two serving arms)',
        'outputs': {
            'sm_scores_overlay.json': sha(os.path.join(ARM4096, 'sm_scores_overlay.json')),
            'sm_rerun_summary.json': sha(os.path.join(ARM4096, 'sm_rerun_summary.json')),
        },
        'scripts': {os.path.relpath(p, ROOT): sha(p) for p in [
            os.path.join(ARM4096, 'run_sm_rerun.py'),
            os.path.join(ARM4096, 'summarize_sm_rerun.py'),
            os.path.join(ARM4096, 'job.sbatch'),
            os.path.join(ARM4096, 'build_sm_manifest.py'),
        ] + [p for p in [os.path.join(ARMDEF, 'job.sbatch'),
                         os.path.join(ARMDEF, 'compare_serving_arms.py')]
             if os.path.exists(p)]},
        'manifests': {
            'rerun_manifest.json': sha(os.path.join(
                ROOT, 'results', 'reanalysis', 'sm_20260718', 'rerun_manifest.json')),
            'source_monitoring_exclusions.json': sha(os.path.join(
                ROOT, 'results', 'reanalysis', 'sm_20260718', 'source_monitoring_exclusions.json')),
        },
        'inputs_local_only': {
            'note': 'raw per-item responses are untracked; tree hashes below let local '
                    'holders verify byte identity. The zero-mismatch rescore gate (G4) '
                    'is therefore local verification; a clean checkout re-verifies '
                    'outputs, scripts, and manifests only.',
            'rescore_20260702_overlay_tree': tree_sha(os.path.join(
                ROOT, 'results', 'rescore_20260702', 'new_scores', '*.json')),
            'frozen_sm_item_tree': tree_sha(os.path.join(
                ROOT, 'results', 'full_eval_*', 'openai_*', 'text',
                'episodic_memory', 'source_monitoring', '*.json')),
            'arm_4096_item_tree': tree_sha(os.path.join(
                ARM4096, 'openai_*', 'text', 'episodic_memory', 'source_monitoring', '*.json')),
            'arm_default_item_tree': (tree_sha(os.path.join(
                ARMDEF, 'openai_*', 'text', 'episodic_memory', 'source_monitoring', '*.json'))
                if os.path.isdir(ARMDEF) else None),
        },
        'generator_scorer_provenance': {
            'bug_origin_commit': 'ed15dea',
            'fixed_generator_commit': '9cfad4f',
            'episodic_memory_module_blob_now': git('rev-parse', 'HEAD:cogarena/dimensions/episodic_memory.py'),
            'repo_head_at_manifest': git('rev-parse', '--short', 'HEAD'),
        },
        'serving': {
            'job_6803_partial_first_32_models': {
                'context': 'ollama default (no override)', 'flash_attention_env': '1',
                'note': 'cancelled at llama3.1:70b after n_ctx=131072 KV spill; its 352 files kept'},
            'job_6805_resume_23_models': {
                'context': 'OLLAMA_CONTEXT_LENGTH=4096', 'flash_attention_env': 'unset (auto)',
                'matches_original_contract_for': ['llama3.1:70b', 'mixtral:8x22b']},
            'job_6807_default_arm_21_models': {
                'context': 'ollama default (no override)', 'flash_attention_env': 'unset (auto)',
                'purpose': 'serving alignment for the 21 ordinary resumed models; '
                           '4096 results kept as sensitivity arm'},
            'ollama_model_digests': {
                'arm_default': digests_from_capture(ARMDEF),
                'arm_4096': digests_from_capture(ARM4096) or
                            'not captured in-job (jobs 6803/6805 predate the capture step); '
                            'models served from the same shared store as the default arm',
            },
        },
        'adoption': _adoption_block(),
    }
    # Two independent conditions, reported separately so a clean archive
    # (raw trees absent) can never claim final:
    #   adoption_status      the serving source map is schema-valid and the
    #                        arm comparison artifact exists;
    #   local_provenance_verified  every raw tree hashed above is present
    #                        with its expected file count and the context
    #                        capture exists.
    trees = man['inputs_local_only']
    expected = {'rescore_20260702_overlay_tree': lambda n: n > 0,
                'frozen_sm_item_tree': lambda n: n >= 2750,
                'arm_4096_item_tree': lambda n: n == 605,
                'arm_default_item_tree': lambda n: n == 231}
    local_ok = all(trees.get(k) and chk(trees[k]['n_files']) for k, chk in expected.items())

    # per-model checks: fp16 files must not be able to mask a missing canon
    # model - every one of the 55 canon models needs its static overlay file
    # and exactly 50 frozen SM items
    canon = [l.strip() for l in open(os.path.join(ARM4096, 'modellist.txt')) if l.strip()]
    per_model_ok = True
    for m in canon:
        ovs = glob.glob(os.path.join(ROOT, 'results', 'rescore_20260702', 'new_scores',
                                     f'full_eval_*__openai_{m}.json'))
        ovs = [p for p in ovs if '__multiturn' not in p]
        frozen = glob.glob(os.path.join(ROOT, 'results', 'full_eval_*', f'openai_{m}',
                                        'text', 'episodic_memory', 'source_monitoring', '*.json'))
        if len(ovs) != 1 or len(frozen) != 50:
            per_model_ok = False
            break
        # content check, not just shape: every frozen file must parse with a
        # coherent identity and a finite in-range accuracy
        for fp in frozen:
            try:
                d = json.load(open(fp))
            except Exception:
                per_model_ok = False
                break
            acc = (d.get('score') or {}).get('accuracy')
            if (d.get('task_id') != os.path.basename(fp)[:-5]
                    or d.get('paradigm') != 'source_monitoring'
                    or not isinstance(acc, (int, float)) or isinstance(acc, bool)
                    or not (0.0 <= float(acc) <= 1.0)):
                per_model_ok = False
                break
        if not per_model_ok:
            break
    local_ok = local_ok and per_model_ok

    # context capture must contain the 21 per-model serving sections and the
    # completion marker, not merely exist
    ctx_ok = False
    log_path = os.path.join(ARMDEF, 'slurm_6807.log')
    if man['adoption'] and os.path.exists(log_path):
        log = open(log_path, errors='replace').read()
        dctx_models = [l.strip() for l in open(os.path.join(
            ARMDEF, 'modellist_default_ctx.txt')) if l.strip()]
        ctx_ok = (all(f">>> {m}" in log for m in dctx_models)
                  and log.count('serving state for') >= len(dctx_models)
                  and 'ollama ready on' in log
                  and 'ALL GATES PASSED' in log and '\nDONE ' in log)
    man['adoption_status'] = ('adopted' if man['adoption'] and man['adoption'].get('schema_valid')
                              else 'pending')
    man['local_provenance_verified'] = bool(local_ok and ctx_ok)
    ov_ok, ov_msg = (_overlay_content_verified()
                     if man['adoption_status'] == 'adopted' and local_ok
                     else (False, 'skipped: adoption/local provenance not satisfied'))
    man['overlay_content_verified'] = {'ok': bool(ov_ok), 'detail': ov_msg}
    man['status'] = ('final' if man['adoption_status'] == 'adopted'
                     and man['local_provenance_verified'] and ov_ok
                     else 'PROVISIONAL: adoption_status=%s, local_provenance_verified=%s, '
                          'overlay_content_verified=%s'
                     % (man['adoption_status'], local_ok and ctx_ok, bool(ov_ok)))
    out = os.path.join(ARM4096, 'SM_MANIFEST.json')
    tmp = out + '.tmp'
    json.dump(man, open(tmp, 'w'), indent=1)
    os.replace(tmp, out)
    print('status:', man['status'])
    print('arm trees:', man['inputs_local_only']['arm_4096_item_tree'],
          man['inputs_local_only']['arm_default_item_tree'])
    print('written:', out)


if __name__ == "__main__":
    main()
