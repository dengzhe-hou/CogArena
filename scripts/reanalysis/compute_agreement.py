#!/usr/bin/env python3
"""Dual-annotator agreement, disagreement manifest, and consensus provenance.

Inputs (in results/reanalysis/ospan_blind_20260718/):
  annotator_A.json, annotator_B.json
    independent full-coverage annotations, adjudications.json schema
Optional consensus (in results/reanalysis/aplus_20260718/):
  adjudications.json  joint resolution of the disagreements

Outputs:
  ospan_blind_20260718/agreement_report.json      label + token agreement
  ospan_blind_20260718/disagreement_manifest.json entries needing joint resolution
  aplus_20260718/ADJUDICATION_MANIFEST.json       (only when the consensus file
    exists and covers every disagreement) A/B/consensus hashes + agreement -
    the provenance record the suite's COGARENA_REQUIRE_ADJ=1 mode demands

Consensus rule: where A and B agree exactly (label, and tokens when
'sequence'), the consensus MUST equal that agreed value; disagreements take
the jointly resolved value from adjudications.json.
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.environ.get("COGARENA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BLIND = f"{ROOT}/results/reanalysis/ospan_blind_20260718"
APLUS = f"{ROOT}/results/reanalysis/aplus_20260718"
LABELS = ('sequence', 'ambiguous', 'refusal', 'echo', 'non-answer')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def _req(cond, msg):
    """Schema gates raise unconditionally (assert would be stripped by
    python -O; this module is part of the formal provenance path)."""
    if not cond:
        raise RuntimeError(f"annotation schema: {msg}")


def load_annotation(path, bids):
    d = json.load(open(path))
    extra = set(d) - bids
    missing = bids - set(d)
    _req(not extra, f"{os.path.basename(path)}: unknown blind_ids {sorted(extra)[:5]}")
    _req(not missing, f"{os.path.basename(path)}: missing blind_ids {sorted(missing)[:5]}")
    for bid, a in d.items():
        _req(a.get('label') in LABELS, f"{path}: bad label for {bid}")
        toks = a.get('tokens')
        if a['label'] == 'sequence':
            _req(isinstance(toks, list) and 1 <= len(toks) <= 40
                 and all(isinstance(t, str) and re.fullmatch(r'[A-Z]', t) for t in toks),
                 f"{path}: bad tokens for {bid}")
        else:
            _req(not toks, f"{path}: tokens on non-sequence for {bid}")
    return d


def verify_frozen_protocol():
    """The annotators must have worked on the exact frozen package: verify
    every bundle artifact against PACKAGE_MANIFEST before touching A/B."""
    pm_path = f"{BLIND}/PACKAGE_MANIFEST.json"
    pm = json.load(open(pm_path))
    checks = {
        'blind_package_sha256': f"{BLIND}/annotator_bundle/blind_package.jsonl",
        'instructions_sha256': f"{BLIND}/annotator_bundle/INSTRUCTIONS.md",
        'blind_mapping_sha256': f"{BLIND}/blind_mapping.json",
        'redaction_audit_sha256': f"{BLIND}/redaction_audit.json",
    }
    for key, path in checks.items():
        _req(os.path.exists(path), f"protocol artifact missing: {path}")
        _req(sha(path) == pm[key], f"{os.path.basename(path)} drifted from PACKAGE_MANIFEST")
    return pm_path


def main():
    pm_path = verify_frozen_protocol()
    bmap = json.load(open(f"{BLIND}/blind_mapping.json"))
    bids = set(bmap)
    pa, pb = f"{BLIND}/annotator_A.json", f"{BLIND}/annotator_B.json"
    for p in (pa, pb):
        _req(os.path.exists(p), f"missing {p}")
    A = load_annotation(pa, bids)
    B = load_annotation(pb, bids)

    lab_agree = tok_agree = seq_both = 0
    disagreements = {}
    for bid in sorted(bids):
        a, b = A[bid], B[bid]
        same_label = a['label'] == b['label']
        if same_label:
            lab_agree += 1
        if a['label'] == b['label'] == 'sequence':
            seq_both += 1
            if a['tokens'] == b['tokens']:
                tok_agree += 1
            else:
                disagreements[bid] = {'A': a, 'B': b, 'kind': 'tokens'}
        elif not same_label:
            disagreements[bid] = {'A': a, 'B': b, 'kind': 'label'}

    report = {
        'n_blind_ids': len(bids),
        'label_agreement': lab_agree / len(bids),
        'n_both_sequence': seq_both,
        'sequence_token_exact_agreement': (tok_agree / seq_both) if seq_both else None,
        'n_disagreements': len(disagreements),
        'annotator_A_sha256': sha(pa),
        'annotator_B_sha256': sha(pb),
    }
    json.dump(report, open(f"{BLIND}/agreement_report.json", 'w'), indent=1)
    json.dump({'spec': 'entries requiring joint resolution', 'entries': disagreements},
              open(f"{BLIND}/disagreement_manifest.json", 'w'), indent=1)
    print(json.dumps(report, indent=1))
    print(f"disagreements -> {BLIND}/disagreement_manifest.json")

    cons_path = f"{APLUS}/adjudications.json"
    if not os.path.exists(cons_path):
        print("no consensus adjudications.json yet; resolve disagreements jointly, "
              "write it, then rerun this script to mint ADJUDICATION_MANIFEST.json")
        return
    C = load_annotation(cons_path, bids)
    violations = []
    for bid in sorted(bids):
        a, b, c = A[bid], B[bid], C[bid]
        agreed = a['label'] == b['label'] and (
            a['label'] != 'sequence' or a['tokens'] == b['tokens'])
        if agreed and (c['label'] != a['label'] or
                       (a['label'] == 'sequence' and c.get('tokens') != a['tokens'])):
            violations.append(bid)
    _req(not violations,
         f"consensus overrides agreed annotations for {violations[:5]} - not allowed")
    manifest = {
        'spec': 'dual-annotation provenance for the frozen OSpan adjudication',
        'annotator_A_sha256': report['annotator_A_sha256'],
        'annotator_B_sha256': report['annotator_B_sha256'],
        'agreement': {k: report[k] for k in
                      ('label_agreement', 'sequence_token_exact_agreement',
                       'n_both_sequence', 'n_disagreements')},
        'disagreement_manifest_sha256': sha(f"{BLIND}/disagreement_manifest.json"),
        'consensus_sha256': sha(cons_path),
        'blind_mapping_sha256': sha(f"{BLIND}/blind_mapping.json"),
        # what the annotators actually saw, all pinned through the package
        # manifest verified by verify_frozen_protocol() at load time
        'package_manifest_sha256': sha(pm_path),
        'blind_package_sha256': sha(f"{BLIND}/annotator_bundle/blind_package.jsonl"),
        'instructions_sha256': sha(f"{BLIND}/annotator_bundle/INSTRUCTIONS.md"),
        'redaction_audit_sha256': sha(f"{BLIND}/redaction_audit.json"),
        'agreement_report_sha256': sha(f"{BLIND}/agreement_report.json"),
    }
    json.dump(manifest, open(f"{APLUS}/ADJUDICATION_MANIFEST.json", 'w'), indent=1)
    print(f"ADJUDICATION_MANIFEST.json written; formal freeze may now run with "
          f"COGARENA_REQUIRE_ADJ=1")


if __name__ == "__main__":
    main()
