"""Production-package regression tests over the REAL blind IDs.

These read the locally built annotator bundle (untracked; tests skip on a
clean checkout without it) and pin every case named across the
re-verification rounds by blind_id/line. Assertions are predicate-based
(mask marks, letter-token counts) so no battery or gold content appears in
this file.
"""
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
BLIND = os.path.join(ROOT, 'results', 'reanalysis', 'ospan_blind_20260718')
PKG = os.path.join(BLIND, 'annotator_bundle', 'blind_package.jsonl')

needs_package = pytest.mark.skipif(
    not os.path.exists(PKG),
    reason="local annotator bundle not built (raw responses are untracked; "
           "these package-content tests are LOCAL verification only - "
           "the hash-consistency tests below run on every checkout)")

CAP = re.compile(r'\b[A-Z]\b')
MASK = '▮'

# (blind_id, line) -> expectation
MASKED_LINES = [
    ('b0038', 7),    # instructional example embedding the full sequence
    ('b0070', 9), ('b0070', 13),      # bare plural per-trial echoes
    ('b0102', 10),   # 'Remembering the letter: The letter "C"' scaffold
    ('b0104', 2),    # Sample Response teaching example
    # b0150: all eight instruction-voice walk-throughs whose arithmetic
    # 'correct answer is YES/NO' tail must not shield the letter head
    ('b0150', 0), ('b0150', 2), ('b0150', 4), ('b0150', 6),
    ('b0150', 8), ('b0150', 10), ('b0150', 12), ('b0150', 14),
    ('b0218', 1), ('b0218', 2), ('b0218', 3),  # articleless per-trial echoes
    ('b0283', 49), ('b0283', 71),     # imperative full-sequence restatements
    ('b0288', 2), ('b0288', 6),       # plural per-trial echoes on math lines
    ('b0365', 9), ('b0365', 10),      # e.g. bullets quoting transcript
    ('b0384', 0),    # intro restating required letters
    ('b0417', 9), ('b0417', 12),      # plural per-trial echoes
]
KEPT_CLAIM_LINES = [
    ('b0054', 0),    # sole claimed sequence in recall-phase phrasing
    ('b0147', 2),    # claim with per-item provenance
    ('b0161', 88),   # final gerund claim, conflicts with earlier candidates
    ('b0167', 2),    # first-person summary claim
    ('b0268', 2), ('b0268', 4),  # 'correct order/answer is ...' claims
    ('b0278', 1),    # final self-correction supersedes earlier candidate
    ('b0425', 0), ('b0425', 1),  # per-item 'remembered letter is J/B' claims
]


def _load():
    return {json.loads(l)['blind_id']: json.loads(l)['response']
            for l in open(PKG)}


def _letters(line):
    return [c for c in CAP.findall(line) if c not in ('I', 'A')]


@needs_package
def test_named_leak_lines_are_masked():
    pkg = _load()
    for bid, li in MASKED_LINES:
        ln = pkg[bid].splitlines()[li]
        assert MASK in ln, f"{bid} L{li} carries no mask"
        assert not _letters(ln), f"{bid} L{li} still carries letters: {_letters(ln)}"


@needs_package
def test_named_claim_lines_are_kept():
    pkg = _load()
    any_letters = re.compile(r'\b[A-Z]{1,8}\b')
    for bid, li in KEPT_CLAIM_LINES:
        ln = pkg[bid].splitlines()[li]
        assert MASK not in ln, f"{bid} L{li} was masked"
        assert any_letters.search(ln), f"{bid} L{li} lost its letters"


@needs_package
def test_no_surviving_transcript_echo_with_letters():
    pkg = _load()
    # article-optional; the remember-phrase is case-insensitive (inline
    # flag) but the leaked-letter class stays STRICTLY uppercase - a global
    # re.I would make [A-Z] match prose lowercase and flood false positives
    imperative = re.compile(
        r'(?i:remember\w*\s+(?:(?:the|a)\s+)?letters?)\b[^\n]*\b[A-Z]\b')
    claim_voice = re.compile(
        r'\b(?:I|[Ww]e)\b[^.\n]{0,60}\b(?:remember|recall)\w*'
        r'|\bRemembering\s+the\s+letters\b'
        r'|\b(?:correct|final)\s+(?:order|answer|sequence)\b[^.\n]{0,40}\bis\b'
        r'|\b(?:sequence|list)\s+of\s+remembered'
        r'|\bremembered\s+letters?\s+(?:is|are|in\s+order)\b', re.I)
    bad = []
    for bid, t in pkg.items():
        for li, ln in enumerate(t.splitlines()):
            if imperative.search(ln) and not claim_voice.search(ln):
                bad.append((bid, li))
    assert not bad, f"transcript echoes with letters survive: {bad[:5]}"


def test_counts_match_package_manifest():
    # runs on every checkout: committed-artifact internal consistency
    pm = json.load(open(os.path.join(BLIND, 'PACKAGE_MANIFEST.json')))
    aud = json.load(open(os.path.join(BLIND, 'redaction_audit.json')))
    assert pm['n_texts_semantically_changed'] == aud['n_texts_changed']
    assert pm['n_changed_lines'] == aud['n_changed_lines']
    assert pm['n_noop_rule_hits'] == aud['n_noop_rule_hits']
    listed = sum(len(v) for v in aud['redactions'].values())
    assert listed == aud['n_changed_lines']


def test_committed_artifact_hashes_bind():
    # runs on every checkout: the committed mapping, audit, manual
    # decisions, and instructions must hash-match PACKAGE_MANIFEST, and the
    # audit's masked lines must be disjoint from manual 'keep' decisions.
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for c in iter(lambda: f.read(1 << 20), b''):
                h.update(c)
        return h.hexdigest()

    pm = json.load(open(os.path.join(BLIND, 'PACKAGE_MANIFEST.json')))
    for key, rel in (('blind_mapping_sha256', 'blind_mapping.json'),
                     ('redaction_audit_sha256', 'redaction_audit.json'),
                     ('manual_decisions_sha256', 'manual_redaction_decisions.json'),
                     ('instructions_sha256', 'annotator_bundle/INSTRUCTIONS.md')):
        assert sha(os.path.join(BLIND, rel)) == pm[key], f"{rel} drifted"
    aud = json.load(open(os.path.join(BLIND, 'redaction_audit.json')))['redactions']
    man = json.load(open(os.path.join(BLIND, 'manual_redaction_decisions.json')))['decisions']
    for bid, dec in man.items():
        for li, d in dec.items():
            if d['action'] == 'keep':
                masked = {r['line'] for r in aud.get(bid, [])}
                assert int(li) not in masked, f"manual keep {bid} L{li} was masked anyway"


def test_manual_decisions_cover_all_named_cases():
    man = json.load(open(os.path.join(BLIND, 'manual_redaction_decisions.json')))['decisions']
    for bid, li in MASKED_LINES + KEPT_CLAIM_LINES:
        assert bid in man, f"{bid} missing from manual_redaction_decisions"
