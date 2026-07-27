#!/usr/bin/env python3
"""Build the OSpan blind-adjudication package.

Splits the 2,750 recall responses with the frozen classify_response() from
aplus_rescore_20260718 (single source of truth).  Machine classes are
zero-prose answer formats scored automatically; every other response goes
to blind human adjudication with model, family, and current score hidden.

Gold leakage: responses that echo the letter-presentation prompt verbatim
("Remember the letter: X") reveal gold letters, so the packaged text masks
the letter in exactly that pattern (deterministic MASK_RE below) with U+25AE.
Masked echo lines still read as echoes; a model whose whole "answer" is
the echo series is labeled `echo` either way, so masking does not change
any adjudication outcome.  The mapping records the sha16 of the RAW
(unmasked) response; the suite re-hashes raw episode text against it.

The annotator receives ONLY results/reanalysis/ospan_blind_20260718/
annotator_bundle/ (masked package + instructions).  blind_mapping.json
lives outside the bundle and unblinds the package: never ship it with the
bundle, and annotators must never open it.

Outputs under results/reanalysis/ospan_blind_20260718/:
  annotator_bundle/blind_package.jsonl  {"blind_id","response"} one unique
                        MASKED text per line, deterministic shuffle (seed 42)
  annotator_bundle/INSTRUCTIONS.md      annotation protocol
  blind_mapping.json    blind_id -> {text_sha16 (raw), episodes:[{model,task_id}]}
  machine_scores.json   per-episode machine class and tokens
  PACKAGE_MANIFEST.json sha256 of package/mapping/instructions + masking spec
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aplus_rescore_20260718 import classify_response, ROOT  # noqa: E402

OUT = f"{ROOT}/results/reanalysis/ospan_blind_20260718"
APLUS = f"{ROOT}/results/reanalysis/aplus_20260718"

# Deterministic gold-echo mask: the letter directly after the verbatim
# letter-presentation prompt pattern is replaced by U+25AE.
MASK_RE = re.compile(r'(remember\w*[^\S\n]+the[^\S\n]+letter[^\S\n]*:[^\S\n]*)([A-Za-z])\b',
                     re.IGNORECASE)
MASK_CHAR = '▮'


def mask_gold_echoes(text):
    return MASK_RE.sub(lambda m: m.group(1) + MASK_CHAR, text)


# --------------------------------------------------------------------------- #
# Semantic redaction audit (2026-07-18, third re-verification round).
#
# Principle: mask letters only in lines that speak in the TRANSCRIPT'S or an
# INSTRUCTOR'S voice (restating what was shown - a trustworthy gold source);
# keep every line where letters are the model's own recall claim (possibly
# wrong, and exactly what the annotator must judge).  KEEP rules override
# MASK rules; each decision is recorded per line in redaction_audit.json.
# The rule set encodes a manual review of every candidate line surfaced by
# broad keyword/sequence detectors over the full 483-text package.

KEEP_RULES = [
    ('R1_first_person', re.compile(
        r'\b[Ii]\s+(?:(?:correctly|successfully|now|then|also|just|only|first|'
        r'have|will|can|am)\s+){0,2}(?:remember\w*|recall\w*|respond\w*|note\w*)\b'
        r'|\b[Ii]\s+have\b[^.\n]{0,60}\b(?:recall|remember|respond)\w*')),
    ('R2_we_claim', re.compile(
        r'\b[Ww]e\s+have\b.*\brecall'
        r'|\b[Ww]e\b[^.\n]{0,60}\b(?:recalled|remembered|verified)\b')),
    ('R3_claim_head', re.compile(
        r'\b(?:remembered|recalled)\s+letters?\b[^:\n]*\b(?:is|are)\b'
        r'|\bletters?\s+(?:is|are|was|were)\s+recalled\b', re.I)),
    ('R4_recall_answer', re.compile(
        r'\brecall(?:ed)?\s+all\s+(?:the\s+)?letters\s+in\s+(?:the\s+)?order\s*:\s*"?[A-Z]\b')),
    ('R6_enum_bare', re.compile(
        r'^\s*(?:[-*]|\d+[.)])\s*(?:Item|Trial)?\s*\d*\s*:?\s*"?[A-Z]"?\s*'
        r'(?:from\s+(?:Item|Trial)\s*\d+)?\s*\.?\s*$', re.I)),
    # Gerund self-narration is claim voice ("Remembering the letters in
    # order: LHE" is the model's answer, cf. b0161); the imperative form
    # ("Remember the letters E, H, ...") stays a transcript recap (R15).
    ('R18_gerund_claim', re.compile(r'\bRemembering\s+the\s+letters\b')),
]
# Worked-example voice: mask regardless of how many letters the line
# carries (a worked example can restate the full sequence, cf.
# b0038/b0283/b0104/b0365).  This tier outranks every KEEP: teaching text
# can quote first-person or claim-shaped fragments.
EXAMPLE_RULES = [
    ('R7_usage_example', re.compile(r'^\s*Usage\s*:', re.I)),
    ('R16_worked_example', re.compile(
        r'\bsample\s+response\b|\bfor\s+example\b|\be\.?g\.?\s*[,.:(]', re.I)),
]
# Self-correction claims ("the correct answer/order/sequence is X Z Y")
# supersede earlier candidates per the adjudication protocol and must stay
# visible (b0268/b0278); checked between the example tier and the
# third-person tier because such lines often also mention the participant.
# The claimed value must itself be LETTER content: "The correct answer is
# YES/NO" is an arithmetic claim and must not shield an instruction-voice
# line from masking (b0150 regression).
CORRECT_CLAIM_RULE = re.compile(
    r'\bcorrect\s+(?:order|answer|sequence)\b[^.\n]{0,40}\bis\b'
    r'[^.\n]{0,12}?(?:▮|(?-i:\b(?!YES\b|NO\b)[A-Z]{1,8}\b))', re.I)
THIRD_PERSON_RULES = [
    ('R13_third_person', re.compile(
        r'\b(?:the\s+)?(?:user|reader|subject|person|participant)\b'
        r'[^.\n]{0,50}\b(?:remember|recall)\w*\b'
        r'|\bthey\s+(?:will\s+be|were|are)\s+asked\b[^.\n]{0,40}\bremember\w*\b', re.I)),
]
# Transcript-form recaps: masked unless a KEEP rule already claimed the line.
# R11 takes an optional article ("remember letter: S" leaks exactly like
# "remember the letter: S", cf. b0218) and requires an actual single-letter
# payload nearby so claim sentences that merely open with the echo scaffold
# ("Remembering the letter: The correct order ...") are not swallowed.
MASK_ALWAYS_RULES = [
    ('R8_math_trial_echo', re.compile(
        r'\b(?:Item|Trial)s?\s*\d+\b.*(?:\bIs\s*\(|\bIs\s+\d|=\s*\d+\s*\?'
        r'|\(\s*-?\d+\s*[+*/x-]|arithmetic\s+equation)'
        r'|^\s*\d+[.)]\s*(?:Is|Next\s+item\s+is)\s*\(', re.I)),
    ('R11_singular_remember', re.compile(
        r'\bremember\w*\s+(?:(?:the|a)\s+)?letter\b(?!s)'
        r'(?=[^A-Za-z\n]{0,6}(?:[Tt]he\s+letter\s+)?"?[A-Za-z]\b)', re.I)),
    ('R15_plural_remember_echo', re.compile(
        r'\bremember\w*\s+the\s+letters\b', re.I)),
    ('R12_saw_the_letter', re.compile(r'\bremember\s+that\s+you\s+saw\b', re.I)),
]
# Recall-phase phrasing: mask per-trial fragments, but a line carrying the
# model's full claimed sequence (>= 3 single-letter tokens) is its ANSWER
# and stays (cf. b0054/b0147; over-masking would flip sequence -> echo=0
# and move the primary estimand).
MASK_UNLESS_CLAIMRUN_RULES = [
    ('R9_shown_presented', re.compile(
        r'\bletters?\s+(?:shown|presented|displayed|given)\b'
        r'|\b(?:shown|presented)\s+in\s+(?:Item|Trial)'
        r'|\bfirst\s+letter\s+presented\b', re.I)),
    ('R10_second_person', re.compile(
        r'\byou\s+(?:were\s+asked|are\s+asked|saw|remembered|need\s+to\s+remember)\b', re.I)),
    ('R14_presented_tail', re.compile(
        r'\bthey\s+were\s+presented\s*:\s*"?[A-Z]\b', re.I)),
]
CLAIM_RUN_MIN = 3
SINGLE_CAP_RE = re.compile(r'\b[A-Z]\b')
LETTER_TOKEN_RE = re.compile(r'\b[A-Z]\b|\b(?!YES\b|NO\b)[A-Z]{2,8}\b')


def redact_line(line):
    return LETTER_TOKEN_RE.sub(lambda m: MASK_CHAR * len(m.group(0)), line)


def _line_decision(ln):
    """-> (action, rule) with action in {'mask','keep',None}.
    Tier order: worked example (outranks everything) > correct-claim keep
    > third-person instructional > KEEP > transcript recap > recall-phase
    phrasing with the >=3-letter claim-run exemption."""
    for name, p in EXAMPLE_RULES:
        if p.search(ln):
            return 'mask', name
    if CORRECT_CLAIM_RULE.search(ln):
        return 'keep', 'R19_correct_claim'
    for name, p in THIRD_PERSON_RULES:
        if p.search(ln):
            return 'mask', name
    if any(p.search(ln) for _, p in KEEP_RULES):
        return 'keep', None
    for name, p in MASK_ALWAYS_RULES:
        if p.search(ln):
            return 'mask', name
    for name, p in MASK_UNLESS_CLAIMRUN_RULES:
        if p.search(ln):
            if len(SINGLE_CAP_RE.findall(ln)) >= CLAIM_RUN_MIN:
                return 'keep', name + '+claimrun'
            return 'mask', name
    return None, None


def semantic_redactions(text, manual=None):
    """-> list of (line_idx, rule) for lines whose letters must be masked.
    *manual* maps line_idx (int) -> {'action': 'mask'|'keep', ...} and
    overrides the rules."""
    out = []
    for li, ln in enumerate(text.splitlines()):
        if not re.search(r'\b[A-Z]\b|\b[A-Z]{2,8}\b', ln):
            continue
        if manual and li in manual:
            if manual[li]['action'] == 'mask':
                out.append((li, 'manual:' + manual[li].get('reason', '')))
            continue
        action, rule = _line_decision(ln)
        if action == 'mask':
            out.append((li, rule))
    return out


def apply_redactions(text, redactions):
    lines = text.splitlines(keepends=True)
    for li, _ in redactions:
        stripped = lines[li].rstrip('\n')
        nl = lines[li][len(stripped):]
        lines[li] = redact_line(stripped) + nl
    return ''.join(lines)


def ep_paths(model):
    for base in (f"{ROOT}/results/multiturn_eval_v3/openai_{model}/working_memory/operation_span",
                 f"{ROOT}/results/multiturn_expansion/openai_{model}/text/working_memory/operation_span"):
        fs = sorted(glob.glob(base + "/*.json"))
        if fs:
            return fs
    return []


def main():
    models = [r.split(',')[0] for r in
              open(f"{APLUS}/matrix_aplus_strict.csv").read().splitlines()[1:]]
    assert len(models) == 55

    episodes = []
    for m in models:
        fs = ep_paths(m)
        assert fs, f"no OSpan episodes for {m}"
        for f in fs:
            d = json.load(open(f))
            episodes.append((m, d['task_id'], d['responses'][-1]['response'] or ''))
    assert len(episodes) == 2750, f"walked {len(episodes)} episodes"

    machine = {}
    blind_by_text = {}
    counts = {}
    for m, tid, text in sorted(episodes, key=lambda x: (x[0], x[1])):
        cls, toks = classify_response(text)
        counts[cls] = counts.get(cls, 0) + 1
        if cls == 'blind':
            blind_by_text.setdefault(text, []).append({'model': m, 'task_id': tid})
        else:
            machine.setdefault(m, {})[tid] = {'class': cls, 'tokens': toks}

    texts = sorted(blind_by_text)  # deterministic pre-shuffle order
    rng = np.random.default_rng(42)
    order = rng.permutation(len(texts))

    bundle = f"{OUT}/annotator_bundle"
    os.makedirs(bundle, exist_ok=True)
    stale = f"{OUT}/blind_package.jsonl"
    if os.path.exists(stale):
        os.remove(stale)  # pre-masking package location; superseded
    if os.path.exists(f"{OUT}/INSTRUCTIONS.md"):
        os.remove(f"{OUT}/INSTRUCTIONS.md")

    manual_path = f"{OUT}/manual_redaction_decisions.json"
    manual_all = (json.load(open(manual_path))['decisions']
                  if os.path.exists(manual_path) else {})

    mapping = {}
    audit = {}
    n_masked = 0
    n_changed_lines = 0
    n_noop_hits = 0
    n_manual_applied = 0
    with open(f"{bundle}/blind_package.jsonl", 'w') as f:
        for rank, ti in enumerate(order):
            text = texts[ti]
            bid = f"b{rank:04d}"
            manual = {int(k): v for k, v in manual_all.get(bid, {}).items()}
            n_manual_applied += len(manual)
            reds = semantic_redactions(text, manual=manual)
            masked = apply_redactions(mask_gold_echoes(text), reds)
            if masked != text:
                n_masked += 1
            raw_lines = text.splitlines()
            mask_lines = masked.splitlines()
            changed = []
            for li, rule in reds:
                if mask_lines[li] != raw_lines[li]:
                    changed.append({'line': li, 'rule': rule,
                                    'orig_line_sha16': hashlib.sha256(raw_lines[li].encode()).hexdigest()[:16]})
                else:
                    n_noop_hits += 1
            if changed:
                audit[bid] = changed
                n_changed_lines += len(changed)
            assert not MASK_RE.search(masked), f"unmasked echo survives in {bid}"
            for li, _ in reds:
                assert not LETTER_TOKEN_RE.search(mask_lines[li]), \
                    f"letters survive redacted line in {bid}"
            f.write(json.dumps({'blind_id': bid, 'response': masked}, ensure_ascii=False) + "\n")
            mapping[bid] = {
                'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'episodes': blind_by_text[text],
            }
    json.dump({'spec': 'semantic redaction audit: transcript/instructor-voice lines whose '
                       'letters are masked; KEEP rules and the >=3-letter claim-run '
                       'exemption protect answer lines; manual_redaction_decisions.json '
                       'overrides both; only lines whose text actually changed are listed',
               'n_texts_changed': len(audit),
               'n_changed_lines': n_changed_lines,
               'n_noop_rule_hits': n_noop_hits,
               'n_manual_decisions_applied': n_manual_applied,
               'redactions': audit},
              open(f"{OUT}/redaction_audit.json", 'w'), indent=1)
    json.dump(mapping, open(f"{OUT}/blind_mapping.json", 'w'), indent=1)
    json.dump({'spec': 'machine-class OSpan episodes (zero-prose formats), frozen classifier',
               'class_counts': counts, 'models': machine},
              open(f"{OUT}/machine_scores.json", 'w'), indent=1)

    n_blind_eps = sum(len(v) for v in blind_by_text.values())
    with open(f"{bundle}/INSTRUCTIONS.md", 'w') as f:
        f.write(f"""# OSpan blind adjudication

This bundle is the ONLY material for annotation: `blind_package.jsonl`
({len(texts)} unique response texts covering {n_blind_eps} episodes) plus
this file. The package hides model identity, family, and current scores.
Letters inside transcript-voiced or instructional restatements of the
stimuli (verbatim "Remember the letter:" echoes, per-trial math+letter
walk-throughs, "the letter shown is ..." phrasings, worked examples) are
masked with {MASK_CHAR}; the model's own claimed recall sequences are
never masked. {n_masked} of {len(texts)} texts contain masks. Residual
task structure (set sizes, math questions) remains visible and is fine.
Do not open `blind_mapping.json`, `redaction_audit.json`, any results
directory, or any battery file while annotating; annotators must not
have previously seen the mapping or gold.

## Protocol (two independent annotators)

1. Annotator A writes `annotator_A.json`; annotator B independently
   writes `annotator_B.json` (same schema below, full coverage of every
   blind_id, no communication before both files are frozen).
2. Run `scripts/reanalysis/compute_agreement.py`. It reports label
   agreement and exact sequence-token agreement, and writes
   `disagreement_manifest.json`.
3. Both annotators jointly resolve each disagreement; the consensus is
   written to `adjudications.json` (in
   `results/reanalysis/aplus_20260718/`). The script then records all
   file hashes into `ADJUDICATION_MANIFEST.json`.
4. Freeze everything (no further edits) and run the suite with
   `COGARENA_REQUIRE_ADJ=1`.

## Schema (one JSON object per blind_id)

```json
{{"b0000": {{"label": "sequence", "tokens": ["M", "J", "F"]}},
 "b0001": {{"label": "refusal"}},
 "b0002": {{"label": "ambiguous", "note": "two conflicting final sequences"}}}}
```

Labels:
- `sequence`: the response commits to exactly one final letter sequence.
  Put that sequence, in order, into `tokens` (single uppercase letters
  A-Z). {MASK_CHAR} is never a token: if the only candidate letters are
  masked, the response is restating stimuli, not answering - label it
  `echo`. Interim or corrected sequences are superseded by the final one
  the response commits to.
- `ambiguous`: more than one candidate sequence with no clear final
  commitment, or it is genuinely unclear which letters are claimed.
- `refusal`: declines to answer or states it cannot recall.
- `echo`: repeats prompt/task text (including masked {MASK_CHAR} lines)
  without providing an answer.
- `non-answer`: anything else with no claimed sequence (explanations,
  code, unrelated text).

Scoring consequences (pre-specified, do not optimize for them):
`sequence` scores by serial-position credit against gold; `ambiguous`,
`refusal`, `echo`, and `non-answer` score 0 in the primary metric; an
ambiguous-excluded sensitivity is also computed.
""")

    pkg_manifest = {
        'spec': 'OSpan blind adjudication package, masked, seed-42 shuffle',
        'masking': 'narrow verbatim echo mask (MASK_RE) plus semantic redaction of '
                   'transcript/instructor-voiced lines (KEEP/MASK rules and per-line '
                   'audit frozen in build_ospan_blind_package.py + redaction_audit.json); '
                   'text_sha256 in blind_mapping.json is over the RAW unmasked response',
        'n_unique_texts': len(texts),
        'n_blind_episodes': n_blind_eps,
        'n_masked_texts': n_masked,
        'n_texts_semantically_changed': len(audit),
        'n_changed_lines': n_changed_lines,
        'n_noop_rule_hits': n_noop_hits,
        'n_manual_decisions_applied': n_manual_applied,
        'class_counts': counts,
        'blind_package_sha256': _sha(f"{bundle}/blind_package.jsonl"),
        'instructions_sha256': _sha(f"{bundle}/INSTRUCTIONS.md"),
        'blind_mapping_sha256': _sha(f"{OUT}/blind_mapping.json"),
        'redaction_audit_sha256': _sha(f"{OUT}/redaction_audit.json"),
        'manual_decisions_sha256': (_sha(manual_path) if os.path.exists(manual_path) else None),
    }
    json.dump(pkg_manifest, open(f"{OUT}/PACKAGE_MANIFEST.json", 'w'), indent=1)

    print('episodes:', len(episodes), '| class counts:', counts)
    print('unique blind texts:', len(texts), '| blind episodes:', n_blind_eps)
    print(f'texts changed: {n_masked} | changed lines: {n_changed_lines} '
          f'| no-op rule hits: {n_noop_hits} | manual decisions: {n_manual_applied}')
    print('annotator bundle:', bundle)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
