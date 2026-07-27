# OSpan blind adjudication

This bundle is the ONLY material for annotation: `blind_package.jsonl`
(483 unique response texts covering 516 episodes) plus
this file. The package hides model identity, family, and current scores.
Letters inside transcript-voiced or instructional restatements of the
stimuli (verbatim "Remember the letter:" echoes, per-trial math+letter
walk-throughs, "the letter shown is ..." phrasings, worked examples) are
masked with ▮; the model's own claimed recall sequences are
never masked. 126 of 483 texts contain masks. Residual
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
{"b0000": {"label": "sequence", "tokens": ["M", "J", "F"]},
 "b0001": {"label": "refusal"},
 "b0002": {"label": "ambiguous", "note": "two conflicting final sequences"}}
```

Labels:
- `sequence`: the response commits to exactly one final letter sequence.
  Put that sequence, in order, into `tokens` (single uppercase letters
  A-Z). ▮ is never a token: if the only candidate letters are
  masked, the response is restating stimuli, not answering - label it
  `echo`. Interim or corrected sequences are superseded by the final one
  the response commits to.
- `ambiguous`: more than one candidate sequence with no clear final
  commitment, or it is genuinely unclear which letters are claimed.
- `refusal`: declines to answer or states it cannot recall.
- `echo`: repeats prompt/task text (including masked ▮ lines)
  without providing an answer.
- `non-answer`: anything else with no claimed sequence (explanations,
  code, unrelated text).

Scoring consequences (pre-specified, do not optimize for them):
`sequence` scores by serial-position credit against gold; `ambiguous`,
`refusal`, `echo`, and `non-answer` score 0 in the primary metric; an
ambiguous-excluded sensitivity is also computed.
