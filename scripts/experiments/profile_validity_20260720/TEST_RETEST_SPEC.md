# Frozen adjacent-administration stability specification

Status: implementation-ready; analysis not yet executed.

## Scope and exclusions

- Occasions: `results/_archive/full_eval_20260525_1522` and
  `results/full_eval_20260526_2208`.
- Sampling frame: the same 20 text models present in both roots.
- Eligible static paradigms: Digit Span, Stroop, Flanker, DRM, Source
  Monitoring, False Belief, Confidence Calibration, and Post-decision
  Wagering.
- Go/No-Go and EPITOME are excluded because their production forms were later
  superseded.
- The 11 Source Monitoring episodes regenerated after the episode-wide
  de-duplication fix are excluded. The retained 39 episodes are those recorded
  as byte-identical in the frozen exclusion manifest.
- Exact expected size: 421 paired task IDs per model and 8,420 pairs overall.

This is an administration-stability analysis for the eligible static subset,
not a test-retest estimate for all 13 paradigms or for the five-dimensional
profile as a fully balanced instrument.

## Frozen primary analysis

For each occasion, raw stored text is replayed through the currently registered
paradigm scorer. The primary unit is the model-level mean score within a
paradigm, yielding 20 paired observations per paradigm. Report:

- absolute-agreement, single-measure ICC(A,1);
- Pearson and Spearman correlations;
- mean absolute difference and RMSE;
- mean occasion shift and Bland-Altman limits.

In addition to raw model-profile correlations, report pooled profile-cell
agreement after removing each model's grand mean separately on each occasion.
This profile-shape estimand cannot be driven by stable overall competence.

Merged model families are sampled with replacement as clusters, retaining all
checkpoints in each selected family. Percentile 95% intervals use 20,000
replicates and seed 42. Item-level paired results are descriptive because item
difficulty is shared across occasions.

## Secondary analysis

Rebuild the 20 x 8 score matrix on each occasion and report the within-minus-
cross grouping contrast, exact label-assignment p-values, PC1 variance share,
and their occasion differences. This analysis is explicitly secondary because
Working Memory and Theory of Mind each have only one eligible paradigm.

## Fail-closed and privacy gates

- exact model, paradigm, task-ID, identity-field, and count checks;
- direct scorer invocation with no silent exact-match fallback;
- formal-occasion item scores checked against the final Source Monitoring,
  corrected-static, and scorer-replayed Wagering overlays;
- the Wagering overlay hash, scorer-replay PASS manifest, source revision, and
  construct-representability gate are revalidated before use;
- seven comparable formal-occasion paradigm means checked against the frozen
  strict-primary matrix;
- consumed-file tree hashes, generator/gold fingerprint hash, input hashes,
  script hash, and output hashes persisted;
- output contains scores and identifiers but no raw responses, stimuli, answer
  keys, or multi-turn transcripts;
- execution only through the c01 Slurm job.
