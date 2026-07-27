# Causal Selectivity Matrix (2026-07-20)

This isolated experiment asks whether five answer-free cognitive scaffolds
produce *selective* gains on their matched CogArena groupings beyond a
length-matched neutral placebo. It is deliberately distinct from a generic
prompting or chain-of-thought sweep: the pre-specified estimand is the diagonal
advantage of a 5 intervention × 5 grouping matrix, with global gain, shuffled
mapping, family consistency, and response-format controls.

The design retains CogBench's useful commitments, including procedurally
generated psychology tasks, separate performance/behavioral metrics, nested
model-lineage analysis, and hypothesis-driven prompt interventions. It changes
the scientific target. CogBench phenotypes models and reports task-specific
prompt effects; here every scaffold is crossed with every grouping on the same
held-out items, compared with a length-matched placebo, and evaluated for a
family-replicated diagonal interaction. Generic prompt improvement therefore
cannot satisfy the primary estimand.

## Current state

`PREPILOT_SPEC.json` is an outcome-blind engineering specification, not a public
preregistration. The operational pilot and capacity gates, final formal freeze,
12-model run, independent replay, raw closure, and amended fail-closed analyzer
are complete. `RUN_MANIFEST_formal.json` binds 19,656 records, 12 verified model
guards, the frozen item and model-manifest trees, the exact source revision, and
the serving/transport checks. The primary analyzer completed with
$\Gamma=.019920$ and an all-nine confirmation decision of `FAIL` (six gates
passed; family-held-out prediction and two minimum-cell response-format
sensitivities failed). `ANALYSIS_AMENDMENT.json` records the reporting-only rule
for unestimable minimum-cell sensitivities; it changes no record, prompt,
scorer, estimand, threshold, seed, or estimable result. Post-hoc aggregate-only
audits are isolated under `analyze_exploratory.py` and cannot alter that decision.

Earlier pilot, capacity, and incomplete formal attempts remain documented as
superseded operational history. Their response outcomes were not reused in the
final run. The authoritative final inputs and outputs are the frozen spec and
amendment, `RUN_MANIFEST_formal.json`, `analysis/analysis_results.json`,
`analysis/ANALYSIS_MANIFEST.json`, and the exploratory spec/result/manifest.

The frozen manifests live in
`results/causal_selectivity_20260720/`:

- formal: 234 episodes = 13 paradigms × 3 difficulties × 6 items;
- pilot: 39 sacrificial episodes = one per paradigm/difficulty;
- every model receives baseline, neutral placebo, and five targeted arms;
- formal: 1,638 task records/model, 19,656 records total;
- pilot/formal model IDs, task IDs, complete gold-bound item fingerprints,
  and complete presented-stimulus hashes are disjoint.

## Gates before any job

Submit the zero-inference preparation job from the repository root. It runs on
the c01 CPU node, regenerates manifests twice, checks byte-level idempotence,
and runs the infrastructure tests; no Python analysis runs on the login node:

```bash
mkdir -p results/causal_selectivity_20260720/slurm
PREP_JOB=$(sbatch --parsable \
  scripts/experiments/causal_selectivity_20260720/prepare.sbatch)
```

Do not run model inference on a login node. The Python runner requires a Slurm
job, a CUDA allocation, a node-local OpenAI-compatible endpoint, an Ollama tag
digest, a selected-model `ollama ps` row whose PROCESSOR column is exactly
`100% GPU` (split CPU/GPU offload is rejected), and an actual context of 4,096.
It writes each record atomically, but never resumes a record or model attempt.
The runner atomically creates a model root that must not already exist and
writes `EXECUTION_GUARD.json` before serving provenance or study requests. The
guard advances only from `in_progress` to `records_complete` and then, after
same-job independent replay, to `verified_complete`. An interruption leaves the
root occupied; recovery requires quarantining the entire profile attempt and
restarting every model in a fresh profile root. Each guard also binds
`SLURM_ARRAY_JOB_ID`, and closure requires one shared array identity across all
pilot or formal models, so a later array cannot fill only a failed model.
Every OpenAI-compatible request also carries the frozen top-level field
`reasoning_effort=none` (via `extra_body` in the Python client and explicitly
in both force-load requests). The request value, absence of an exposed
reasoning channel, and any reported reasoning-token count are persisted and
replayed; a missing/different request or a positive reported reasoning-token
count fails closed. This follows Ollama's documented OpenAI-compatible
reasoning control while making the limited claim that the control was sent and
audited, not that an opaque server could never ignore it. Each request also
tells the model to append the answer-free delimiter
`<END_COGARENA_RESPONSE>` only after every requested answer line or field.
Transport is routed only by the paradigm's declared format: nine single-line
paradigms stop at newline or the marker, while DRM, source monitoring,
confidence, and wagering stop at a blank line or the marker. CVLT's parser
already treats commas and newlines identically, so a fixed paradigm-only suffix
requests its full list on one comma-separated line. Internal newlines remain
legal for the four required-multiline paradigms, whereas repetitive single-line
output cannot consume the budget. Natural EOS is also valid; a
leaked delimiter, a different per-paradigm stop list, or a missing policy
binding fails closed. An HTTP-200 response is not accepted merely because it
contains text: it must also contain an admissible terminal finish reason and
real, positive usage metadata. A listed terminal/usage-metadata fault is retried
using the same canonical normalized request payload (identical SHA-256) for
three total attempts. If all three attempts have listed faults and none is a
request exception, the logical call is
stored as `transport_incomplete`; the complete task record bypasses the native
scorer and receives zero, and later multi-turn history receives the fixed
sentinel `[INVALID_TRANSPORT_COMPLETION]`. A later valid response is consumed
normally. Request exceptions may be retried, but an exhausted sequence
containing one is fatal rather than converted to `transport_incomplete`. No
alternate API or endpoint fallback is permitted. Physical attempts, logical
calls, retries, HTTP-200 terminal-metadata faults, request errors, and
usage-valid logical calls are recorded and replayed. Attempt accounting is an
exact partition: physical attempts equal accepted usage-valid logical calls
plus terminal-metadata-fault attempts plus request-error attempts.
Independently, a complete task record is marked
*recovered-terminal-metadata-fault exposed* when any ultimately valid logical
call had at least one preceding HTTP-200 terminal-metadata-fault attempt.
Request-error-only recovery is not exposure under this definition. Model and
run manifests report exposed-record totals plus exact per-condition counts and
rates. SDK-internal retries are disabled so the explicit three-attempt loop is
the only retry layer.
Both the conservative model-independent prompt estimate and, for every
transport-valid logical call, server-reported token usage must leave the full
512-token completion budget inside the uniform 4,096-token context.

Batch nodes are not assumed to have Git. On the login node, inject the exact
committed revision at submission time; no runtime script calls `git`:

```bash
mkdir -p results/causal_selectivity_20260720/{slurm,ollama}
HEAD=$(git rev-parse HEAD)
PILOT_JOB=$(sbatch --parsable --export=ALL,COGARENA_GIT_HEAD="$HEAD" \
  scripts/experiments/causal_selectivity_20260720/pilot.sbatch)
sbatch --dependency=afterok:"$PILOT_JOB" \
  --export=ALL,COGARENA_GIT_HEAD="$HEAD" \
  scripts/experiments/causal_selectivity_20260720/finalize.sbatch pilot
```

The shared model store must already contain all three pilot tags. Jobs never
pull. Qwen2.5-1.5B, Llama3.2-3B, and Gemma3-1B exercise three ordinary format
paths at a uniform 4,096-token context, 512-token completion limit,
`reasoning_effort=none`, and the paradigm-routed/end-marker contract. Earlier
operational-only attempts (jobs 6902 and 6917) encountered length finishes in
Qwen3, SmolLM2, Phi-3, and Mistral before any intervention contrast was
inspected. Before formal freeze, the protocol was therefore made deterministic:
a successful call with `finish_reason=length` is persisted on its first
occurrence, is never retried, and makes the complete task record score zero.
For multi-turn tasks, later turns see the fixed history sentinel
`[INVALID_COMPLETION_AT_TOKEN_LIMIT]`, never the partial completion. Closure
counts these records by condition rather than rejecting them. Exhausted
terminal-metadata faults follow the parallel zero-and-sentinel rule above. The
bound pilot passes its operational freeze gate only when every condition's
union task-record protocol-invalid rate (length or `transport_incomplete`) is at
most the structured 1% ceiling in the specification.
Closure itself enforces this limit before writing a PASS manifest. No pilot
outcome aggregate is written to a model manifest. Formal per-model closure is
also operational-only: no arm-by-paradigm outcome aggregate is written before
the single downstream aggregate analyzer runs.

Job 6961 was an operational-only run at the earlier 1,024-token ceiling. No
accuracy or intervention contrast was inspected, but condition-level finish
accounting found two conditions at 2/117 truncated task records (1.71%), above
the already specified 1% gate. The attempt was quarantined. That run also
revealed that the closure writer did not yet enforce the gate, although formal
preflight would have rejected its manifest. Before formal freeze, closure was
made fail-closed and the completion ceiling was revised to 2,048; the task-zero
and paired-exclusion rules remained unchanged. Job 6972 then showed that one
condition still had 2/117 length-limited task records. No effect estimate was
inspected. The attempt was quarantined and the final pre-freeze budget was set
once more to 4,096 completion tokens inside a 16,384-token served context. A
conservative prompt-plus-completion runtime check prevents hidden context
truncation. Job 6979 then retained the same 2/117 failure in one condition:
the affected calls had empty visible content but reported exactly 4,096
completion tokens, consistent with server-side reasoning-token expenditure.
No intervention outcome was inspected; that pilot and its same-revision
capacity gate were invalidated. Before formal freeze, the request contract was
extended to `reasoning_effort=none` without changing the panel, items, prompts,
scorers, context, completion budget, or 1% gate. The operational pilot and
capacity gate were rerun under that exact contract as jobs 6989--6992. The
same 2/117 condition-level failure remained. A response-shape-only audit (no
answer or score inspection) showed that every limited response contained 289
to 2,048 visible lines of repetition. A uniform newline stop was then added,
and jobs 7006--7009 passed their operational truncation gate. A later static
audit showed that this stop would truncate required multiline outputs (up to
35 DRM lines, 22 source-monitoring lines, and two metacognitive fields), so
those otherwise passing closures were superseded without inspecting outcome
accuracy or contrasts.

A subsequent formal-runtime metadata audit found that the retained Gemma2 tags
serve at most 8,192 tokens and the retained Llama2 tags at 4,096, making the
then-uniform 16,384 setting incompatible with the frozen panel. Jobs
7018--7020 were canceled; every generated record, including two complete
Qwen2.5 checkpoints, was quarantined without reuse or aggregate analysis. The
first compatibility amendment set a panel-wide 4,096 context, a 512-token
completion ceiling, and the answer-free end marker. Its first pilot was stopped
as soon as Qwen2.5-1.5B alone made the final 1% gate mathematically impossible:
three task records in one condition had already reached the ceiling, so the
best possible final rate was 3/117 (2.56%). Shape-only inspection found
205--256-line repetitions in calls whose declared response is a single line,
and blank-line-separated repetition in every limited CVLT call. All partial
records were quarantined without accuracy or contrast inspection. The final
format amendment therefore added the fixed paradigm-only routing above. That
routing yielded zero length finishes across all 973 Qwen pilot calls, but a Llama pilot then made
the unchanged gate impossible through CVLT alone (at least 2/117 in one
condition). It too was stopped and quarantined without outcome inspection.
Because comma- and newline-separated CVLT words are scorer-equivalent, CVLT was
standardized to the single-line comma format and moved into the single-line
transport class. The model panel, hardware, items, conditions, intervention
text, seeds, scorers, estimands, thresholds, and analysis code remain unchanged.
Preparation now
machine-audits the marker against all item/scaffold content, the largest
declared response shape, and a 1,024-character CVLT-filler history stress test
before any GPU job can run.

### Superseded operational chronology

The first formal array under that contract completed 11 of 12 model tasks.
Qwen2.5-14B then reproduced an HTTP-200 CVLT response whose body lacked both a
terminal event and valid usage under the OpenAI-compatible endpoint, native
streaming, the legacy engine, and a current official server release. No closure
or aggregate analyzer ran. Before inspecting any intervention outcome, the
specification was reopened and the whole formal attempt was declared
non-reusable. The transport rule above was added without changing the panel,
items, conditions, prompts, seeds, scorers, estimands, thresholds, hardware,
context, or completion budget; pilot and capacity closures were subsequently
rerun and rebound before the final formal inference.

In parallel, the workflow verified that a sacrificial, nonformal Qwen2.5-32B model---larger
than the maximum 27B formal checkpoint---serves fully on the single eligible
RTX PRO 6000 class at the same 4,096-token context. The experiment is
restricted to c04, avoiding cross-hardware variation and an A100 queue blocked
by unrelated multi-day allocations. Node selection is explicit; jobs never
infer hardware or change models:

```bash
PRO_JOB=$(sbatch --parsable --nodelist=c04 \
  --export=ALL,COGARENA_GIT_HEAD="$HEAD" \
  scripts/experiments/causal_selectivity_20260720/capacity_probe.sbatch pro6000)
sbatch --dependency=afterok:"$PRO_JOB" \
  --export=ALL,COGARENA_GIT_HEAD="$HEAD" \
  scripts/experiments/causal_selectivity_20260720/finalize_capacity.sbatch
```

The capacity model is `qwen2.5:32b`, which is excluded from formal inference.
The capacity gate requires its model digest, source revision, actual context,
and exact 100% GPU service on RTX PRO 6000. Pilot and formal launchers are
pinned to `c04`, so no unverified hardware can enter inference. Adjust
`--nodelist` only if cluster node assignments change, then revise and rerun the
pre-freeze capacity gate; never weaken the hardware-name content gate.

The final sequence after both closures was:

1. inspect only completeness, finish reasons, parser coverage, context, GPU,
   digest, and latency gates rather than intervention contrasts;
2. add the actual freeze timestamp, `RUN_MANIFEST_pilot.json` SHA-256, and
   `CAPACITY_GATE_MANIFEST.json` SHA-256 to the spec;
3. set `status` to `formal_frozen_after_pilot`;
4. rerun `prepare.sbatch`, then commit the frozen spec, manifests, analysis
   implementation, and runtime sources;
5. capture the resulting commit once as `FORMAL_HEAD`, submit `full.sbatch`
   with that exact revision injected as `COGARENA_GIT_HEAD`, then close it
   with `finalize.sbatch formal` using an `afterok` dependency and the same
   `FORMAL_HEAD` (never recapture HEAD between these stages).

```bash
FORMAL_HEAD=$(git rev-parse HEAD)
FORMAL_JOB=$(sbatch --parsable --nodelist=c04 --nodes=1 --ntasks=1 \
  --gres=gpu:pro_6000:1 --array=0-11%2 --mem=12G \
  --export=ALL,COGARENA_GIT_HEAD="$FORMAL_HEAD" \
  scripts/experiments/causal_selectivity_20260720/full.sbatch)
FORMAL_CLOSE=$(sbatch --parsable --dependency=afterok:"$FORMAL_JOB" \
  --export=ALL,COGARENA_GIT_HEAD="$FORMAL_HEAD" \
  scripts/experiments/causal_selectivity_20260720/finalize.sbatch formal)
```

## Frozen aggregate analysis

Only after `finalize.sbatch formal` emits a `RUN_MANIFEST_formal.json` with
`status=formal_raw_complete`, submit the CPU-only aggregate analyzer to c01:

```bash
ANALYSIS_JOB=$(sbatch --parsable \
  --dependency=afterok:"$FORMAL_CLOSE" \
  --export=ALL,COGARENA_GIT_HEAD="$FORMAL_HEAD" \
  scripts/experiments/causal_selectivity_20260720/analyze.sbatch)
```

`analyze.py` re-enumerates and scorer-replays all 19,656 formal records before
computing anything. The primary contrast uses paired targeted-minus-placebo
item gains, equal paradigm weights within matched and nonmatched sets, and an
equal-weight mean across the five interventions. Inference is a fixed 20,000-
draw crossed family-by-item bootstrap (six families, both checkpoints retained,
shared item resamples across models and arms) plus the exact 5! intervention-to-
group mapping permutation. It also reports family consistency, BH-adjusted
intervention contrasts, fixed-lambda leave-one-family-out predictive log
likelihood, the nine-rule no-override success vector, canonical OSpan,
hard-minus-easy difficulty interaction, condition-wise PC1 share, descriptive
OSpan math compliance, and frozen primary-response-unit empty/parse/protocol-
invalid sensitivities. Length finishes and exhausted terminal-metadata faults
enter the primary as task-level zeroes; the gate additionally requires every
condition's union task-record invalid rate to be at most 1% and a paired
protocol-invalid-exclusion analysis to retain at least half of Gamma with at
least three items per cell. Response-character adjustment fits and estimates
only on those same protocol-valid target-placebo pairs, so a
private partial completion cannot drive its confirmatory sensitivity. The
analyzer also reports an outcome-blind pre-rerun descriptive paired-exclusion sensitivity
for recovered terminal-metadata-fault exposure: a target-placebo item pair is
removed when either task record is exposed, and the output reports excluded
pairs, Gamma, the preservation ratio versus primary Gamma, and the minimum
retained items per model/intervention/paradigm. This diagnostic adds no success
gate. Request-error-only recovery remains descriptively counted at the attempt
level and is not part of this exclusion. The exact centered two-sided
permutation convention, ridge feature design, solver, seeds,
and every gate threshold are structured numeric fields in the pre-pilot spec
rather than being selected after results are seen or duplicated in the analyzer.

The only analysis outputs are aggregate statistics and hashes in
`results/causal_selectivity_20260720/analysis/`; no response, stimulus, gold,
task identifier, or item-level score is copied there. `ANALYSIS_MANIFEST.json`
binds the frozen spec, formal run closure, item manifest, all 12 model
manifests, analyzer/launcher, seeds, and output hash. The analyzer refuses a
login node, any node other than c01, a stale source tree, an incomplete raw
closure, or any missing, extra, malformed, non-finite, incorrectly transformed,
or identity-mismatched record.

### Post-freeze reporting amendment

The closed 12-model formal array and raw replay passed, but the first frozen
analysis job (`7148`) stopped before emitting results when the empty-response
paired-exclusion sensitivity contained a zero retained cell. The frozen rule
already requires a defined preservation ratio and at least three retained
items in every model/intervention/paradigm cell, so such a sensitivity is a
confirmatory failure rather than grounds to drop a model, impute observations,
or rerun inference. The original hash-bound `analyze.py` is therefore retained
unchanged. `ANALYSIS_AMENDMENT.json` binds its hash, the immutable formal run,
the failed job/log, and the outcome-blind reporting decision. The separately
versioned `analyze_amended.py` changes only fail-closed reporting: an
unestimable exclusion sensitivity emits `estimable=false`, retained/excluded
counts, a failure reason, and null estimates/bootstrap; its gate is false while
all primary, secondary, and estimable calculations remain frozen. Run it only
on c01 with both revisions injected:

```bash
sbatch --export=ALL,COGARENA_GIT_HEAD="$FORMAL_HEAD",\
COGARENA_ANALYSIS_GIT_HEAD="$AMENDMENT_HEAD" \
  scripts/experiments/causal_selectivity_20260720/analyze_amended.sbatch
```

### Post-hoc exploratory robustness analyses

`EXPLORATORY_ANALYSIS_SPEC.json` defines five aggregate-only analyses authored
after the primary outcome was inspected: all-paradigm leave-one-out influence,
a family-by-paradigm-within-group-by-item bootstrap, observable response-
evaluability and accuracy-contribution diagnostics, neutral-placebo versus
baseline auditing, and exact small-G family inference. These analyses have no
confirmatory role, cannot change the frozen primary result or its nine-gate
FAIL decision, and write only to `analysis/exploratory/`. In particular, the
paradigm bootstrap has only two or three observed paradigms per theoretical
stratum and is reported as a weak design-sensitivity diagnostic rather than
strong population-of-paradigms inference.

Do not run the replay or bootstrap on a login node. After committing the
exploratory code and spec, submit the CPU job to c01 with both the immutable
formal revision and the new analysis revision injected:

```bash
FORMAL_HEAD=$(jq -r .source_revision \
  results/causal_selectivity_20260720/RUN_MANIFEST_formal.json)
EXPLORATORY_HEAD=$(git rev-parse HEAD)
sbatch --export=ALL,COGARENA_GIT_HEAD="$FORMAL_HEAD",\
COGARENA_EXPLORATORY_GIT_HEAD="$EXPLORATORY_HEAD" \
  scripts/experiments/causal_selectivity_20260720/analyze_exploratory.sbatch
```

## Fail-closed provenance

Each result binds the specification, item manifest, complete item/gold hash,
presented-stimulus hash, system-prompt hash and length counts, source revision,
served model digest, actual context, schedule seed, finish reason, and scorer
contract. Existing model roots or records are never reused. Independent
verification reconstructs every static or multi-turn prompt hash (including
either fixed sentinel after a protocol-invalid call), binds the raw-record tree,
serving provenance and run summary into the no-resume execution guard, and must
finish in the same Slurm job/task/node that acquired the root. Accepted
server completions require `stop` or `length`; a listed fault attempt may lack a
finish reason only under the replayable classifier. `transport_incomplete` is a
derived logical-call status permitted only after three replay-verifiable
attempts with the same canonical request-payload SHA-256 all show listed faults
and none is a request exception. A length or transport-incomplete call is valid only under its
persisted task-zero completion contract and exact replay. Unknown, missing,
malformed, unaccounted-protocol-invalid, stale-digest, stale-revision, CPU-served,
context-mismatched, or temporary files stop closure.
Closure and formal analysis independently replay every physical attempt from
the private raw records. They reconcile length calls/records,
`transport_incomplete` calls/records, the union-invalid mask, recovered-fault
exposure, request-error and terminal-fault attempts, valid-usage calls, response
units, empty responses, and usage extrema against both model and run manifests;
any disagreement fails closed.

The commit value is not embedded inside the commit it names (which would be
circular). Instead it is mandatory submission metadata, while the item
manifest independently pins every runtime source file by SHA-256.

No paper file is read or modified by this experiment.
