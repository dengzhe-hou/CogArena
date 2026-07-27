# Scaffold wording replication

This is a post-hoc exploratory replication of the fully crossed intervention
study. It changes only the five targeted scaffold wordings and reruns the same
baseline, neutral placebo, five targeted conditions, 234 held-out items, 12
models, scorers, and family-aware analysis. It does not update or replace the
parent study's frozen all-nine-gate decision.

The parent operational pilot and capacity check are reused only for transport,
response-format, context-window, prompt-length, and full-GPU feasibility. The
replication remains a separate study with a separate specification, output
root, raw closure, and analysis manifest.

The inference records remain bound to the launch-injected inference
revision. Final analysis reuses the parent analyzer but replaces only its
confirmatory pilot-freeze check with a replication-specific provenance gate.
The analysis-adapter revision is injected separately at launch. All
record, replay, scorer, manifest, and aggregate-output checks remain active.

Launch sequence

```bash
INFERENCE_HEAD=$(git rev-parse HEAD)
PREP=$(sbatch --parsable \
  scripts/experiments/scaffold_wording_replication_20260725/prepare.sbatch)
RAW=$(sbatch --parsable --dependency=afterok:"${PREP}" \
  --export=ALL,COGARENA_GIT_HEAD="${INFERENCE_HEAD}" \
  scripts/experiments/scaffold_wording_replication_20260725/full.sbatch)
ANALYSIS_ADAPTER_HEAD=$(git rev-parse HEAD)
sbatch --dependency=afterok:"${RAW}" \
  --export=ALL,COGARENA_GIT_HEAD="${INFERENCE_HEAD}",COGARENA_ANALYSIS_ADAPTER_HEAD="${ANALYSIS_ADAPTER_HEAD}" \
  scripts/experiments/scaffold_wording_replication_20260725/finalize.sbatch
```

The frozen aggregate result is reported as a post-hoc wording sensitivity
in the paper. It does not replace the outcome-frozen parent study. The
public release includes the frozen specification, item manifests,
provenance, run manifest, aggregate result, and analysis manifest. Raw
responses and scheduler logs remain excluded.

## Transport amendment

Repeated full execution showed that OLMo2 7B can return only the requested
transport delimiter while also reporting `finish_reason=stop`. GPU diagnostics
reproduced two such delimiter-only bodies, while a later occurrence was not
reproduced on immediate replay. This is treated as a model-server transport
shape, not as a property of particular task requests. The amended replication
runner maps only an exact delimiter-only body from OLMo2 7B to an empty
response, which receives zero credit, and writes a separate request-hashed
incident record. Any answer-plus-delimiter body and all delimiter leakage from
other models remain fatal. Because the amendment changes executed source, all
twelve models are restarted together in a new Slurm array and no records from
failed attempts are reused.
