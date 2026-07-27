# Frozen VLM remediation run

This run replaces the invalid May 2026 image aggregate after two audit
findings. Empty responses had been credited by the legacy scorer, and missing
system fonts had silently reduced nominal 55 to 60 pixel glyphs to a tiny
bitmap fallback.

The authority remediation freezes one seed-42 stimulus set before inference.
It contains 100 Stroop trials, 100 Flanker trials, and 50 false-belief
stories. Stroop ink and word labels are matched across conditions. Flanker
target directions form an exact direction-by-congruency factorial. Each
false-belief story is rendered as one readable 2-by-2 montage, avoiding the
four-image context saturation observed in the diagnostic arm.
Six exact Ollama tags each consume the same 250 tasks. Per-item files retain the
full OpenAI-compatible response object, finish reason, usage data, request
fingerprint, stimulus hashes, strict parser output, and immutable serving
session. Valid blank completions are preserved and scored as incorrect. They
are not retried into a different scientific answer.

The model array is GPU-only and refuses to run on a login host. It uses cached
tags and never pulls. Context length and flash-attention overrides are
explicitly unset to match the original tag-default serving contract.
Operational readiness uses one frozen out-of-sample image endpoint probe.
Any transport-valid completion, including a blank completion, passes the
health check; no scientific item is used to decide whether inference proceeds.

```bash
export COGARENA_VLM_RUN_ROOT="$PWD/results/vlm_rerun_20260724_authority"
python -m scripts.experiments.vlm_rerun_20260724.build_stimuli \
  --design balanced_montage_v2
python -m scripts.experiments.vlm_rerun_20260724.build_scoring_contract
mkdir -p "$COGARENA_VLM_RUN_ROOT/slurm"
sbatch scripts/experiments/vlm_rerun_20260724/run_array.sbatch
sbatch --dependency=afterok:<array-job-id> \
  scripts/experiments/vlm_rerun_20260724/finalize.sbatch
```
