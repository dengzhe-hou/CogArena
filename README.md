# CogArena

**A benchmark of 13 cognitive-science paradigms for evaluating LLMs and agents.**

CogArena adapts validated experimental paradigms from cognitive science — working
memory, cognitive control, episodic memory, theory of mind, and metacognition — into
a procedurally generated benchmark you can run against **any** language model in one
command. Items are generated on the fly from seeded templates, reducing reliance
on a fixed public item bank. Procedural generation mitigates rather than eliminates
contamination from familiar task templates.

> Paper: *CogArena: A Multimethod Evaluation of Cognitive Ability Structure in Large
> Language Models* (Hou, Jiang, Lin, and Yamada, 2026).
>
> Headline finding: across 55 open-weight models, a broad competence axis dominates and
> the five theory-motivated groupings show only a modest, inference-sensitive advantage.
> A separate fully crossed intervention study finds a small matched-scaffold tendency,
> but the frozen all-gates confirmation rule fails, and selective terms do not improve
> held-out-family prediction. The groupings are therefore useful organizing labels,
> not yet validated transportable dimensions. CogArena is the instrument and validation
> workflow behind that result.

---

## Install

```bash
git clone https://github.com/dengzhe-hou/CogArena.git
cd cogarena
pip install -e .            # core (gymnasium)
pip install -e ".[openai]"  # + OpenAI / any OpenAI-compatible endpoint
# extras: ".[anthropic]", ".[google]", ".[image]", ".[all]", ".[analysis]"
```

Requires Python ≥ 3.10.

## Quickstart — evaluate *your* model

The single-turn battery (10 paradigms across 5 groupings) runs out of the box.

```bash
# 0. Check your install (no API key needed)
cogarena eval --dry-run --model test

# 1. Any OpenAI-compatible endpoint — Ollama, vLLM, TGI, LM Studio, OpenRouter, Together, ...
cogarena eval --provider local --base-url http://localhost:11434/v1 --model qwen2.5:7b

# 2. Hosted APIs (set the matching key in your environment)
OPENAI_API_KEY=...    cogarena eval --provider openai    --model gpt-4o-mini
ANTHROPIC_API_KEY=... cogarena eval --provider anthropic --model claude-3-5-sonnet-20241022
GOOGLE_API_KEY=...     cogarena eval --provider google    --model gemini-1.5-pro

cogarena list   # show the paradigms
```

Useful flags: `--n` (items per paradigm, default 50), `--seed`, `--paradigms stroop flanker ...`,
`--temperature`, `--max-tokens`, `--output DIR`.

### Bring your own model

Three options, easiest first:

1. **OpenAI-compatible server** (covers most local/hosted models): `--provider local --base-url <url>`.
2. **Native SDKs**: `--provider {openai,anthropic,google}`.
3. **Anything else**: subclass the client and override one method —
   ```python
   from cogarena.llm_client import LLMClient
   class MyClient(LLMClient):
       def _dispatch(self, prompt, system_prompt=None, temperature=None,
                     max_tokens=None, images=None):
           return my_model.generate(prompt)   # return the response text
   ```
   Then drive the battery from Python (see `cogarena/cli.py` for a ~30-line example, or use
   `cogarena.scoring.score_static` + the generators in `cogarena.generators`).

## Output

Results are written to `cogarena_results/<model>/`:

- `aggregate.json` — overall accuracy, per-paradigm accuracy, and the 5 grouping means.
- `details.json` — every item: prompt id, paradigm, the model's response, and the full score dict.

The console prints a summary table (per-paradigm + grouping + overall).

## The battery

| Grouping | Single-turn paradigms (CLI) |
|---|---|
| Working Memory | digit span |
| Cognitive Control | Stroop, Flanker, Go/No-Go |
| Episodic Memory | DRM false memory, source monitoring |
| Theory of Mind | false belief, EPITOME |
| Metacognition | confidence calibration, post-decision wagering |

Three further paradigms are **multi-turn** (n-back, operation span, CVLT) and the agent/VLM
modes use the Gymnasium API:

```python
import gymnasium as gym
import cogarena.gym_env            # registers the envs
env = gym.make("CogArena/NBack-v0", seed=42)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step("NO MATCH")
print(env.unwrapped.score())
```

## Reproduce the paper

**What is in this repo.** The analysis code and the derived, anonymous result artifacts are
committed. These include the observational reanalyses under `results/reanalysis/`, corrected
matrices and inference outputs, the fully crossed intervention study under
`results/causal_selectivity_20260720/`, and the profile-stability and family-held-out diagnostics.
Their manifests bind the inputs, specifications, code, seeds, and outputs used for the reported
numbers, so the results can be inspected without rerunning model inference.
The committed target-versus-baseline sensitivity is inspectable and output-bound by its
manifest. Replaying it from its frozen inputs also requires the formal raw records and frozen
runtime manifest distributed with the separate raw-output bundle.

**What is not in this repo.** The *raw* per-item evaluation outputs (every model's
`details.json` / `aggregate.json`, several hundred MB across the `results/full_eval_*` and
`results/multiturn_*` directories) are not committed, because of their size. The analysis
scripts read from those directories, so re-running an analysis end-to-end needs the raw outputs.
Two ways to obtain them:

1. **Regenerate** them by running the benchmark (`cogarena eval`, or the `scripts/run_*.sh`
   drivers) over the model set; items are seeded, so generation is deterministic.
2. Use the separately archived raw-output bundle (large; to be released after anonymous
   review, subject to the venue's artifact policy).

Scripts resolve the project root from their own location or from a `COGARENA_ROOT` environment
variable, so they run from any checkout. The reanalysis scripts need the `analysis` extra
(`pip install -e ".[analysis]"`: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`,
`statsmodels`, `cairosvg`, and `PyMuPDF`).
The paper figures are generated by the scripts in `paper/figures/`, including
`generate_all.py`, `fig_manifold.py`, and `generate_causal_selectivity.py`. They consume the
committed derived artifacts (and, where noted, the local raw-output bundle) and write the
tracked PDF figures next to the generators.

## Citation

```bibtex
@article{cogarena2026,
  title   = {CogArena: A Multimethod Evaluation of Cognitive Ability Structure in Large Language Models},
  author  = {Hou, Dengzhe and Jiang, Lingyu and Lin, Fangzhou and Yamada, Kazunori D},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
