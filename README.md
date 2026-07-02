# CogArena

**A benchmark of 13 cognitive-science paradigms for evaluating LLMs and agents.**

CogArena adapts validated experimental paradigms from cognitive science — working
memory, cognitive control, episodic memory, theory of mind, and metacognition — into
a procedurally generated benchmark you can run against **any** language model in one
command. Items are generated on the fly (seeded), so there is nothing to leak or
memorize.

> Paper: *CogArena: Cognitive-Science Paradigms Reveal Broad Competence, Not Separable
> Cognitive Domains in LLMs* (Anonymous, 2026).
>
> Headline finding: across 55 open-weight models, the five theory-motivated groupings
> are **not** empirically separable — LLM behavior is dominated by a single
> overall-competence (positive-manifold) axis. The groupings are an organizing taxonomy,
> not validated cognitive dimensions. CogArena is the instrument behind that result and a
> reusable battery for probing your own models.

---

## Install

```bash
git clone https://github.com/dengzhe-hou/cogarena.git
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

**What is in this repo.** The analysis scripts (`scripts/` and `scripts/reanalysis/`) and the
*derived* result files (`results/reanalysis/*.json` and `results/predictive_validity.json`) are
committed. Those JSONs hold the paper's headline numbers: the dimensional-separability test
(`b2_expanded.json`), the positive-manifold / general-factor analysis (`pca_partialcorr.json`),
the restricted-range robustness (`restricted_range_robustness.json`), and the scaling,
scorer-robustness, split-half-reliability, and signature-significance results. The reported
values can therefore be inspected directly, without re-running anything.

**What is not in this repo.** The *raw* per-item evaluation outputs (every model's
`details.json` / `aggregate.json`, several hundred MB across the `results/full_eval_*` and
`results/multiturn_*` directories) are not committed, because of their size. The analysis
scripts read from those directories, so re-running an analysis end-to-end needs the raw outputs.
Two ways to obtain them:

1. **Regenerate** them by running the benchmark (`cogarena eval`, or the `scripts/run_*.sh`
   drivers) over the model set; items are seeded, so generation is deterministic.
2. Use the separately archived raw-output bundle (large; released alongside the paper).

Scripts resolve the project root from their own location or from a `COGARENA_ROOT` environment
variable, so they run from any checkout. The reanalysis scripts need the `analysis` extra
(`pip install -e ".[analysis]"`: `numpy`, `scipy`, `pandas`, `scikit-learn`, `statsmodels`).
Figure generation lives with the paper sources and is not part of this code release; figures are
produced from the same derived JSONs.

## Citation

```bibtex
@article{cogarena2026,
  title   = {CogArena: Cognitive-Science Paradigms Reveal Broad Competence, Not Separable Cognitive Domains in LLMs},
  author  = {Anonymous},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
