"""Real Gymnasium environments for the 13 CogArena cognitive paradigms.

Wraps the in-package :class:`cogarena.core.CogArenaEnv` in a proper
:class:`gymnasium.Env` (declared Text spaces, modern 5-tuple ``step``) and
registers one environment per paper paradigm, so each can be created with
``gymnasium.make``::

    import gymnasium as gym
    import cogarena.gym_env            # registers the CogArena/* environments
    env = gym.make("CogArena/NBack-v0")
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step("NO MATCH")
    print(env.unwrapped.score())

The environment exposes a generic partial-match reward (the per-turn expected
answer appearing in the response); the paper's paradigm-specific scoring lives
in :func:`cogarena.core.score_item`.  Registered ids cover all 13 paradigms:
``CogArena/{DigitSpan,NBack,OperationSpan,Stroop,Flanker,GoNoGo,DRM,
SourceMonitoring,CVLT,FalseBelief,EPITOME,ConfidenceCalibration,Wagering}-v0``.
"""
from __future__ import annotations

import importlib
import inspect
from string import printable
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from cogarena.core import CogArenaEnv, TaskMetadata, EvalMode, ScoringConfig

# paradigm -> (dimensions module, generator class, dimension, env-id stem)
_SPEC: Dict[str, Tuple[str, str, str, str]] = {
    "digit_span":             ("working_memory",   "DigitSpanGenerator",            "working_memory",  "DigitSpan"),
    "n_back":                 ("working_memory",   "NBackGenerator",                "working_memory",  "NBack"),
    "operation_span":         ("working_memory",   "OperationSpanGenerator",        "working_memory",  "OperationSpan"),
    "stroop":                 ("cognitive_control","StroopParadigm",                "cognitive_control","Stroop"),
    "flanker":                ("cognitive_control","FlankerParadigm",               "cognitive_control","Flanker"),
    "go_nogo":                ("cognitive_control","GoNoGoParadigm",                "cognitive_control","GoNoGo"),
    "drm_false_memory":       ("episodic_memory",  "DRMGenerator",                  "episodic_memory", "DRM"),
    "source_monitoring":      ("episodic_memory",  "SourceMonitoringGenerator",     "episodic_memory", "SourceMonitoring"),
    "cvlt_word_list":         ("episodic_memory",  "CVLTGenerator",                 "episodic_memory", "CVLT"),
    "false_belief":           ("theory_of_mind",   "FalseBeliefGenerator",          "theory_of_mind",  "FalseBelief"),
    "epitome_tom":            ("theory_of_mind",   "EpitomeToMGenerator",           "theory_of_mind",  "EPITOME"),
    "confidence_calibration": ("metacognition",    "ConfidenceCalibrationGenerator","metacognition",   "ConfidenceCalibration"),
    "post_decision_wagering": ("metacognition",    "PostDecisionWageringGenerator", "metacognition",   "Wagering"),
}
_CHARSET = frozenset(printable)


def _format_nback_stimulus(turn: dict) -> str:
    token = turn.get("stimulus", turn.get("token", "?"))
    return f"Token: {token}\nRespond MATCH or NO MATCH."


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _call_generate(gen_cls, seed: int, difficulty: str):
    """Call a generator's ``generate`` passing only the kwargs it accepts."""
    fn = gen_cls().generate
    params = inspect.signature(fn).parameters
    kw: Dict[str, Any] = {}
    if "seed" in params:
        kw["seed"] = seed
    for nk in ("n_items", "n_per_paradigm"):
        if nk in params:
            kw[nk] = 1
    if "n_congruent" in params:
        kw["n_congruent"] = 1
    if "n_incongruent" in params:
        kw["n_incongruent"] = 1
    if "n_trials" in params:
        kw["n_trials"] = 3
    if "difficulty" in params:
        kw["difficulty"] = difficulty
    if "include_contamination_probes" in params:
        kw["include_contamination_probes"] = False
    return fn(**kw)


def make_trial_generator(paradigm: str, seed: int, difficulty: str = "easy"):
    """Return ``generator(config) -> list[trial dict]`` for one paradigm."""
    mod, cls, _dim, _stem = _SPEC[paradigm]

    def generator(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        gen_cls = getattr(importlib.import_module(f"cogarena.dimensions.{mod}"), cls)
        items = _call_generate(gen_cls, seed, difficulty)
        if not items:
            return []
        item = items[0]
        meta = _attr(item, "metadata", None)
        params = _attr(meta, "parameters", {}) or {}
        turns = params.get("turns", []) if isinstance(params, dict) else (getattr(params, "turns", []) or [])
        if turns:
            trials: List[Dict[str, Any]] = []
            for t in turns:
                stim = _format_nback_stimulus(t) if paradigm == "n_back" else (t.get("stimulus") or t.get("prompt") or str(t))
                trials.append({"stimulus": stim, "expected": str(t.get("expected", t.get("correct_answer", "")))})
            return trials
        stim = _attr(item, "stimulus", "") or ""
        exp = _attr(item, "expected_response", None)
        if exp is None:
            exp = _attr(item, "expected", "")
        return [{"stimulus": stim, "expected": str(exp or "")}]

    return generator


class CogArenaGymEnv(gym.Env):
    """Gymnasium environment for one CogArena cognitive paradigm (text in/out)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        paradigm: str = "n_back",
        seed: int = 42,
        difficulty: str = "easy",
        max_obs_len: int = 16384,
        max_act_len: int = 512,
    ) -> None:
        super().__init__()
        if paradigm not in _SPEC:
            raise ValueError(f"paradigm must be one of {sorted(_SPEC)}, got {paradigm!r}")
        self.paradigm = paradigm
        self.difficulty = difficulty
        self._default_seed = seed
        self.observation_space = spaces.Text(max_length=max_obs_len, charset=_CHARSET)
        self.action_space = spaces.Text(max_length=max_act_len, charset=_CHARSET)
        self._meta = TaskMetadata(
            dimension=_SPEC[paradigm][2],
            paradigm=paradigm,
            mode=EvalMode.AGENT_INTERACTIVE,
            scoring=ScoringConfig(method="partial_match", params={"direction": "expected_in_response"}),
        )
        self._env: Optional[CogArenaEnv] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed=seed)
        use_seed = self._default_seed if seed is None else seed
        gen = make_trial_generator(self.paradigm, use_seed, self.difficulty)
        self._env = CogArenaEnv(gen, self._meta)
        obs = self._env.reset(seed=use_seed, config=options or {})
        return self._obs_text(obs), {"raw": obs, "paradigm": self.paradigm}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Call reset() before step().")
        obs, reward, done, info = self._env.step(str(action))
        return self._obs_text(obs), float(reward), bool(done), False, info

    def score(self) -> Dict[str, float]:
        return self._env.score() if self._env is not None else {"accuracy": 0.0}

    @staticmethod
    def _obs_text(obs: Dict[str, Any]) -> str:
        text = "\n\n".join(p for p in (obs.get("instructions"), obs.get("feedback"), obs.get("stimulus")) if p)
        return text or "[episode complete]"


# -- registration (one env per paper paradigm) ------------------------------
for _para, (_m, _c, _d, _stem) in _SPEC.items():
    _id = f"CogArena/{_stem}-v0"
    if _id not in gym.registry:
        gym.register(
            id=_id,
            entry_point="cogarena.gym_env:CogArenaGymEnv",
            kwargs={"paradigm": _para},
            max_episode_steps=200,
            order_enforce=True,
        )
