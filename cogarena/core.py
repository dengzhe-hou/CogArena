"""CogArena core data models and evaluation API.

Defines:
- TaskMetadata, TaskInstance, EpisodeTrace, CognitiveProfile — core data models
- CogArenaEnv — Gymnasium-style environment for multi-turn cognitive tasks
- LLMEvaluator — wrapper for running static/multi-turn tasks on LLMs
- Scoring helpers — item-level and dimension-level scoring
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvalMode(str, Enum):
    """How the task is presented to the model."""
    LLM_STATIC = "llm_static"
    AGENT_INTERACTIVE = "agent_interactive"


class AdaptationDistance(str, Enum):
    """How far the LLM adaptation deviates from the original human version."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DifficultyLevel(str, Enum):
    """Coarse difficulty bin used for item selection."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Core data-classes
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """Describes how a task should be scored."""
    method: str  # e.g. "exact_match", "partial_match", "edit_distance", "custom"
    params: Dict[str, Any] = field(default_factory=dict)
    # For custom scoring, `params` may contain a dotted path to a callable.


@dataclass
class TaskMetadata:
    """Describes a cognitive task type (paradigm-level metadata)."""
    dimension: str              # CHC dimension, e.g. "working_memory"
    paradigm: str               # e.g. "n_back", "stroop", "wcst"
    mode: EvalMode              # llm_static or agent_interactive
    parameters: Dict[str, Any] = field(default_factory=dict)
    scoring: ScoringConfig = field(
        default_factory=lambda: ScoringConfig(method="exact_match")
    )
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    human_anchor: Optional[Dict[str, Any]] = None  # published human norm data
    adaptation_distance: AdaptationDistance = AdaptationDistance.LOW
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    description: str = ""


@dataclass
class TaskInstance:
    """A concrete, evaluable item (one stimulus + expected answer)."""
    task_id: str
    metadata: TaskMetadata
    stimulus: str                            # text prompt shown to the model
    image_path: Optional[str] = None         # optional image for VLM tasks
    expected_response: Optional[Any] = None  # ground-truth (None for open-ended)
    scoring_fn: Optional[Callable[..., Dict[str, float]]] = None  # override

    def __post_init__(self) -> None:
        if not self.task_id:
            h = hashlib.sha256(
                f"{self.metadata.dimension}:{self.metadata.paradigm}:{self.stimulus}".encode()
            ).hexdigest()[:12]
            self.task_id = f"{self.metadata.paradigm}_{h}"

    def score(self, response: Any) -> Dict[str, float]:
        """Score a response against this task instance."""
        if self.scoring_fn is not None:
            return self.scoring_fn(response, self.expected_response, self.metadata)
        return score_item(response, self.expected_response, self.metadata.scoring)


@dataclass
class EpisodeTrace:
    """Records every detail of one evaluation episode (single- or multi-turn)."""
    task_id: str
    model_id: str
    responses: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    token_counts: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    })
    timestamps: Dict[str, float] = field(default_factory=lambda: {
        "start": 0.0,
        "end": 0.0,
    })
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def wall_time(self) -> float:
        return self.timestamps.get("end", 0.0) - self.timestamps.get("start", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "responses": self.responses,
            "scores": self.scores,
            "confidence": self.confidence,
            "token_counts": self.token_counts,
            "timestamps": self.timestamps,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeTrace":
        return cls(
            task_id=d["task_id"],
            model_id=d["model_id"],
            responses=d.get("responses", []),
            scores=d.get("scores", {}),
            confidence=d.get("confidence"),
            token_counts=d.get("token_counts", {}),
            timestamps=d.get("timestamps", {}),
            extra=d.get("extra", {}),
        )


@dataclass
class CognitiveProfile:
    """Aggregated cognitive profile for a model across all dimensions."""
    model_id: str
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    # e.g. {"working_memory": 0.72, "cognitive_control": 0.58, ...}
    latent_traits: Dict[str, float] = field(default_factory=dict)
    # IRT-estimated latent ability parameters
    raw_data: Dict[str, Any] = field(default_factory=dict)
    # Full per-paradigm breakdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dimension_scores": self.dimension_scores,
            "latent_traits": self.latent_traits,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CognitiveProfile":
        return cls(
            model_id=d["model_id"],
            dimension_scores=d.get("dimension_scores", {}),
            latent_traits=d.get("latent_traits", {}),
            raw_data=d.get("raw_data", {}),
        )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_item(
    response: Any,
    expected: Any,
    config: ScoringConfig,
) -> Dict[str, float]:
    """Score a single item according to its ScoringConfig.

    Returns a dict with at least an ``accuracy`` key in [0, 1].
    """
    method = config.method
    if method == "exact_match":
        return _score_exact(response, expected)
    elif method == "partial_match":
        return _score_partial(response, expected, config.params)
    elif method == "edit_distance":
        return _score_edit_distance(response, expected, config.params)
    elif method == "list_recall":
        return _score_list_recall(response, expected, config.params)
    elif method == "custom":
        fn_path = config.params.get("fn")
        if fn_path is not None:
            fn = _import_callable(fn_path)
            return fn(response, expected, config.params)
        raise ValueError("custom scoring requires 'fn' in params")
    else:
        raise ValueError(f"Unknown scoring method: {method}")


def _score_exact(response: Any, expected: Any) -> Dict[str, float]:
    resp_str = str(response).strip().lower()
    exp_str = str(expected).strip().lower()
    correct = float(resp_str == exp_str)
    return {"accuracy": correct}


def _score_partial(
    response: Any, expected: Any, params: Dict[str, Any]
) -> Dict[str, float]:
    """Check if expected answer is contained in the response."""
    resp_str = str(response).strip().lower()
    exp_str = str(expected).strip().lower()
    # Guard against empty-string false positives
    if not exp_str or not resp_str:
        return {"accuracy": 0.0}
    if params.get("direction", "expected_in_response") == "expected_in_response":
        match = float(exp_str in resp_str)
    else:
        match = float(resp_str in exp_str)
    return {"accuracy": match}


def _score_edit_distance(
    response: Any, expected: Any, params: Dict[str, Any]
) -> Dict[str, float]:
    """Normalised Levenshtein similarity; accuracy = 1 if above threshold."""
    a = str(response).strip().lower()
    b = str(expected).strip().lower()
    dist = _levenshtein(a, b)
    max_len = max(len(a), len(b), 1)
    similarity = 1.0 - dist / max_len
    threshold = params.get("threshold", 0.8)
    return {
        "accuracy": float(similarity >= threshold),
        "similarity": similarity,
    }


def _score_list_recall(
    response: Any, expected: Any, params: Dict[str, Any]
) -> Dict[str, float]:
    """Score recall of a word list.

    ``expected`` should be a list; ``response`` is either a list or a
    comma/newline-separated string.
    """
    if isinstance(response, str):
        items = [
            w.strip().lower()
            for w in response.replace(",", "\n").split("\n")
            if w.strip()
        ]
    else:
        items = [str(w).strip().lower() for w in response]

    target = [str(w).strip().lower() for w in expected]
    target_remaining = list(target)  # consumed-set to prevent duplicate counting

    hits = 0
    intrusions = 0
    for w in items:
        if w in target_remaining:
            hits += 1
            target_remaining.remove(w)  # consume so duplicates don't double-count
        else:
            intrusions += 1
    total = max(len(target), 1)

    # Check order preservation (serial position)
    target_set = set(target)
    order_score = 0.0
    if hits > 1:
        matched_positions = [target.index(w) for w in items if w in target_set]
        pairs = sum(
            1
            for i in range(len(matched_positions) - 1)
            if matched_positions[i] < matched_positions[i + 1]
        )
        order_score = pairs / max(len(matched_positions) - 1, 1)

    return {
        "accuracy": hits / total,
        "recall": hits / total,
        "precision": hits / max(len(items), 1),
        "intrusions": float(intrusions),
        "order_preservation": order_score,
    }


def _levenshtein(a: str, b: str) -> int:
    """Standard DP Levenshtein distance."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[-1]


def _import_callable(dotted_path: str) -> Callable:
    """Import a callable from ``'pkg.mod.func'`` style path."""
    import importlib
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Invalid callable path: {dotted_path}")
    module_path, attr_name = parts
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def aggregate_dimension_scores(
    traces: List[EpisodeTrace],
    dimension: str,
) -> Dict[str, float]:
    """Aggregate item-level scores into a dimension-level summary.

    Returns mean accuracy plus item count.
    """
    if not traces:
        return {"accuracy": 0.0, "n_items": 0}

    accuracies = [t.scores.get("accuracy", 0.0) for t in traces]
    return {
        "accuracy": sum(accuracies) / len(accuracies),
        "n_items": float(len(traces)),
    }


# ---------------------------------------------------------------------------
# CogArenaEnv — Gymnasium-style environment for multi-turn cognitive tasks
# ---------------------------------------------------------------------------

class CogArenaEnv:
    """Gymnasium-style environment for multi-turn cognitive task episodes.

    Usage::

        env = CogArenaEnv(task_generator_fn, metadata)
        obs = env.reset(seed=42, config={"n": 2, "length": 24})
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        result = env.score()
    """

    def __init__(
        self,
        task_generator: Callable[..., List[Dict[str, Any]]],
        metadata: Optional[TaskMetadata] = None,
    ) -> None:
        """
        Args:
            task_generator: Callable(config_dict) -> list of trial dicts.
                Each trial dict must have at least ``"stimulus"`` (str) and
                ``"expected"`` (Any).  May include ``"image_path"``,
                ``"feedback_template"``, ``"system_prompt"``, ``"instructions"``.
            metadata: Optional paradigm-level metadata.
        """
        self.task_generator = task_generator
        self.metadata = metadata

        # Episode state
        self._trials: List[Dict[str, Any]] = []
        self._step_idx: int = 0
        self._done: bool = True
        self._responses: List[str] = []
        self._trial_scores: List[Dict[str, float]] = []
        self._config: Dict[str, Any] = {}
        self._seed: Optional[int] = None
        self._trace: Optional[EpisodeTrace] = None
        self._start_time: float = 0.0

    # -- Gymnasium interface ------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Reset the environment and return the first observation.

        Returns:
            Observation dict with ``stimulus``, ``step``, ``total_steps``,
            and optionally ``image_path``, ``instructions``.
        """
        self._seed = seed
        self._config = config or {}
        if seed is not None:
            import random
            random.seed(seed)
        self._trials = self.task_generator(self._config)
        if not self._trials:
            raise ValueError("task_generator returned no trials")
        self._step_idx = 0
        self._done = False
        self._responses = []
        self._trial_scores = []
        self._trace = None
        self._start_time = time.time()
        return self._make_observation()

    def step(
        self, action: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Process the model's response and advance one trial.

        Returns:
            ``(observation, reward, done, info)``
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        trial = self._trials[self._step_idx]
        self._responses.append(action)

        # Score this trial
        expected = trial.get("expected")
        scoring_cfg = trial.get("scoring")
        if scoring_cfg is None and self.metadata is not None:
            scoring_cfg = self.metadata.scoring
        if scoring_cfg is None:
            scoring_cfg = ScoringConfig(method="exact_match")
        if isinstance(scoring_cfg, dict):
            scoring_cfg = ScoringConfig(**scoring_cfg)

        trial_score = score_item(action, expected, scoring_cfg)
        self._trial_scores.append(trial_score)
        reward = trial_score.get("accuracy", 0.0)

        self._step_idx += 1
        self._done = self._step_idx >= len(self._trials)

        if self._done:
            obs = self._make_terminal_observation()
        else:
            obs = self._make_observation()
            feedback = trial.get("feedback_template")
            if feedback is not None:
                obs["feedback"] = feedback.format(
                    correct=expected, response=action, score=reward
                )

        info: Dict[str, Any] = {
            "trial_score": trial_score,
            "step": self._step_idx,
            "total_steps": len(self._trials),
        }
        return obs, reward, self._done, info

    def score(self, trace: Optional[EpisodeTrace] = None) -> Dict[str, float]:
        """Compute aggregate scores for the completed episode."""
        scores_list = self._trial_scores
        if not scores_list:
            return {"accuracy": 0.0}

        acc = sum(s.get("accuracy", 0.0) for s in scores_list) / len(scores_list)
        result: Dict[str, float] = {
            "accuracy": acc,
            "n_trials": float(len(scores_list)),
        }

        # Average any extra numeric metrics across trials
        all_keys: set[str] = set()
        for s in scores_list:
            all_keys.update(s.keys())
        all_keys.discard("accuracy")
        for k in sorted(all_keys):
            vals = [s[k] for s in scores_list if k in s]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                result[k] = sum(vals) / len(vals)

        return result

    @property
    def trace(self) -> EpisodeTrace:
        """Build an EpisodeTrace from the current episode state."""
        if self._trace is not None:
            return self._trace
        task_id = ""
        if self.metadata:
            task_id = f"{self.metadata.paradigm}_{self._seed or 0}"
        end_time = time.time()
        t = EpisodeTrace(
            task_id=task_id,
            model_id="",  # filled by LLMEvaluator
            responses=list(self._responses),
            scores=self.score() if self._done else {},
            timestamps={"start": self._start_time, "end": end_time},
        )
        if self._done:
            self._trace = t
        return t

    # -- Internal helpers ---------------------------------------------------

    def _make_observation(self) -> Dict[str, Any]:
        trial = self._trials[self._step_idx]
        obs: Dict[str, Any] = {
            "stimulus": trial["stimulus"],
            "step": self._step_idx,
            "total_steps": len(self._trials),
        }
        if trial.get("image_path"):
            obs["image_path"] = trial["image_path"]
        if trial.get("system_prompt"):
            obs["system_prompt"] = trial["system_prompt"]
        if trial.get("instructions") and self._step_idx == 0:
            obs["instructions"] = trial["instructions"]
        return obs

    def _make_terminal_observation(self) -> Dict[str, Any]:
        return {
            "stimulus": "[EPISODE COMPLETE]",
            "step": self._step_idx,
            "total_steps": len(self._trials),
            "done": True,
        }


# ---------------------------------------------------------------------------
# LLMEvaluator — run static or multi-turn evaluations
# ---------------------------------------------------------------------------

class LLMEvaluator:
    """Evaluates LLMs on CogArena tasks, producing EpisodeTraces.

    Requires an ``llm_client`` with a ``generate()`` method (see
    :class:`cogarena.llm_client.LLMClient`).
    """

    def __init__(
        self,
        llm_client: Any,
        checkpoint_manager: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        self.client = llm_client
        self.checkpoint = checkpoint_manager
        self.verbose = verbose

    # -- Single static task -------------------------------------------------

    def run_single(
        self,
        model_id: str,
        task: TaskInstance,
    ) -> EpisodeTrace:
        """Run a single static task and return the trace."""
        # Check cache
        if self.checkpoint and self.checkpoint.has_result(task.task_id, model_id):
            cached = self.checkpoint.load_result(task.task_id, model_id)
            if cached is not None:
                return EpisodeTrace.from_dict(cached)

        start = time.time()
        images = [task.image_path] if task.image_path else None
        response = self.client.generate(
            prompt=task.stimulus,
            images=images,
        )
        end = time.time()

        scores = task.score(response)

        trace = EpisodeTrace(
            task_id=task.task_id,
            model_id=model_id,
            responses=[response],
            scores=scores,
            token_counts=getattr(self.client, "last_token_counts", {}) or {},
            timestamps={"start": start, "end": end},
        )

        if self.checkpoint:
            self.checkpoint.save_result(task.task_id, model_id, trace.to_dict())

        if self.verbose:
            _log(f"[{model_id}] {task.task_id}: acc={scores.get('accuracy', '?')}")

        return trace

    # -- Multi-turn episode -------------------------------------------------

    def run_multi_turn(
        self,
        model_id: str,
        env: CogArenaEnv,
        max_steps: int = 256,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> EpisodeTrace:
        """Run a multi-turn episode in a CogArenaEnv."""
        obs = env.reset(seed=seed, config=config)
        start = time.time()

        step_system = system_prompt or obs.get("system_prompt")
        if obs.get("instructions"):
            first_prompt = obs["instructions"] + "\n\n" + obs["stimulus"]
        else:
            first_prompt = obs["stimulus"]

        done = False
        step_count = 0
        prompt = first_prompt

        while not done and step_count < max_steps:
            images = [obs["image_path"]] if obs.get("image_path") else None
            response = self.client.generate(
                prompt=prompt,
                system_prompt=step_system,
                images=images,
            )
            obs, reward, done, info = env.step(response)

            if not done:
                feedback_part = obs.get("feedback", "")
                stimulus_part = obs["stimulus"]
                prompt = (feedback_part + "\n" + stimulus_part).strip()

            step_count += 1

        end = time.time()
        trace = env.trace
        trace.model_id = model_id
        trace.timestamps = {"start": start, "end": end}
        trace.token_counts = getattr(self.client, "last_token_counts", {}) or {}

        if self.checkpoint:
            self.checkpoint.save_result(trace.task_id, model_id, trace.to_dict())

        if self.verbose:
            _log(
                f"[{model_id}] {trace.task_id}: "
                f"steps={step_count}, acc={trace.scores.get('accuracy', '?')}"
            )

        return trace

    # -- Batch evaluation ---------------------------------------------------

    def run_batch(
        self,
        model_id: str,
        tasks: List[TaskInstance],
        skip_existing: bool = True,
    ) -> List[EpisodeTrace]:
        """Run a batch of static tasks sequentially."""
        traces: List[EpisodeTrace] = []
        for i, task in enumerate(tasks):
            if skip_existing and self.checkpoint:
                if self.checkpoint.has_result(task.task_id, model_id):
                    cached = self.checkpoint.load_result(task.task_id, model_id)
                    if cached is not None:
                        traces.append(EpisodeTrace.from_dict(cached))
                        continue
            trace = self.run_single(model_id, task)
            traces.append(trace)
            if self.verbose and (i + 1) % 50 == 0:
                _log(f"  ... completed {i + 1}/{len(tasks)} tasks")
        return traces

    # -- Profile construction -----------------------------------------------

    def build_profile(
        self,
        model_id: str,
        traces: List[EpisodeTrace],
        dimension_map: Optional[Dict[str, str]] = None,
    ) -> CognitiveProfile:
        """Aggregate traces into a CognitiveProfile.

        Args:
            model_id: Model identifier.
            traces: All traces for this model.
            dimension_map: Optional mapping from paradigm name to dimension
                name.  If None, dimension scores are left empty and only
                per-paradigm raw_data is populated.
        """
        # Group by paradigm (prefix of task_id before last underscore)
        paradigm_traces: Dict[str, List[EpisodeTrace]] = {}
        for t in traces:
            paradigm = t.task_id.rsplit("_", 1)[0] if "_" in t.task_id else t.task_id
            paradigm_traces.setdefault(paradigm, []).append(t)

        raw_data: Dict[str, Any] = {}
        for paradigm, pts in paradigm_traces.items():
            accs = [t.scores.get("accuracy", 0.0) for t in pts]
            mean_acc = sum(accs) / len(accs) if accs else 0.0
            raw_data[paradigm] = {
                "mean_accuracy": mean_acc,
                "n_items": len(pts),
                "all_scores": [t.scores for t in pts],
            }

        # Roll up to dimension level if mapping is provided
        dimension_scores: Dict[str, float] = {}
        if dimension_map:
            dim_accs: Dict[str, List[float]] = {}
            for paradigm, info in raw_data.items():
                dim = dimension_map.get(paradigm)
                if dim:
                    dim_accs.setdefault(dim, []).append(info["mean_accuracy"])
            for dim, accs in dim_accs.items():
                dimension_scores[dim] = sum(accs) / len(accs)

        return CognitiveProfile(
            model_id=model_id,
            dimension_scores=dimension_scores,
            raw_data=raw_data,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[CogArena {ts}] {msg}")
