"""Task and dimension registry for CogArena.

Provides:
- :class:`DimensionRegistry` — register cognitive dimensions and their paradigms
- :class:`TaskRegistry` — register task generators, load from YAML configs
- Module-level convenience functions for a shared default registry
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .core import (
    AdaptationDistance,
    DifficultyLevel,
    EvalMode,
    ScoringConfig,
    TaskInstance,
    TaskMetadata,
)


# ---------------------------------------------------------------------------
# Data models for the registry
# ---------------------------------------------------------------------------

@dataclass
class ParadigmInfo:
    """Metadata about a single cognitive paradigm."""
    name: str                               # e.g. "n_back"
    dimension: str                          # e.g. "working_memory"
    description: str = ""
    mode: EvalMode = EvalMode.LLM_STATIC
    adaptation_distance: AdaptationDistance = AdaptationDistance.LOW
    human_anchor: Optional[Dict[str, Any]] = None
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    scoring: ScoringConfig = field(
        default_factory=lambda: ScoringConfig(method="exact_match")
    )
    tags: List[str] = field(default_factory=list)
    multi_turn: bool = False

    def to_metadata(self, **overrides: Any) -> TaskMetadata:
        """Create a TaskMetadata from this paradigm info, with optional overrides."""
        kwargs: Dict[str, Any] = {
            "dimension": self.dimension,
            "paradigm": self.name,
            "mode": self.mode,
            "parameters": dict(self.default_parameters),
            "scoring": self.scoring,
            "adaptation_distance": self.adaptation_distance,
            "human_anchor": self.human_anchor,
            "tags": list(self.tags),
        }
        kwargs.update(overrides)
        return TaskMetadata(**kwargs)


@dataclass
class DimensionInfo:
    """Metadata about a cognitive dimension."""
    name: str                   # e.g. "working_memory"
    display_name: str = ""      # e.g. "Working Memory"
    theory: str = ""            # e.g. "CHC: Gwm; Miyake: Updating"
    description: str = ""
    paradigms: Dict[str, ParadigmInfo] = field(default_factory=dict)

    def add_paradigm(self, paradigm: ParadigmInfo) -> None:
        self.paradigms[paradigm.name] = paradigm

    def list_paradigms(self) -> List[str]:
        return list(self.paradigms.keys())


# ---------------------------------------------------------------------------
# DimensionRegistry
# ---------------------------------------------------------------------------

class DimensionRegistry:
    """Registry for cognitive dimensions and their paradigms."""

    def __init__(self) -> None:
        self._dimensions: Dict[str, DimensionInfo] = {}

    def register_dimension(
        self,
        name: str,
        display_name: str = "",
        theory: str = "",
        description: str = "",
    ) -> DimensionInfo:
        """Register a new cognitive dimension.

        Returns the DimensionInfo (creates it if it does not exist).
        """
        if name in self._dimensions:
            dim = self._dimensions[name]
            if display_name:
                dim.display_name = display_name
            if theory:
                dim.theory = theory
            if description:
                dim.description = description
            return dim
        dim = DimensionInfo(
            name=name,
            display_name=display_name or name.replace("_", " ").title(),
            theory=theory,
            description=description,
        )
        self._dimensions[name] = dim
        return dim

    def register_paradigm(
        self,
        dimension: str,
        paradigm: ParadigmInfo,
    ) -> None:
        """Register a paradigm under a dimension (creates dimension if needed)."""
        if dimension not in self._dimensions:
            self.register_dimension(dimension)
        paradigm.dimension = dimension
        self._dimensions[dimension].add_paradigm(paradigm)

    def get_dimension(self, name: str) -> Optional[DimensionInfo]:
        """Look up a dimension by name."""
        return self._dimensions.get(name)

    def list_dimensions(self) -> List[str]:
        """Return all registered dimension names."""
        return list(self._dimensions.keys())

    def get_paradigm(
        self, dimension: str, paradigm: str
    ) -> Optional[ParadigmInfo]:
        """Look up a specific paradigm under a dimension."""
        dim = self._dimensions.get(dimension)
        if dim is None:
            return None
        return dim.paradigms.get(paradigm)

    def list_paradigms(self, dimension: str) -> List[str]:
        """List all paradigm names under a dimension."""
        dim = self._dimensions.get(dimension)
        if dim is None:
            return []
        return dim.list_paradigms()

    def get_all_paradigms(self) -> List[ParadigmInfo]:
        """Return all registered paradigms across all dimensions."""
        result: List[ParadigmInfo] = []
        for dim in self._dimensions.values():
            result.extend(dim.paradigms.values())
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full registry to a dict."""
        out: Dict[str, Any] = {}
        for name, dim in self._dimensions.items():
            out[name] = {
                "display_name": dim.display_name,
                "theory": dim.theory,
                "description": dim.description,
                "paradigms": {
                    p.name: {
                        "description": p.description,
                        "mode": p.mode.value,
                        "adaptation_distance": p.adaptation_distance.value,
                        "multi_turn": p.multi_turn,
                        "default_parameters": p.default_parameters,
                        "tags": p.tags,
                    }
                    for p in dim.paradigms.values()
                },
            }
        return out


# ---------------------------------------------------------------------------
# TaskRegistry
# ---------------------------------------------------------------------------

# Type alias for a task generator function: (metadata, config) -> list[TaskInstance]
TaskGeneratorFn = Callable[[TaskMetadata, Dict[str, Any]], List[TaskInstance]]


class TaskRegistry:
    """Registry for task generator functions.

    Each paradigm can have a registered generator that produces
    ``TaskInstance`` objects programmatically (needed to avoid contamination).

    Generators can also be loaded from YAML configuration files.
    """

    def __init__(self, dimension_registry: Optional[DimensionRegistry] = None) -> None:
        self._generators: Dict[str, TaskGeneratorFn] = {}
        self._dim_registry = dimension_registry or DimensionRegistry()

    @property
    def dimension_registry(self) -> DimensionRegistry:
        return self._dim_registry

    # -- Registration -------------------------------------------------------

    def register(
        self,
        paradigm: str,
        generator: TaskGeneratorFn,
        dimension: Optional[str] = None,
        paradigm_info: Optional[ParadigmInfo] = None,
    ) -> None:
        """Register a task generator for a paradigm.

        Args:
            paradigm: Paradigm name (e.g. ``"n_back"``).
            generator: Callable ``(metadata, config) -> [TaskInstance, ...]``.
            dimension: If provided, also register the paradigm under this
                dimension in the DimensionRegistry.
            paradigm_info: Optional ParadigmInfo to register alongside.
        """
        self._generators[paradigm] = generator
        if paradigm_info and dimension:
            self._dim_registry.register_paradigm(dimension, paradigm_info)

    def register_decorator(
        self,
        paradigm: str,
        dimension: Optional[str] = None,
    ) -> Callable[[TaskGeneratorFn], TaskGeneratorFn]:
        """Decorator form of :meth:`register`.

        Usage::

            @task_registry.register_decorator("n_back", "working_memory")
            def generate_nback(metadata, config):
                ...
        """
        def decorator(fn: TaskGeneratorFn) -> TaskGeneratorFn:
            self.register(paradigm, fn, dimension=dimension)
            return fn
        return decorator

    # -- Generation ---------------------------------------------------------

    def generate(
        self,
        paradigm: str,
        config: Optional[Dict[str, Any]] = None,
        metadata_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[TaskInstance]:
        """Generate task instances for a paradigm.

        Args:
            paradigm: Registered paradigm name.
            config: Generation parameters (e.g. ``{"n_items": 100}``).
            metadata_overrides: Override fields on the TaskMetadata.

        Returns:
            List of TaskInstance objects.
        """
        if paradigm not in self._generators:
            raise KeyError(
                f"No generator registered for paradigm '{paradigm}'. "
                f"Available: {list(self._generators.keys())}"
            )

        # Build metadata from dimension registry if available
        pinfo = self._find_paradigm_info(paradigm)
        if pinfo is not None:
            metadata = pinfo.to_metadata(**(metadata_overrides or {}))
        else:
            metadata = TaskMetadata(
                dimension="_unknown",
                paradigm=paradigm,
                mode=EvalMode.LLM_STATIC,
                **(metadata_overrides or {}),
            )

        generator = self._generators[paradigm]
        return generator(metadata, config or {})

    def has_generator(self, paradigm: str) -> bool:
        return paradigm in self._generators

    def list_generators(self) -> List[str]:
        return list(self._generators.keys())

    # -- YAML loading -------------------------------------------------------

    def load_from_yaml(self, path: str | Path) -> None:
        """Load dimension and paradigm definitions from a YAML config file.

        Expected YAML structure::

            dimensions:
              working_memory:
                display_name: "Working Memory"
                theory: "CHC: Gwm; Miyake: Updating"
                paradigms:
                  n_back:
                    description: "..."
                    mode: "agent_interactive"
                    adaptation_distance: "low"
                    multi_turn: true
                    default_parameters:
                      n: 2
                      length: 24
                    scoring:
                      method: "exact_match"
                    tags: ["wm", "updating"]
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configs. "
                "Install with: pip install pyyaml"
            )
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a dict, got {type(data)}")

        dims_data = data.get("dimensions", data)
        for dim_name, dim_cfg in dims_data.items():
            if not isinstance(dim_cfg, dict):
                continue
            self._dim_registry.register_dimension(
                name=dim_name,
                display_name=dim_cfg.get("display_name", ""),
                theory=dim_cfg.get("theory", ""),
                description=dim_cfg.get("description", ""),
            )
            paradigms = dim_cfg.get("paradigms", {})
            for para_name, para_cfg in paradigms.items():
                if not isinstance(para_cfg, dict):
                    continue
                scoring_raw = para_cfg.get("scoring", {})
                if isinstance(scoring_raw, dict):
                    scoring = ScoringConfig(
                        method=scoring_raw.get("method", "exact_match"),
                        params=scoring_raw.get("params", {}),
                    )
                else:
                    scoring = ScoringConfig(method="exact_match")

                mode_str = para_cfg.get("mode", "llm_static")
                try:
                    mode = EvalMode(mode_str)
                except ValueError:
                    mode = EvalMode.LLM_STATIC

                ad_str = para_cfg.get("adaptation_distance", "low")
                try:
                    ad = AdaptationDistance(ad_str)
                except ValueError:
                    ad = AdaptationDistance.LOW

                pinfo = ParadigmInfo(
                    name=para_name,
                    dimension=dim_name,
                    description=para_cfg.get("description", ""),
                    mode=mode,
                    adaptation_distance=ad,
                    human_anchor=para_cfg.get("human_anchor"),
                    default_parameters=para_cfg.get("default_parameters", {}),
                    scoring=scoring,
                    tags=para_cfg.get("tags", []),
                    multi_turn=para_cfg.get("multi_turn", False),
                )
                self._dim_registry.register_paradigm(dim_name, pinfo)

    def load_from_json(self, path: str | Path) -> None:
        """Load from a JSON config (same structure as YAML)."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Re-use the YAML loader logic after parsing
        self._load_from_dict(data)

    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """Internal: load from a parsed dict (shared by YAML and JSON loaders)."""
        dims_data = data.get("dimensions", data)
        for dim_name, dim_cfg in dims_data.items():
            if not isinstance(dim_cfg, dict):
                continue
            self._dim_registry.register_dimension(
                name=dim_name,
                display_name=dim_cfg.get("display_name", ""),
                theory=dim_cfg.get("theory", ""),
                description=dim_cfg.get("description", ""),
            )
            paradigms = dim_cfg.get("paradigms", {})
            for para_name, para_cfg in paradigms.items():
                if not isinstance(para_cfg, dict):
                    continue
                scoring_raw = para_cfg.get("scoring", {})
                if isinstance(scoring_raw, dict):
                    scoring = ScoringConfig(
                        method=scoring_raw.get("method", "exact_match"),
                        params=scoring_raw.get("params", {}),
                    )
                else:
                    scoring = ScoringConfig(method="exact_match")

                mode_str = para_cfg.get("mode", "llm_static")
                try:
                    mode = EvalMode(mode_str)
                except ValueError:
                    mode = EvalMode.LLM_STATIC

                ad_str = para_cfg.get("adaptation_distance", "low")
                try:
                    ad = AdaptationDistance(ad_str)
                except ValueError:
                    ad = AdaptationDistance.LOW

                pinfo = ParadigmInfo(
                    name=para_name,
                    dimension=dim_name,
                    description=para_cfg.get("description", ""),
                    mode=mode,
                    adaptation_distance=ad,
                    human_anchor=para_cfg.get("human_anchor"),
                    default_parameters=para_cfg.get("default_parameters", {}),
                    scoring=scoring,
                    tags=para_cfg.get("tags", []),
                    multi_turn=para_cfg.get("multi_turn", False),
                )
                self._dim_registry.register_paradigm(dim_name, pinfo)

    # -- Helpers ------------------------------------------------------------

    def _find_paradigm_info(self, paradigm: str) -> Optional[ParadigmInfo]:
        """Search all dimensions for a paradigm by name."""
        for dim in self._dim_registry._dimensions.values():
            if paradigm in dim.paradigms:
                return dim.paradigms[paradigm]
        return None


# ---------------------------------------------------------------------------
# Module-level default registry (singleton)
# ---------------------------------------------------------------------------

_default_dim_registry = DimensionRegistry()
_default_task_registry = TaskRegistry(dimension_registry=_default_dim_registry)


def get_dimension_registry() -> DimensionRegistry:
    """Return the module-level default DimensionRegistry."""
    return _default_dim_registry


def get_task_registry() -> TaskRegistry:
    """Return the module-level default TaskRegistry."""
    return _default_task_registry


# Convenience wrappers

def register_dimension(
    name: str, display_name: str = "", theory: str = "", description: str = ""
) -> DimensionInfo:
    return _default_dim_registry.register_dimension(name, display_name, theory, description)


def register_paradigm(dimension: str, paradigm: ParadigmInfo) -> None:
    _default_dim_registry.register_paradigm(dimension, paradigm)


def get_dimension(name: str) -> Optional[DimensionInfo]:
    return _default_dim_registry.get_dimension(name)


def list_dimensions() -> List[str]:
    return _default_dim_registry.list_dimensions()


def get_paradigm(dimension: str, paradigm: str) -> Optional[ParadigmInfo]:
    return _default_dim_registry.get_paradigm(dimension, paradigm)


def register_generator(
    paradigm: str,
    generator: TaskGeneratorFn,
    dimension: Optional[str] = None,
) -> None:
    _default_task_registry.register(paradigm, generator, dimension=dimension)


def generate_tasks(
    paradigm: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[TaskInstance]:
    return _default_task_registry.generate(paradigm, config)


# ---------------------------------------------------------------------------
# Pre-register CogArena v1 dimensions (metadata only, no generators yet)
# ---------------------------------------------------------------------------

def _register_v1_dimensions() -> None:
    """Register the six core CogArena v1 dimensions."""
    _dims = [
        ("working_memory", "Working Memory", "CHC: Gwm; Miyake: Updating"),
        ("cognitive_control", "Cognitive Control / Inhibition", "Miyake: Inhibition"),
        ("set_shifting", "Set Shifting / Cognitive Flexibility", "Miyake: Shifting"),
        ("episodic_memory", "Episodic Memory", "CHC: Glr"),
        ("theory_of_mind", "Theory of Mind", "Social cognition"),
        ("metacognition", "Metacognitive Monitoring", "Nelson & Narens (1990)"),
    ]
    for name, display, theory in _dims:
        _default_dim_registry.register_dimension(name, display, theory)

    # Working Memory paradigms
    _wm_paradigms = [
        ParadigmInfo(
            name="n_back",
            dimension="working_memory",
            description="N-back task: identify items matching N positions back",
            mode=EvalMode.AGENT_INTERACTIVE,
            adaptation_distance=AdaptationDistance.LOW,
            multi_turn=True,
            default_parameters={"n": 2, "length": 24, "alphabet": "ABCDEFGHIJ"},
            scoring=ScoringConfig(method="exact_match"),
            tags=["wm", "updating"],
        ),
        ParadigmInfo(
            name="digit_span",
            dimension="working_memory",
            description="Digit span (forward and backward): recall digit sequences",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={"min_length": 3, "max_length": 12, "direction": "forward"},
            scoring=ScoringConfig(method="exact_match"),
            tags=["wm", "capacity"],
        ),
        ParadigmInfo(
            name="operation_span",
            dimension="working_memory",
            description="Operation span: solve math while remembering letters",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.MEDIUM,
            default_parameters={"set_size_range": [2, 7]},
            scoring=ScoringConfig(method="list_recall"),
            tags=["wm", "complex_span"],
        ),
    ]
    for p in _wm_paradigms:
        _default_dim_registry.register_paradigm("working_memory", p)

    # Cognitive Control paradigms
    _cc_paradigms = [
        ParadigmInfo(
            name="stroop",
            dimension="cognitive_control",
            description="Stroop task: name the ink colour, ignore the word",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.MEDIUM,
            default_parameters={"conditions": ["congruent", "incongruent", "neutral"]},
            scoring=ScoringConfig(method="exact_match"),
            tags=["inhibition", "interference"],
        ),
        ParadigmInfo(
            name="flanker",
            dimension="cognitive_control",
            description="Flanker task: identify the central target amid distractors",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={"conditions": ["congruent", "incongruent"]},
            scoring=ScoringConfig(method="exact_match"),
            tags=["attention", "selective"],
        ),
        ParadigmInfo(
            name="go_nogo",
            dimension="cognitive_control",
            description="Go/No-Go task: respond to Go stimuli, withhold on No-Go",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={"go_ratio": 0.75},
            scoring=ScoringConfig(method="exact_match"),
            tags=["inhibition", "response"],
        ),
    ]
    for p in _cc_paradigms:
        _default_dim_registry.register_paradigm("cognitive_control", p)

    # Set Shifting paradigms
    _ss_paradigms = [
        ParadigmInfo(
            name="wcst",
            dimension="set_shifting",
            description="Wisconsin Card Sorting Test: discover and shift sorting rules",
            mode=EvalMode.AGENT_INTERACTIVE,
            adaptation_distance=AdaptationDistance.MEDIUM,
            multi_turn=True,
            default_parameters={"n_trials": 64, "rule_shift_every": 10},
            scoring=ScoringConfig(method="exact_match"),
            tags=["shifting", "flexibility"],
        ),
        ParadigmInfo(
            name="reversal_learning",
            dimension="set_shifting",
            description="Reversal learning: learn then adapt to reversed reward contingencies",
            mode=EvalMode.AGENT_INTERACTIVE,
            adaptation_distance=AdaptationDistance.LOW,
            multi_turn=True,
            default_parameters={"n_trials": 30, "reversal_after": 15},
            scoring=ScoringConfig(method="exact_match"),
            tags=["shifting", "reward"],
        ),
    ]
    for p in _ss_paradigms:
        _default_dim_registry.register_paradigm("set_shifting", p)

    # Episodic Memory paradigms
    _em_paradigms = [
        ParadigmInfo(
            name="cvlt",
            dimension="episodic_memory",
            description="California Verbal Learning Test style: word list learning and recall",
            mode=EvalMode.AGENT_INTERACTIVE,
            adaptation_distance=AdaptationDistance.LOW,
            multi_turn=True,
            default_parameters={"list_length": 16, "learning_trials": 5},
            scoring=ScoringConfig(method="list_recall"),
            tags=["memory", "verbal_learning"],
        ),
        ParadigmInfo(
            name="drm",
            dimension="episodic_memory",
            description="DRM false memory paradigm: detect false recall of critical lures",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={"list_length": 12},
            scoring=ScoringConfig(method="list_recall"),
            tags=["memory", "false_memory"],
        ),
        ParadigmInfo(
            name="source_monitoring",
            dimension="episodic_memory",
            description="Source monitoring: attribute information to correct source",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={},
            scoring=ScoringConfig(method="exact_match"),
            tags=["memory", "source_attribution"],
        ),
    ]
    for p in _em_paradigms:
        _default_dim_registry.register_paradigm("episodic_memory", p)

    # Theory of Mind paradigms
    _tom_paradigms = [
        ParadigmInfo(
            name="sally_anne",
            dimension="theory_of_mind",
            description="Sally-Anne false belief task variants",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={"order": 1},
            scoring=ScoringConfig(method="exact_match"),
            tags=["tom", "false_belief"],
        ),
        ParadigmInfo(
            name="epitome",
            dimension="theory_of_mind",
            description="EPITOME: multi-dimensional Theory of Mind assessment",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={},
            scoring=ScoringConfig(method="partial_match"),
            tags=["tom", "multi_aspect"],
        ),
    ]
    for p in _tom_paradigms:
        _default_dim_registry.register_paradigm("theory_of_mind", p)

    # Metacognition paradigms
    _meta_paradigms = [
        ParadigmInfo(
            name="confidence_calibration",
            dimension="metacognition",
            description="Confidence calibration: answer + confidence rating",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={"scale": "0-100"},
            scoring=ScoringConfig(method="custom", params={"fn": "cogarena.scoring.metacognition.score_calibration"}),
            tags=["metacognition", "monitoring"],
        ),
        ParadigmInfo(
            name="post_decision_wagering",
            dimension="metacognition",
            description="Post-decision wagering: bet on correctness of own answer",
            mode=EvalMode.LLM_STATIC,
            adaptation_distance=AdaptationDistance.LOW,
            default_parameters={"wager_options": [1, 2, 5, 10]},
            scoring=ScoringConfig(method="custom", params={"fn": "cogarena.scoring.metacognition.score_wagering"}),
            tags=["metacognition", "control"],
        ),
    ]
    for p in _meta_paradigms:
        _default_dim_registry.register_paradigm("metacognition", p)


# Run on import
_register_v1_dimensions()
