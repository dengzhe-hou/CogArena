"""CogArena: Extensible Cognitive Evaluation Platform for LLMs and Agents."""

__version__ = "0.1.0"

from .core import (
    AdaptationDistance,
    CogArenaEnv,
    CognitiveProfile,
    DifficultyLevel,
    EpisodeTrace,
    EvalMode,
    LLMEvaluator,
    ScoringConfig,
    TaskInstance,
    TaskMetadata,
    aggregate_dimension_scores,
    score_item,
)
from .llm_client import LLMClient
from .registry import (
    DimensionRegistry,
    ParadigmInfo,
    TaskRegistry,
    get_dimension,
    get_dimension_registry,
    get_paradigm,
    get_task_registry,
    list_dimensions,
    register_dimension,
    register_generator,
    register_paradigm,
)
from .utils.checkpoint import CheckpointManager
