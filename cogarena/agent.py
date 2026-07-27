"""ReAct-style agent scaffold for CogArena.

Wraps an LLM with a think-act-observe loop and simple tools (memory,
arithmetic, note-taking) so it can interact with :class:`CogArenaEnv`
via the Gymnasium API.

Usage::

    from cogarena.llm_client import LLMClient
    from cogarena.core import CogArenaEnv
    from cogarena.agent import CogArenaAgent

    client = LLMClient(config={"provider": "local", "model": "llama3"})
    agent = CogArenaAgent(client)

    env = CogArenaEnv(task_generator, metadata)
    obs = env.reset(seed=42)
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
    trace = env.trace
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class Tool(ABC):
    """Base class for agent tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, args: str, context: Dict[str, Any]) -> str:
        """Run the tool and return a string result.

        Args:
            args: Raw argument string from the LLM output.
            context: Shared mutable state (contains ``scratchpad``, etc.).
        """


class MemoryStore(Tool):
    """Store a key-value pair for later recall.

    Format: ``memory_store key=<key> value=<value>``
    """

    name = "memory_store"
    description = "memory_store key=<key> value=<value>: Store information for later"

    def execute(self, args: str, context: Dict[str, Any]) -> str:
        key, value = _parse_kv(args, "key", "value")
        if key is None:
            return "ERROR: must provide key=... value=..."
        context.setdefault("scratchpad", {})[key] = value
        return f"Stored: {key} = {value}"


class MemoryRecall(Tool):
    """Recall a previously stored value.

    Format: ``memory_recall key=<key>``
    """

    name = "memory_recall"
    description = "memory_recall key=<key>: Recall stored information"

    def execute(self, args: str, context: Dict[str, Any]) -> str:
        key, _ = _parse_kv(args, "key")
        if key is None:
            # Fallback: treat entire args as the key
            key = args.strip()
        scratchpad = context.get("scratchpad", {})
        if key in scratchpad:
            return str(scratchpad[key])
        return f"(nothing stored under '{key}')"


class Calculate(Tool):
    """Evaluate a simple arithmetic expression.

    Format: ``calculate <expression>``
    Supports ``+  -  *  /  //  %  **`` and parentheses on integers/floats.
    """

    name = "calculate"
    description = "calculate <expression>: Compute arithmetic"

    _ALLOWED = re.compile(r"^[\d\s\+\-\*\/\.\(\)\%]+$")

    def execute(self, args: str, context: Dict[str, Any]) -> str:
        expr = args.strip()
        if not expr:
            return "ERROR: empty expression"
        if not self._ALLOWED.match(expr):
            return f"ERROR: unsafe expression '{expr}'"
        try:
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
            return str(result)
        except Exception as exc:
            return f"ERROR: {exc}"


class NoteTake(Tool):
    """Append a note to the scratchpad notes list.

    Format: ``note text=<text>``
    """

    name = "note"
    description = "note text=<text>: Take a note"

    def execute(self, args: str, context: Dict[str, Any]) -> str:
        _, text = _parse_kv(args, "text")
        if text is None:
            # Fallback: treat entire args as the note
            text = args.replace("text=", "").strip()
        if not text:
            return "ERROR: empty note"
        context.setdefault("notes", []).append(text)
        return f"Note recorded ({len(context['notes'])} total)"


def default_tools() -> List[Tool]:
    """Return the default tool set."""
    return [MemoryStore(), MemoryRecall(), Calculate(), NoteTake()]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an agent solving a cognitive task. You SHOULD think step by step and use tools to help you.

Available tools:
{tool_descriptions}

Format your response EXACTLY as follows:
THINK: <your reasoning about the current observation>
TOOL: <tool_name> <args>  (use tools to store/recall information!)
ANSWER: <your final response to the task>

Rules:
- You MUST end with an ANSWER line containing your response.
- Use THINK to reason about what you observe.
- Use TOOL memory_store to save important information (e.g., which rules worked, what you learned from feedback).
- Use TOOL memory_recall to retrieve previously stored information.
- Use TOOL calculate for any arithmetic.
- Only the text after "ANSWER: " is sent to the environment.
- Keep your ANSWER short. Use a number, word, or short phrase when possible.
"""

_TURN_TEMPLATE = """\
Current observation (step {step}/{total_steps}):
{stimulus}

{context_block}\
Respond with THINK/TOOL/ANSWER as described."""


def _build_system_prompt(tools: List[Tool]) -> str:
    descs = "\n".join(f"- {t.description}" for t in tools)
    return _SYSTEM_PROMPT.format(tool_descriptions=descs)


def _build_context_block(
    memory: List[Dict[str, str]],
    scratchpad: Dict[str, Any],
    notes: List[str],
    max_history: int = 5,
) -> str:
    """Build the context section injected into the user prompt."""
    parts: List[str] = []

    # Recent conversation history
    recent = memory[-max_history:] if memory else []
    if recent:
        history_lines = []
        for turn in recent:
            history_lines.append(f"  Obs: {_truncate(turn.get('obs', ''), 120)}")
            history_lines.append(f"  You: {_truncate(turn.get('answer', ''), 120)}")
            if turn.get("reward") is not None:
                history_lines.append(f"  Reward: {turn['reward']}")
        parts.append("Previous turns:\n" + "\n".join(history_lines))

    # Scratchpad contents
    if scratchpad:
        kvs = ", ".join(f"{k}={v}" for k, v in scratchpad.items())
        parts.append(f"Stored memory: {kvs}")

    # Notes
    if notes:
        parts.append("Notes: " + "; ".join(notes[-5:]))

    if parts:
        return "\n".join(parts) + "\n\n"
    return ""


# ---------------------------------------------------------------------------
# CogArenaAgent
# ---------------------------------------------------------------------------

class CogArenaAgent:
    """ReAct-style agent for CogArena cognitive tasks.

    Wraps an :class:`~cogarena.llm_client.LLMClient` with a
    think-action-observation loop and simple tools.

    Args:
        llm_client: An ``LLMClient`` instance (any provider).
        tools: List of :class:`Tool` instances.  Defaults to
            ``[MemoryStore, MemoryRecall, Calculate, NoteTake]``.
        max_think_steps: Maximum tool-use iterations per ``act()`` call
            before forcing an answer.
        max_history: Number of past turns to include in the prompt context.
        verbose: If True, print debug info to stdout.
    """

    def __init__(
        self,
        llm_client: Any,
        tools: Optional[List[Tool]] = None,
        max_think_steps: int = 3,
        max_history: int = 5,
        verbose: bool = False,
    ) -> None:
        self.client = llm_client
        self.tools = tools if tools is not None else default_tools()
        self.max_think_steps = max_think_steps
        self.max_history = max_history
        self.verbose = verbose

        # Per-episode state
        self.memory: List[Dict[str, str]] = []
        self.scratchpad: Dict[str, Any] = {}
        self.notes: List[str] = []
        self.tool_call_count: int = 0

        # Build tool lookup
        self._tool_map: Dict[str, Tool] = {t.name: t for t in self.tools}
        self._system_prompt = _build_system_prompt(self.tools)

    # -- Public API ---------------------------------------------------------

    def act(self, observation: Dict[str, Any]) -> str:
        """Given an observation from CogArenaEnv, return an action string.

        Runs a ReAct loop: prompt the LLM, parse THINK/TOOL/ANSWER lines,
        execute any tool calls, and re-prompt until an ANSWER is produced
        or ``max_think_steps`` is reached.

        Args:
            observation: Dict with at least ``stimulus``, ``step``,
                ``total_steps``.  May also contain ``feedback``,
                ``instructions``, ``image_path``.

        Returns:
            The answer string to pass to ``env.step()``.
        """
        stimulus = observation.get("stimulus", "")
        step = observation.get("step", 0)
        total = observation.get("total_steps", 1)

        # Prepend instructions if present (first step)
        if observation.get("instructions"):
            stimulus = observation["instructions"] + "\n\n" + stimulus
        # Prepend feedback from previous step if present
        if observation.get("feedback"):
            stimulus = observation["feedback"] + "\n\n" + stimulus

        context_block = _build_context_block(
            self.memory, self.scratchpad, self.notes, self.max_history
        )
        user_prompt = _TURN_TEMPLATE.format(
            step=step + 1,
            total_steps=total,
            stimulus=stimulus,
            context_block=context_block,
        )

        images = [observation["image_path"]] if observation.get("image_path") else None
        tool_context = {
            "scratchpad": self.scratchpad,
            "notes": self.notes,
        }

        # ReAct loop
        answer: Optional[str] = None
        accumulated_tool_results: List[str] = []

        for think_step in range(self.max_think_steps):
            # Append tool results from previous iteration
            if accumulated_tool_results:
                user_prompt += "\n\nTool results:\n" + "\n".join(accumulated_tool_results)
                user_prompt += "\n\nContinue with THINK/TOOL/ANSWER."
                accumulated_tool_results = []

            raw = self.client.generate(
                prompt=user_prompt,
                system_prompt=self._system_prompt,
                images=images,
            )

            if self.verbose:
                print(f"[Agent step {think_step}] Raw LLM output:\n{raw}\n")

            # Parse response
            parsed = _parse_react(raw)
            answer = parsed.get("answer")
            tool_calls = parsed.get("tools", [])

            # Execute tool calls FIRST, even if answer is also present
            if tool_calls:
                for tool_name, tool_args in tool_calls:
                    tool = self._tool_map.get(tool_name)
                    if tool is not None:
                        result = tool.execute(tool_args, tool_context)
                        accumulated_tool_results.append(f"  {tool_name}: {result}")
                        self.tool_call_count += 1
                        if self.verbose:
                            print(f"  [Tool] {tool_name}({tool_args}) -> {result}")
                    else:
                        accumulated_tool_results.append(
                            f"  {tool_name}: ERROR: unknown tool"
                        )

            # If we got an answer (with or without tools), we're done
            if answer is not None:
                break

            # If no tools and no answer, use the raw response
            if not tool_calls:
                answer = raw.strip()
                break

        # Fallback if no answer was produced
        if answer is None:
            # Last resort: re-prompt asking for just the answer
            raw = self.client.generate(
                prompt=user_prompt + "\n\nYou must respond with ANSWER: <response> now.",
                system_prompt=self._system_prompt,
                images=images,
            )
            parsed = _parse_react(raw)
            answer = parsed.get("answer") or raw.strip()

        # Record this turn
        self.memory.append({
            "obs": stimulus[:200],
            "answer": answer,
        })

        return answer

    def record_reward(self, reward: float) -> None:
        """Attach a reward to the most recent turn (call after env.step)."""
        if self.memory:
            self.memory[-1]["reward"] = str(reward)

    def reset(self) -> None:
        """Clear all per-episode state for a new episode."""
        self.memory.clear()
        self.scratchpad.clear()
        self.notes.clear()
        self.tool_call_count = 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"^THINK:\s*(.+)", re.MULTILINE)
_TOOL_RE = re.compile(r"^TOOL:\s*(\S+)\s*(.*)", re.MULTILINE)
_ANSWER_RE = re.compile(r"^ANSWER:\s*(.+)", re.MULTILINE)


def _parse_react(raw: str) -> Dict[str, Any]:
    """Parse a ReAct-formatted LLM response.

    Returns a dict with optional keys:
        - ``think``: list of reasoning strings
        - ``tools``: list of ``(tool_name, args_str)`` tuples
        - ``answer``: final answer string (or None)
    """
    result: Dict[str, Any] = {}

    thinks = _THINK_RE.findall(raw)
    if thinks:
        result["think"] = thinks

    tools = _TOOL_RE.findall(raw)
    if tools:
        result["tools"] = [(name.strip(), args.strip()) for name, args in tools]

    answer_match = _ANSWER_RE.findall(raw)
    if answer_match:
        # Use the last ANSWER line (in case the model wrote multiple)
        result["answer"] = answer_match[-1].strip()

    return result


def _parse_kv(args: str, *keys: str):
    """Parse ``key=value`` pairs from an argument string.

    Returns a tuple of values in the order of *keys*.
    Missing keys yield None.
    """
    results = []
    for key in keys:
        pattern = re.compile(rf"{key}=(\S+(?:\s+(?!\S+=)\S+)*)")
        match = pattern.search(args)
        results.append(match.group(1).strip() if match else None)
    if len(results) == 1:
        return results[0], None
    return tuple(results)


def _truncate(s: str, max_len: int = 120) -> str:
    """Truncate a string for display."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."
