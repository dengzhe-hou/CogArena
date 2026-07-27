"""Unified LLM API client for CogArena.

Supports OpenAI-compatible APIs, Anthropic, Google GenAI, and local servers
through a single ``LLMClient.generate()`` interface.

Features:
- Retry with exponential backoff (configurable)
- Token-bucket rate limiting
- Per-response checkpointing to disk
- Configuration via env vars or dict
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Simple token-bucket rate limiter (requests per second)."""

    def __init__(self, rate: float, capacity: int) -> None:
        """
        Args:
            rate: Tokens added per second.
            capacity: Maximum burst size.
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last = time.monotonic()

    def acquire(self, tokens: int = 1) -> None:
        """Block until *tokens* are available."""
        while True:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            # Sleep for estimated wait
            deficit = tokens - self._tokens
            time.sleep(deficit / self.rate)


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified LLM client supporting multiple providers.

    Supported providers (set via ``provider`` key in config or
    ``COGARENA_LLM_PROVIDER`` env var):

    - ``"openai"`` uses OpenAI Chat Completions (GPT-4o, etc.)
    - ``"anthropic"`` uses the Anthropic Messages API (Claude)
    - ``"google"`` uses Google Generative AI (Gemini)
    - ``"local"`` uses an OpenAI-compatible server (e.g. vLLM, Ollama)

    Example::

        client = LLMClient(config={
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-...",
        })
        reply = client.generate("What is 2+2?")
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        cfg = config or {}
        self.provider: str = cfg.get(
            "provider", os.getenv("COGARENA_LLM_PROVIDER", "openai")
        )
        self.model: str = cfg.get(
            "model", os.getenv("COGARENA_LLM_MODEL", "gpt-4o")
        )
        self.api_key: str = cfg.get("api_key", "")
        self.base_url: Optional[str] = cfg.get("base_url")
        self.max_retries: int = int(cfg.get("max_retries", 3))
        self.initial_backoff: float = float(cfg.get("initial_backoff", 1.0))
        self.temperature: float = float(cfg.get("temperature", 0.0))
        self.max_tokens: int = int(cfg.get("max_tokens", 1024))

        # Rate limiter: default 10 requests/sec, burst of 20
        rate = float(cfg.get("rate_limit_rps", 10))
        burst = int(cfg.get("rate_limit_burst", 20))
        self._limiter = _TokenBucket(rate=rate, capacity=burst)

        # Checkpointing
        self._checkpoint_dir: Optional[Path] = None
        if checkpoint_dir is not None:
            self._checkpoint_dir = Path(checkpoint_dir)
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Populated after each call for the evaluator to read
        self.last_token_counts: Dict[str, int] = {}

        # Lazy-loaded provider clients
        self._openai_client: Any = None
        self._anthropic_client: Any = None
        self._google_model: Any = None

    # -- Public API ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Generate a single completion.

        Args:
            prompt: User message text.
            system_prompt: Optional system message.
            temperature: Sampling temperature (default from config).
            max_tokens: Max tokens to generate (default from config).
            images: Optional list of image file paths for VLM tasks.

        Returns:
            The model's response text.
        """
        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens

        # Rate limit
        self._limiter.acquire()

        # Retry loop
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response_text = self._dispatch(
                    prompt, system_prompt, temp, mt, images
                )
                self._save_checkpoint(prompt, response_text)
                return response_text
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = self.initial_backoff * (2 ** attempt)
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_err}"
        ) from last_err

    # -- Provider dispatch --------------------------------------------------

    def _dispatch(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        images: Optional[List[str]],
    ) -> str:
        provider = self.provider.lower()
        if provider in ("openai", "local"):
            return self._call_openai(
                prompt, system_prompt, temperature, max_tokens, images
            )
        elif provider == "anthropic":
            return self._call_anthropic(
                prompt, system_prompt, temperature, max_tokens, images
            )
        elif provider in ("google", "google-genai", "gemini"):
            return self._call_google(
                prompt, system_prompt, temperature, max_tokens, images
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    # -- OpenAI / local -----------------------------------------------------

    def _get_openai_client(self) -> Any:
        if self._openai_client is not None:
            return self._openai_client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for the 'openai' / 'local' provider. "
                "Install it with: pip install openai"
            )
        api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.provider.lower() == "local":
            base_url = self.base_url or os.getenv(
                "COGARENA_LOCAL_BASE_URL", "http://localhost:8000/v1"
            )
            kwargs["base_url"] = base_url
        elif self.base_url:
            kwargs["base_url"] = self.base_url
        self._openai_client = openai.OpenAI(**kwargs)
        return self._openai_client

    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        images: Optional[List[str]],
    ) -> str:
        client = self._get_openai_client()
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message content
        if images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_path in images:
                b64 = _encode_image_base64(img_path)
                ext = Path(img_path).suffix.lstrip(".").lower()
                mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "gif", "webp") else "image/png"
                if ext == "jpg":
                    mime = "image/jpeg"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        text = choice.message.content or ""

        # Record token usage
        usage = getattr(response, "usage", None)
        if usage:
            self.last_token_counts = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            }
        else:
            self.last_token_counts = {}

        return text

    # -- Anthropic ----------------------------------------------------------

    def _get_anthropic_client(self) -> Any:
        if self._anthropic_client is not None:
            return self._anthropic_client
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for the 'anthropic' provider. "
                "Install it with: pip install anthropic"
            )
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        images: Optional[List[str]],
    ) -> str:
        client = self._get_anthropic_client()

        # Build content blocks
        content: List[Dict[str, Any]] = []
        if images:
            for img_path in images:
                b64 = _encode_image_base64(img_path)
                ext = Path(img_path).suffix.lstrip(".").lower()
                mime = f"image/{ext}" if ext in ("png", "jpeg", "gif", "webp") else "image/png"
                if ext == "jpg":
                    mime = "image/jpeg"
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                })
        content.append({"type": "text", "text": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        usage = getattr(response, "usage", None)
        if usage:
            self.last_token_counts = {
                "prompt_tokens": getattr(usage, "input_tokens", 0),
                "completion_tokens": getattr(usage, "output_tokens", 0),
            }
        else:
            self.last_token_counts = {}

        return text

    # -- Google GenAI -------------------------------------------------------

    def _get_google_model(self) -> Any:
        if self._google_model is not None:
            return self._google_model
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for the 'google' provider. "
                "Install it with: pip install google-generativeai"
            )
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY", "")
        genai.configure(api_key=api_key)
        self._google_model = genai.GenerativeModel(self.model)
        return self._google_model

    def _call_google(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        images: Optional[List[str]],
    ) -> str:
        model = self._get_google_model()
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai is required")

        parts: List[Any] = []

        if images:
            import PIL.Image  # type: ignore
            for img_path in images:
                img = PIL.Image.open(img_path)
                parts.append(img)

        full_prompt = prompt
        if system_prompt:
            full_prompt = system_prompt + "\n\n" + prompt
        parts.append(full_prompt)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = model.generate_content(parts, generation_config=generation_config)
        text = response.text or ""

        # Google GenAI usage metadata
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.last_token_counts = {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
            }
        else:
            self.last_token_counts = {}

        return text

    # -- Checkpointing ------------------------------------------------------

    def _save_checkpoint(self, prompt: str, response: str) -> None:
        """Persist each response to disk immediately."""
        if self._checkpoint_dir is None:
            return
        h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        ts = int(time.time() * 1000)
        filename = f"{ts}_{h}.json"
        path = self._checkpoint_dir / filename
        data = {
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": h,
            "prompt_preview": prompt[:200],
            "response": response,
            "token_counts": self.last_token_counts,
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image_base64(path: str) -> str:
    """Read an image file and return its base64-encoded contents."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
