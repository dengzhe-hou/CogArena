"""Deterministic font loading for CogArena image stimuli.

The image generators must never silently fall back to Pillow's tiny bitmap
font.  That fallback previously reduced nominal 55--60 pixel glyphs to roughly
10 pixels on hosts without system DejaVu fonts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

from PIL import ImageFont


DEJAVU_SANS_BOLD_SHA256 = (
    "b184b89e3c1075f22f6b71575b6fc20d4972b3cfd3b23322ca6fd596dcaef167"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_font_path() -> Path:
    """Return the exact DejaVu Sans Bold asset or fail closed.

    ``COGARENA_IMAGE_FONT`` may point to the same font asset in a cold-start
    environment.  Otherwise the copy distributed with Matplotlib is used.
    Both paths are accepted only when the file hash matches the frozen asset.
    """

    candidates: list[Path] = []
    override = os.environ.get("COGARENA_IMAGE_FONT")
    if override:
        candidates.append(Path(override).expanduser())

    matplotlib_spec = importlib.util.find_spec("matplotlib")
    if matplotlib_spec and matplotlib_spec.origin:
        candidates.append(
            Path(matplotlib_spec.origin).resolve().parent
            / "mpl-data"
            / "fonts"
            / "ttf"
            / "DejaVuSans-Bold.ttf"
        )

    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"),
        ]
    )

    checked: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            checked.append(f"{candidate} (missing)")
            continue
        actual = _sha256(candidate)
        if actual != DEJAVU_SANS_BOLD_SHA256:
            checked.append(f"{candidate} (sha256={actual})")
            continue
        return candidate.resolve()

    detail = "; ".join(checked) if checked else "no candidate font paths"
    raise RuntimeError(
        "CogArena requires the frozen DejaVuSans-Bold.ttf asset with sha256 "
        f"{DEJAVU_SANS_BOLD_SHA256}; checked {detail}"
    )


def load_frozen_font(size: int) -> ImageFont.FreeTypeFont:
    """Load the frozen font at ``size`` without any fallback."""

    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"font size must be a positive integer, got {size!r}")
    return ImageFont.truetype(str(resolve_font_path()), size)


def font_provenance() -> dict[str, object]:
    """Return portable provenance for the frozen font asset."""

    path = resolve_font_path()
    return {
        "family": "DejaVu Sans Bold",
        "filename": path.name,
        "sha256": _sha256(path),
        "source": "matplotlib-or-system asset verified by content hash",
    }
