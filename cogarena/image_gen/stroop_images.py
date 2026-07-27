"""Generate Stroop task images: colored text for VLM evaluation.

Creates images where a color word (e.g., "RED") is rendered in a
different ink color (e.g., blue). The VLM must report the ink color,
not the word meaning.
"""

import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

from PIL import Image, ImageDraw

from .font_utils import load_frozen_font


# Color name → RGB
COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 50, 220),
    "green": (50, 180, 50),
    "yellow": (200, 200, 0),
    "purple": (150, 50, 200),
    "orange": (240, 150, 30),
    "pink": (240, 100, 150),
    "brown": (140, 80, 30),
}

COLOR_NAMES = list(COLORS.keys())

# Image settings
WIDTH, HEIGHT = 400, 150
BG_COLOR = (255, 255, 255)
FONT_SIZE = 60


def _get_font(size: int = FONT_SIZE):
    """Load the frozen font and fail rather than rendering tiny fallback text."""
    return load_frozen_font(size)


def generate_stroop_image(
    word: str,
    ink_color_name: str,
    out_path: str,
) -> str:
    """Generate a single Stroop image.

    Args:
        word: The color word to display (e.g., "RED")
        ink_color_name: The actual ink color (e.g., "blue")
        out_path: Where to save the PNG

    Returns:
        Path to the saved image
    """
    ink_rgb = COLORS[ink_color_name]
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = _get_font()

    # Center the text
    bbox = draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if tw < 2 * FONT_SIZE or th < int(0.65 * FONT_SIZE):
        raise RuntimeError(
            f"Stroop glyph geometry is too small: {tw}x{th} for size {FONT_SIZE}"
        )
    if tw > WIDTH or th > HEIGHT:
        raise RuntimeError(f"Stroop glyph geometry overflows canvas: {tw}x{th}")
    x = (WIDTH - tw) // 2
    y = (HEIGHT - th) // 2
    draw.text((x, y), word, fill=ink_rgb, font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def generate_stroop_set(
    seed: int = 42,
    n_congruent: int = 10,
    n_incongruent: int = 10,
    out_dir: str = "data/images/stroop",
) -> List[Dict[str, Any]]:
    """Generate a full set of Stroop image trials.

    Returns list of dicts with: image_path, word, ink_color, congruent,
    expected_response, stimulus_text (prompt for VLM).
    """
    rng = random.Random(seed)
    trials = []
    out_dir = Path(out_dir)
    idx = 0

    # Congruent trials: word matches ink color
    for _ in range(n_congruent):
        color = rng.choice(COLOR_NAMES)
        word = color.upper()
        fname = f"stroop_cong_{idx:03d}.png"
        img_path = generate_stroop_image(word, color, str(out_dir / fname))
        trials.append({
            "image_path": img_path,
            "word": word,
            "ink_color": color,
            "congruent": True,
            "expected_response": color,
            "stimulus_text": (
                "Look at the image. A color word is displayed in colored ink.\n"
                "What color is the INK the word is printed in?\n"
                "Answer with exactly one word (the ink color, not the word itself)."
            ),
        })
        idx += 1

    # Incongruent trials: word does NOT match ink color
    for _ in range(n_incongruent):
        word_color = rng.choice(COLOR_NAMES)
        ink_color = rng.choice([c for c in COLOR_NAMES if c != word_color])
        word = word_color.upper()
        fname = f"stroop_incong_{idx:03d}.png"
        img_path = generate_stroop_image(word, ink_color, str(out_dir / fname))
        trials.append({
            "image_path": img_path,
            "word": word,
            "ink_color": ink_color,
            "congruent": False,
            "expected_response": ink_color,
            "stimulus_text": (
                "Look at the image. A color word is displayed in colored ink.\n"
                "What color is the INK the word is printed in?\n"
                "Answer with exactly one word (the ink color, not the word itself)."
            ),
        })
        idx += 1

    rng.shuffle(trials)
    return trials


def generate_balanced_stroop_set(
    seed: int = 42,
    n_per_condition: int = 50,
    out_dir: str = "data/images/stroop_balanced",
) -> List[Dict[str, Any]]:
    """Generate matched congruent/incongruent ink-color distributions.

    The same color-count multiset is used in both conditions.  Incongruent
    word labels are a deterministic derangement of the ink-color multiset, so
    neither ink-response frequency nor word frequency is condition-confounded.
    """

    if n_per_condition < len(COLOR_NAMES):
        raise ValueError("n_per_condition must cover every color")
    rng = random.Random(seed)
    color_order = list(COLOR_NAMES)
    rng.shuffle(color_order)
    base, remainder = divmod(n_per_condition, len(color_order))
    counts = {
        color: base + int(index < remainder)
        for index, color in enumerate(color_order)
    }
    ink_labels = [
        color for color in color_order for _ in range(counts[color])
    ]
    word_labels = list(ink_labels)
    for _ in range(10000):
        rng.shuffle(word_labels)
        if all(word != ink for word, ink in zip(word_labels, ink_labels)):
            break
    else:
        raise RuntimeError("failed to construct a color-label derangement")

    out_dir_path = Path(out_dir)
    trials: List[Dict[str, Any]] = []
    prompt = (
        "Look at the image. A color word is displayed in colored ink.\n"
        "What color is the INK the word is printed in?\n"
        "Answer with exactly one word (the ink color, not the word itself)."
    )
    for index, ink_color in enumerate(ink_labels):
        word = ink_color.upper()
        image_path = generate_stroop_image(
            word,
            ink_color,
            str(out_dir_path / f"stroop_cong_{index:03d}.png"),
        )
        trials.append(
            {
                "image_path": image_path,
                "word": word,
                "ink_color": ink_color,
                "congruent": True,
                "expected_response": ink_color,
                "stimulus_text": prompt,
            }
        )
    for index, (word_color, ink_color) in enumerate(
        zip(word_labels, ink_labels), start=n_per_condition
    ):
        image_path = generate_stroop_image(
            word_color.upper(),
            ink_color,
            str(out_dir_path / f"stroop_incong_{index:03d}.png"),
        )
        trials.append(
            {
                "image_path": image_path,
                "word": word_color.upper(),
                "ink_color": ink_color,
                "congruent": False,
                "expected_response": ink_color,
                "stimulus_text": prompt,
            }
        )
    rng.shuffle(trials)
    return trials


if __name__ == "__main__":
    trials = generate_stroop_set(seed=42, n_congruent=5, n_incongruent=5)
    print(f"Generated {len(trials)} Stroop images")
    for t in trials[:3]:
        print(f"  {t['word']:8s} in {t['ink_color']:8s} (cong={t['congruent']}) → {t['image_path']}")
