"""Generate WCST (Wisconsin Card Sorting Test) images for VLM evaluation.

Each trial shows a target card at the top and 4 reference cards below.
Cards have 3 visual dimensions: shape, color, and count. The VLM must
figure out the hidden sorting rule from feedback.
"""

import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

from PIL import Image, ImageDraw, ImageFont


# Visual dimensions
SHAPES = ["circle", "triangle", "square", "star"]
CARD_COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 50, 220),
    "green": (50, 180, 50),
    "yellow": (200, 200, 0),
}
COUNTS = [1, 2, 3, 4]

# Image settings
WIDTH, HEIGHT = 600, 400
BG_COLOR = (255, 255, 255)
CARD_BG = (250, 250, 250)
CARD_BORDER = (80, 80, 80)
LABEL_COLOR = (60, 60, 60)
PROMPT_COLOR = (40, 40, 40)
FONT_SIZE = 12
LABEL_FONT_SIZE = 14
PROMPT_FONT_SIZE = 10


def _get_font(size: int = FONT_SIZE):
    """Get a font, falling back to default if no TTF available."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_shape(draw: ImageDraw.Draw, shape: str, cx: int, cy: int,
                size: int, color: Tuple[int, int, int]):
    """Draw a single shape centered at (cx, cy)."""
    r = size // 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                     outline=(0, 0, 0), width=1)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color,
                       outline=(0, 0, 0), width=1)
    elif shape == "triangle":
        points = [
            (cx, cy - r),
            (cx - r, cy + r),
            (cx + r, cy + r),
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0))
    elif shape == "star":
        # 5-pointed star
        import math
        pts = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            rad = r if i % 2 == 0 else r * 0.4
            pts.append((cx + rad * math.cos(angle),
                        cy + rad * math.sin(angle)))
        draw.polygon(pts, fill=color, outline=(0, 0, 0))


def _draw_card(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int,
               card: Dict[str, Any], label: str = ""):
    """Draw a card with shapes inside a bordered rectangle.

    Args:
        draw: ImageDraw instance
        x, y: top-left corner of the card
        w, h: card width and height
        card: dict with keys 'shape', 'color', 'count'
        label: optional label below the card (e.g., "1", "2")
    """
    # Card background and border
    draw.rectangle([x, y, x + w, y + h], fill=CARD_BG, outline=CARD_BORDER,
                   width=2)

    shape = card["shape"]
    color_rgb = CARD_COLORS[card["color"]]
    count = card["count"]
    shape_size = min(w, h) // 4  # shape radius scales with card size

    # Arrange shapes in a row within the card
    total_shape_width = count * shape_size * 2
    spacing = (w - total_shape_width) // (count + 1) if count < w // (shape_size * 2) else 4
    start_x = x + spacing + shape_size
    step = shape_size * 2 + spacing
    cy = y + h // 2

    for i in range(count):
        sx = start_x + i * step
        _draw_shape(draw, shape, sx, cy, shape_size, color_rgb)

    # Label below card
    if label:
        font = _get_font(LABEL_FONT_SIZE)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        lx = x + (w - tw) // 2
        ly = y + h + 3
        draw.text((lx, ly), label, fill=LABEL_COLOR, font=font)


def _random_card(rng: random.Random) -> Dict[str, Any]:
    """Generate a random card specification."""
    return {
        "shape": rng.choice(SHAPES),
        "color": rng.choice(list(CARD_COLORS.keys())),
        "count": rng.choice(COUNTS),
    }


def generate_wcst_trial_image(
    target_card: Dict[str, Any],
    reference_cards: List[Dict[str, Any]],
    out_path: str,
) -> str:
    """Generate a single WCST trial image.

    Args:
        target_card: dict with 'shape', 'color', 'count'
        reference_cards: list of 4 dicts, each with 'shape', 'color', 'count'
        out_path: where to save the PNG

    Returns:
        Path to the saved image
    """
    assert len(reference_cards) == 4, "Need exactly 4 reference cards"

    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = _get_font(PROMPT_FONT_SIZE)

    # Prompt text at the very top
    prompt = ("Which reference card (1-4) matches the target card? "
              "The sorting rule is hidden - figure it out from feedback.")
    bbox = draw.textbbox((0, 0), prompt, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, 5), prompt, fill=PROMPT_COLOR, font=font)

    # Target card (centered, top area)
    card_w, card_h = 110, 80
    target_x = (WIDTH - card_w) // 2
    target_y = 30

    # "Target" label above the card
    label_font = _get_font(LABEL_FONT_SIZE)
    bbox = draw.textbbox((0, 0), "Target", font=label_font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, target_y - 2), "Target",
              fill=LABEL_COLOR, font=label_font)
    target_y += 18
    _draw_card(draw, target_x, target_y, card_w, card_h, target_card)

    # Separator line
    sep_y = target_y + card_h + 25
    draw.line([(30, sep_y), (WIDTH - 30, sep_y)], fill=(180, 180, 180),
              width=1)

    # "Reference Cards" label
    ref_label_y = sep_y + 8
    bbox = draw.textbbox((0, 0), "Reference Cards", font=label_font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, ref_label_y), "Reference Cards",
              fill=LABEL_COLOR, font=label_font)

    # 4 reference cards in a row
    ref_y = ref_label_y + 22
    ref_card_w, ref_card_h = 110, 80
    total_w = 4 * ref_card_w
    gap = (WIDTH - total_w) // 5

    for i, rcard in enumerate(reference_cards):
        rx = gap + i * (ref_card_w + gap)
        _draw_card(draw, rx, ref_y, ref_card_w, ref_card_h, rcard,
                   label=str(i + 1))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def generate_wcst_set(
    seed: int = 42,
    n_trials: int = 20,
    out_dir: str = "data/images/wcst",
) -> List[Dict[str, Any]]:
    """Generate a full set of WCST trials.

    The hidden sorting rule cycles through shape, color, count every
    few trials. For each trial, exactly one reference card matches the
    target on the current rule dimension; the others differ on that
    dimension but may share other features.

    Returns list of dicts with: image_path, target_card, reference_cards,
    correct_answer (1-4), rule, stimulus_text.
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    trials = []
    rules = ["shape", "color", "count"]
    # Rule changes every 4-8 trials
    rule_idx = 0
    trials_until_switch = rng.randint(4, 8)

    for idx in range(n_trials):
        # Possibly switch rule
        if trials_until_switch <= 0:
            rule_idx = (rule_idx + 1) % len(rules)
            trials_until_switch = rng.randint(4, 8)
        trials_until_switch -= 1
        current_rule = rules[rule_idx]

        target = _random_card(rng)

        # Build 4 reference cards: one matches on the current rule
        correct_pos = rng.randint(0, 3)
        reference_cards = []
        for ri in range(4):
            if ri == correct_pos:
                # Match on the current rule dimension only
                card = _random_card(rng)
                card[current_rule] = target[current_rule]
                # Make sure the other dimensions differ from target
                # (to avoid ambiguity)
                for dim in rules:
                    if dim != current_rule:
                        options = {
                            "shape": [s for s in SHAPES if s != target["shape"]],
                            "color": [c for c in CARD_COLORS if c != target["color"]],
                            "count": [n for n in COUNTS if n != target["count"]],
                        }
                        card[dim] = rng.choice(options[dim])
                reference_cards.append(card)
            else:
                # Ensure this card does NOT match on the current rule
                card = _random_card(rng)
                if card[current_rule] == target[current_rule]:
                    options = {
                        "shape": [s for s in SHAPES if s != target["shape"]],
                        "color": [c for c in CARD_COLORS if c != target["color"]],
                        "count": [n for n in COUNTS if n != target["count"]],
                    }
                    card[current_rule] = rng.choice(options[current_rule])
                reference_cards.append(card)

        fname = f"wcst_{idx:03d}.png"
        img_path = generate_wcst_trial_image(
            target, reference_cards, str(out_dir / fname)
        )

        trials.append({
            "image_path": img_path,
            "target_card": target,
            "reference_cards": reference_cards,
            "correct_answer": correct_pos + 1,  # 1-indexed
            "rule": current_rule,
            "stimulus_text": (
                "Look at the image. There is a target card at the top and "
                "4 reference cards (labeled 1-4) at the bottom.\n"
                "Each card has shapes that vary in shape type, color, and count.\n"
                "Which reference card (1-4) matches the target card based on "
                "the hidden sorting rule?\n"
                "The rule could be shape, color, or count. "
                "Answer with exactly one number: 1, 2, 3, or 4."
            ),
        })

    return trials


if __name__ == "__main__":
    trials = generate_wcst_set(seed=42, n_trials=10)
    print(f"Generated {len(trials)} WCST images")
    for t in trials[:3]:
        print(f"  rule={t['rule']:6s} correct={t['correct_answer']} "
              f"target={t['target_card']} -> {t['image_path']}")
