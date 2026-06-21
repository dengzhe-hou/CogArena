"""Generate Flanker task images: target arrow flanked by distractors.

Creates images with a row of arrows where the center arrow may point
in a different direction from the flanking arrows.
"""

import random
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 500, 120
BG_COLOR = (255, 255, 255)
ARROW_COLOR = (30, 30, 30)
FONT_SIZE = 55


def _get_font(size: int = FONT_SIZE):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# Arrow characters
LEFT_ARROW = "<"
RIGHT_ARROW = ">"


def generate_flanker_image(
    target_dir: str,
    flanker_dir: str,
    n_flankers: int,
    out_path: str,
) -> str:
    """Generate a single Flanker image.

    Args:
        target_dir: "left" or "right" (center arrow)
        flanker_dir: "left" or "right" (surrounding arrows)
        n_flankers: number of flankers on each side
        out_path: where to save

    Returns:
        Path to saved image
    """
    target = LEFT_ARROW if target_dir == "left" else RIGHT_ARROW
    flanker = LEFT_ARROW if flanker_dir == "left" else RIGHT_ARROW

    # Build the arrow string: flankers + target + flankers
    display = flanker * n_flankers + " " + target + " " + flanker * n_flankers

    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = _get_font()

    bbox = draw.textbbox((0, 0), display, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (WIDTH - tw) // 2
    y = (HEIGHT - th) // 2
    draw.text((x, y), display, fill=ARROW_COLOR, font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def generate_flanker_set(
    seed: int = 42,
    n_congruent: int = 10,
    n_incongruent: int = 10,
    n_flankers: int = 3,
    out_dir: str = "data/images/flanker",
) -> List[Dict[str, Any]]:
    """Generate a full set of Flanker image trials.

    Returns list of dicts with: image_path, target_dir, flanker_dir,
    congruent, expected_response, stimulus_text.
    """
    rng = random.Random(seed)
    trials = []
    out_dir = Path(out_dir)
    idx = 0

    # Congruent: target and flankers point same direction
    for _ in range(n_congruent):
        direction = rng.choice(["left", "right"])
        fname = f"flanker_cong_{idx:03d}.png"
        img_path = generate_flanker_image(direction, direction, n_flankers, str(out_dir / fname))
        trials.append({
            "image_path": img_path,
            "target_dir": direction,
            "flanker_dir": direction,
            "congruent": True,
            "expected_response": direction,
            "stimulus_text": (
                "Look at the image. There is a row of arrows.\n"
                "What direction does the CENTER arrow point?\n"
                "Answer with exactly one word: left or right."
            ),
        })
        idx += 1

    # Incongruent: target and flankers point different directions
    for _ in range(n_incongruent):
        target = rng.choice(["left", "right"])
        flanker = "right" if target == "left" else "left"
        fname = f"flanker_incong_{idx:03d}.png"
        img_path = generate_flanker_image(target, flanker, n_flankers, str(out_dir / fname))
        trials.append({
            "image_path": img_path,
            "target_dir": target,
            "flanker_dir": flanker,
            "congruent": False,
            "expected_response": target,
            "stimulus_text": (
                "Look at the image. There is a row of arrows.\n"
                "What direction does the CENTER arrow point?\n"
                "Answer with exactly one word: left or right."
            ),
        })
        idx += 1

    rng.shuffle(trials)
    return trials


if __name__ == "__main__":
    trials = generate_flanker_set(seed=42, n_congruent=5, n_incongruent=5)
    print(f"Generated {len(trials)} Flanker images")
    for t in trials[:3]:
        cong = "CONG" if t["congruent"] else "INCONG"
        print(f"  target={t['target_dir']:5s} flanker={t['flanker_dir']:5s} ({cong}) → {t['image_path']}")
