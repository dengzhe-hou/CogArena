"""Generate Sally-Anne false belief task images for VLM evaluation.

Creates sequences of simple scene illustrations showing characters,
containers, and an object. Tests whether the VLM can reason about
characters' false beliefs about object locations.
"""

import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from PIL import Image, ImageDraw

from .font_utils import load_frozen_font


# Scene settings
WIDTH, HEIGHT = 500, 300
BG_COLOR = (255, 255, 255)
FLOOR_COLOR = (180, 180, 180)
FLOOR_Y = 250
WALL_COLOR = (220, 220, 220)
TEXT_COLOR = (40, 40, 40)
CONTAINER_BORDER = (80, 80, 80)
CONTAINER_FILL = (240, 240, 230)
FADED_ALPHA = 80  # alpha for absent characters
FONT_SIZE = 13
CAPTION_FONT_SIZE = 11
LABEL_FONT_SIZE = 11

# Character colors
CHAR_COLORS = [
    (50, 100, 200),   # blue
    (200, 50, 50),    # red
    (50, 160, 50),    # green
    (180, 100, 30),   # orange
]

# Object options
OBJECT_OPTIONS = [
    ("marble", (100, 200, 100), "circle"),
    ("ball", (200, 80, 80), "circle"),
    ("toy", (80, 80, 200), "square"),
    ("key", (200, 180, 50), "square"),
    ("cookie", (180, 130, 60), "circle"),
    ("coin", (200, 200, 60), "circle"),
]

# Container options
CONTAINER_OPTIONS = [
    ("basket", (160, 120, 80)),
    ("box", (100, 100, 160)),
    ("cupboard", (120, 160, 100)),
    ("drawer", (140, 100, 120)),
    ("bag", (100, 140, 160)),
    ("jar", (160, 140, 100)),
]

# Name pools
NAMES = ["Sally", "Anne", "Max", "Lily", "Tom", "Emma", "Sam", "Mia"]


def _get_font(size: int = FONT_SIZE):
    """Load the same frozen font used by the other VLM paradigms."""
    return load_frozen_font(size)


def _draw_stick_figure(draw: ImageDraw.Draw, cx: int, base_y: int,
                       color: Tuple[int, int, int], name: str,
                       faded: bool = False, *, label_font_size: int = LABEL_FONT_SIZE,
                       small_font_size: int = 9):
    """Draw a simple stick figure with a name label above.

    Args:
        draw: ImageDraw instance
        cx: center x position
        base_y: y position for feet (bottom of figure)
        color: RGB color for the figure
        name: name label drawn above
        faded: if True, draw in lighter shade to show absence
    """
    if faded:
        # Lighten the color toward white
        color = tuple(min(255, c + 160) for c in color)
        text_color = (200, 200, 200)
        line_w = 1
    else:
        text_color = TEXT_COLOR
        line_w = 2

    head_r = 10
    body_len = 40
    arm_len = 20
    leg_len = 25

    head_y = base_y - leg_len - body_len - head_r * 2
    neck_y = head_y + head_r * 2
    hip_y = neck_y + body_len

    # Head
    draw.ellipse([cx - head_r, head_y, cx + head_r, head_y + head_r * 2],
                 outline=color, width=line_w)
    # Body
    draw.line([(cx, neck_y), (cx, hip_y)], fill=color, width=line_w)
    # Arms
    draw.line([(cx - arm_len, neck_y + 15), (cx + arm_len, neck_y + 15)],
              fill=color, width=line_w)
    # Left leg
    draw.line([(cx, hip_y), (cx - 12, base_y)], fill=color, width=line_w)
    # Right leg
    draw.line([(cx, hip_y), (cx + 12, base_y)], fill=color, width=line_w)

    # Name label above head
    font = _get_font(label_font_size)
    bbox = draw.textbbox((0, 0), name, font=font)
    tw = bbox[2] - bbox[0]
    name_y = (
        head_y - label_font_size - 21
        if label_font_size > LABEL_FONT_SIZE
        else head_y - 18
    )
    draw.text((cx - tw // 2, name_y), name, fill=text_color, font=font)
    name_box = draw.textbbox((cx - tw // 2, name_y), name, font=font)

    if faded:
        # Draw "(away)" under name
        away_font = _get_font(small_font_size)
        bbox = draw.textbbox((0, 0), "(away)", font=away_font)
        tw = bbox[2] - bbox[0]
        away_y = (
            head_y - small_font_size - 1
            if label_font_size > LABEL_FONT_SIZE
            else head_y - 7
        )
        away_box = draw.textbbox(
            (cx - tw // 2, away_y), "(away)", font=away_font
        )
        if label_font_size > LABEL_FONT_SIZE and away_box[1] < name_box[3] + 2:
            raise RuntimeError("false-belief name and away labels overlap")
        draw.text((cx - tw // 2, away_y), "(away)",
                  fill=(180, 180, 180), font=away_font)


def _draw_container(draw: ImageDraw.Draw, cx: int, base_y: int,
                    w: int, h: int, label: str,
                    border_color: Tuple[int, int, int],
                    has_object: bool = False,
                    obj_label: str = "",
                    obj_color: Tuple[int, int, int] = (0, 0, 0),
                    obj_shape: str = "circle", *,
                    label_font_size: int = LABEL_FONT_SIZE,
                    small_font_size: int = 9):
    """Draw a labeled container, optionally with an object inside."""
    x0 = cx - w // 2
    y0 = base_y - h
    # Container body
    draw.rectangle([x0, y0, x0 + w, base_y], fill=CONTAINER_FILL,
                   outline=border_color, width=2)
    # Label below
    font = _get_font(label_font_size)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, base_y + 4), label, fill=TEXT_COLOR, font=font)

    if has_object:
        # Draw small object inside the container
        obj_cx = cx
        obj_cy = base_y - h // 2
        obj_r = 8
        if obj_shape == "circle":
            draw.ellipse([obj_cx - obj_r, obj_cy - obj_r,
                          obj_cx + obj_r, obj_cy + obj_r],
                         fill=obj_color, outline=(0, 0, 0), width=1)
        else:
            draw.rectangle([obj_cx - obj_r, obj_cy - obj_r,
                            obj_cx + obj_r, obj_cy + obj_r],
                           fill=obj_color, outline=(0, 0, 0), width=1)
        # Object label
        small_font = _get_font(small_font_size)
        bbox = draw.textbbox((0, 0), obj_label, font=small_font)
        tw = bbox[2] - bbox[0]
        draw.text((obj_cx - tw // 2, obj_cy + obj_r + 1), obj_label,
                  fill=TEXT_COLOR, font=small_font)


def generate_false_belief_scene(
    characters: List[str],
    object_name: str,
    container_a: str,
    container_b: str,
    scene_state: Dict[str, Any],
    out_path: str,
    *,
    montage_readable: bool = False,
) -> str:
    """Generate a single scene image for a false belief story.

    Args:
        characters: list of 2 character names
        object_name: name of the object being moved
        container_a: name of container A (left)
        container_b: name of container B (right)
        scene_state: dict with keys:
            - char_present: list of bools (who is visible)
            - char_faded: list of bools (who is shown faded / away)
            - object_in: "a" or "b" (which container has the object)
            - caption: str (text description of this scene)
        out_path: where to save

    Returns:
        Path to saved image
    """
    canvas_height = 360 if montage_readable else HEIGHT
    img = Image.new("RGB", (WIDTH, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Floor line
    draw.line([(0, FLOOR_Y), (WIDTH, FLOOR_Y)], fill=FLOOR_COLOR, width=2)

    # Room walls (subtle)
    draw.line([(0, 30), (0, FLOOR_Y)], fill=WALL_COLOR, width=1)
    draw.line([(WIDTH - 1, 30), (WIDTH - 1, FLOOR_Y)], fill=WALL_COLOR, width=1)
    draw.line([(0, 30), (WIDTH - 1, 30)], fill=WALL_COLOR, width=1)

    # Look up object info
    obj_color = (100, 200, 100)
    obj_shape = "circle"
    for oname, ocol, oshp in OBJECT_OPTIONS:
        if oname == object_name:
            obj_color = ocol
            obj_shape = oshp
            break

    # Look up container colors
    cont_a_color = CONTAINER_BORDER
    cont_b_color = CONTAINER_BORDER
    for cname, ccol in CONTAINER_OPTIONS:
        if cname == container_a:
            cont_a_color = ccol
        if cname == container_b:
            cont_b_color = ccol

    # Layout positions
    # Characters at x=100 and x=400, containers at x=220 and x=330
    char_positions = [100, 400]
    container_positions = [210, 340]
    container_w, container_h = 70, 55

    label_font_size = 24 if montage_readable else LABEL_FONT_SIZE
    caption_font_size = 24 if montage_readable else CAPTION_FONT_SIZE
    small_font_size = 16 if montage_readable else 9

    # Draw containers
    obj_in = scene_state.get("object_in", "a")
    _draw_container(draw, container_positions[0], FLOOR_Y, container_w,
                    container_h, container_a, cont_a_color,
                    has_object=(obj_in == "a"),
                    obj_label=object_name, obj_color=obj_color,
                    obj_shape=obj_shape,
                    label_font_size=label_font_size,
                    small_font_size=small_font_size)
    _draw_container(draw, container_positions[1], FLOOR_Y, container_w,
                    container_h, container_b, cont_b_color,
                    has_object=(obj_in == "b"),
                    obj_label=object_name, obj_color=obj_color,
                    obj_shape=obj_shape,
                    label_font_size=label_font_size,
                    small_font_size=small_font_size)

    # Draw characters
    char_present = scene_state.get("char_present", [True, True])
    char_faded = scene_state.get("char_faded", [False, False])
    for i, name in enumerate(characters):
        if char_present[i]:
            _draw_stick_figure(draw, char_positions[i], FLOOR_Y,
                               CHAR_COLORS[i % len(CHAR_COLORS)], name,
                               faded=char_faded[i],
                               label_font_size=label_font_size,
                               small_font_size=small_font_size)

    # Caption at the bottom
    caption = scene_state.get("caption", "")
    if caption:
        font = _get_font(caption_font_size)
        bbox = draw.textbbox((0, 0), caption, font=font)
        tw = bbox[2] - bbox[0]
        if montage_readable:
            if tw > WIDTH - 30:
                words = caption.split()
                lines: list[str] = []
                current = ""
                for word in words:
                    candidate = f"{current} {word}".strip()
                    candidate_width = draw.textbbox(
                        (0, 0), candidate, font=font
                    )[2]
                    if current and candidate_width > WIDTH - 30:
                        lines.append(current)
                        current = word
                    else:
                        current = candidate
                if current:
                    lines.append(current)
                if len(lines) > 2:
                    raise RuntimeError(f"false-belief caption needs >2 lines: {caption}")
            else:
                lines = [caption]
            for line_index, line in enumerate(lines):
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                draw.text(
                    ((WIDTH - line_width) // 2, 296 + line_index * 27),
                    line,
                    fill=TEXT_COLOR,
                    font=font,
                )
        else:
            draw.text(((WIDTH - tw) // 2, HEIGHT - 25), caption,
                      fill=TEXT_COLOR, font=font)

    # Scene number in top-left if provided
    scene_num = scene_state.get("scene_num")
    if scene_num is not None:
        font = _get_font(label_font_size)
        draw.text((10, 8), f"Scene {scene_num}", fill=TEXT_COLOR, font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def generate_false_belief_set(
    seed: int = 42,
    n_items: int = 10,
    out_dir: str = "data/images/false_belief",
    *,
    montage_readable: bool = False,
) -> List[Dict[str, Any]]:
    """Generate a full set of Sally-Anne false belief stories.

    Each story produces 4 scene images:
      1. Both characters present, object in container A
      2. Character B leaves (shown faded)
      3. Character A moves object to container B
      4. Character B returns

    The false belief question: Where will character B look for the object?
    Correct answer: container A (where B last saw it).

    Returns list of dicts with: image_paths, characters, object,
    containers, correct_answer, stimulus_text.
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    items = []

    for idx in range(n_items):
        # Pick characters
        char_pair = rng.sample(NAMES, 2)
        char_a, char_b = char_pair

        # Pick object
        obj_name, _, _ = rng.choice(OBJECT_OPTIONS)

        # Pick containers
        cont_pair = rng.sample(CONTAINER_OPTIONS, 2)
        cont_a_name = cont_pair[0][0]
        cont_b_name = cont_pair[1][0]

        story_dir = out_dir / f"story_{idx:03d}"
        image_paths = []

        # Scene 1: Both present, object in container A
        state1 = {
            "char_present": [True, True],
            "char_faded": [False, False],
            "object_in": "a",
            "caption": (f"{char_a} and {char_b} see the {obj_name} "
                        f"placed in the {cont_a_name}."),
            "scene_num": 1,
        }
        p1 = generate_false_belief_scene(
            char_pair, obj_name, cont_a_name, cont_b_name, state1,
            str(story_dir / "scene_1.png"),
            montage_readable=montage_readable)
        image_paths.append(p1)

        # Scene 2: Character B leaves
        state2 = {
            "char_present": [True, True],
            "char_faded": [False, True],
            "object_in": "a",
            "caption": f"{char_b} leaves the room.",
            "scene_num": 2,
        }
        p2 = generate_false_belief_scene(
            char_pair, obj_name, cont_a_name, cont_b_name, state2,
            str(story_dir / "scene_2.png"),
            montage_readable=montage_readable)
        image_paths.append(p2)

        # Scene 3: Character A moves object to container B
        state3 = {
            "char_present": [True, True],
            "char_faded": [False, True],
            "object_in": "b",
            "caption": (f"{char_a} moves the {obj_name} from the "
                        f"{cont_a_name} to the {cont_b_name}."),
            "scene_num": 3,
        }
        p3 = generate_false_belief_scene(
            char_pair, obj_name, cont_a_name, cont_b_name, state3,
            str(story_dir / "scene_3.png"),
            montage_readable=montage_readable)
        image_paths.append(p3)

        # Scene 4: Character B returns
        state4 = {
            "char_present": [True, True],
            "char_faded": [False, False],
            "object_in": "b",
            "caption": f"{char_b} comes back. Where will {char_b} look?",
            "scene_num": 4,
        }
        p4 = generate_false_belief_scene(
            char_pair, obj_name, cont_a_name, cont_b_name, state4,
            str(story_dir / "scene_4.png"),
            montage_readable=montage_readable)
        image_paths.append(p4)

        # Correct answer: B will look in container A (where they last saw it)
        items.append({
            "image_paths": image_paths,
            "characters": char_pair,
            "object": obj_name,
            "containers": [cont_a_name, cont_b_name],
            "correct_answer": cont_a_name,
            "stimulus_text": (
                "Look at the 4 scene images showing a story in order.\n"
                f"Scene 1: {char_a} and {char_b} both see the {obj_name} "
                f"placed in the {cont_a_name}.\n"
                f"Scene 2: {char_b} leaves the room.\n"
                f"Scene 3: {char_a} moves the {obj_name} from the "
                f"{cont_a_name} to the {cont_b_name}. {char_b} does not "
                f"see this.\n"
                f"Scene 4: {char_b} returns.\n\n"
                f"Question: When {char_b} returns, where will {char_b} "
                f"FIRST look for the {obj_name}?\n"
                f"Answer with exactly the container name: "
                f"{cont_a_name} or {cont_b_name}."
            ),
        })

    return items


if __name__ == "__main__":
    items = generate_false_belief_set(seed=42, n_items=3)
    print(f"Generated {len(items)} false belief stories")
    for it in items:
        print(f"  {it['characters'][0]} & {it['characters'][1]}: "
              f"{it['object']} in {it['containers'][0]}/{it['containers'][1]} "
              f"-> answer={it['correct_answer']}")
        for p in it["image_paths"]:
            print(f"    {p}")
