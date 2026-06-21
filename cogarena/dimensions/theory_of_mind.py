"""Theory of Mind dimension for CogArena.

Implements two paradigms with procedural generation:
  1. False Belief (Sally-Anne style)  -- 1st- and 2nd-order belief attribution
  2. EPITOME-style Multi-aspect ToM   -- belief / desire / intention / emotion

All items are procedurally generated from random seeds using diverse
character-name pools, object pools, and location pools to minimise
contamination from training corpora.  The classic Sally-Anne scenario
is ONLY used as a contamination probe.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from cogarena.core import (
    AdaptationDistance,
    DifficultyLevel,
    EvalMode,
    ScoringConfig,
    TaskInstance,
    TaskMetadata,
)

# ---------------------------------------------------------------------------
# Shared pools for procedural generation
# ---------------------------------------------------------------------------

# Diverse cultural name pool (60 names from many traditions)
_NAME_POOL: list[str] = [
    # East Asian
    "Mei", "Haruto", "Jia", "Yuki", "Wei", "Sora", "Hana", "Ren",
    "Linh", "Tao",
    # South Asian
    "Ananya", "Arjun", "Priya", "Rohan", "Devi", "Kiran", "Nisha",
    "Vikram", "Lakshmi", "Amir",
    # African
    "Amina", "Kwame", "Zara", "Kofi", "Fatima", "Chidi", "Nia",
    "Tendai", "Ayo", "Jabari",
    # European
    "Elena", "Mateo", "Ingrid", "Luca", "Freya", "Henrik", "Astrid",
    "Dmitri", "Sofia", "Pavel",
    # Latin American
    "Camila", "Diego", "Valentina", "Santiago", "Isabela", "Rafael",
    "Lucia", "Andres", "Mariana", "Carlos",
    # Middle Eastern
    "Leila", "Omar", "Yasmin", "Tariq", "Nadia", "Karim", "Samira",
    "Hassan", "Dalal", "Idris",
]

# Objects that can be moved / hidden
_OBJECT_POOL: list[str] = [
    "red marble", "blue keychain", "silver ring", "wooden toy car",
    "green notebook", "small bronze figurine", "pink eraser",
    "yellow scarf", "striped pencil case", "ceramic mug",
    "origami crane", "brass compass", "velvet pouch",
    "glass paperweight", "leather wallet", "miniature globe",
    "porcelain cat", "rubber duck", "stone chess piece",
    "knitted bookmark", "tin lunchbox", "crystal snowflake",
    "plastic dinosaur", "gold coin", "seashell necklace",
    "cork coaster", "silk ribbon", "copper bell",
    "bamboo flute", "woven bracelet",
]

# Container locations (where objects can be hidden)
_CONTAINER_POOL: list[str] = [
    "the wicker basket", "the cardboard box", "the top drawer",
    "the glass jar", "the coat pocket", "the backpack",
    "the filing cabinet", "the shoebox", "the ceramic bowl",
    "the metal tin", "the paper bag", "the wooden chest",
    "the desk drawer", "the lunchbox", "the tote bag",
    "the toolbox", "the hat box", "the cookie jar",
    "the suitcase", "the cupboard", "the pencil case",
    "the storage bin", "the handbag", "the music box",
]

# Scenario settings
_SCENARIO_POOL: list[str] = [
    "a sunlit kitchen", "a busy office", "a school classroom",
    "a park pavilion", "a community garden", "an art studio",
    "a library reading room", "a hospital waiting area",
    "a train station cafe", "a rooftop terrace",
    "a woodworking workshop", "a university lab",
    "a seaside cottage", "a mountain cabin", "a marketplace stall",
    "a pottery studio", "a music rehearsal room",
    "a photography darkroom", "a greenhouse", "a bakery kitchen",
]

# ---------------------------------------------------------------------------
# Reason pools for 2nd-order false belief
# ---------------------------------------------------------------------------

_ABSENCE_REASONS: list[str] = [
    "went to get water",
    "stepped outside for fresh air",
    "left to answer a phone call",
    "went to the restroom",
    "was called away by a colleague",
    "walked to the next room to fetch something",
    "left to check the mail",
    "went to park the car",
    "stepped out to greet a visitor",
    "left to buy something from the store nearby",
]

_MOVE_REASONS: list[str] = [
    "wanted to keep it safe",
    "thought it would be better stored there",
    "was tidying up and relocated it",
    "needed the space where it was",
    "decided it belonged there instead",
    "moved it while cleaning",
    "put it there absent-mindedly",
    "reorganised the area",
]


def _make_rng(seed: int) -> random.Random:
    """Return a seeded Random instance (reproducible, thread-safe)."""
    return random.Random(seed)


def _difficulty_enum(s: str) -> DifficultyLevel:
    return DifficultyLevel(s.lower())


# ===================================================================
# PARADIGM 1 -- FALSE BELIEF (Sally-Anne style, procedurally generated)
# ===================================================================

class FalseBeliefGenerator:
    """Procedural generator for False Belief tasks.

    First-order: Character A moves an object while Character B is away.
      Question -- "Where will B look for the object?"
      Correct  -- the ORIGINAL location (where B last saw it).
      Egocentric error -- the CURRENT location (true state of world).

    Second-order: A moves object; B secretly observes; A does NOT know
      B observed.
      Question -- "Where does A think B will look for the object?"
      Correct  -- the ORIGINAL location (A thinks B doesn't know).
      Egocentric error -- the CURRENT location.
    """

    PARADIGM = "false_belief"
    DIMENSION = "theory_of_mind"

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 20,
        order: int = 1,
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> list[TaskInstance]:
        """Generate False Belief task items.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of items to generate.
        order : int
            1 = first-order false belief, 2 = second-order.
        difficulty : str
            "easy" = simpler language, shorter stories;
            "medium" = standard; "hard" = additional irrelevant detail.
        contamination_probe : bool
            If True, use the classic Sally-Anne scenario.
        """
        rng = _make_rng(seed)
        items: list[TaskInstance] = []

        for idx in range(n_items):
            ep_seed = rng.randint(0, 2**31)
            ep_rng = _make_rng(ep_seed)

            if contamination_probe:
                task = cls._generate_classic_sally_anne(ep_rng, order)
            else:
                task = cls._generate_novel(ep_rng, order, difficulty)

            stimulus_text = task["story"]
            correct_answer = task["correct_answer"]
            egocentric_answer = task["egocentric_answer"]

            task_id = (
                f"tom_fb_order{order}"
                f"_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_s{ep_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.LLM_STATIC,
                parameters={
                    "order": order,
                    "correct_answer": correct_answer,
                    "egocentric_answer": egocentric_answer,
                    "original_location": task["original_location"],
                    "new_location": task["new_location"],
                    "episode_seed": ep_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": False,
                    "characters": task["characters"],
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.theory_of_mind.score_false_belief",
                        "order": order,
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=(
                    AdaptationDistance.HIGH if contamination_probe
                    else AdaptationDistance.LOW
                ),
                description=(
                    f"{'Classic ' if contamination_probe else ''}"
                    f"Order-{order} false belief task"
                ),
            )

            items.append(TaskInstance(
                task_id=task_id,
                metadata=metadata,
                stimulus=stimulus_text,
                expected_response=correct_answer,
            ))

        return items

    # ------------------------------------------------------------------
    # Story generators
    # ------------------------------------------------------------------

    @classmethod
    def _generate_novel(
        cls,
        rng: random.Random,
        order: int,
        difficulty: str,
    ) -> Dict[str, Any]:
        """Generate a novel false-belief vignette."""
        names = rng.sample(_NAME_POOL, k=3)
        char_a = names[0]  # The mover
        char_b = names[1]  # The one with the false belief
        char_c = names[2]  # Optional witness (for adding complexity)

        obj = rng.choice(_OBJECT_POOL)
        loc_original = rng.choice(_CONTAINER_POOL)
        remaining_locs = [l for l in _CONTAINER_POOL if l != loc_original]
        loc_new = rng.choice(remaining_locs)
        scenario = rng.choice(_SCENARIO_POOL)
        absence_reason = rng.choice(_ABSENCE_REASONS)
        move_reason = rng.choice(_MOVE_REASONS)

        if order == 1:
            story = cls._build_first_order_story(
                char_a, char_b, obj, loc_original, loc_new,
                scenario, absence_reason, move_reason, difficulty, rng,
            )
            correct = loc_original
            egocentric = loc_new
        else:
            story = cls._build_second_order_story(
                char_a, char_b, char_c, obj, loc_original, loc_new,
                scenario, absence_reason, move_reason, difficulty, rng,
            )
            correct = loc_original
            egocentric = loc_new

        return {
            "story": story,
            "correct_answer": correct,
            "egocentric_answer": egocentric,
            "original_location": loc_original,
            "new_location": loc_new,
            "characters": names[:2] if order == 1 else names[:3],
        }

    @classmethod
    def _build_first_order_story(
        cls, char_a, char_b, obj, loc_orig, loc_new,
        scenario, absence_reason, move_reason, difficulty, rng,
    ) -> str:
        """Build a first-order false belief story."""
        # Core story
        lines = [
            f"{char_a} and {char_b} are together in {scenario}.",
            f"{char_b} places a {obj} in {loc_orig}.",
            f"{char_b} then leaves the room because they {absence_reason}.",
            f"While {char_b} is away, {char_a} moves the {obj} "
            f"from {loc_orig} to {loc_new}.",
        ]

        if difficulty == "hard":
            # Add irrelevant detail to increase complexity
            extra_obj = rng.choice([o for o in _OBJECT_POOL if o != obj])
            extra_loc = rng.choice([l for l in _CONTAINER_POOL
                                    if l not in (loc_orig, loc_new)])
            lines.insert(2, (
                f"There is also a {extra_obj} sitting on the table "
                f"near {extra_loc}."
            ))
            lines.append(
                f"{char_a} also rearranges some other items in the room."
            )

        lines.append(f"{char_b} now returns to the room.")
        lines.append("")
        lines.append(
            f"Question: {char_b} wants to find the {obj}. "
            f"Where will {char_b} FIRST look for it?\n"
            f"Answer with the location only (e.g., \"{loc_orig}\" "
            f"or \"{loc_new}\")."
        )
        return "\n".join(lines)

    @classmethod
    def _build_second_order_story(
        cls, char_a, char_b, char_c, obj, loc_orig, loc_new,
        scenario, absence_reason, move_reason, difficulty, rng,
    ) -> str:
        """Build a second-order false belief story.

        Setup: B puts object in loc_orig. B leaves. A moves object to
        loc_new. HOWEVER, B secretly peeks through the window and sees
        A move it. A does NOT know that B saw.

        Question: Where does A think B will look for the object?
        Answer: loc_orig (because A thinks B doesn't know about the move).
        """
        lines = [
            f"{char_a} and {char_b} are together in {scenario}.",
            f"{char_b} places a {obj} in {loc_orig}.",
            f"{char_b} then leaves the room because they {absence_reason}.",
            f"While {char_b} is away, {char_a} moves the {obj} "
            f"from {loc_orig} to {loc_new} because they {move_reason}.",
            f"However, {char_b} secretly peeks through the window "
            f"and sees {char_a} move the {obj} to {loc_new}.",
            f"{char_a} does NOT know that {char_b} saw the move.",
        ]

        if difficulty == "hard":
            lines.insert(3, (
                f"{char_c} walks through the room during this time "
                f"but does not touch anything."
            ))
            extra_action = rng.choice([
                f"{char_a} then closes the curtains.",
                f"{char_a} sits back down at the table.",
                f"A clock chimes in the background.",
            ])
            lines.append(extra_action)

        lines.append(f"{char_b} now comes back into the room.")
        lines.append("")
        lines.append(
            f"Question: Where does {char_a} think {char_b} will look "
            f"for the {obj}?\n"
            f"Answer with the location only (e.g., \"{loc_orig}\" "
            f"or \"{loc_new}\")."
        )
        return "\n".join(lines)

    # Classic false-belief scenario variants — all well-known in the
    # cognitive science literature and likely in LLM training data.
    _CLASSIC_VARIANTS = [
        # Original Sally-Anne (Baron-Cohen et al., 1985)
        {"char_a": "Sally", "char_b": "Anne", "object": "marble",
         "loc_a": "the basket", "loc_b": "the box", "setting": "a room"},
        # Maxi chocolate (Wimmer & Perner, 1983)
        {"char_a": "Maxi", "char_b": "his mother", "object": "chocolate",
         "loc_a": "the blue cupboard", "loc_b": "the green cupboard",
         "setting": "the kitchen"},
        # Smarties task variant
        {"char_a": "Sam", "char_b": "Lisa", "object": "toy car",
         "loc_a": "the drawer", "loc_b": "the cupboard",
         "setting": "the playroom"},
        # Band-Aid box variant
        {"char_a": "John", "char_b": "Mary", "object": "pencil",
         "loc_a": "the pencil case", "loc_b": "the backpack",
         "setting": "the classroom"},
        # Classic location-change variant
        {"char_a": "Emma", "char_b": "Tom", "object": "ball",
         "loc_a": "the box", "loc_b": "the bag",
         "setting": "the garden"},
    ]

    @classmethod
    def _generate_classic_sally_anne(
        cls, rng: random.Random, order: int,
    ) -> Dict[str, Any]:
        """Classic false-belief scenarios from the literature (contamination probes).

        Uses well-known variants (Sally-Anne, Maxi, etc.) that are highly
        likely to appear in LLM training data. Each call picks a random
        variant to avoid identical items.
        """
        v = rng.choice(cls._CLASSIC_VARIANTS)
        char_a, char_b = v["char_a"], v["char_b"]
        obj, loc_a, loc_b = v["object"], v["loc_a"], v["loc_b"]
        setting = v["setting"]

        if order == 1:
            story = (
                f"{char_a} and {char_b} are together in {setting}.\n"
                f"{char_a} places a {obj} in {loc_a}.\n"
                f"{char_a} leaves the room.\n"
                f"While {char_a} is away, {char_b} moves the {obj} from "
                f"{loc_a} to {loc_b}.\n"
                f"{char_a} returns to the room.\n\n"
                f"Question: Where will {char_a} look for the {obj}?\n"
                f"Answer with the location only "
                f"(e.g., \"{loc_a}\" or \"{loc_b}\")."
            )
            correct = loc_a
            egocentric = loc_b
        else:
            story = (
                f"{char_a} and {char_b} are together in {setting}.\n"
                f"{char_a} places a {obj} in {loc_a}.\n"
                f"{char_a} leaves the room.\n"
                f"While {char_a} is away, {char_b} moves the {obj} from "
                f"{loc_a} to {loc_b}.\n"
                f"However, {char_a} secretly peeks through the window "
                f"and sees {char_b} move the {obj} to {loc_b}.\n"
                f"{char_b} does NOT know that {char_a} saw the move.\n"
                f"{char_a} comes back into the room.\n\n"
                f"Question: Where does {char_b} think {char_a} will look "
                f"for the {obj}?\n"
                f"Answer with the location only "
                f"(e.g., \"{loc_a}\" or \"{loc_b}\")."
            )
            correct = loc_a
            egocentric = loc_b

        return {
            "story": story,
            "correct_answer": correct,
            "egocentric_answer": egocentric,
            "original_location": loc_a,
            "new_location": loc_b,
            "characters": [char_a, char_b],
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, float]:
        """Score a single False Belief trial.

        Returns accuracy, and egocentric_error flag.
        """
        resp_lower = response.strip().lower()
        correct_lower = str(task.expected_response).strip().lower()
        ego_answer = task.metadata.parameters["egocentric_answer"].lower()

        # Flexible matching: check if correct answer is contained in response
        is_correct = correct_lower in resp_lower
        is_egocentric = ego_answer in resp_lower and not is_correct

        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "egocentric_error": 1.0 if is_egocentric else 0.0,
            "order": float(task.metadata.parameters["order"]),
        }


def score_false_belief(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Module-level scoring function for False Belief (custom fn path)."""
    resp_lower = str(response).strip().lower()
    correct_lower = str(expected).strip().lower()
    ego_answer = metadata.parameters.get("egocentric_answer", "").lower()

    is_correct = correct_lower in resp_lower
    is_egocentric = ego_answer in resp_lower and not is_correct

    return {
        "accuracy": 1.0 if is_correct else 0.0,
        "egocentric_error": 1.0 if is_egocentric else 0.0,
    }


# ===================================================================
# PARADIGM 2 -- EPITOME-style Multi-aspect ToM
# ===================================================================

# Vignette templates for each sub-capacity.
# Each template has: scenario, question, correct_answer, wrong_answer

_BELIEF_SCENARIOS: list[Dict[str, str]] = [
    {
        "template": (
            "{char_a} told {char_b} that the meeting starts at {time_wrong}. "
            "In reality, the meeting starts at {time_real}. "
            "{char_b} did not check the schedule themselves."
        ),
        "question": "What time does {char_b} believe the meeting starts?",
        "correct_key": "time_wrong",
        "wrong_key": "time_real",
    },
    {
        "template": (
            "{char_a} put a {item} in {location_a} and told {char_b} about it. "
            "Later, without {char_b} knowing, {char_a} moved the {item} "
            "to {location_b}."
        ),
        "question": "Where does {char_b} believe the {item} is?",
        "correct_key": "location_a",
        "wrong_key": "location_b",
    },
    {
        "template": (
            "{char_b} heard from {char_a} that the store on {street} "
            "is closed today. Actually, the store is open; {char_a} "
            "was mistaken."
        ),
        "question": "Does {char_b} believe the store is open or closed?",
        "correct": "closed",
        "wrong": "open",
    },
    {
        "template": (
            "{char_a} sent {char_b} a message saying the project deadline "
            "was extended to {date_wrong}. However, the actual deadline "
            "remains {date_real}. {char_b} has not seen any other updates."
        ),
        "question": "What does {char_b} believe the project deadline is?",
        "correct_key": "date_wrong",
        "wrong_key": "date_real",
    },
    {
        "template": (
            "{char_b} was told by {char_a} that {char_c} is arriving "
            "by {transport_wrong}. In fact, {char_c} is arriving "
            "by {transport_real}. {char_b} has no other information."
        ),
        "question": "How does {char_b} believe {char_c} is arriving?",
        "correct_key": "transport_wrong",
        "wrong_key": "transport_real",
    },
]

_DESIRE_SCENARIOS: list[Dict[str, str]] = [
    {
        "template": (
            "{char_a} has been talking all week about wanting to visit "
            "{place}. {char_a} has been looking up directions and "
            "checking the weather forecast for {place}."
        ),
        "question": "What does {char_a} want to do?",
        "correct": "visit {place}",
    },
    {
        "template": (
            "{char_a} keeps looking at {item} in the shop window every day "
            "on the way to work. {char_a} has been saving money and "
            "comparing prices online."
        ),
        "question": "What does {char_a} want?",
        "correct": "to buy {item}",
    },
    {
        "template": (
            "{char_a} skipped lunch to keep practising the {skill}. "
            "{char_a} has signed up for extra lessons and stays late "
            "every day to practise."
        ),
        "question": "What does {char_a} want to achieve?",
        "correct": "to improve at the {skill}",
    },
    {
        "template": (
            "{char_a} has been studying {subject} late into the night, "
            "turning down invitations to social events. {char_a} "
            "mentioned hoping for a top score on the upcoming exam."
        ),
        "question": "What does {char_a} want?",
        "correct": "to get a top score on the {subject} exam",
    },
    {
        "template": (
            "{char_a} has been preparing {food} for {char_b}'s surprise "
            "birthday party. {char_a} bought decorations and invited "
            "all of {char_b}'s friends in secret."
        ),
        "question": "What does {char_a} want to do?",
        "correct": "to throw a surprise birthday party for {char_b}",
    },
]

_INTENTION_SCENARIOS: list[Dict[str, str]] = [
    {
        "template": (
            "{char_a} put on a raincoat and grabbed an umbrella "
            "before heading out the door, even though it was not "
            "raining yet."
        ),
        "question": "Why did {char_a} take the raincoat and umbrella?",
        "correct": "because {char_a} expected it to rain",
    },
    {
        "template": (
            "{char_a} quietly closed the laptop and tiptoed out of "
            "the room where {char_b} was sleeping."
        ),
        "question": "Why did {char_a} tiptoe?",
        "correct": "to avoid waking {char_b}",
    },
    {
        "template": (
            "{char_a} brought a gift-wrapped box and a card to "
            "{char_b}'s front door and rang the bell."
        ),
        "question": "Why did {char_a} go to {char_b}'s door?",
        "correct": "to give {char_b} a gift",
    },
    {
        "template": (
            "{char_a} hid behind the door and waited quietly as "
            "{char_b} approached the room. When {char_b} walked in, "
            "{char_a} jumped out and shouted."
        ),
        "question": "Why did {char_a} hide behind the door?",
        "correct": "to surprise {char_b}",
    },
    {
        "template": (
            "{char_a} cleared the table, set out plates, lit candles, "
            "and opened a bottle of wine before {char_b} arrived home."
        ),
        "question": "Why did {char_a} prepare the table this way?",
        "correct": "to set up a special dinner for {char_b}",
    },
]

_EMOTION_SCENARIOS: list[Dict[str, str]] = [
    {
        "template": (
            "{char_a} had been preparing for the presentation for weeks. "
            "When {char_a} finished presenting, the audience gave a "
            "standing ovation."
        ),
        "question": "How does {char_a} most likely feel?",
        "correct": "proud",
        "options": ["proud", "anxious", "indifferent", "angry"],
    },
    {
        "template": (
            "{char_a} arrived at the airport and discovered their flight "
            "had been cancelled with no alternative available that day. "
            "{char_a} was supposed to attend an important family event."
        ),
        "question": "How does {char_a} most likely feel?",
        "correct": "frustrated",
        "options": ["frustrated", "relieved", "amused", "calm"],
    },
    {
        "template": (
            "{char_a} spent the entire afternoon baking a cake for "
            "{char_b}. When {char_a} took it out of the oven, it had "
            "collapsed and was burnt."
        ),
        "question": "How does {char_a} most likely feel?",
        "correct": "disappointed",
        "options": ["disappointed", "delighted", "bored", "suspicious"],
    },
    {
        "template": (
            "{char_a} had not heard from {char_b} in several days. "
            "Then {char_a} received a late-night message saying "
            "{char_b} was in the hospital."
        ),
        "question": "How does {char_a} most likely feel?",
        "correct": "worried",
        "options": ["worried", "cheerful", "envious", "neutral"],
    },
    {
        "template": (
            "{char_a} worked overtime for a month on a project. "
            "At the team meeting, the manager gave all the credit "
            "to {char_b}, who had barely contributed."
        ),
        "question": "How does {char_a} most likely feel?",
        "correct": "resentful",
        "options": ["resentful", "grateful", "relaxed", "indifferent"],
    },
    {
        "template": (
            "{char_a} found a handwritten thank-you note from {char_b} "
            "tucked inside a book. The note expressed deep gratitude "
            "for {char_a}'s help during a difficult time."
        ),
        "question": "How does {char_a} most likely feel?",
        "correct": "touched",
        "options": ["touched", "annoyed", "confused", "bored"],
    },
    {
        "template": (
            "{char_a} walked into the surprise party that {char_b} "
            "had secretly organised. All of {char_a}'s friends were "
            "there, cheering."
        ),
        "question": "How does {char_a} most likely feel?",
        "correct": "surprised and happy",
        "options": ["surprised and happy", "angry", "sad", "suspicious"],
    },
    {
        "template": (
            "{char_a} confided a personal secret to {char_b}, asking "
            "them to keep it private. The next day, {char_a} overheard "
            "{char_b} telling the secret to others."
        ),
        "question": "How does {char_a} most likely feel?",
        "correct": "betrayed",
        "options": ["betrayed", "proud", "amused", "relieved"],
    },
]

# Filler variable pools for template instantiation
_TIME_POOL = [
    "9:00 AM", "10:30 AM", "2:00 PM", "3:15 PM", "4:45 PM",
    "8:00 AM", "11:00 AM", "1:00 PM", "5:30 PM", "6:00 PM",
]
_DATE_POOL = [
    "next Monday", "Friday", "March 15th", "the end of the month",
    "next Wednesday", "April 3rd", "two weeks from now", "tomorrow",
]
_TRANSPORT_POOL = ["train", "bus", "car", "plane", "bicycle", "taxi", "boat"]
_PLACE_POOL = [
    "the botanical garden", "the history museum", "the seaside pier",
    "the mountain trail", "the night market", "the old castle ruins",
    "the national park", "the local art gallery", "the lakeside cabin",
    "the hot springs resort",
]
_ITEM_SHOP_POOL = [
    "a vintage camera", "a leather jacket", "a pair of running shoes",
    "a ceramic tea set", "a mechanical wristwatch", "a silk scarf",
    "a handcrafted vase", "a hardcover cookbook", "a telescope",
    "a portable record player",
]
_SKILL_POOL = [
    "piano", "pottery", "archery", "calligraphy", "chess",
    "juggling", "origami", "fencing", "photography", "coding",
]
_SUBJECT_POOL = [
    "organic chemistry", "linear algebra", "microeconomics",
    "classical literature", "quantum mechanics", "statistics",
    "molecular biology", "music theory", "philosophy", "linguistics",
]
_FOOD_POOL = [
    "a three-layer cake", "homemade pasta", "a fruit tart",
    "sushi rolls", "a roast dinner", "dumplings", "a chocolate souffle",
    "a vegetable stew", "grilled salmon", "a cheese board",
]
_STREET_POOL = [
    "Elm Street", "Maple Avenue", "Oak Drive", "Pine Road",
    "Cedar Lane", "Birch Boulevard", "Willow Way", "Spruce Court",
]


class EpitomeToMGenerator:
    """Procedural generator for EPITOME-style multi-aspect ToM tasks.

    Tests four sub-capacities:
      - belief: What does X believe about Y?
      - desire: What does X want?
      - intention: Why did X do Z?
      - emotion: How does X feel?
    """

    PARADIGM = "epitome_tom"
    DIMENSION = "theory_of_mind"

    SUB_CAPACITIES = ("belief", "desire", "intention", "emotion")

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 20,
        sub_capacity: str = "all",
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> list[TaskInstance]:
        """Generate EPITOME-style ToM items.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Total number of items. If sub_capacity="all", distributed
            evenly across the four sub-capacities.
        sub_capacity : str
            "belief", "desire", "intention", "emotion", or "all".
        difficulty : str
            "easy", "medium", "hard".
        contamination_probe : bool
            Not directly applicable; flag stored for consistency.
        """
        rng = _make_rng(seed)
        items: list[TaskInstance] = []

        if sub_capacity == "all":
            caps = list(cls.SUB_CAPACITIES)
            per_cap = max(1, n_items // len(caps))
            remainder = n_items - per_cap * len(caps)
        else:
            caps = [sub_capacity]
            per_cap = n_items
            remainder = 0

        for cap_idx, cap in enumerate(caps):
            count = per_cap + (1 if cap_idx < remainder else 0)
            for _ in range(count):
                ep_seed = rng.randint(0, 2**31)
                ep_rng = _make_rng(ep_seed)

                vignette = cls._generate_vignette(ep_rng, cap, difficulty)

                task_id = (
                    f"tom_epitome_{cap}"
                    f"_{'probe' if contamination_probe else 'gen'}"
                    f"_{difficulty}_s{ep_seed}"
                )

                metadata = TaskMetadata(
                    dimension=cls.DIMENSION,
                    paradigm=cls.PARADIGM,
                    mode=EvalMode.LLM_STATIC,
                    parameters={
                        "sub_capacity": cap,
                        "correct_answer": vignette["correct_answer"],
                        "episode_seed": ep_seed,
                        "contamination_probe": contamination_probe,
                        "multi_turn": False,
                    },
                    scoring=ScoringConfig(
                        method="custom",
                        params={
                            "fn": "cogarena.dimensions.theory_of_mind.score_epitome_tom",
                            "sub_capacity": cap,
                        },
                    ),
                    difficulty=_difficulty_enum(difficulty),
                    adaptation_distance=AdaptationDistance.LOW,
                    description=f"EPITOME ToM -- {cap} attribution",
                )

                stimulus = vignette["stimulus"]

                items.append(TaskInstance(
                    task_id=task_id,
                    metadata=metadata,
                    stimulus=stimulus,
                    expected_response=vignette["correct_answer"],
                ))

        return items

    @classmethod
    def _generate_vignette(
        cls,
        rng: random.Random,
        sub_capacity: str,
        difficulty: str,
    ) -> Dict[str, Any]:
        """Generate a single vignette for the given sub-capacity."""
        names = rng.sample(_NAME_POOL, k=3)
        char_a, char_b, char_c = names[0], names[1], names[2]

        if sub_capacity == "belief":
            return cls._gen_belief(rng, char_a, char_b, char_c, difficulty)
        elif sub_capacity == "desire":
            return cls._gen_desire(rng, char_a, char_b, difficulty)
        elif sub_capacity == "intention":
            return cls._gen_intention(rng, char_a, char_b, difficulty)
        elif sub_capacity == "emotion":
            return cls._gen_emotion(rng, char_a, char_b, difficulty)
        else:
            raise ValueError(f"Unknown sub_capacity: {sub_capacity}")

    @classmethod
    def _gen_belief(cls, rng, char_a, char_b, char_c, difficulty):
        tmpl = rng.choice(_BELIEF_SCENARIOS)
        # Generate fill values
        time_wrong = rng.choice(_TIME_POOL)
        time_real_options = [t for t in _TIME_POOL if t != time_wrong]
        time_real = rng.choice(time_real_options)
        location_a = rng.choice(_CONTAINER_POOL)
        location_b = rng.choice([l for l in _CONTAINER_POOL if l != location_a])
        item = rng.choice(_OBJECT_POOL)
        date_wrong = rng.choice(_DATE_POOL)
        date_real = rng.choice([d for d in _DATE_POOL if d != date_wrong])
        transport_wrong = rng.choice(_TRANSPORT_POOL)
        transport_real = rng.choice(
            [t for t in _TRANSPORT_POOL if t != transport_wrong]
        )
        street = rng.choice(_STREET_POOL)

        fill = {
            "char_a": char_a, "char_b": char_b, "char_c": char_c,
            "time_wrong": time_wrong, "time_real": time_real,
            "location_a": location_a, "location_b": location_b,
            "item": item,
            "date_wrong": date_wrong, "date_real": date_real,
            "transport_wrong": transport_wrong, "transport_real": transport_real,
            "street": street,
        }

        story = tmpl["template"].format(**fill)
        question = tmpl["question"].format(**fill)

        if "correct" in tmpl:
            correct = tmpl["correct"]
            wrong = tmpl.get("wrong", "")
        else:
            correct = fill[tmpl["correct_key"]]
            wrong = fill.get(tmpl.get("wrong_key", ""), "")

        if difficulty == "hard":
            # Add distracting detail
            distractor = rng.choice([
                f"Earlier that day, {char_c} had mentioned something "
                f"about a different schedule.",
                f"The weather outside was unusually warm for the season.",
                f"There was construction noise coming from across the street.",
            ])
            story = story + " " + distractor

        # Convert to A/B multiple choice
        if wrong:
            options = [correct, wrong]
            rng.shuffle(options)
            labels = ["A", "B"]
            correct_label = labels[options.index(correct)]
            options_text = "\n".join(f"  ({labels[i]}) {options[i]}" for i in range(len(options)))
            stimulus = f"{story}\n\n{question}\n{options_text}\nAnswer with the letter only (A or B)."
        else:
            correct_label = correct
            options = None
            stimulus = f"{story}\n\n{question}\nAnswer concisely."

        return {
            "stimulus": stimulus,
            "correct_answer": correct_label,
            "wrong_answer": wrong,
            "options": options,
        }

    @classmethod
    def _gen_desire(cls, rng, char_a, char_b, difficulty):
        tmpl = rng.choice(_DESIRE_SCENARIOS)
        place = rng.choice(_PLACE_POOL)
        item = rng.choice(_ITEM_SHOP_POOL)
        skill = rng.choice(_SKILL_POOL)
        subject = rng.choice(_SUBJECT_POOL)
        food = rng.choice(_FOOD_POOL)

        fill = {
            "char_a": char_a, "char_b": char_b,
            "place": place, "item": item, "skill": skill,
            "subject": subject, "food": food,
        }

        story = tmpl["template"].format(**fill)
        question = tmpl["question"].format(**fill)
        correct = tmpl["correct"].format(**fill)

        # Generate 3 distractors from other desire templates
        distractors = []
        other_tmpls = [t for t in _DESIRE_SCENARIOS if t is not tmpl]
        rng.shuffle(other_tmpls)
        for dt in other_tmpls[:3]:
            dfill = dict(fill)
            dfill["place"] = rng.choice([p for p in _PLACE_POOL if p != place])
            dfill["item"] = rng.choice([i for i in _ITEM_SHOP_POOL if i != item])
            dfill["skill"] = rng.choice([s for s in _SKILL_POOL if s != skill])
            dfill["subject"] = rng.choice([s for s in _SUBJECT_POOL if s != subject])
            dfill["food"] = rng.choice([f for f in _FOOD_POOL if f != food])
            distractors.append(dt["correct"].format(**dfill))

        options = [correct] + distractors[:3]
        rng.shuffle(options)
        labels = ["A", "B", "C", "D"]
        correct_label = labels[options.index(correct)]
        options_text = "\n".join(f"  ({labels[i]}) {options[i]}" for i in range(len(options)))

        if difficulty == "hard":
            distractor = rng.choice([
                f"{char_a} also briefly mentioned {rng.choice(_PLACE_POOL)} "
                f"but did not seem very interested.",
                f"{char_b} suggested a different activity, but {char_a} "
                f"politely declined.",
            ])
            story = story + " " + distractor

        stimulus = f"{story}\n\n{question}\n{options_text}\nAnswer with the letter only (A, B, C, or D)."
        return {
            "stimulus": stimulus,
            "correct_answer": correct_label,
            "options": options,
        }

    @classmethod
    def _gen_intention(cls, rng, char_a, char_b, difficulty):
        tmpl = rng.choice(_INTENTION_SCENARIOS)
        fill = {"char_a": char_a, "char_b": char_b}

        story = tmpl["template"].format(**fill)
        question = tmpl["question"].format(**fill)
        correct = tmpl["correct"].format(**fill)

        # Generate 3 distractors from other intention templates
        distractors = []
        other_tmpls = [t for t in _INTENTION_SCENARIOS if t is not tmpl]
        rng.shuffle(other_tmpls)
        for dt in other_tmpls[:3]:
            distractors.append(dt["correct"].format(**fill))

        options = [correct] + distractors[:3]
        rng.shuffle(options)
        labels = ["A", "B", "C", "D"]
        correct_label = labels[options.index(correct)]
        options_text = "\n".join(f"  ({labels[i]}) {options[i]}" for i in range(len(options)))

        if difficulty == "hard":
            distractor = rng.choice([
                f"The hallway was dimly lit at the time.",
                f"There were other people nearby who did not notice.",
                f"It was late in the evening and the building was quiet.",
            ])
            story = story + " " + distractor

        stimulus = f"{story}\n\n{question}\n{options_text}\nAnswer with the letter only (A, B, C, or D)."
        return {
            "stimulus": stimulus,
            "correct_answer": correct_label,
            "options": options,
        }

    @classmethod
    def _gen_emotion(cls, rng, char_a, char_b, difficulty):
        tmpl = rng.choice(_EMOTION_SCENARIOS)
        fill = {"char_a": char_a, "char_b": char_b}

        story = tmpl["template"].format(**fill)
        question = tmpl["question"].format(**fill)
        correct = tmpl["correct"]
        options = list(tmpl.get("options", [correct]))

        # Shuffle options and assign letter labels
        rng.shuffle(options)
        labels = ["A", "B", "C", "D"][:len(options)]
        correct_label = labels[options.index(correct)]
        options_text = "\n".join(f"  ({labels[i]}) {options[i]}" for i in range(len(options)))

        if difficulty == "hard":
            distractor = rng.choice([
                f"{char_a} had also had a long day at work.",
                f"Earlier, {char_a} had been in a neutral mood.",
                f"The setting was otherwise unremarkable.",
            ])
            story = story + " " + distractor

        stimulus = f"{story}\n\n{question}\n{options_text}\nAnswer with the letter only ({', '.join(labels)})."
        return {
            "stimulus": stimulus,
            "correct_answer": correct_label,
            "options": options,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, float]:
        """Score a single EPITOME ToM trial (multiple-choice format)."""
        resp_clean = response.strip().upper()
        correct = str(task.expected_response).strip().upper()
        sub_cap = task.metadata.parameters["sub_capacity"]

        # Match on letter (A, B, C, D) — extract first letter from response
        resp_letter = ""
        for ch in resp_clean:
            if ch in "ABCD":
                resp_letter = ch
                break

        is_correct = resp_letter == correct

        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "sub_capacity": sub_cap,
        }


def score_epitome_tom(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Module-level scoring function for EPITOME ToM (multiple-choice)."""
    resp_clean = str(response).strip().upper()
    correct = str(expected).strip().upper()

    # Extract first A/B/C/D letter from response
    resp_letter = ""
    for ch in resp_clean:
        if ch in "ABCD":
            resp_letter = ch
            break

    is_correct = resp_letter == correct

    return {
        "accuracy": 1.0 if is_correct else 0.0,
    }


# ===================================================================
# Convenience dispatch
# ===================================================================

_GENERATORS: Dict[str, type] = {
    "false_belief": FalseBeliefGenerator,
    "epitome_tom": EpitomeToMGenerator,
}


def generate(
    paradigm: str,
    seed: int,
    n_items: int = 20,
    difficulty: str = "medium",
    contamination_probe: bool = False,
    **kwargs: Any,
) -> list[TaskInstance]:
    """Unified entry-point for generating Theory of Mind items.

    Parameters
    ----------
    paradigm : str
        One of "false_belief", "epitome_tom".
    seed, n_items, difficulty, contamination_probe
        Forwarded to the paradigm generator.
    **kwargs
        Extra keyword arguments (e.g., ``order`` for false_belief,
        ``sub_capacity`` for epitome_tom).
    """
    gen_cls = _GENERATORS.get(paradigm)
    if gen_cls is None:
        raise ValueError(
            f"Unknown paradigm '{paradigm}'. Choose from {list(_GENERATORS)}"
        )
    return gen_cls.generate(
        seed=seed,
        n_items=n_items,
        difficulty=difficulty,
        contamination_probe=contamination_probe,
        **kwargs,
    )


def score(task: TaskInstance, response: Any) -> Dict[str, float]:
    """Unified scoring dispatcher."""
    gen_cls = _GENERATORS.get(task.metadata.paradigm)
    if gen_cls is None:
        raise ValueError(f"Unknown paradigm '{task.metadata.paradigm}'")
    return gen_cls.score(task, response)
