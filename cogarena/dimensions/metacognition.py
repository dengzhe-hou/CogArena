"""Metacognitive Monitoring dimension for CogArena.

Implements two paradigms with procedural generation:
  1. Confidence Calibration  -- answer + confidence rating (0-100%)
  2. Post-Decision Wagering  -- answer + bet (YES/NO) with payoffs

Questions are procedurally generated from hardcoded template pools
across diverse domains to minimise contamination.
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from cogarena.core import (
    AdaptationDistance,
    DifficultyLevel,
    EvalMode,
    ScoringConfig,
    TaskInstance,
    TaskMetadata,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _difficulty_enum(s: str) -> DifficultyLevel:
    return DifficultyLevel(s.lower())


# ---------------------------------------------------------------------------
# Knowledge question pools (~250 questions across diverse domains)
# Each entry: (question, answer, domain, difficulty)
#   difficulty: "easy" / "medium" / "hard"
# ---------------------------------------------------------------------------

_QUESTION_POOL: list[Tuple[str, str, str, str]] = [
    # ===== GEOGRAPHY -- Capitals =====
    ("What is the capital of France?", "Paris", "geography", "easy"),
    ("What is the capital of Japan?", "Tokyo", "geography", "easy"),
    ("What is the capital of Brazil?", "Brasilia", "geography", "easy"),
    ("What is the capital of Australia?", "Canberra", "geography", "medium"),
    ("What is the capital of Canada?", "Ottawa", "geography", "easy"),
    ("What is the capital of South Korea?", "Seoul", "geography", "easy"),
    ("What is the capital of Turkey?", "Ankara", "geography", "medium"),
    ("What is the capital of Nigeria?", "Abuja", "geography", "medium"),
    ("What is the capital of Myanmar?", "Naypyidaw", "geography", "hard"),
    ("What is the capital of Sri Lanka?", "Sri Jayawardenepura Kotte", "geography", "hard"),
    ("What is the capital of Pakistan?", "Islamabad", "geography", "medium"),
    ("What is the capital of Morocco?", "Rabat", "geography", "medium"),
    ("What is the capital of Switzerland?", "Bern", "geography", "medium"),
    ("What is the capital of New Zealand?", "Wellington", "geography", "medium"),
    ("What is the capital of Ivory Coast?", "Yamoussoukro", "geography", "hard"),
    ("What is the capital of Mongolia?", "Ulaanbaatar", "geography", "hard"),
    ("What is the capital of Bolivia?", "Sucre", "geography", "hard"),
    ("What is the capital of Belize?", "Belmopan", "geography", "hard"),
    ("What is the capital of Palau?", "Ngerulmud", "geography", "hard"),
    ("What is the capital of Malta?", "Valletta", "geography", "hard"),

    # ===== CHEMISTRY -- Elements =====
    ("What is the chemical symbol for gold?", "Au", "chemistry", "easy"),
    ("What is the chemical symbol for iron?", "Fe", "chemistry", "easy"),
    ("What is the chemical symbol for sodium?", "Na", "chemistry", "easy"),
    ("What is the chemical symbol for silver?", "Ag", "chemistry", "medium"),
    ("What is the chemical symbol for potassium?", "K", "chemistry", "medium"),
    ("What is the chemical symbol for tungsten?", "W", "chemistry", "medium"),
    ("What is the chemical symbol for mercury?", "Hg", "chemistry", "medium"),
    ("What is the chemical symbol for lead?", "Pb", "chemistry", "medium"),
    ("What is the chemical symbol for tin?", "Sn", "chemistry", "medium"),
    ("What is the chemical symbol for antimony?", "Sb", "chemistry", "hard"),
    ("What is the chemical symbol for copper?", "Cu", "chemistry", "easy"),
    ("What is the atomic number of carbon?", "6", "chemistry", "easy"),
    ("What is the atomic number of oxygen?", "8", "chemistry", "easy"),
    ("What is the atomic number of neon?", "10", "chemistry", "medium"),
    ("What is the atomic number of iron?", "26", "chemistry", "medium"),
    ("What is the atomic number of uranium?", "92", "chemistry", "hard"),
    ("What is the atomic number of gold?", "79", "chemistry", "hard"),
    ("What is the atomic number of zinc?", "30", "chemistry", "medium"),
    ("What element has the highest melting point?", "tungsten", "chemistry", "hard"),
    ("What is the lightest noble gas?", "helium", "chemistry", "easy"),

    # ===== HISTORY -- Years =====
    ("In what year did World War II end?", "1945", "history", "easy"),
    ("In what year did the Berlin Wall fall?", "1989", "history", "easy"),
    ("In what year was the Declaration of Independence signed?", "1776", "history", "easy"),
    ("In what year did the French Revolution begin?", "1789", "history", "medium"),
    ("In what year did the Titanic sink?", "1912", "history", "easy"),
    ("In what year was the Magna Carta signed?", "1215", "history", "medium"),
    ("In what year did humans first walk on the Moon?", "1969", "history", "easy"),
    ("In what year did the Russian Revolution occur?", "1917", "history", "medium"),
    ("In what year did the printing press become widely used in Europe?", "1440", "history", "hard"),
    ("In what year was the Treaty of Westphalia signed?", "1648", "history", "hard"),
    ("In what year did the Ottoman Empire fall?", "1922", "history", "medium"),
    ("In what year was Machu Picchu built (approximately)?", "1450", "history", "hard"),
    ("In what year did the Spanish Armada sail?", "1588", "history", "hard"),
    ("In what year was the United Nations established?", "1945", "history", "medium"),
    ("In what year did the Chernobyl disaster occur?", "1986", "history", "medium"),
    ("In what year was the Suez Canal completed?", "1869", "history", "hard"),
    ("In what year did India gain independence?", "1947", "history", "medium"),
    ("In what year did the Korean War begin?", "1950", "history", "medium"),
    ("In what year was the Panama Canal completed?", "1914", "history", "hard"),
    ("In what year did the Chinese Cultural Revolution begin?", "1966", "history", "hard"),

    # ===== BIOLOGY =====
    ("How many chromosomes do humans have?", "46", "biology", "easy"),
    ("What is the largest organ in the human body?", "skin", "biology", "easy"),
    ("What is the powerhouse of the cell?", "mitochondria", "biology", "easy"),
    ("How many bones are in the adult human body?", "206", "biology", "medium"),
    ("What is the smallest bone in the human body?", "stapes", "biology", "medium"),
    ("What protein carries oxygen in red blood cells?", "hemoglobin", "biology", "medium"),
    ("What is the largest living organism by area?", "honey fungus", "biology", "hard"),
    ("What is the main pigment in plant photosynthesis?", "chlorophyll", "biology", "easy"),
    ("How many chambers does the human heart have?", "4", "biology", "easy"),
    ("What is the longest bone in the human body?", "femur", "biology", "easy"),
    ("What blood type is known as the universal donor?", "O negative", "biology", "medium"),
    ("What organelle contains the cell's genetic material?", "nucleus", "biology", "easy"),
    ("What is the fastest land animal?", "cheetah", "biology", "easy"),
    ("What is the average resting heart rate for adults (in bpm)?", "72", "biology", "medium"),
    ("What vitamin does the skin produce when exposed to sunlight?", "vitamin D", "biology", "medium"),
    ("What is the name of the process by which cells divide?", "mitosis", "biology", "medium"),
    ("What enzyme breaks down starch in saliva?", "amylase", "biology", "hard"),
    ("What is the pH of pure water?", "7", "biology", "easy"),
    ("What part of the brain controls balance?", "cerebellum", "biology", "medium"),
    ("What is the name of the pigment that determines skin color?", "melanin", "biology", "medium"),

    # ===== PHYSICS =====
    ("What is the speed of light in a vacuum (in km/s, approximately)?", "300000", "physics", "easy"),
    ("What is the SI unit of force?", "newton", "physics", "easy"),
    ("What is the acceleration due to gravity on Earth (in m/s^2)?", "9.8", "physics", "easy"),
    ("What is absolute zero in Celsius?", "-273.15", "physics", "medium"),
    ("What is the SI unit of electric current?", "ampere", "physics", "easy"),
    ("What particle has a positive charge in an atom?", "proton", "physics", "easy"),
    ("What is the charge of an electron (in coulombs, approximately)?", "1.6e-19", "physics", "hard"),
    ("What is Planck's constant (in J*s, order of magnitude)?", "6.626e-34", "physics", "hard"),
    ("What is the boiling point of water at standard pressure (in Celsius)?", "100", "physics", "easy"),
    ("What is the SI unit of energy?", "joule", "physics", "easy"),
    ("What is the SI unit of power?", "watt", "physics", "easy"),
    ("What is the SI unit of frequency?", "hertz", "physics", "easy"),
    ("What is the SI unit of pressure?", "pascal", "physics", "medium"),
    ("How many planets are in our solar system?", "8", "physics", "easy"),
    ("What is the closest star to Earth (besides the Sun)?", "Proxima Centauri", "physics", "medium"),
    ("What is the escape velocity from Earth (in km/s, approximately)?", "11.2", "physics", "hard"),
    ("What is the half-life of carbon-14 (in years, approximately)?", "5730", "physics", "hard"),
    ("What subatomic particle was discovered by James Chadwick?", "neutron", "physics", "medium"),
    ("What is the most abundant element in the universe?", "hydrogen", "physics", "medium"),
    ("What is the Schwarzschild radius formula proportional to?", "mass", "physics", "hard"),

    # ===== MATHEMATICS =====
    ("What is the value of pi to two decimal places?", "3.14", "math", "easy"),
    ("What is the square root of 144?", "12", "math", "easy"),
    ("What is 17 squared?", "289", "math", "medium"),
    ("What is the sum of angles in a triangle (in degrees)?", "180", "math", "easy"),
    ("How many faces does a dodecahedron have?", "12", "math", "medium"),
    ("What is the value of e (Euler's number) to two decimal places?", "2.72", "math", "medium"),
    ("What is the derivative of x^3?", "3x^2", "math", "medium"),
    ("What is the integral of 1/x?", "ln(x)", "math", "medium"),
    ("What is the 10th prime number?", "29", "math", "medium"),
    ("What is 2^10?", "1024", "math", "easy"),
    ("How many degrees are in a radian (approximately)?", "57.3", "math", "hard"),
    ("What is the factorial of 7?", "5040", "math", "medium"),
    ("What is the sum of the first 10 positive integers?", "55", "math", "easy"),
    ("What is log base 2 of 256?", "8", "math", "medium"),
    ("How many edges does a cube have?", "12", "math", "easy"),
    ("What is the golden ratio to two decimal places?", "1.62", "math", "hard"),
    ("What is the square root of 2 to two decimal places?", "1.41", "math", "medium"),
    ("How many vertices does an icosahedron have?", "12", "math", "hard"),
    ("What is the Fibonacci number at position 10 (starting from 1,1)?", "55", "math", "hard"),
    ("What is the cube root of 27?", "3", "math", "easy"),

    # ===== LITERATURE =====
    ("Who wrote 'Romeo and Juliet'?", "Shakespeare", "literature", "easy"),
    ("Who wrote '1984'?", "George Orwell", "literature", "easy"),
    ("Who wrote 'One Hundred Years of Solitude'?", "Gabriel Garcia Marquez", "literature", "medium"),
    ("Who wrote 'War and Peace'?", "Leo Tolstoy", "literature", "medium"),
    ("Who wrote 'The Divine Comedy'?", "Dante Alighieri", "literature", "medium"),
    ("Who wrote 'Crime and Punishment'?", "Fyodor Dostoevsky", "literature", "medium"),
    ("Who wrote 'Don Quixote'?", "Miguel de Cervantes", "literature", "medium"),
    ("Who wrote 'The Tale of Genji'?", "Murasaki Shikibu", "literature", "hard"),
    ("Who wrote 'Things Fall Apart'?", "Chinua Achebe", "literature", "medium"),
    ("Who wrote 'The Stranger'?", "Albert Camus", "literature", "medium"),
    ("Who wrote 'Beloved'?", "Toni Morrison", "literature", "medium"),
    ("Who wrote 'Metamorphosis'?", "Franz Kafka", "literature", "medium"),
    ("Who wrote 'In Search of Lost Time'?", "Marcel Proust", "literature", "hard"),
    ("Who wrote 'The Odyssey'?", "Homer", "literature", "easy"),
    ("Who wrote 'Moby-Dick'?", "Herman Melville", "literature", "easy"),
    ("Who wrote 'The Brothers Karamazov'?", "Fyodor Dostoevsky", "literature", "medium"),
    ("Who wrote 'Invisible Man'?", "Ralph Ellison", "literature", "hard"),
    ("Who wrote 'Ulysses'?", "James Joyce", "literature", "medium"),
    ("Who wrote 'The Iliad'?", "Homer", "literature", "easy"),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen", "literature", "easy"),

    # ===== MUSIC =====
    ("How many keys does a standard piano have?", "88", "music", "easy"),
    ("What instrument has four strings and is played with a bow?", "violin", "music", "easy"),
    ("How many symphonies did Beethoven compose?", "9", "music", "medium"),
    ("What is the highest female singing voice type?", "soprano", "music", "easy"),
    ("How many strings does a standard guitar have?", "6", "music", "easy"),
    ("What musical term means 'gradually getting louder'?", "crescendo", "music", "medium"),
    ("What key has no sharps or flats?", "C major", "music", "medium"),
    ("How many notes are in a chromatic scale?", "12", "music", "medium"),
    ("What is the lowest brass instrument in an orchestra?", "tuba", "music", "medium"),
    ("What musical period did Bach belong to?", "Baroque", "music", "medium"),
    ("How many strings does a cello have?", "4", "music", "easy"),
    ("What does 'fortissimo' mean?", "very loud", "music", "medium"),
    ("What instrument is Yo-Yo Ma famous for playing?", "cello", "music", "easy"),
    ("How many flats are in the key of B-flat major?", "2", "music", "hard"),
    ("What tempo marking means 'at a walking pace'?", "andante", "music", "hard"),
    ("What is the Italian term for a gradual decrease in tempo?", "ritardando", "music", "hard"),
    ("How many movements typically make up a classical symphony?", "4", "music", "medium"),
    ("What woodwind instrument uses a double reed?", "oboe", "music", "medium"),
    ("What note is concert pitch tuned to (in Hz)?", "440", "music", "hard"),
    ("What is the term for two notes played simultaneously?", "interval", "music", "medium"),

    # ===== LANGUAGE =====
    ("What is the most spoken native language in the world?", "Mandarin Chinese", "language", "easy"),
    ("How many letters are in the English alphabet?", "26", "language", "easy"),
    ("What language family does Japanese belong to?", "Japonic", "language", "hard"),
    ("How many tones does Mandarin Chinese have?", "4", "language", "medium"),
    ("What is the official language of Brazil?", "Portuguese", "language", "easy"),
    ("What script is used to write Hindi?", "Devanagari", "language", "medium"),
    ("How many cases does Russian have?", "6", "language", "hard"),
    ("What language family does Finnish belong to?", "Uralic", "language", "hard"),
    ("How many vowels are in the Hawaiian alphabet?", "5", "language", "hard"),
    ("What is the most widely spoken Bantu language?", "Swahili", "language", "hard"),
    ("What language is 'Esperanto' classified as?", "constructed language", "language", "medium"),
    ("How many letters are in the Greek alphabet?", "24", "language", "medium"),
    ("What is the writing system used for Korean?", "Hangul", "language", "medium"),
    ("How many basic stroke types exist in Chinese calligraphy?", "8", "language", "hard"),
    ("What is the oldest known written language?", "Sumerian", "language", "hard"),

    # ===== TECHNOLOGY =====
    ("What does 'HTTP' stand for?", "HyperText Transfer Protocol", "technology", "easy"),
    ("What does 'CPU' stand for?", "Central Processing Unit", "technology", "easy"),
    ("In what year was the World Wide Web invented?", "1989", "technology", "medium"),
    ("What programming language was created by Guido van Rossum?", "Python", "technology", "easy"),
    ("What does 'RAM' stand for?", "Random Access Memory", "technology", "easy"),
    ("What does 'SQL' stand for?", "Structured Query Language", "technology", "medium"),
    ("How many bits are in a byte?", "8", "technology", "easy"),
    ("What company created the Java programming language?", "Sun Microsystems", "technology", "medium"),
    ("What does 'GPU' stand for?", "Graphics Processing Unit", "technology", "easy"),
    ("What is the time complexity of binary search?", "O(log n)", "technology", "medium"),
    ("What does 'DNS' stand for?", "Domain Name System", "technology", "medium"),
    ("What year was the first iPhone released?", "2007", "technology", "medium"),
    ("What does 'API' stand for?", "Application Programming Interface", "technology", "easy"),
    ("What protocol is used for secure web connections?", "HTTPS", "technology", "easy"),
    ("What does 'BIOS' stand for?", "Basic Input Output System", "technology", "medium"),

    # ===== ART =====
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "art", "easy"),
    ("Who painted 'Starry Night'?", "Vincent van Gogh", "art", "easy"),
    ("Who sculpted 'David' (the famous Renaissance sculpture)?", "Michelangelo", "art", "easy"),
    ("Who painted 'The Persistence of Memory' (melting clocks)?", "Salvador Dali", "art", "medium"),
    ("Who painted 'Guernica'?", "Pablo Picasso", "art", "medium"),
    ("Who painted 'The Great Wave off Kanagawa'?", "Hokusai", "art", "medium"),
    ("Who painted 'Girl with a Pearl Earring'?", "Johannes Vermeer", "art", "medium"),
    ("Who painted the ceiling of the Sistine Chapel?", "Michelangelo", "art", "easy"),
    ("Who painted 'The Birth of Venus'?", "Sandro Botticelli", "art", "medium"),
    ("What art movement was Claude Monet associated with?", "Impressionism", "art", "medium"),
    ("Who painted 'The Scream'?", "Edvard Munch", "art", "medium"),
    ("Who painted 'Water Lilies' (series)?", "Claude Monet", "art", "easy"),
    ("Who created the sculpture 'The Thinker'?", "Auguste Rodin", "art", "medium"),
    ("What art movement was Andy Warhol associated with?", "Pop Art", "art", "medium"),
    ("Who painted 'Las Meninas'?", "Diego Velazquez", "art", "hard"),

    # ===== FOOD & COOKING =====
    ("What grain is sake made from?", "rice", "food", "easy"),
    ("What country does kimchi originate from?", "Korea", "food", "easy"),
    ("What is the main ingredient in hummus?", "chickpeas", "food", "easy"),
    ("What Italian dish is made from arborio rice?", "risotto", "food", "medium"),
    ("What Japanese dish consists of vinegared rice with toppings?", "sushi", "food", "easy"),
    ("What spice gives curry its yellow color?", "turmeric", "food", "medium"),
    ("What is the French term for a stock made from meat bones?", "fond", "food", "hard"),
    ("What Mexican dish wraps filling in a corn husk?", "tamale", "food", "medium"),
    ("What is the primary ingredient in miso paste?", "soybeans", "food", "medium"),
    ("What cheese is traditionally used in a Greek salad?", "feta", "food", "easy"),

    # ===== SPORTS =====
    ("How many players are on a standard soccer team on the field?", "11", "sports", "easy"),
    ("How many points is a touchdown worth in American football?", "6", "sports", "easy"),
    ("How long is a marathon (in miles, approximately)?", "26.2", "sports", "easy"),
    ("How many periods are in a standard ice hockey game?", "3", "sports", "medium"),
    ("What is the diameter of a basketball hoop (in inches)?", "18", "sports", "hard"),
    ("How many sets are needed to win a men's Grand Slam tennis match?", "3", "sports", "medium"),
    ("What is the maximum break in snooker?", "147", "sports", "hard"),
    ("How many players are on a water polo team in the water?", "7", "sports", "hard"),
    ("How long is an Olympic swimming pool (in meters)?", "50", "sports", "medium"),
    ("In which year were the first modern Olympic Games held?", "1896", "sports", "medium"),
]

# Separate pools by difficulty for sampling
_EASY_QS = [q for q in _QUESTION_POOL if q[3] == "easy"]
_MEDIUM_QS = [q for q in _QUESTION_POOL if q[3] == "medium"]
_HARD_QS = [q for q in _QUESTION_POOL if q[3] == "hard"]


# ===================================================================
# PARADIGM 1 -- CONFIDENCE CALIBRATION
# ===================================================================

class ConfidenceCalibrationGenerator:
    """Procedural generator for Confidence Calibration tasks.

    Present a knowledge question, collect the answer, then ask for a
    confidence rating (0-100%).  Scoring uses Expected Calibration
    Error (ECE), overconfidence index, and Brier decomposition.
    """

    PARADIGM = "confidence_calibration"
    DIMENSION = "metacognition"

    # Difficulty distribution maps
    DIFFICULTY_MAP: Dict[str, Dict[str, float]] = {
        "easy":   {"easy_frac": 0.60, "medium_frac": 0.30, "hard_frac": 0.10},
        "medium": {"easy_frac": 0.25, "medium_frac": 0.50, "hard_frac": 0.25},
        "hard":   {"easy_frac": 0.10, "medium_frac": 0.30, "hard_frac": 0.60},
    }

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 50,
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> list[TaskInstance]:
        """Generate Confidence Calibration items.

        Each TaskInstance presents a two-part interaction:
          1. A knowledge question.
          2. A confidence prompt (0-100%).

        These are stored as a single static item with the full prompt.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of items (50-100 recommended).
        difficulty : str
            Controls question difficulty distribution.
        contamination_probe : bool
            If True, all questions are well-known "textbook" items.
        """
        rng = _make_rng(seed)
        items: list[TaskInstance] = []

        fracs = cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"])

        if contamination_probe:
            # Use only easy (well-known) questions
            pool = list(_EASY_QS)
        else:
            # Sample from all pools according to difficulty distribution
            n_easy = int(round(n_items * fracs["easy_frac"]))
            n_medium = int(round(n_items * fracs["medium_frac"]))
            n_hard = n_items - n_easy - n_medium

            pool = []
            pool.extend(rng.sample(_EASY_QS, min(n_easy, len(_EASY_QS))))
            pool.extend(rng.sample(_MEDIUM_QS, min(n_medium, len(_MEDIUM_QS))))
            pool.extend(rng.sample(_HARD_QS, min(n_hard, len(_HARD_QS))))

            # If we don't have enough, fill from the full pool
            while len(pool) < n_items:
                extra = rng.choice(_QUESTION_POOL)
                if extra not in pool:
                    pool.append(extra)

        rng.shuffle(pool)
        pool = pool[:n_items]

        for idx, (question, answer, domain, q_diff) in enumerate(pool):
            ep_seed = rng.randint(0, 2**31)

            stimulus = (
                f"Question: {question}\n\n"
                f"Please provide:\n"
                f"1. Your answer to the question.\n"
                f"2. Your confidence in your answer as a percentage "
                f"(0-100%), where 0% means you are completely unsure "
                f"and 100% means you are absolutely certain.\n\n"
                f"Format your response as:\n"
                f"Answer: <your answer>\n"
                f"Confidence: <number>%"
            )

            task_id = (
                f"mc_confcal_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_{domain}_s{ep_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.LLM_STATIC,
                parameters={
                    "question": question,
                    "correct_answer": answer,
                    "domain": domain,
                    "question_difficulty": q_diff,
                    "episode_seed": ep_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": False,
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.metacognition.score_confidence_calibration",
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=(
                    AdaptationDistance.HIGH if contamination_probe
                    else AdaptationDistance.LOW
                ),
                description=(
                    f"Confidence calibration -- {domain} ({q_diff})"
                ),
            )

            items.append(TaskInstance(
                task_id=task_id,
                metadata=metadata,
                stimulus=stimulus,
                expected_response=answer,
            ))

        return items

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, float]:
        """Score a single Confidence Calibration trial.

        Parses the response for an answer and confidence value.
        Returns per-item accuracy and confidence.
        """
        answer_text, confidence = _parse_answer_confidence(response)
        correct_answer = str(task.expected_response).strip().lower()

        is_correct = _flexible_match(answer_text, correct_answer)

        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "confidence": confidence,
            "is_correct": 1.0 if is_correct else 0.0,
        }

    @staticmethod
    def aggregate(scored: list[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate calibration metrics across items.

        Returns: calibration_error (ECE), overconfidence_index,
        resolution, accuracy, mean_confidence.
        """
        if not scored:
            return {
                "calibration_error": 0.0,
                "overconfidence_index": 0.0,
                "resolution": 0.0,
                "accuracy": 0.0,
                "mean_confidence": 0.0,
            }

        accuracies = [s["accuracy"] for s in scored]
        confidences = [s["confidence"] for s in scored]
        n = len(scored)

        mean_acc = sum(accuracies) / n
        mean_conf = sum(confidences) / n

        # Expected Calibration Error (ECE) with 10 bins
        ece = _compute_ece(accuracies, confidences, n_bins=10)

        # Overconfidence index: mean(confidence - accuracy) when conf > acc
        overconf_diffs = [
            c - a for c, a in zip(confidences, accuracies)
        ]
        overconf_index = sum(max(0, d) for d in overconf_diffs) / n

        # Resolution (Brier decomposition: variance of bin accuracies)
        resolution = _compute_resolution(accuracies, confidences, n_bins=10)

        return {
            "calibration_error": round(ece, 4),
            "overconfidence_index": round(overconf_index, 4),
            "resolution": round(resolution, 4),
            "accuracy": round(mean_acc, 4),
            "mean_confidence": round(mean_conf, 4),
        }


def _parse_answer_confidence(response: str) -> Tuple[str, float]:
    """Parse response for answer and confidence.

    Expected format:
        Answer: <text>
        Confidence: <number>%

    Falls back to heuristics if format doesn't match exactly.
    """
    answer_text = ""
    confidence = 0.5  # default if unparseable

    resp_lower = response.lower()

    # Try to extract answer
    for prefix in ["answer:", "answer :", "a:"]:
        if prefix in resp_lower:
            idx = resp_lower.index(prefix) + len(prefix)
            # Extract until newline or "confidence"
            rest = response[idx:]
            for sep in ["\n", "confidence", "Confidence"]:
                if sep in rest:
                    answer_text = rest[:rest.index(sep)].strip()
                    break
            else:
                answer_text = rest.strip()
            break
    else:
        # No "Answer:" prefix found; use first line
        lines = response.strip().split("\n")
        answer_text = lines[0].strip()

    # Try to extract confidence
    for prefix in ["confidence:", "confidence :"]:
        if prefix in resp_lower:
            idx = resp_lower.index(prefix) + len(prefix)
            rest = response[idx:].strip()
            # Extract number
            num_str = ""
            for ch in rest:
                if ch.isdigit() or ch == ".":
                    num_str += ch
                elif num_str:
                    break
            if num_str:
                try:
                    val = float(num_str)
                    if val > 1.0:
                        val = val / 100.0  # Convert percentage to fraction
                    confidence = max(0.0, min(1.0, val))
                except ValueError:
                    pass
            break

    return answer_text.strip(), confidence


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _as_number(s: str):
    """First number in s as float, tolerating thousands separators; None if none."""
    m = re.search(r"-?\d[\d,]*\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _flexible_match(response_text: str, correct_answer: str) -> bool:
    """Flexible matching: case- and accent-insensitive substring match, plus
    numeric equality tolerating thousands separators (e.g. 300,000 == 300000).
    No numeric tolerance is applied, so off-by-one counts and adjacent years
    still count as wrong."""
    resp = _strip_accents(response_text.strip().lower())
    correct = _strip_accents(correct_answer.strip().lower())
    if not resp or not correct:
        return False
    if correct in resp or resp in correct:
        return True
    rn, cn = _as_number(resp), _as_number(correct)
    if rn is not None and cn is not None:
        return abs(rn - cn) < 1e-9
    return False


def _compute_ece(
    accuracies: list[float],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins: list[list[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for acc, conf in zip(accuracies, confidences):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((acc, conf))

    ece = 0.0
    n = len(accuracies)
    for bin_items in bins:
        if not bin_items:
            continue
        bin_acc = sum(a for a, _ in bin_items) / len(bin_items)
        bin_conf = sum(c for _, c in bin_items) / len(bin_items)
        ece += len(bin_items) / n * abs(bin_acc - bin_conf)

    return ece


def _compute_resolution(
    accuracies: list[float],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """Compute resolution component of Brier decomposition.

    Resolution measures how well confidence discriminates between
    correct and incorrect answers.
    """
    n = len(accuracies)
    overall_acc = sum(accuracies) / max(n, 1)

    bins: list[list[float]] = [[] for _ in range(n_bins)]
    for acc, conf in zip(accuracies, confidences):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(acc)

    resolution = 0.0
    for bin_accs in bins:
        if not bin_accs:
            continue
        bin_mean = sum(bin_accs) / len(bin_accs)
        resolution += len(bin_accs) / n * (bin_mean - overall_acc) ** 2

    return resolution


def score_confidence_calibration(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Module-level scoring function for Confidence Calibration."""
    answer_text, confidence = _parse_answer_confidence(str(response))
    correct_answer = str(expected).strip().lower()

    is_correct = _flexible_match(answer_text, correct_answer)

    return {
        "accuracy": 1.0 if is_correct else 0.0,
        "confidence": confidence,
    }


# ===================================================================
# PARADIGM 2 -- POST-DECISION WAGERING
# ===================================================================

class PostDecisionWageringGenerator:
    """Procedural generator for Post-Decision Wagering tasks.

    Same knowledge questions as confidence calibration, but with a
    betting framing:
      - After answering, choose BET (YES) or PASS (NO).
      - BET + correct = +10 points;  BET + wrong = -10 points.
      - PASS = +2 points regardless.

    This tests risk-sensitive metacognitive control.
    """

    PARADIGM = "post_decision_wagering"
    DIMENSION = "metacognition"

    DEFAULT_PAYOFF: Dict[str, int] = {
        "bet_correct": 10,
        "bet_wrong": -10,
        "pass_any": 2,
    }

    DIFFICULTY_MAP: Dict[str, Dict[str, float]] = {
        "easy":   {"easy_frac": 0.60, "medium_frac": 0.30, "hard_frac": 0.10},
        "medium": {"easy_frac": 0.25, "medium_frac": 0.50, "hard_frac": 0.25},
        "hard":   {"easy_frac": 0.10, "medium_frac": 0.30, "hard_frac": 0.60},
    }

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 50,
        difficulty: str = "medium",
        bet_payoff_structure: Optional[Dict[str, int]] = None,
        contamination_probe: bool = False,
    ) -> list[TaskInstance]:
        """Generate Post-Decision Wagering items.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of items (50-100 recommended).
        difficulty : str
            Controls question difficulty distribution.
        bet_payoff_structure : dict, optional
            Override default payoffs. Keys: bet_correct, bet_wrong, pass_any.
        contamination_probe : bool
            If True, use only well-known questions.
        """
        rng = _make_rng(seed)
        payoff = bet_payoff_structure or cls.DEFAULT_PAYOFF
        items: list[TaskInstance] = []

        fracs = cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"])

        if contamination_probe:
            pool = list(_EASY_QS)
        else:
            n_easy = int(round(n_items * fracs["easy_frac"]))
            n_medium = int(round(n_items * fracs["medium_frac"]))
            n_hard = n_items - n_easy - n_medium

            pool = []
            pool.extend(rng.sample(_EASY_QS, min(n_easy, len(_EASY_QS))))
            pool.extend(rng.sample(_MEDIUM_QS, min(n_medium, len(_MEDIUM_QS))))
            pool.extend(rng.sample(_HARD_QS, min(n_hard, len(_HARD_QS))))

            while len(pool) < n_items:
                extra = rng.choice(_QUESTION_POOL)
                if extra not in pool:
                    pool.append(extra)

        rng.shuffle(pool)
        pool = pool[:n_items]

        bet_correct = payoff.get("bet_correct", 10)
        bet_wrong = payoff.get("bet_wrong", -10)
        pass_any = payoff.get("pass_any", 2)

        for idx, (question, answer, domain, q_diff) in enumerate(pool):
            ep_seed = rng.randint(0, 2**31)

            stimulus = (
                f"Question: {question}\n\n"
                f"Please provide:\n"
                f"1. Your answer to the question.\n"
                f"2. Would you BET {abs(bet_correct)} points that your answer "
                f"is correct?\n"
                f"   - If you bet YES and are correct: +{bet_correct} points\n"
                f"   - If you bet YES and are wrong: {bet_wrong} points\n"
                f"   - If you bet NO (pass): +{pass_any} points regardless\n\n"
                f"Format your response as:\n"
                f"Answer: <your answer>\n"
                f"Bet: YES or NO"
            )

            task_id = (
                f"mc_pdw_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_{domain}_s{ep_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.LLM_STATIC,
                parameters={
                    "question": question,
                    "correct_answer": answer,
                    "domain": domain,
                    "question_difficulty": q_diff,
                    "bet_correct": bet_correct,
                    "bet_wrong": bet_wrong,
                    "pass_any": pass_any,
                    "episode_seed": ep_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": False,
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.metacognition.score_post_decision_wagering",
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=(
                    AdaptationDistance.HIGH if contamination_probe
                    else AdaptationDistance.MEDIUM
                ),
                description=(
                    f"Post-decision wagering -- {domain} ({q_diff})"
                ),
            )

            items.append(TaskInstance(
                task_id=task_id,
                metadata=metadata,
                stimulus=stimulus,
                expected_response=answer,
            ))

        return items

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, float]:
        """Score a single Post-Decision Wagering trial."""
        answer_text, did_bet = _parse_answer_bet(response)
        correct_answer = str(task.expected_response).strip().lower()

        is_correct = _flexible_match(answer_text, correct_answer)

        bet_correct = task.metadata.parameters.get("bet_correct", 10)
        bet_wrong = task.metadata.parameters.get("bet_wrong", -10)
        pass_any = task.metadata.parameters.get("pass_any", 2)

        if did_bet:
            points = bet_correct if is_correct else bet_wrong
        else:
            points = pass_any

        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "did_bet": 1.0 if did_bet else 0.0,
            "is_correct": 1.0 if is_correct else 0.0,
            "points": float(points),
        }

    @staticmethod
    def aggregate(scored: list[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate wagering metrics across items.

        Returns: expected_value_ratio, optimal_betting_rate,
        wagering_sensitivity, total_score, accuracy.
        """
        if not scored:
            return {
                "expected_value_ratio": 0.0,
                "optimal_betting_rate": 0.0,
                "wagering_sensitivity": 0.0,
                "total_score": 0.0,
                "accuracy": 0.0,
            }

        n = len(scored)
        total_points = sum(s["points"] for s in scored)
        mean_acc = sum(s["accuracy"] for s in scored) / n

        # Wagering sensitivity: bet rate when correct vs when incorrect
        correct_items = [s for s in scored if s["is_correct"] > 0.5]
        incorrect_items = [s for s in scored if s["is_correct"] <= 0.5]

        bet_rate_correct = (
            sum(s["did_bet"] for s in correct_items) / len(correct_items)
            if correct_items else 0.0
        )
        bet_rate_incorrect = (
            sum(s["did_bet"] for s in incorrect_items) / len(incorrect_items)
            if incorrect_items else 0.0
        )
        wagering_sensitivity = bet_rate_correct - bet_rate_incorrect

        # Optimal betting rate: should bet YES on correct, NO on incorrect
        # i.e., optimal bet rate = accuracy
        actual_bet_rate = sum(s["did_bet"] for s in scored) / n
        optimal_bet_rate = mean_acc
        # How close actual is to optimal (1.0 = perfect, negative = worse)
        # Use expected value ratio: actual EV / optimal EV
        # Optimal EV per item: acc * bet_correct + (1-acc) * pass_any
        bet_correct = scored[0].get("points", 10)  # approximate
        pass_any = 2  # default
        optimal_ev = mean_acc * 10 + (1 - mean_acc) * 2  # simplified
        actual_ev = total_points / n
        ev_ratio = actual_ev / max(optimal_ev, 0.01)

        return {
            "expected_value_ratio": round(ev_ratio, 4),
            "optimal_betting_rate": round(optimal_bet_rate, 4),
            "wagering_sensitivity": round(wagering_sensitivity, 4),
            "total_score": round(total_points, 2),
            "accuracy": round(mean_acc, 4),
            "bet_rate_correct": round(bet_rate_correct, 4),
            "bet_rate_incorrect": round(bet_rate_incorrect, 4),
        }


def _parse_answer_bet(response: str) -> Tuple[str, bool]:
    """Parse response for answer and bet decision.

    Expected format:
        Answer: <text>
        Bet: YES or NO
    """
    answer_text = ""
    did_bet = False

    resp_lower = response.lower()

    # Extract answer
    for prefix in ["answer:", "answer :", "a:"]:
        if prefix in resp_lower:
            idx = resp_lower.index(prefix) + len(prefix)
            rest = response[idx:]
            for sep in ["\n", "bet", "Bet"]:
                if sep in rest:
                    answer_text = rest[:rest.index(sep)].strip()
                    break
            else:
                answer_text = rest.strip()
            break
    else:
        lines = response.strip().split("\n")
        answer_text = lines[0].strip()

    # Extract bet
    for prefix in ["bet:", "bet :", "wager:"]:
        if prefix in resp_lower:
            idx = resp_lower.index(prefix) + len(prefix)
            rest = response[idx:].strip().lower()
            did_bet = rest.startswith("yes") or rest.startswith("y")
            break
    else:
        # Fallback: look for YES or NO anywhere after answer
        after_answer = resp_lower.split("answer", 1)[-1] if "answer" in resp_lower else resp_lower
        if "yes" in after_answer:
            did_bet = True

    return answer_text.strip(), did_bet


def score_post_decision_wagering(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Module-level scoring function for Post-Decision Wagering."""
    answer_text, did_bet = _parse_answer_bet(str(response))
    correct_answer = str(expected).strip().lower()

    is_correct = _flexible_match(answer_text, correct_answer)

    bet_correct = metadata.parameters.get("bet_correct", 10)
    bet_wrong = metadata.parameters.get("bet_wrong", -10)
    pass_any = metadata.parameters.get("pass_any", 2)

    if did_bet:
        points = bet_correct if is_correct else bet_wrong
    else:
        points = pass_any

    return {
        "accuracy": 1.0 if is_correct else 0.0,
        "did_bet": 1.0 if did_bet else 0.0,
        "points": float(points),
    }


# ===================================================================
# Convenience dispatch
# ===================================================================

_GENERATORS: Dict[str, type] = {
    "confidence_calibration": ConfidenceCalibrationGenerator,
    "post_decision_wagering": PostDecisionWageringGenerator,
}


def generate(
    paradigm: str,
    seed: int,
    n_items: int = 50,
    difficulty: str = "medium",
    contamination_probe: bool = False,
    **kwargs: Any,
) -> list[TaskInstance]:
    """Unified entry-point for generating Metacognition items.

    Parameters
    ----------
    paradigm : str
        One of "confidence_calibration", "post_decision_wagering".
    seed, n_items, difficulty, contamination_probe
        Forwarded to the paradigm generator.
    **kwargs
        Extra keyword arguments (e.g., ``bet_payoff_structure``).
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
