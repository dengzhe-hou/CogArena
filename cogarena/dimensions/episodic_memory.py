"""Episodic Memory dimension for CogArena.

Implements three paradigms with procedural generation:
  1. CVLT-style Word List Learning  -- multi-turn encoding/recall/interference
  2. DRM False Memory               -- static recognition with critical lures
  3. Source Monitoring               -- static source attribution

All items are procedurally generated from random seeds to minimise
contamination from training corpora.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

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
    """Return a seeded Random instance (reproducible, thread-safe)."""
    return random.Random(seed)


def _dprime(hit_rate: float, fa_rate: float) -> float:
    """Compute d' (d-prime) from hit rate and false-alarm rate.

    Rates are clipped to (0.01, 0.99) to avoid infinite z-scores.
    """

    def _z(p: float) -> float:
        """Inverse of the standard normal CDF (probit) via rational approx."""
        p = max(0.01, min(0.99, p))
        if p < 0.5:
            t = math.sqrt(-2.0 * math.log(p))
        else:
            t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        val = t - (c0 + c1 * t + c2 * t * t) / (
            1.0 + d1 * t + d2 * t * t + d3 * t * t * t
        )
        return val if p >= 0.5 else -val

    return _z(hit_rate) - _z(fa_rate)


def _difficulty_enum(s: str) -> DifficultyLevel:
    return DifficultyLevel(s.lower())


def _parse_word_list(response: Any) -> List[str]:
    """Parse a response string into a list of lowercase words.

    Handles comma-separated, newline-separated, numbered lists, and
    space-separated formats.
    """
    if isinstance(response, list):
        return [str(w).strip().lower() for w in response if str(w).strip()]
    text = str(response).strip()
    # Remove common numbering patterns (e.g., "1. apple", "1) apple", "- apple")
    import re
    lines = text.replace(",", "\n").split("\n")
    words: List[str] = []
    for line in lines:
        line = line.strip()
        line = re.sub(r"^\d+[\.\)\-]\s*", "", line)
        line = re.sub(r"^[\-\*\+]\s*", "", line)
        line = line.strip()
        if line:
            # Could be multiple space-separated words on one line
            for w in line.split():
                cleaned = w.strip().lower().strip(".,;:!?\"'()")
                if cleaned:
                    words.append(cleaned)
    return words


# ---------------------------------------------------------------------------
# Category word pools for CVLT-style list learning
# ---------------------------------------------------------------------------

_CATEGORY_POOLS: Dict[str, List[str]] = {
    "fruits": [
        "apple", "banana", "orange", "grape", "mango", "peach", "plum",
        "cherry", "pear", "melon", "kiwi", "papaya", "fig", "lemon",
        "lime", "coconut", "apricot", "guava", "lychee", "pomegranate",
        "tangerine", "nectarine", "blueberry", "raspberry", "strawberry",
    ],
    "animals": [
        "tiger", "eagle", "whale", "cobra", "bison", "otter", "falcon",
        "panda", "moose", "raven", "shark", "camel", "gecko", "crane",
        "horse", "zebra", "lynx", "ferret", "badger", "robin", "salmon",
        "parrot", "jaguar", "donkey", "turtle",
    ],
    "tools": [
        "hammer", "wrench", "pliers", "chisel", "drill", "saw", "clamp",
        "level", "rasp", "mallet", "anvil", "vise", "awl", "plane",
        "screwdriver", "crowbar", "trowel", "file", "lathe", "shears",
        "grinder", "sander", "router", "gouge", "jigsaw",
    ],
    "vegetables": [
        "carrot", "broccoli", "spinach", "celery", "pepper", "onion",
        "garlic", "potato", "tomato", "cabbage", "lettuce", "radish",
        "turnip", "squash", "fennel", "kale", "beet", "leek", "pea",
        "corn", "yam", "okra", "parsnip", "endive", "arugula",
    ],
    "clothing": [
        "jacket", "sweater", "trousers", "scarf", "gloves", "boots",
        "shirt", "vest", "belt", "hat", "socks", "shorts", "blouse",
        "skirt", "coat", "poncho", "sandals", "mittens", "hoodie",
        "cardigan", "tunic", "apron", "cape", "overalls", "robe",
    ],
    "furniture": [
        "chair", "table", "couch", "shelf", "desk", "stool", "dresser",
        "cabinet", "bench", "bed", "ottoman", "bookcase", "armoire",
        "futon", "hutch", "nightstand", "rocker", "cradle", "wardrobe",
        "credenza", "hamper", "trunk", "buffet", "vanity", "crib",
    ],
    "instruments": [
        "guitar", "piano", "violin", "trumpet", "drums", "flute", "harp",
        "cello", "banjo", "oboe", "tuba", "clarinet", "sitar", "organ",
        "accordion", "harmonica", "mandolin", "bassoon", "cymbal",
        "tambourine", "ukulele", "marimba", "bugle", "lute", "fiddle",
    ],
    "sports": [
        "soccer", "tennis", "hockey", "cricket", "rugby", "golf",
        "boxing", "fencing", "rowing", "surfing", "skiing", "cycling",
        "archery", "polo", "diving", "judo", "karate", "lacrosse",
        "squash", "curling", "bobsled", "handball", "wrestling",
        "biathlon", "triathlon",
    ],
    "kitchen_items": [
        "spatula", "whisk", "colander", "skillet", "ladle", "grater",
        "peeler", "tongs", "kettle", "blender", "toaster", "mortar",
        "pestle", "strainer", "saucepan", "wok", "teapot", "mug",
        "pitcher", "corkscrew", "trivet", "cleaver", "mandoline",
        "steamer", "griddle",
    ],
    "occupations": [
        "teacher", "doctor", "lawyer", "pilot", "chef", "nurse",
        "plumber", "farmer", "painter", "welder", "tailor", "baker",
        "dentist", "barber", "cashier", "mechanic", "janitor",
        "librarian", "florist", "surgeon", "chemist", "architect",
        "sculptor", "auditor", "ranger",
    ],
    "body_parts": [
        "elbow", "ankle", "wrist", "shoulder", "knee", "forehead",
        "chin", "palm", "thumb", "heel", "spine", "ribcage", "temple",
        "nostril", "eyelid", "knuckle", "calf", "thigh", "forearm",
        "collarbone", "sternum", "pelvis", "shin", "jaw", "navel",
    ],
    "weather": [
        "thunder", "lightning", "blizzard", "tornado", "hurricane",
        "drizzle", "hailstorm", "monsoon", "cyclone", "sleet",
        "gust", "breeze", "frost", "fog", "mist", "downpour",
        "rainbow", "drought", "tempest", "squall", "avalanche",
        "flurry", "whirlwind", "heatwave", "tsunami",
    ],
    "transportation": [
        "bicycle", "trolley", "canoe", "subway", "airplane", "scooter",
        "sailboat", "helicopter", "ferry", "gondola", "rickshaw",
        "chariot", "kayak", "tanker", "monorail", "tractor", "blimp",
        "sled", "yacht", "tugboat", "glider", "moped", "raft",
        "hovercraft", "catamaran",
    ],
    "flowers": [
        "daisy", "tulip", "orchid", "sunflower", "jasmine", "lily",
        "iris", "dahlia", "peony", "violet", "carnation", "hibiscus",
        "marigold", "poppy", "lavender", "daffodil", "chrysanthemum",
        "magnolia", "bluebell", "heather", "primrose", "pansy",
        "begonia", "azalea", "lotus",
    ],
    "trees": [
        "oak", "maple", "pine", "birch", "cedar", "willow", "elm",
        "spruce", "walnut", "cherry", "ash", "poplar", "cypress",
        "redwood", "sycamore", "hickory", "juniper", "magnolia",
        "beech", "fir", "hemlock", "sequoia", "chestnut", "alder",
        "larch",
    ],
    "minerals": [
        "quartz", "granite", "marble", "obsidian", "topaz", "jade",
        "amber", "opal", "garnet", "onyx", "basalt", "slate",
        "pumice", "flint", "agate", "jasper", "turquoise", "ruby",
        "sapphire", "emerald", "diamond", "amethyst", "pyrite",
        "feldspar", "gypsum",
    ],
    "fabrics": [
        "cotton", "silk", "linen", "wool", "denim", "velvet", "satin",
        "flannel", "chiffon", "tweed", "cashmere", "corduroy", "nylon",
        "polyester", "rayon", "fleece", "burlap", "organza", "crepe",
        "taffeta", "muslin", "canvas", "suede", "lace", "jersey",
    ],
    "spices": [
        "cinnamon", "turmeric", "cumin", "paprika", "ginger",
        "cardamom", "clove", "nutmeg", "saffron", "oregano",
        "rosemary", "thyme", "basil", "parsley", "coriander",
        "cayenne", "fennel", "anise", "dill", "sage",
        "tarragon", "marjoram", "chili", "vanilla", "pepper",
    ],
    "insects": [
        "beetle", "butterfly", "cricket", "dragonfly", "firefly",
        "grasshopper", "ladybug", "mosquito", "moth", "wasp",
        "ant", "bee", "flea", "locust", "termite",
        "hornet", "cicada", "mantis", "caterpillar", "cockroach",
        "aphid", "earwig", "gnat", "maggot", "weevil",
    ],
    "gemstones": [
        "diamond", "emerald", "ruby", "sapphire", "amethyst",
        "topaz", "opal", "garnet", "peridot", "aquamarine",
        "tourmaline", "zircon", "alexandrite", "tanzanite", "morganite",
        "spinel", "moonstone", "sunstone", "labradorite", "citrine",
        "kunzite", "iolite", "chrysoberyl", "apatite", "sphene",
    ],
}

# All available category names for random selection
_ALL_CATEGORIES: List[str] = list(_CATEGORY_POOLS.keys())


# ---------------------------------------------------------------------------
# DRM (Deese-Roediger-McDermott) theme-associate pools
# ---------------------------------------------------------------------------
# Each key is the critical lure (the word NOT presented), and the value
# is a list of strong associates (in roughly descending associative
# strength) drawn from psychology literature and norms.

_DRM_POOLS: Dict[str, List[str]] = {
    "sleep": [
        "bed", "rest", "awake", "tired", "dream", "wake", "snooze",
        "blanket", "doze", "slumber", "snore", "nap", "peace", "yawn",
        "drowsy", "pillow", "night", "quiet", "dark", "cozy",
    ],
    "needle": [
        "thread", "pin", "eye", "sewing", "sharp", "point", "prick",
        "thimble", "haystack", "thorn", "hurt", "injection", "syringe",
        "cloth", "knitting", "stitch", "fabric", "pierce", "thin", "steel",
    ],
    "mountain": [
        "hill", "valley", "climb", "summit", "peak", "glacier", "range",
        "steep", "top", "mole", "goat", "bike", "climber", "snow",
        "altitude", "ridge", "trail", "boulder", "cliff", "hike",
    ],
    "river": [
        "water", "stream", "lake", "mississippi", "boat", "tide",
        "swim", "flow", "run", "barge", "creek", "brook", "fish",
        "bridge", "winding", "bank", "current", "rapids", "dam", "shore",
    ],
    "music": [
        "note", "sound", "piano", "sing", "radio", "band", "melody",
        "horn", "concert", "instrument", "symphony", "jazz", "orchestra",
        "art", "rhythm", "tune", "song", "harmony", "beat", "dance",
    ],
    "sweet": [
        "sour", "candy", "sugar", "bitter", "good", "taste", "tooth",
        "nice", "honey", "chocolate", "heart", "cake", "tart", "pie",
        "caramel", "syrup", "treat", "fudge", "dessert", "vanilla",
    ],
    "cold": [
        "hot", "snow", "warm", "winter", "ice", "wet", "frigid",
        "chilly", "heat", "weather", "freeze", "air", "shiver",
        "arctic", "frost", "wind", "blizzard", "coat", "icicle", "polar",
    ],
    "anger": [
        "mad", "fear", "hate", "rage", "temper", "fury", "ire",
        "wrath", "happy", "fight", "hatred", "mean", "calm", "emotion",
        "enrage", "hostile", "bitter", "irritate", "resent", "scream",
    ],
    "doctor": [
        "nurse", "sick", "lawyer", "medicine", "health", "hospital",
        "dentist", "physician", "ill", "patient", "office", "stethoscope",
        "surgeon", "clinic", "cure", "practice", "examine", "treat",
        "diagnosis", "prescription",
    ],
    "rough": [
        "smooth", "bumpy", "road", "tough", "sandpaper", "jagged",
        "ready", "coarse", "uneven", "riders", "rugged", "gravel",
        "sand", "ground", "texture", "terrain", "bristle", "harsh",
        "gritty", "ragged",
    ],
    "window": [
        "door", "glass", "pane", "shade", "ledge", "sill", "house",
        "open", "curtain", "frame", "view", "breeze", "screen",
        "shutter", "blind", "light", "room", "wall", "outside", "clear",
    ],
    "king": [
        "queen", "crown", "prince", "palace", "throne", "royal",
        "ruler", "kingdom", "reign", "castle", "lord", "power",
        "monarch", "knight", "empire", "duke", "scepter", "robe",
        "court", "sovereign",
    ],
    "smoke": [
        "fire", "cigarette", "puff", "pipe", "chimney", "ashes",
        "cigar", "fog", "cloud", "screen", "blaze", "flame", "burn",
        "signal", "alarm", "haze", "fumes", "exhaust", "tobacco", "soot",
    ],
    "bread": [
        "butter", "food", "eat", "sandwich", "rye", "jam", "milk",
        "flour", "jelly", "dough", "crust", "slice", "wine", "loaf",
        "toast", "wheat", "roll", "bake", "yeast", "crumb",
    ],
    "chair": [
        "table", "sit", "legs", "seat", "couch", "desk", "recliner",
        "sofa", "wood", "cushion", "stool", "sitting", "rocking",
        "bench", "arm", "swivel", "back", "padded", "office", "lawn",
    ],
    "thief": [
        "steal", "robber", "crook", "burglar", "money", "cop", "bad",
        "rob", "jail", "gun", "villain", "crime", "bandit", "criminal",
        "sneak", "mask", "heist", "loot", "pickpocket", "outlaw",
    ],
    "spider": [
        "web", "insect", "bug", "fright", "fly", "arachnid", "crawl",
        "tarantula", "poison", "bite", "creepy", "animal", "ugly",
        "feeler", "small", "venom", "silk", "spin", "leg", "black",
    ],
    "lion": [
        "tiger", "circus", "jungle", "wild", "animal", "mane", "fierce",
        "den", "roar", "pride", "cage", "cat", "africa", "hunter",
        "king", "safari", "strength", "cub", "predator", "savanna",
    ],
    "fruit": [
        "apple", "vegetable", "orange", "kiwi", "citrus", "ripe",
        "pear", "banana", "berry", "cherry", "basket", "juice",
        "salad", "bowl", "cocktail", "tropical", "melon", "harvest",
        "orchard", "fresh",
    ],
    "man": [
        "woman", "husband", "uncle", "lady", "mouse", "male", "father",
        "strong", "friend", "beard", "person", "boy", "gentleman",
        "suit", "brave", "old", "young", "tall", "brother", "fellow",
    ],
    "soft": [
        "hard", "light", "pillow", "plush", "loud", "cotton", "fur",
        "touch", "fluffy", "feather", "comfortable", "skin", "baby",
        "tender", "silk", "smooth", "gentle", "quiet", "velvet",
        "cushion",
    ],
    "city": [
        "town", "crowded", "state", "capital", "county", "village",
        "new york", "streets", "subway", "country", "building", "urban",
        "traffic", "bus", "metropolis", "downtown", "skyscraper",
        "population", "noise", "park",
    ],
    "high": [
        "low", "clouds", "up", "tall", "tower", "jump", "above",
        "building", "noon", "cliff", "sky", "dive", "airplane",
        "elevate", "kite", "mountain", "rise", "altitude", "soar",
        "peak",
    ],
    "slow": [
        "fast", "lethargic", "stop", "listless", "snail", "cautious",
        "delay", "traffic", "turtle", "hesitant", "wait", "sluggish",
        "motion", "crawl", "pace", "gradual", "steady", "drag",
        "plod", "creep",
    ],
}

# All available DRM themes
_ALL_DRM_THEMES: List[str] = list(_DRM_POOLS.keys())


# ---------------------------------------------------------------------------
# Name and statement pools for Source Monitoring
# ---------------------------------------------------------------------------

_FIRST_NAMES: List[str] = [
    "James", "Maria", "David", "Sarah", "Robert", "Linda", "Thomas",
    "Karen", "Daniel", "Susan", "William", "Nancy", "Richard", "Betty",
    "Charles", "Helen", "Joseph", "Sandra", "Mark", "Donna",
    "Paul", "Carol", "Steven", "Ruth", "Edward", "Sharon", "Brian",
    "Michelle", "Ronald", "Laura", "Kenneth", "Jessica", "Andrew",
    "Dorothy", "Kevin", "Lisa", "Joshua", "Emily", "George", "Deborah",
]

_LAST_NAMES: List[str] = [
    "Walker", "Chen", "Patel", "Nakamura", "Hoffman", "Silva",
    "Kowalski", "Bergstrom", "Okafor", "Reyes", "Thornton", "Larsen",
    "Moreau", "Kimura", "Hassan", "Petrov", "Andersen", "Vasquez",
    "Fischer", "Brennan", "Yamamoto", "Oliveira", "Nielsen", "Kumar",
    "Sullivan", "Lehmann", "Tanaka", "Muller", "Gonzalez", "Jensen",
]

_TITLES: List[str] = [
    "Dr.", "Professor", "Dr.", "Professor", "Dr.", "Professor",
]

_STATEMENT_TEMPLATES: List[str] = [
    "The best time to plant {plant} is during {season}.",
    "The average {animal} can live for about {number} years.",
    "The city of {city} was founded in {year}.",
    "The chemical symbol for {element} is {symbol}.",
    "The distance from Earth to {celestial} is approximately {distance}.",
    "The traditional recipe for {dish} requires {ingredient} as the key ingredient.",
    "The invention of the {invention} is credited to a team in {country}.",
    "The {landmark} was constructed using primarily {material}.",
    "Studies show that {activity} can improve {benefit} by {percent} percent.",
    "The population of {place} exceeded {population} in {decade}.",
    "The {instrument} was first developed in {region} during the {era} era.",
    "The maximum speed of a {vehicle} is approximately {speed} km/h.",
    "The {river} flows through {num_countries} different countries.",
    "The {festival} celebration traditionally lasts for {days} days.",
    "The primary export of {island} is {export}.",
    "The {mountain} has been successfully climbed by fewer than {climbers} people.",
    "The {bird} migrates approximately {distance_km} kilometers each year.",
    "The {mineral} is primarily found in deposits near {location}.",
    "The process of {process} typically takes about {duration} to complete.",
    "The discovery of {discovery} changed our understanding of {field}.",
    "The {building} was designed in the {style} architectural style.",
    "The traditional {craft} technique requires at least {years} years to master.",
    "Research indicates that {food} contains high levels of {nutrient}.",
    "The {species} was first classified by scientists in {century}.",
    "The average {object} weighs approximately {weight} kilograms.",
]

_FILL_VALUES: Dict[str, List[str]] = {
    "plant": ["lavender", "basil", "rosemary", "sage", "thyme", "mint"],
    "season": ["early spring", "late autumn", "midsummer", "early winter"],
    "animal": ["parrot", "tortoise", "dolphin", "elephant", "owl", "eagle"],
    "number": ["25", "40", "60", "80", "120", "15"],
    "city": ["Ravenna", "Bruges", "Tallinn", "Lucerne", "Salzburg", "Kyoto"],
    "year": ["1247", "1389", "1502", "1635", "1742", "1821"],
    "element": ["Rhodium", "Iridium", "Osmium", "Rhenium", "Hafnium", "Thallium"],
    "symbol": ["Rh", "Ir", "Os", "Re", "Hf", "Tl"],
    "celestial": ["Mars", "Jupiter", "Saturn", "Neptune", "Venus", "Mercury"],
    "distance": ["225 million km", "778 million km", "1.4 billion km", "4.5 billion km"],
    "dish": ["bouillabaisse", "ratatouille", "goulash", "tagine", "paella", "kimchi"],
    "ingredient": ["saffron", "cardamom", "tamarind", "miso", "paprika", "turmeric"],
    "invention": ["astrolabe", "printing press", "compass", "telescope", "pendulum clock"],
    "country": ["Portugal", "Germany", "Japan", "India", "Egypt", "Norway"],
    "landmark": ["Colosseum", "Parthenon", "Alhambra", "Angkor Wat", "Petra"],
    "material": ["travertine", "sandstone", "marble", "granite", "limestone"],
    "activity": ["meditation", "swimming", "reading", "gardening", "cycling"],
    "benefit": ["memory", "flexibility", "focus", "endurance", "creativity"],
    "percent": ["12", "18", "23", "31", "7", "42"],
    "place": ["Reykjavik", "Tallinn", "Ljubljana", "Bratislava", "Valletta"],
    "population": ["500,000", "1 million", "200,000", "750,000", "2 million"],
    "decade": ["the 2010s", "the 1990s", "the 2000s", "the 1980s"],
    "instrument": ["theremin", "hurdy-gurdy", "sitar", "didgeridoo", "erhu"],
    "region": ["Central Asia", "West Africa", "Southeast Asia", "Scandinavia"],
    "era": ["Renaissance", "Medieval", "Baroque", "Classical", "Romantic"],
    "vehicle": ["maglev train", "hydrofoil", "hovercraft", "airship", "catamaran"],
    "speed": ["450", "120", "280", "85", "350", "190"],
    "river": ["Danube", "Mekong", "Zambezi", "Rhine", "Ganges", "Niger"],
    "num_countries": ["4", "6", "8", "3", "5", "10"],
    "festival": ["Diwali", "Carnival", "Hanami", "Songkran", "Midsommar"],
    "days": ["3", "5", "7", "9", "12", "15"],
    "island": ["Mauritius", "Fiji", "Madagascar", "Borneo", "Crete"],
    "export": ["vanilla", "cinnamon", "copra", "palm oil", "teak wood"],
    "mountain": ["K2", "Makalu", "Lhotse", "Cho Oyu", "Manaslu"],
    "climbers": ["500", "300", "150", "80", "200"],
    "bird": ["Arctic Tern", "Bar-tailed Godwit", "Sooty Shearwater", "Osprey"],
    "distance_km": ["35,000", "11,000", "64,000", "18,000"],
    "mineral": ["beryllium", "tantalum", "lithium", "cobalt", "tungsten"],
    "location": ["the Andes", "the Urals", "the Himalayas", "the Rockies"],
    "process": ["fermentation", "photosynthesis", "crystallization", "calcification"],
    "duration": ["72 hours", "14 days", "6 weeks", "3 months", "48 hours"],
    "discovery": ["penicillin", "DNA structure", "X-rays", "radioactivity"],
    "field": ["medicine", "genetics", "physics", "chemistry", "biology"],
    "building": ["Guggenheim Museum", "Sydney Opera House", "Fallingwater"],
    "style": ["Brutalist", "Art Deco", "Gothic Revival", "Neoclassical"],
    "craft": ["glassblowing", "pottery", "weaving", "blacksmithing"],
    "years": ["5", "7", "10", "15", "3"],
    "food": ["quinoa", "spirulina", "kale", "acai", "turmeric"],
    "nutrient": ["vitamin K", "zinc", "manganese", "selenium", "folate"],
    "species": ["coelacanth", "okapi", "axolotl", "pangolin", "tardigrade"],
    "century": ["the 18th century", "the 19th century", "the 17th century"],
    "object": ["bowling ball", "watermelon", "car tire", "bale of hay"],
    "weight": ["6.4", "8.2", "11.5", "3.7", "15.0", "22.3"],
}


def _generate_statement(rng: random.Random) -> str:
    """Generate a plausible-sounding factoid statement using templates."""
    template = rng.choice(_STATEMENT_TEMPLATES)
    import re
    placeholders = re.findall(r"\{(\w+)\}", template)
    values: Dict[str, str] = {}
    for ph in placeholders:
        if ph in _FILL_VALUES and ph not in values:
            values[ph] = rng.choice(_FILL_VALUES[ph])
    result = template
    for ph, val in values.items():
        result = result.replace("{" + ph + "}", val, 1)
    return result


def _generate_source_name(rng: random.Random, used: Set[str]) -> str:
    """Generate a unique source name like 'Dr. Walker' or 'Professor Chen'."""
    for _ in range(100):
        title = rng.choice(_TITLES)
        last = rng.choice(_LAST_NAMES)
        name = f"{title} {last}"
        if name not in used:
            used.add(name)
            return name
    # Fallback with first name
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)
    name = f"Dr. {first} {last}"
    used.add(name)
    return name


# ===================================================================
# PARADIGM 1 -- CVLT-STYLE WORD LIST LEARNING (Multi-turn)
# ===================================================================

class CVLTGenerator:
    """Procedural generator for CVLT-style word list learning.

    Produces multi-turn episodes with:
      - Multiple learning trials (present list, free recall)
      - Interference list presentation + recall
      - Short-delay free recall of the original list
      - Long-delay free recall (after interposed filler tasks)

    Parameters are fully procedurally generated using category word pools.
    """

    PARADIGM = "cvlt_word_list"
    DIMENSION = "episodic_memory"

    DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
        "easy": {
            "list_length": 12,
            "n_learning_trials": 3,
            "interference": False,
            "long_delay": False,
        },
        "medium": {
            "list_length": 14,
            "n_learning_trials": 5,
            "interference": True,
            "long_delay": False,
        },
        "hard": {
            "list_length": 16,
            "n_learning_trials": 5,
            "interference": True,
            "long_delay": True,
        },
    }

    # Filler tasks interposed between short and long delay recall
    _FILLER_TASKS: List[str] = [
        "Please count backwards from 100 by 7s. Write out each number.",
        "Name as many countries in Europe as you can in one response.",
        "List the months of the year in reverse order.",
        "What are the first 10 prime numbers? List them.",
        "Name as many US state capitals as you can.",
    ]

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 10,
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> List[TaskInstance]:
        """Generate CVLT-style word list learning episodes.

        Each episode is a multi-turn task. Turn-level data is stored in
        ``metadata.parameters["turns"]``.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of independent episodes to generate.
        difficulty : str
            "easy" | "medium" | "hard"
        contamination_probe : bool
            If True, uses classic fruit/animal categories.
        """
        params = dict(cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"]))
        list_length: int = params["list_length"]
        n_learning_trials: int = params["n_learning_trials"]
        use_interference: bool = params["interference"]
        use_long_delay: bool = params["long_delay"]

        rng = _make_rng(seed)
        items: List[TaskInstance] = []

        for idx in range(n_items):
            ep_seed = rng.randint(0, 2**31)
            ep_rng = _make_rng(ep_seed)

            # Pick two different categories
            if contamination_probe:
                primary_cat = "fruits"
                interference_cat = "animals"
            else:
                cats = ep_rng.sample(_ALL_CATEGORIES, 2)
                primary_cat, interference_cat = cats[0], cats[1]

            # Sample word lists
            primary_pool = list(_CATEGORY_POOLS[primary_cat])
            ep_rng.shuffle(primary_pool)
            primary_list = primary_pool[:list_length]

            interference_pool = list(_CATEGORY_POOLS[interference_cat])
            ep_rng.shuffle(interference_pool)
            interference_list = interference_pool[:list_length]

            # Build turns
            turns: List[Dict[str, Any]] = []
            turn_idx = 0

            # --- Learning trials ---
            for trial in range(n_learning_trials):
                # Present list
                word_str = ", ".join(primary_list)
                turns.append({
                    "position": turn_idx,
                    "type": "learning_trial",
                    "trial_number": trial + 1,
                    "stimulus": (
                        f"Learning Trial {trial + 1}/{n_learning_trials}.\n"
                        f"Study the following list of words carefully:\n\n"
                        f"{word_str}\n\n"
                        f"Now recall as many words from the list as you can. "
                        f"Write one word per line or separate with commas."
                    ),
                    "expected_words": list(primary_list),
                    "category": primary_cat,
                })
                turn_idx += 1

            # --- Interference trial (if applicable) ---
            if use_interference:
                interf_str = ", ".join(interference_list)
                turns.append({
                    "position": turn_idx,
                    "type": "interference_trial",
                    "stimulus": (
                        f"Now study this NEW list of words (different from before):\n\n"
                        f"{interf_str}\n\n"
                        f"Recall as many words from THIS new list as you can."
                    ),
                    "expected_words": list(interference_list),
                    "category": interference_cat,
                })
                turn_idx += 1

                # --- Short-delay free recall of original list ---
                turns.append({
                    "position": turn_idx,
                    "type": "short_delay_recall",
                    "stimulus": (
                        f"Now go back to the FIRST list of words you studied "
                        f"(the {primary_cat} words, NOT the {interference_cat} words).\n"
                        f"Recall as many words from that FIRST list as you can."
                    ),
                    "expected_words": list(primary_list),
                    "category": primary_cat,
                })
                turn_idx += 1

            # --- Long-delay free recall (with filler tasks) ---
            if use_long_delay:
                # Interpose filler tasks
                filler = ep_rng.choice(cls._FILLER_TASKS)
                turns.append({
                    "position": turn_idx,
                    "type": "filler_task",
                    "stimulus": (
                        f"Before we continue, please complete this task:\n"
                        f"{filler}"
                    ),
                    "expected_words": None,
                })
                turn_idx += 1

                # A second filler
                filler2 = ep_rng.choice(
                    [f for f in cls._FILLER_TASKS if f != filler]
                )
                turns.append({
                    "position": turn_idx,
                    "type": "filler_task",
                    "stimulus": (
                        f"One more task:\n{filler2}"
                    ),
                    "expected_words": None,
                })
                turn_idx += 1

                # Long-delay recall
                turns.append({
                    "position": turn_idx,
                    "type": "long_delay_recall",
                    "stimulus": (
                        f"Now think back to the VERY FIRST list of words you "
                        f"studied at the beginning (the {primary_cat} words).\n"
                        f"Recall as many of those words as you can."
                    ),
                    "expected_words": list(primary_list),
                    "category": primary_cat,
                })
                turn_idx += 1

            # Build stimulus overview
            stimulus_text = (
                f"CVLT Word List Learning Task.\n"
                f"You will study a list of {list_length} words across "
                f"{n_learning_trials} learning trials.\n"
                f"After each presentation, recall as many words as you can.\n"
            )
            if use_interference:
                stimulus_text += (
                    f"You will then study a second (interference) list and recall it.\n"
                    f"After that, recall the original list again.\n"
                )
            if use_long_delay:
                stimulus_text += (
                    f"Finally, after some unrelated tasks, recall the original list "
                    f"once more (long delay).\n"
                )
            stimulus_text += (
                f"\n--- Learning Trial 1/{n_learning_trials} ---\n"
                f"Study these words:\n\n"
                f"{', '.join(primary_list)}\n\n"
                f"Now recall as many words as you can."
            )

            task_id = (
                f"em_cvlt_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_{primary_cat}_s{ep_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.AGENT_INTERACTIVE,
                parameters={
                    "list_length": list_length,
                    "n_learning_trials": n_learning_trials,
                    "primary_category": primary_cat,
                    "interference_category": interference_cat,
                    "primary_list": primary_list,
                    "interference_list": interference_list,
                    "use_interference": use_interference,
                    "use_long_delay": use_long_delay,
                    "episode_seed": ep_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": True,
                    "turns": turns,
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.episodic_memory.score_cvlt",
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=AdaptationDistance.LOW,
                description=(
                    f"CVLT word list learning, {list_length} words "
                    f"({primary_cat}), {n_learning_trials} trials, "
                    f"{'with' if use_interference else 'no'} interference"
                ),
            )

            items.append(TaskInstance(
                task_id=task_id,
                metadata=metadata,
                stimulus=stimulus_text,
                expected_response=primary_list,
            ))

        return items

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(
        task: TaskInstance, responses: List[str]
    ) -> Dict[str, float]:
        """Score a completed CVLT episode.

        Parameters
        ----------
        task : TaskInstance
            The generated CVLT task.
        responses : list[str]
            Model responses, one per turn.

        Returns
        -------
        dict with keys: learning_curve (per-trial recall counts),
        total_learning, short_delay_recall, long_delay_recall,
        intrusions, perseverations, primacy_effect, recency_effect.
        """
        turns = task.metadata.parameters["turns"]
        primary_list = task.metadata.parameters["primary_list"]
        primary_set = set(w.lower() for w in primary_list)
        list_length = len(primary_list)

        # Pad responses if needed
        while len(responses) < len(turns):
            responses.append("")

        learning_curve: List[int] = []
        total_intrusions = 0
        total_perseverations = 0
        short_delay_recall_count = 0
        long_delay_recall_count = 0
        interference_recall_count = 0

        # Track serial position effects across learning trials
        primacy_hits = 0  # first quartile
        recency_hits = 0  # last quartile
        middle_hits = 0
        learning_trial_count = 0
        quartile_size = max(1, list_length // 4)

        for turn, resp in zip(turns, responses):
            if turn["type"] == "filler_task":
                continue

            recalled = _parse_word_list(resp)
            expected_set = set(
                w.lower() for w in (turn.get("expected_words") or [])
            )

            if turn["type"] == "learning_trial":
                hits = sum(1 for w in recalled if w in primary_set)
                learning_curve.append(hits)
                learning_trial_count += 1

                # Intrusions: words not in primary list
                intrusions = sum(1 for w in recalled if w not in primary_set)
                total_intrusions += intrusions

                # Perseverations: duplicate responses within a single trial
                seen: Set[str] = set()
                for w in recalled:
                    if w in seen:
                        total_perseverations += 1
                    seen.add(w)

                # Serial position: check which positions were recalled
                for w in recalled:
                    if w in primary_set:
                        try:
                            pos = [x.lower() for x in primary_list].index(w)
                            if pos < quartile_size:
                                primacy_hits += 1
                            elif pos >= list_length - quartile_size:
                                recency_hits += 1
                            else:
                                middle_hits += 1
                        except ValueError:
                            pass

            elif turn["type"] == "interference_trial":
                interference_set = set(
                    w.lower() for w in (turn.get("expected_words") or [])
                )
                interference_recall_count = sum(
                    1 for w in recalled if w in interference_set
                )

            elif turn["type"] == "short_delay_recall":
                short_delay_recall_count = sum(
                    1 for w in recalled if w in primary_set
                )
                # Count intrusions from interference list
                interference_list = task.metadata.parameters.get(
                    "interference_list", []
                )
                interference_set = set(w.lower() for w in interference_list)
                interference_intrusions = sum(
                    1 for w in recalled if w in interference_set
                )
                total_intrusions += interference_intrusions
                # Other intrusions
                total_intrusions += sum(
                    1 for w in recalled
                    if w not in primary_set and w not in interference_set
                )

            elif turn["type"] == "long_delay_recall":
                long_delay_recall_count = sum(
                    1 for w in recalled if w in primary_set
                )

        # Compute serial position effect
        total_position_hits = primacy_hits + recency_hits + middle_hits
        if total_position_hits > 0 and learning_trial_count > 0:
            # Normalize by number of items in each region
            middle_size = max(1, list_length - 2 * quartile_size)
            primacy_rate = (primacy_hits / learning_trial_count) / quartile_size
            recency_rate = (recency_hits / learning_trial_count) / quartile_size
            middle_rate = (middle_hits / learning_trial_count) / middle_size
            # Serial position effect: (primacy + recency) / 2 vs middle
            avg_ends = (primacy_rate + recency_rate) / 2
            serial_position_effect = (
                (avg_ends - middle_rate) / max(avg_ends, middle_rate, 0.001)
            )
        else:
            primacy_rate = 0.0
            recency_rate = 0.0
            serial_position_effect = 0.0

        # Total learning: sum of all learning trial recalls
        total_learning = sum(learning_curve)

        # Build result
        result: Dict[str, float] = {
            "accuracy": (
                total_learning / max(list_length * len(learning_curve), 1)
            ),
            "total_learning": float(total_learning),
            "n_learning_trials": float(len(learning_curve)),
            "list_length": float(list_length),
            "intrusions": float(total_intrusions),
            "perseverations": float(total_perseverations),
            "primacy_effect": round(primacy_rate, 4),
            "recency_effect": round(recency_rate, 4),
            "serial_position_effect": round(serial_position_effect, 4),
        }

        # Per-trial recall counts
        for i, count in enumerate(learning_curve):
            result[f"trial_{i+1}_recall"] = float(count)

        if task.metadata.parameters.get("use_interference"):
            result["interference_recall"] = float(interference_recall_count)
            result["short_delay_recall"] = float(short_delay_recall_count)
            result["short_delay_recall_rate"] = round(
                short_delay_recall_count / max(list_length, 1), 4
            )

        if task.metadata.parameters.get("use_long_delay"):
            result["long_delay_recall"] = float(long_delay_recall_count)
            result["long_delay_recall_rate"] = round(
                long_delay_recall_count / max(list_length, 1), 4
            )

        return result


# Module-level scoring function referenced by ScoringConfig custom fn path
def score_cvlt(
    response: Any,
    expected: Any,
    metadata: Any,
) -> Dict[str, float]:
    """Scoring function for CVLT (used by TaskInstance.score via custom fn).

    For the custom scoring path, ``response`` is expected to be a list of
    per-turn response strings. If a single string is provided, it is
    treated as a single-trial response.
    """
    if isinstance(metadata, TaskMetadata):
        params = metadata.parameters
    elif isinstance(metadata, dict):
        params = metadata
    else:
        params = getattr(metadata, "parameters", {})

    primary_list = params.get("primary_list", expected or [])
    primary_set = set(w.lower() for w in primary_list)
    list_length = len(primary_list)

    if isinstance(response, str):
        recalled = _parse_word_list(response)
        hits = sum(1 for w in recalled if w in primary_set)
        intrusions = sum(1 for w in recalled if w not in primary_set)
        return {
            "accuracy": hits / max(list_length, 1),
            "recall": float(hits),
            "intrusions": float(intrusions),
        }

    # Multi-turn: delegate to full scoring via a minimal TaskInstance
    turns = params.get("turns", [])
    responses = list(response) if not isinstance(response, list) else response
    while len(responses) < len(turns):
        responses.append("")

    # Simplified scoring for the custom fn path
    learning_recalls: List[int] = []
    for turn, resp in zip(turns, responses):
        if turn.get("type") == "learning_trial":
            recalled = _parse_word_list(resp)
            hits = sum(1 for w in recalled if w in primary_set)
            learning_recalls.append(hits)

    total = sum(learning_recalls)
    n_trials = max(len(learning_recalls), 1)
    return {
        "accuracy": total / max(list_length * n_trials, 1),
        "total_learning": float(total),
    }


# ===================================================================
# PARADIGM 2 -- DRM FALSE MEMORY (Static)
# ===================================================================

class DRMGenerator:
    """Procedural generator for the DRM False Memory paradigm.

    Presents lists of semantically associated words where a critical lure
    (the theme word) is NOT presented.  The model then takes a recognition
    test with targets (presented words), critical lures (theme words),
    and unrelated foils.

    Scoring focuses on false memory rates: do models falsely "recognize"
    the critical lure at higher rates than unrelated foils?
    """

    PARADIGM = "drm_false_memory"
    DIMENSION = "episodic_memory"

    DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
        "easy": {
            "n_lists": 3,
            "list_length": 12,
            "n_recognition_targets": 6,
            "n_critical_lures": 3,
            "n_unrelated_foils": 6,
        },
        "medium": {
            "n_lists": 5,
            "list_length": 12,
            "n_recognition_targets": 8,
            "n_critical_lures": 5,
            "n_unrelated_foils": 10,
        },
        "hard": {
            "n_lists": 8,
            "list_length": 15,
            "n_recognition_targets": 12,
            "n_critical_lures": 8,
            "n_unrelated_foils": 15,
        },
    }

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 10,
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> List[TaskInstance]:
        """Generate DRM false memory recognition test items.

        Each item presents multiple DRM-style word lists, then gives a
        recognition test.  The stimulus contains the study lists and the
        recognition probe words.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of independent DRM episodes to generate.
        difficulty : str
            Controls number of lists, list length, and recognition test size.
        contamination_probe : bool
            If True, uses classic DRM themes (sleep, needle, etc.).
        """
        params = dict(cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"]))
        n_lists: int = params["n_lists"]
        list_length: int = params["list_length"]
        n_rec_targets: int = params["n_recognition_targets"]
        n_critical_lures: int = params["n_critical_lures"]
        n_unrelated_foils: int = params["n_unrelated_foils"]

        rng = _make_rng(seed)
        items: List[TaskInstance] = []

        for idx in range(n_items):
            ep_seed = rng.randint(0, 2**31)
            ep_rng = _make_rng(ep_seed)

            # Select themes
            available_themes = list(_ALL_DRM_THEMES)
            ep_rng.shuffle(available_themes)

            if contamination_probe:
                # Use the most classic themes
                selected_themes = ["sleep", "needle", "mountain", "river",
                                   "music", "sweet", "cold", "anger"][:n_lists]
            else:
                selected_themes = available_themes[:n_lists]

            # Build study lists
            study_lists: List[Dict[str, Any]] = []
            all_presented_words: List[str] = []
            critical_lures: List[str] = []

            for theme in selected_themes:
                associates = list(_DRM_POOLS[theme])
                ep_rng.shuffle(associates)
                presented = associates[:list_length]
                study_lists.append({
                    "theme": theme,
                    "words": presented,
                })
                all_presented_words.extend(presented)
                critical_lures.append(theme)

            all_presented_set = set(w.lower() for w in all_presented_words)

            # Build recognition test
            # Targets: sample from presented words
            target_candidates = list(all_presented_words)
            ep_rng.shuffle(target_candidates)
            recognition_targets = target_candidates[:min(n_rec_targets, len(target_candidates))]

            # Critical lures: the theme words themselves
            rec_critical_lures = critical_lures[:n_critical_lures]

            # Unrelated foils: words that are NOT in any study list or
            # theme pool.  Use words from unused DRM themes' associates.
            unused_themes = [
                t for t in _ALL_DRM_THEMES if t not in selected_themes
            ]
            foil_candidates: List[str] = []
            for ut in unused_themes:
                for w in _DRM_POOLS[ut]:
                    if w.lower() not in all_presented_set and w not in critical_lures:
                        foil_candidates.append(w)
            ep_rng.shuffle(foil_candidates)
            unrelated_foils = list(dict.fromkeys(foil_candidates))[:n_unrelated_foils]

            # If we still need more foils, grab from category pools
            if len(unrelated_foils) < n_unrelated_foils:
                for cat in _ALL_CATEGORIES:
                    for w in _CATEGORY_POOLS[cat]:
                        if (w.lower() not in all_presented_set
                                and w not in critical_lures
                                and w not in unrelated_foils):
                            unrelated_foils.append(w)
                        if len(unrelated_foils) >= n_unrelated_foils:
                            break
                    if len(unrelated_foils) >= n_unrelated_foils:
                        break

            # Combine and shuffle recognition probe words
            all_probes: List[Dict[str, Any]] = []
            for w in recognition_targets:
                all_probes.append({"word": w, "type": "target", "correct": "OLD"})
            for w in rec_critical_lures:
                all_probes.append({"word": w, "type": "critical_lure", "correct": "NEW"})
            for w in unrelated_foils:
                all_probes.append({"word": w, "type": "unrelated_foil", "correct": "NEW"})
            ep_rng.shuffle(all_probes)

            # Build stimulus text
            study_text = "WORD LIST STUDY PHASE\n\n"
            study_text += (
                "You will now study several word lists. "
                "Read each list carefully and try to remember the words.\n\n"
            )
            for li, sl in enumerate(study_lists, 1):
                study_text += f"--- List {li} ---\n"
                study_text += ", ".join(sl["words"]) + "\n\n"

            study_text += (
                "RECOGNITION TEST\n\n"
                "For each word below, respond OLD if you saw the word in "
                "any of the lists above, or NEW if you did not see it.\n"
                "Format your response as one line per word: "
                "WORD: OLD or WORD: NEW\n\n"
            )
            for probe in all_probes:
                study_text += f"  {probe['word']}: ___\n"

            # Expected response (for reference)
            expected_lines: List[str] = []
            for probe in all_probes:
                expected_lines.append(f"{probe['word']}: {probe['correct']}")
            expected_response = "\n".join(expected_lines)

            task_id = (
                f"em_drm_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_n{n_lists}_s{ep_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.LLM_STATIC,
                parameters={
                    "n_lists": n_lists,
                    "list_length": list_length,
                    "study_lists": study_lists,
                    "all_presented_words": all_presented_words,
                    "critical_lures": critical_lures,
                    "recognition_probes": all_probes,
                    "recognition_targets": recognition_targets,
                    "rec_critical_lures": rec_critical_lures,
                    "unrelated_foils": unrelated_foils,
                    "episode_seed": ep_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": False,
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.episodic_memory.score_drm",
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=AdaptationDistance.LOW,
                description=(
                    f"DRM false memory, {n_lists} lists of {list_length} words, "
                    f"{'classic' if contamination_probe else 'procedural'} themes"
                ),
            )

            items.append(TaskInstance(
                task_id=task_id,
                metadata=metadata,
                stimulus=study_text,
                expected_response=expected_response,
            ))

        return items

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, float]:
        """Score a DRM recognition test.

        Parameters
        ----------
        task : TaskInstance
            The generated DRM task.
        response : str
            Model response (one line per probe: "WORD: OLD/NEW").

        Returns
        -------
        dict with: true_positive_rate, false_alarm_to_critical_lures,
        false_alarm_to_unrelated, d_prime, false_memory_index.
        """
        probes = task.metadata.parameters["recognition_probes"]
        return _score_drm_response(probes, response)


def _score_drm_response(
    probes: List[Dict[str, Any]], response: str
) -> Dict[str, float]:
    """Core DRM scoring logic."""
    import re

    # Parse model response: look for "word: OLD" or "word: NEW" patterns
    response_map: Dict[str, str] = {}
    resp_text = str(response).strip()

    # Try line-by-line parsing
    for line in resp_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match patterns like "word: OLD", "word - OLD", "word OLD"
        match = re.match(
            r"^[\s\-\d\.\)]*(.+?)[\s:\-]+\b(OLD|NEW)\b",
            line, re.IGNORECASE,
        )
        if match:
            word = match.group(1).strip().lower().strip(".,;:!?\"'()")
            judgment = match.group(2).upper()
            response_map[word] = judgment

    # Score each probe
    target_hits = 0
    target_total = 0
    critical_lure_fa = 0
    critical_lure_total = 0
    unrelated_fa = 0
    unrelated_total = 0
    correct_total = 0

    for probe in probes:
        word = probe["word"].lower()
        expected = probe["correct"]
        probe_type = probe["type"]

        # Find the model's judgment for this word
        judgment = response_map.get(word, "")

        if probe_type == "target":
            target_total += 1
            if judgment == "OLD":
                target_hits += 1
                correct_total += 1
            elif judgment == "NEW":
                pass  # miss
            # else: no response found

        elif probe_type == "critical_lure":
            critical_lure_total += 1
            if judgment == "OLD":
                critical_lure_fa += 1  # false alarm
            elif judgment == "NEW":
                correct_total += 1  # correct rejection

        elif probe_type == "unrelated_foil":
            unrelated_total += 1
            if judgment == "OLD":
                unrelated_fa += 1  # false alarm
            elif judgment == "NEW":
                correct_total += 1  # correct rejection

    # Compute rates
    hit_rate = target_hits / max(target_total, 1)
    fa_critical = critical_lure_fa / max(critical_lure_total, 1)
    fa_unrelated = unrelated_fa / max(unrelated_total, 1)
    total_probes = target_total + critical_lure_total + unrelated_total
    overall_accuracy = correct_total / max(total_probes, 1)

    # d' using unrelated foil FA as the base false alarm rate
    d_prime_val = _dprime(hit_rate, fa_unrelated)

    # False memory index: difference between critical lure FA and unrelated FA
    false_memory_index = fa_critical - fa_unrelated

    return {
        "accuracy": round(overall_accuracy, 4),
        "true_positive_rate": round(hit_rate, 4),
        "false_alarm_to_critical_lures": round(fa_critical, 4),
        "false_alarm_to_unrelated": round(fa_unrelated, 4),
        "d_prime": round(d_prime_val, 4),
        "false_memory_index": round(false_memory_index, 4),
        "target_hits": float(target_hits),
        "target_total": float(target_total),
        "critical_lure_false_alarms": float(critical_lure_fa),
        "critical_lure_total": float(critical_lure_total),
        "unrelated_false_alarms": float(unrelated_fa),
        "unrelated_total": float(unrelated_total),
    }


# Module-level scoring function referenced by ScoringConfig custom fn path
def score_drm(
    response: Any,
    expected: Any,
    metadata: Any,
) -> Dict[str, float]:
    """Scoring function for DRM (used by TaskInstance.score via custom fn)."""
    if isinstance(metadata, TaskMetadata):
        params = metadata.parameters
    elif isinstance(metadata, dict):
        params = metadata
    else:
        params = getattr(metadata, "parameters", {})

    probes = params.get("recognition_probes", [])
    return _score_drm_response(probes, str(response))


# ===================================================================
# PARADIGM 3 -- SOURCE MONITORING (Static)
# ===================================================================

class SourceMonitoringGenerator:
    """Procedural generator for the Source Monitoring paradigm.

    Presents factual statements attributed to specific named sources.
    The model must later identify which source made each statement.

    This tests episodic binding — remembering not just *what* was said
    but *who* said it.
    """

    PARADIGM = "source_monitoring"
    DIMENSION = "episodic_memory"

    DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
        "easy": {
            "n_sources": 3,
            "n_statements_per_source": 4,
            "n_test_items": 8,
        },
        "medium": {
            "n_sources": 4,
            "n_statements_per_source": 5,
            "n_test_items": 14,
        },
        "hard": {
            "n_sources": 5,
            "n_statements_per_source": 6,
            "n_test_items": 22,
        },
    }

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 10,
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> List[TaskInstance]:
        """Generate Source Monitoring test items.

        Each item presents a set of source-attributed statements, then
        asks the model to identify which source said what.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of independent episodes to generate.
        difficulty : str
            Controls n_sources, statements per source, test items.
        contamination_probe : bool
            If True, uses more stereotypical source names and statements.
        """
        params = dict(cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"]))
        n_sources: int = params["n_sources"]
        n_stmts_per: int = params["n_statements_per_source"]
        n_test: int = params["n_test_items"]

        rng = _make_rng(seed)
        items: List[TaskInstance] = []

        for idx in range(n_items):
            ep_seed = rng.randint(0, 2**31)
            ep_rng = _make_rng(ep_seed)

            # Generate unique source names
            used_names: Set[str] = set()
            sources: List[str] = []
            for _ in range(n_sources):
                name = _generate_source_name(ep_rng, used_names)
                sources.append(name)

            # Generate statements for each source
            source_statements: Dict[str, List[str]] = {}
            all_statements: List[Dict[str, str]] = []  # {source, statement}

            for source in sources:
                stmts: List[str] = []
                for _ in range(n_stmts_per):
                    stmt = _generate_statement(ep_rng)
                    # Ensure no duplicate statements
                    attempts = 0
                    while stmt in stmts and attempts < 20:
                        stmt = _generate_statement(ep_rng)
                        attempts += 1
                    stmts.append(stmt)
                    all_statements.append({
                        "source": source,
                        "statement": stmt,
                    })
                source_statements[source] = stmts

            # Build study phase text
            study_text = "SOURCE MONITORING TASK\n\n"
            study_text += (
                "Read the following statements carefully. "
                "Each statement is attributed to a specific person. "
                "Pay attention to WHO says WHAT.\n\n"
            )
            # Present statements in interleaved order (more challenging)
            presentation_order = list(all_statements)
            ep_rng.shuffle(presentation_order)

            for si, item in enumerate(presentation_order, 1):
                study_text += (
                    f"{si}. {item['source']} says: "
                    f"\"{item['statement']}\"\n"
                )

            # Build test phase
            # Select a subset of statements for testing
            test_candidates = list(all_statements)
            ep_rng.shuffle(test_candidates)
            test_items = test_candidates[:min(n_test, len(test_candidates))]

            study_text += "\n\nSOURCE IDENTIFICATION TEST\n\n"
            study_text += (
                "For each statement below, identify WHO said it. "
                "Choose from: " + ", ".join(sources) + "\n"
                "Format: Statement number. SOURCE NAME\n\n"
            )

            test_data: List[Dict[str, Any]] = []
            expected_lines: List[str] = []
            for ti, test_item in enumerate(test_items, 1):
                study_text += (
                    f"{ti}. \"{test_item['statement']}\"\n"
                    f"   Who said this? ___\n\n"
                )
                test_data.append({
                    "test_position": ti,
                    "statement": test_item["statement"],
                    "correct_source": test_item["source"],
                    "all_sources": list(sources),
                })
                expected_lines.append(
                    f"{ti}. {test_item['source']}"
                )

            expected_response = "\n".join(expected_lines)

            task_id = (
                f"em_srcmon_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_src{n_sources}_s{ep_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.LLM_STATIC,
                parameters={
                    "n_sources": n_sources,
                    "n_statements_per_source": n_stmts_per,
                    "sources": sources,
                    "source_statements": source_statements,
                    "all_statements": all_statements,
                    "presentation_order": [
                        {"source": s["source"], "statement": s["statement"]}
                        for s in presentation_order
                    ],
                    "test_items": test_data,
                    "n_test_items": len(test_data),
                    "episode_seed": ep_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": False,
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.episodic_memory.score_source_monitoring",
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=AdaptationDistance.LOW,
                description=(
                    f"Source monitoring, {n_sources} sources, "
                    f"{n_stmts_per} statements each"
                ),
            )

            items.append(TaskInstance(
                task_id=task_id,
                metadata=metadata,
                stimulus=study_text,
                expected_response=expected_response,
            ))

        return items

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, float]:
        """Score a source monitoring test.

        Parameters
        ----------
        task : TaskInstance
            The generated source monitoring task.
        response : str
            Model response with source attributions.

        Returns
        -------
        dict with: source_accuracy, per-source accuracy,
        within_type_errors, between_type_errors.
        """
        test_items = task.metadata.parameters["test_items"]
        sources = task.metadata.parameters["sources"]
        return _score_source_monitoring_response(test_items, sources, response)


def _score_source_monitoring_response(
    test_items: List[Dict[str, Any]],
    sources: List[str],
    response: str,
) -> Dict[str, float]:
    """Core source monitoring scoring logic."""
    import re

    resp_text = str(response).strip()

    # Parse response: look for "N. Source Name" patterns
    response_map: Dict[int, str] = {}
    for line in resp_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match "1. Dr. Walker" or "1) Professor Chen" etc.
        match = re.match(
            r"^(\d+)[\.\)\-\s]+(.+)$", line,
        )
        if match:
            num = int(match.group(1))
            attribution = match.group(2).strip().rstrip(".,;")
            response_map[num] = attribution

    # Score each test item
    correct = 0
    total = len(test_items)

    # Confusion matrix: confusion[correct_source][attributed_source] = count
    source_set = set(s.lower() for s in sources)
    confusion: Dict[str, Dict[str, int]] = {
        s: {s2: 0 for s2 in sources} for s in sources
    }
    per_source_correct: Dict[str, int] = {s: 0 for s in sources}
    per_source_total: Dict[str, int] = {s: 0 for s in sources}

    for item in test_items:
        pos = item["test_position"]
        correct_source = item["correct_source"]
        per_source_total[correct_source] = per_source_total.get(correct_source, 0) + 1

        attributed = response_map.get(pos, "")

        # Try to match the attributed source to one of the known sources
        matched_source = None
        attr_lower = attributed.lower()
        if not attr_lower.strip():
            # Missing/unparseable attribution earns no credit: an empty
            # string would otherwise substring-match the first source.
            continue
        for s in sources:
            if s.lower() in attr_lower or attr_lower in s.lower():
                matched_source = s
                break
        # Fallback: check last name only
        if matched_source is None:
            for s in sources:
                last_name = s.split()[-1].lower()
                if last_name in attr_lower:
                    matched_source = s
                    break

        if matched_source == correct_source:
            correct += 1
            per_source_correct[correct_source] = (
                per_source_correct.get(correct_source, 0) + 1
            )

        if matched_source is not None:
            confusion[correct_source][matched_source] = (
                confusion[correct_source].get(matched_source, 0) + 1
            )

    accuracy = correct / max(total, 1)

    # Compute error types
    # Within-type errors: confusing sources with same title prefix
    # Between-type errors: confusing sources with different title prefix
    within_type_errors = 0
    between_type_errors = 0

    for true_src, attributed_dict in confusion.items():
        true_title = true_src.split()[0]  # "Dr." or "Professor"
        for attr_src, count in attributed_dict.items():
            if attr_src != true_src and count > 0:
                attr_title = attr_src.split()[0]
                if true_title == attr_title:
                    within_type_errors += count
                else:
                    between_type_errors += count

    result: Dict[str, float] = {
        "accuracy": round(accuracy, 4),
        "source_accuracy": round(accuracy, 4),
        "correct": float(correct),
        "total": float(total),
        "within_type_errors": float(within_type_errors),
        "between_type_errors": float(between_type_errors),
    }

    # Per-source accuracy
    for s in sources:
        s_total = per_source_total.get(s, 0)
        s_correct = per_source_correct.get(s, 0)
        safe_key = s.replace(" ", "_").replace(".", "")
        result[f"source_{safe_key}_accuracy"] = round(
            s_correct / max(s_total, 1), 4
        )

    return result


# Module-level scoring function referenced by ScoringConfig custom fn path
def score_source_monitoring(
    response: Any,
    expected: Any,
    metadata: Any,
) -> Dict[str, float]:
    """Scoring function for Source Monitoring (used by TaskInstance.score)."""
    if isinstance(metadata, TaskMetadata):
        params = metadata.parameters
    elif isinstance(metadata, dict):
        params = metadata
    else:
        params = getattr(metadata, "parameters", {})

    test_items = params.get("test_items", [])
    sources = params.get("sources", [])
    return _score_source_monitoring_response(test_items, sources, str(response))


# ===================================================================
# Convenience dispatch
# ===================================================================

_GENERATORS: Dict[str, type] = {
    "cvlt_word_list": CVLTGenerator,
    "drm_false_memory": DRMGenerator,
    "source_monitoring": SourceMonitoringGenerator,
}


def generate(
    paradigm: str,
    seed: int,
    n_items: int = 10,
    difficulty: str = "medium",
    contamination_probe: bool = False,
) -> List[TaskInstance]:
    """Unified entry-point for generating Episodic Memory items.

    Parameters
    ----------
    paradigm : str
        One of "cvlt_word_list", "drm_false_memory", "source_monitoring".
    seed, n_items, difficulty, contamination_probe
        Forwarded to the paradigm generator.
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
    )


def score(task: TaskInstance, response: Any) -> Dict[str, float]:
    """Unified scoring dispatcher.

    *response* should be:
      - ``list[str]`` for multi-turn paradigms (cvlt_word_list)
      - ``str`` for single-turn paradigms (drm_false_memory, source_monitoring)
    """
    gen_cls = _GENERATORS.get(task.metadata.paradigm)
    if gen_cls is None:
        raise ValueError(f"Unknown paradigm '{task.metadata.paradigm}'")
    return gen_cls.score(task, response)
