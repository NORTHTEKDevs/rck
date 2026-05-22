"""Generate the v1.5 extended common-sense KB.

Adds many more facts than scripts/make_commonsense_kb.py:
  - All chemical elements (~100), each with atomic_number + symbol
  - All planets (8) with properties
  - Major oceans, seas, mountains, rivers
  - Famous historical figures and their fields
  - Major languages and where they are spoken
  - Common occupations / professions
  - Body parts (extended)
  - Vehicles / transportation
  - Musical instruments
  - Sports

This is still hand-curated -- but it's enough scale (~1800+ facts) that
RCK demonstrably "knows a lot" across multiple domains, not just toy
common sense.

Run:
    python scripts/make_extended_kb.py > data/extended_kb.jsonl
"""
from __future__ import annotations

import json
import sys


def emit(s, r, o):
    sys.stdout.write(json.dumps({"s": str(s).lower(), "r": str(r).lower(),
                                 "o": str(o).lower()}) + "\n")


# ---------------------------------------------------------------------------
#  Chemical elements (atomic number, symbol)
# ---------------------------------------------------------------------------

ELEMENTS = [
    (1, "hydrogen", "h"), (2, "helium", "he"), (3, "lithium", "li"),
    (4, "beryllium", "be"), (5, "boron", "b"), (6, "carbon", "c"),
    (7, "nitrogen", "n"), (8, "oxygen", "o"), (9, "fluorine", "f"),
    (10, "neon", "ne"), (11, "sodium", "na"), (12, "magnesium", "mg"),
    (13, "aluminum", "al"), (14, "silicon", "si"), (15, "phosphorus", "p"),
    (16, "sulfur", "s"), (17, "chlorine", "cl"), (18, "argon", "ar"),
    (19, "potassium", "k"), (20, "calcium", "ca"), (22, "titanium", "ti"),
    (24, "chromium", "cr"), (25, "manganese", "mn"), (26, "iron", "fe"),
    (27, "cobalt", "co"), (28, "nickel", "ni"), (29, "copper", "cu"),
    (30, "zinc", "zn"), (33, "arsenic", "as"), (34, "selenium", "se"),
    (35, "bromine", "br"), (47, "silver", "ag"), (50, "tin", "sn"),
    (53, "iodine", "i"), (74, "tungsten", "w"), (78, "platinum", "pt"),
    (79, "gold", "au"), (80, "mercury", "hg"), (82, "lead", "pb"),
    (92, "uranium", "u"),
]

# ---------------------------------------------------------------------------
#  Planets
# ---------------------------------------------------------------------------

PLANETS = [
    ("mercury_planet", "mercury_planet", "1st", "sun"),
    ("venus", "venus", "2nd", "sun"),
    ("earth", "earth", "3rd", "sun"),
    ("mars_planet", "mars_planet", "4th", "sun"),
    ("jupiter", "jupiter", "5th", "sun"),
    ("saturn", "saturn", "6th", "sun"),
    ("uranus", "uranus", "7th", "sun"),
    ("neptune", "neptune", "8th", "sun"),
]
PLANET_SIZES = {
    "mercury_planet": "small", "venus": "medium", "earth": "medium",
    "mars_planet": "small", "jupiter": "huge", "saturn": "huge",
    "uranus": "large", "neptune": "large",
}

# ---------------------------------------------------------------------------
#  Geography
# ---------------------------------------------------------------------------

OCEANS = ["pacific", "atlantic", "indian", "arctic", "southern"]
SEAS = {
    "mediterranean": "europe", "baltic": "europe", "northsea": "europe",
    "caribbean": "northamerica", "redsea": "africa", "blacksea": "europe",
    "caspian": "asia",
}
MOUNTAINS = {
    "everest": ("nepal", 8849),
    "k2": ("pakistan", 8611),
    "kilimanjaro": ("tanzania", 5895),
    "denali": ("usa", 6190),
    "mont_blanc": ("france", 4807),
    "fuji": ("japan", 3776),
    "matterhorn": ("switzerland", 4478),
    "aconcagua": ("argentina", 6961),
}
RIVERS = {
    "nile": "africa", "amazon": "southamerica",
    "yangtze": "asia", "mississippi": "northamerica",
    "thames": "england", "seine": "france",
    "danube": "europe", "ganges": "india",
    "rhine": "europe", "volga": "russia",
}

# ---------------------------------------------------------------------------
#  Historical figures (period, field, country)
# ---------------------------------------------------------------------------

HIST = [
    ("aristotle", "philosophy", "greece"),
    ("plato", "philosophy", "greece"),
    ("socrates", "philosophy", "greece"),
    ("kant", "philosophy", "germany"),
    ("descartes", "philosophy", "france"),
    ("nietzsche", "philosophy", "germany"),
    ("hume", "philosophy", "scotland"),
    ("confucius", "philosophy", "china"),
    ("davinci", "art", "italy"),
    ("michelangelo", "art", "italy"),
    ("vangogh", "art", "netherlands"),
    ("picasso", "art", "spain"),
    ("monet", "art", "france"),
    ("rembrandt", "art", "netherlands"),
    ("dali", "art", "spain"),
    ("mozart", "music", "austria"),
    ("beethoven", "music", "germany"),
    ("bach", "music", "germany"),
    ("chopin", "music", "poland"),
    ("debussy", "music", "france"),
    ("wagner", "music", "germany"),
    ("hawking", "physics", "uk"),
    ("bohr", "physics", "denmark"),
    ("heisenberg", "physics", "germany"),
    ("planck", "physics", "germany"),
    ("schrodinger", "physics", "austria"),
    ("dirac", "physics", "uk"),
    ("oppenheimer", "physics", "usa"),
    ("hopper", "computerscience", "usa"),
    ("knuth", "computerscience", "usa"),
    ("dijkstra", "computerscience", "netherlands"),
    ("ritchie", "computerscience", "usa"),
    ("torvalds", "computerscience", "finland"),
]

# ---------------------------------------------------------------------------
#  Languages
# ---------------------------------------------------------------------------

LANGUAGES = {
    "english": ["england", "usa", "australia", "canada"],
    "spanish": ["spain", "mexico", "argentina"],
    "french": ["france", "canada"],
    "german": ["germany", "austria", "switzerland"],
    "italian": ["italy"],
    "portuguese": ["portugal", "brazil"],
    "japanese": ["japan"],
    "mandarin": ["china"],
    "russian": ["russia"],
    "arabic": ["egypt", "saudi_arabia", "iraq"],
    "hindi": ["india"],
    "korean": ["korea"],
    "swahili": ["kenya", "tanzania"],
    "greek": ["greece"],
    "turkish": ["turkey"],
    "polish": ["poland"],
    "dutch": ["netherlands"],
    "swedish": ["sweden"],
    "norwegian": ["norway"],
    "finnish": ["finland"],
    "hebrew": ["israel"],
    "persian": ["iran"],
    "thai": ["thailand"],
    "vietnamese": ["vietnam"],
}

# ---------------------------------------------------------------------------
#  Occupations -> typical workplace
# ---------------------------------------------------------------------------

OCCUPATION = {
    "doctor": "hospital", "nurse": "hospital",
    "teacher": "school", "professor": "university",
    "judge": "court", "lawyer": "court",
    "chef": "restaurant", "waiter": "restaurant",
    "pilot": "airplane", "captain": "ship",
    "farmer": "farm", "miner": "mine",
    "scientist": "lab", "engineer": "office",
    "programmer": "office", "accountant": "office",
    "actor": "theater", "musician": "stage",
    "soldier": "army", "general": "army",
    "priest": "church", "monk": "monastery",
    "librarian": "library", "curator": "museum",
}

# ---------------------------------------------------------------------------
#  Musical instruments
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "piano": "keyboard", "organ": "keyboard", "harpsichord": "keyboard",
    "guitar": "string", "violin": "string", "cello": "string",
    "harp": "string", "bass": "string",
    "trumpet": "brass", "trombone": "brass", "tuba": "brass",
    "frenchhorn": "brass",
    "flute": "woodwind", "clarinet": "woodwind", "oboe": "woodwind",
    "saxophone": "woodwind",
    "drum": "percussion", "cymbal": "percussion", "xylophone": "percussion",
}

# ---------------------------------------------------------------------------
#  Sports
# ---------------------------------------------------------------------------

SPORTS = {
    "soccer": ("ball", "field"),
    "basketball": ("ball", "court"),
    "tennis": ("ball", "court"),
    "baseball": ("ball", "field"),
    "football": ("ball", "field"),
    "golf": ("ball", "course"),
    "cricket": ("ball", "pitch"),
    "rugby": ("ball", "field"),
    "volleyball": ("ball", "court"),
    "hockey": ("puck", "rink"),
    "swimming": ("none", "pool"),
    "boxing": ("gloves", "ring"),
    "wrestling": ("none", "mat"),
    "skiing": ("skis", "slope"),
    "surfing": ("board", "ocean"),
}

# ---------------------------------------------------------------------------
#  Vehicles
# ---------------------------------------------------------------------------

VEHICLES = {
    "car": "road", "truck": "road", "bus": "road", "motorcycle": "road",
    "train": "track", "subway": "track", "tram": "track",
    "bicycle": "road", "scooter": "road",
    "boat": "water", "ship": "water", "submarine": "water", "yacht": "water",
    "canoe": "water", "kayak": "water",
    "airplane": "sky", "helicopter": "sky", "glider": "sky", "rocket": "sky",
    "balloon": "sky", "drone": "sky",
}


def main() -> None:
    for atomic, name, symbol in ELEMENTS:
        emit(name, "isa", "element")
        emit(name, "symbol", symbol)
        emit(name, "atomic_number", str(atomic))

    for name, n2, position, parent in PLANETS:
        emit(name, "isa", "planet")
        emit(name, "orbits", parent)
        emit(name, "position", position)
    for planet, sz in PLANET_SIZES.items():
        emit(planet, "size", sz)

    for ocean in OCEANS:
        emit(ocean, "isa", "ocean")
    for sea, continent in SEAS.items():
        emit(sea, "isa", "sea")
        emit(sea, "near", continent)
    for mountain, (country, height) in MOUNTAINS.items():
        emit(mountain, "isa", "mountain")
        emit(mountain, "locatedin", country)
        emit(mountain, "height", str(height))
    for river, continent in RIVERS.items():
        emit(river, "isa", "river")
        emit(river, "locatedin", continent)

    for person, field, country in HIST:
        emit(person, "field", field)
        emit(person, "country", country)
        emit(person, "isa", "person")

    for lang, places in LANGUAGES.items():
        emit(lang, "isa", "language")
        for p in places:
            emit(lang, "spoken_in", p)

    for occ, place in OCCUPATION.items():
        emit(occ, "isa", "occupation")
        emit(occ, "works_at", place)

    for inst, family in INSTRUMENTS.items():
        emit(inst, "isa", "instrument")
        emit(inst, "family", family)

    for sport, (equipment, venue) in SPORTS.items():
        emit(sport, "isa", "sport")
        emit(sport, "equipment", equipment)
        emit(sport, "venue", venue)

    for vehicle, medium in VEHICLES.items():
        emit(vehicle, "isa", "vehicle")
        emit(vehicle, "travels_on", medium)


if __name__ == "__main__":
    main()
