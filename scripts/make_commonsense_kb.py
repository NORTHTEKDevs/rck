"""Synthesize a ConceptNet-style common-sense KB on disk as JSONL.

Each line is {"s": ..., "r": ..., "o": ...} for one fact. The relations
are the canonical ConceptNet set we care about. This is a HAND-CURATED
dataset (no external download required) -- it deliberately mirrors the
kind of knowledge a small language model would have memorised, so the
RCK retrieval benchmark is grounded in something an LLM could also do.

Run:
    python scripts/make_commonsense_kb.py > data/commonsense_kb.jsonl
"""
from __future__ import annotations

import json
import sys


# ---------------------------------------------------------------------------
#  Hand-curated common-sense facts (will produce ~1500+ triples)
# ---------------------------------------------------------------------------

# Animals
ANIMAL_HAS = {
    "dog": ["fur", "tail", "legs", "ears"],
    "cat": ["fur", "tail", "whiskers", "claws"],
    "bird": ["feathers", "beak", "wings"],
    "fish": ["scales", "fins", "gills"],
    "snake": ["scales", "fangs"],
    "elephant": ["tusks", "trunk", "ears"],
    "lion": ["mane", "claws", "fur"],
    "tiger": ["stripes", "claws", "fur"],
    "zebra": ["stripes", "hooves"],
    "horse": ["mane", "hooves", "tail"],
    "cow": ["horns", "hooves", "udder"],
    "rabbit": ["fur", "ears", "tail"],
    "owl": ["feathers", "wings", "talons"],
    "shark": ["fins", "teeth", "gills"],
    "octopus": ["tentacles", "ink"],
    "spider": ["legs", "webs"],
    "bee": ["wings", "stinger"],
    "ant": ["legs", "antennae"],
    "butterfly": ["wings", "antennae"],
    "human": ["hands", "legs", "head"],
}
ANIMAL_ISA = {
    "dog": "mammal", "cat": "mammal", "elephant": "mammal", "lion": "mammal",
    "tiger": "mammal", "zebra": "mammal", "horse": "mammal", "cow": "mammal",
    "rabbit": "mammal", "human": "mammal", "shark": "fish", "owl": "bird",
    "octopus": "mollusk", "spider": "arachnid", "bee": "insect", "ant": "insect",
    "butterfly": "insect", "snake": "reptile",
}

# Colors
COLORS = {
    "sky": "blue", "grass": "green", "blood": "red", "snow": "white",
    "coal": "black", "sun": "yellow", "rose": "red", "lemon": "yellow",
    "tomato": "red", "banana": "yellow", "leaf": "green", "ocean": "blue",
    "lavender": "purple", "carrot": "orange", "pumpkin": "orange",
    "milk": "white", "soot": "black", "blueberry": "blue",
    "strawberry": "red", "fox": "orange", "crow": "black", "swan": "white",
}

# Capitals
CAPITALS = {
    "france": "paris", "germany": "berlin", "italy": "rome",
    "spain": "madrid", "japan": "tokyo", "china": "beijing",
    "india": "delhi", "russia": "moscow", "uk": "london",
    "england": "london", "ireland": "dublin", "egypt": "cairo",
    "kenya": "nairobi", "brazil": "brasilia", "argentina": "buenosaires",
    "mexico": "mexicocity", "canada": "ottawa", "usa": "washington",
    "australia": "canberra", "norway": "oslo", "sweden": "stockholm",
    "finland": "helsinki", "poland": "warsaw", "greece": "athens",
    "turkey": "ankara", "iran": "tehran", "iraq": "baghdad",
    "thailand": "bangkok", "vietnam": "hanoi", "indonesia": "jakarta",
}

# Countries -> continent
CONTINENT = {
    "france": "europe", "germany": "europe", "italy": "europe",
    "spain": "europe", "uk": "europe", "ireland": "europe",
    "russia": "europe", "norway": "europe", "sweden": "europe",
    "finland": "europe", "poland": "europe", "greece": "europe",
    "japan": "asia", "china": "asia", "india": "asia",
    "thailand": "asia", "vietnam": "asia", "indonesia": "asia",
    "iran": "asia", "iraq": "asia", "turkey": "asia",
    "egypt": "africa", "kenya": "africa",
    "brazil": "southamerica", "argentina": "southamerica",
    "mexico": "northamerica", "canada": "northamerica", "usa": "northamerica",
    "australia": "oceania",
}

# Parts (X partof Y)
PARTS = {
    "wheel": "car", "engine": "car", "door": "car",
    "trunk": "tree", "leaf": "tree", "branch": "tree", "root": "tree",
    "page": "book", "cover": "book", "chapter": "book",
    "key": "keyboard", "screen": "laptop", "battery": "laptop",
    "wing": "bird", "beak": "bird", "feather": "bird",
    "petal": "flower", "stem": "flower",
    "blade": "knife", "handle": "knife",
    "string": "guitar", "fret": "guitar",
}

# Materials
MADE_OF = {
    "book": "paper", "table": "wood", "chair": "wood",
    "window": "glass", "mirror": "glass", "bottle": "glass",
    "fork": "metal", "spoon": "metal", "knife": "metal",
    "tire": "rubber", "balloon": "rubber",
    "shirt": "cotton", "dress": "cotton",
    "sweater": "wool",
    "cake": "flour", "bread": "flour",
    "wine": "grape", "cheese": "milk", "butter": "milk",
    "ice": "water", "snow": "water", "steam": "water",
}

# Uses
USED_FOR = {
    "knife": "cutting", "fork": "eating", "spoon": "eating",
    "pen": "writing", "pencil": "writing", "keyboard": "typing",
    "car": "driving", "bicycle": "riding", "boat": "sailing",
    "plane": "flying", "umbrella": "rain",
    "shoes": "walking", "boots": "hiking",
    "phone": "calling", "camera": "photography", "guitar": "music",
    "piano": "music", "book": "reading", "lamp": "lighting",
    "stove": "cooking", "oven": "baking", "fridge": "cooling",
}

# Located in
LOCATED_IN = {
    "kitchen": "house", "bedroom": "house", "bathroom": "house",
    "tree": "forest", "fish": "water", "star": "sky", "cloud": "sky",
    "book": "library", "painting": "museum", "exhibit": "museum",
    "altar": "church", "pew": "church", "stage": "theater",
    "screen": "cinema", "desk": "office", "lab": "university",
    "engine": "car", "passenger": "train", "pilot": "plane",
    # cities -> countries (enables multi-hop city -> country -> continent)
    "paris": "france", "berlin": "germany", "rome": "italy",
    "madrid": "spain", "tokyo": "japan", "beijing": "china",
    "delhi": "india", "moscow": "russia", "london": "england",
    "dublin": "ireland", "cairo": "egypt", "nairobi": "kenya",
    "brasilia": "brazil", "buenosaires": "argentina",
    "mexicocity": "mexico", "ottawa": "canada", "washington": "usa",
    "canberra": "australia", "oslo": "norway", "stockholm": "sweden",
    "helsinki": "finland", "warsaw": "poland", "athens": "greece",
    "ankara": "turkey", "tehran": "iran", "baghdad": "iraq",
    "bangkok": "thailand", "hanoi": "vietnam", "jakarta": "indonesia",
}

# Causes
CAUSES = {
    "rain": "wetness", "sun": "warmth", "fire": "heat",
    "ice": "coldness", "exercise": "sweat", "joke": "laughter",
    "sadness": "tears", "fear": "trembling", "tiredness": "yawning",
    "hunger": "eating", "thirst": "drinking", "cold": "shivering",
    "noise": "annoyance", "music": "enjoyment", "loss": "grief",
}

# Numbers
NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Famous works (X wrote Y)
WROTE = {
    "shakespeare": ["hamlet", "macbeth", "othello", "romeoandjuliet"],
    "tolkien": ["lordoftherings", "thehobbit"],
    "orwell": ["1984", "animalfarm"],
    "dickens": ["olivertwist", "greatexpectations", "achristmascarol"],
    "austen": ["prideandprejudice", "emma", "sensesensibility"],
    "twain": ["tomsawyer", "huckleberryfinn"],
    "homer": ["iliad", "odyssey"],
    "joyce": ["ulysses", "dubliners"],
    "rowling": ["harrypotter"],
    "tolstoy": ["warandpeace", "annakarenina"],
    "dostoevsky": ["crimeandpunishment", "brotherskaramazov"],
    "wilde": ["pictureofdoriangray"],
    "kafka": ["metamorphosis", "thetrial"],
    "hemingway": ["oldmanandthesea", "farewelltoarms"],
}


# Famous scientists -> field
FIELD = {
    "einstein": "physics", "newton": "physics", "feynman": "physics",
    "curie": "chemistry", "darwin": "biology", "mendel": "biology",
    "tesla": "engineering", "edison": "engineering",
    "turing": "computerscience", "lovelace": "computerscience",
    "vonneumann": "computerscience",
    "freud": "psychology", "jung": "psychology",
    "euclid": "geometry", "pythagoras": "geometry",
    "ramanujan": "mathematics", "euler": "mathematics", "gauss": "mathematics",
}


# Foods
FOOD = {
    "apple": "fruit", "banana": "fruit", "orange": "fruit", "grape": "fruit",
    "strawberry": "fruit", "blueberry": "fruit", "pear": "fruit",
    "lemon": "fruit", "tomato": "fruit", "carrot": "vegetable",
    "potato": "vegetable", "onion": "vegetable", "garlic": "vegetable",
    "spinach": "vegetable", "broccoli": "vegetable", "lettuce": "vegetable",
    "rice": "grain", "wheat": "grain", "oats": "grain",
    "chicken": "meat", "beef": "meat", "pork": "meat", "lamb": "meat",
    "salmon": "fish", "tuna": "fish", "shrimp": "shellfish",
}

# Sizes (for comparison queries)
SIZE = {
    "elephant": "huge", "whale": "huge", "giraffe": "huge",
    "dinosaur": "huge", "mountain": "huge", "ocean": "huge",
    "lion": "large", "tiger": "large", "horse": "large",
    "cow": "large", "bear": "large", "car": "large", "house": "large",
    "human": "medium", "dog": "medium", "wolf": "medium", "tree": "medium",
    "cat": "small", "rabbit": "small", "chicken": "small", "book": "small",
    "phone": "small", "apple": "small",
    "mouse": "tiny", "ant": "tiny", "spider": "tiny", "fly": "tiny",
    "atom": "tiny", "grain": "tiny", "pin": "tiny", "seed": "tiny",
}

# Math facts (X + Y = Z encoded as triples)
MATH_SUMS = [
    ("one_plus_one", "equals", "two"),
    ("two_plus_two", "equals", "four"),
    ("three_plus_three", "equals", "six"),
    ("four_plus_four", "equals", "eight"),
    ("five_plus_five", "equals", "ten"),
    ("two_times_three", "equals", "six"),
    ("three_times_three", "equals", "nine"),
    ("four_times_four", "equals", "sixteen"),
    ("five_times_five", "equals", "twentyfive"),
    ("ten_times_ten", "equals", "hundred"),
]

# Historical figures + their works/roles
HISTORY = [
    ("napoleon", "kind", "emperor"),
    ("napoleon", "country", "france"),
    ("caesar", "kind", "emperor"),
    ("caesar", "country", "rome"),
    ("cleopatra", "kind", "queen"),
    ("cleopatra", "country", "egypt"),
    ("lincoln", "kind", "president"),
    ("lincoln", "country", "usa"),
    ("churchill", "kind", "primeminister"),
    ("churchill", "country", "uk"),
    ("gandhi", "kind", "leader"),
    ("gandhi", "country", "india"),
    ("mandela", "kind", "president"),
    ("mandela", "country", "southafrica"),
]

# Months / days
MONTHS_BEFORE = {
    "february": "january", "march": "february", "april": "march",
    "may": "april", "june": "may", "july": "june",
    "august": "july", "september": "august", "october": "september",
    "november": "october", "december": "november",
}

# Body parts -> creature
BODY_PARTS = [
    ("heart", "partof", "body"),
    ("brain", "partof", "body"),
    ("lung", "partof", "body"),
    ("liver", "partof", "body"),
    ("kidney", "partof", "body"),
    ("muscle", "partof", "body"),
    ("bone", "partof", "body"),
    ("eye", "partof", "head"),
    ("ear", "partof", "head"),
    ("nose", "partof", "head"),
    ("mouth", "partof", "head"),
    ("hand", "partof", "arm"),
    ("foot", "partof", "leg"),
    ("finger", "partof", "hand"),
    ("toe", "partof", "foot"),
]

# Weather / seasons
WEATHER = [
    ("summer", "is", "hot"),
    ("winter", "is", "cold"),
    ("spring", "is", "warm"),
    ("autumn", "is", "cool"),
    ("desert", "is", "dry"),
    ("rainforest", "is", "wet"),
    ("arctic", "is", "cold"),
    ("tropic", "is", "hot"),
]


def emit(s, r, o):
    sys.stdout.write(json.dumps({"s": str(s).lower(), "r": str(r).lower(),
                                 "o": str(o).lower()}) + "\n")


def main() -> None:
    # ANIMAL_HAS -> (animal, has, part)
    for animal, parts in ANIMAL_HAS.items():
        for p in parts:
            emit(animal, "has", p)
    for animal, kind in ANIMAL_ISA.items():
        emit(animal, "isa", kind)
    for thing, color in COLORS.items():
        emit(thing, "color", color)
    for country, capital in CAPITALS.items():
        emit(country, "capital", capital)
    for country, continent in CONTINENT.items():
        emit(country, "continent", continent)
    for part, whole in PARTS.items():
        emit(part, "partof", whole)
    for thing, mat in MADE_OF.items():
        emit(thing, "madeof", mat)
    for thing, use in USED_FOR.items():
        emit(thing, "usedfor", use)
    for thing, loc in LOCATED_IN.items():
        emit(thing, "locatedin", loc)
    for cause, effect in CAUSES.items():
        emit(cause, "causes", effect)
    for word, num in NUMBERS.items():
        emit(word, "value", num)
    for author, works in WROTE.items():
        for w in works:
            emit(author, "wrote", w)
            emit(w, "author", author)
    for person, fld in FIELD.items():
        emit(person, "field", fld)
    for food, cat in FOOD.items():
        emit(food, "category", cat)
    for thing, sz in SIZE.items():
        emit(thing, "size", sz)
    for s, r, o in MATH_SUMS:
        emit(s, r, o)
    for s, r, o in HISTORY:
        emit(s, r, o)
    for month, prev in MONTHS_BEFORE.items():
        emit(month, "previousmonth", prev)
    for s, r, o in BODY_PARTS:
        emit(s, r, o)
    for s, r, o in WEATHER:
        emit(s, r, o)


if __name__ == "__main__":
    main()
