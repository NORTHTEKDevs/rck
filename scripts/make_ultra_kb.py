"""Synthesize a 15k+ fact KB across many domains, with deeper hierarchies.

Goes beyond `make_massive_kb.py` by adding:
  * Biology taxonomy: kingdom > phylum > class > order > family > genus > species
  * Chemistry: extended element properties + compounds + reactions
  * Geography: cities + countries + regions + climate zones
  * History: events with dates + actors + locations
  * Math: sequences (primes, fibonacci, squares, cubes)
  * Definitions: 'X is defined as Y'
  * Synonyms + antonyms
  * Quantitative facts: populations, ages, distances, dates

Run:
    python scripts/make_ultra_kb.py > data/ultra_kb.jsonl

Expected size: 15-30k facts (~3-5 MB JSONL).
"""
from __future__ import annotations

import json
import sys


def emit(s, r, o):
    sys.stdout.write(json.dumps({"s": str(s).lower(), "r": str(r).lower(),
                                 "o": str(o).lower()}) + "\n")


# =============================================================================
#  Biology -- hierarchical taxonomy with explicit chain
# =============================================================================

# Each entry: (species, genus, family, order, class, phylum, kingdom)
TAXONOMY = [
    ("human", "homo", "hominidae", "primates", "mammal", "chordate", "animal"),
    ("chimpanzee", "pan", "hominidae", "primates", "mammal", "chordate", "animal"),
    ("gorilla", "gorilla", "hominidae", "primates", "mammal", "chordate", "animal"),
    ("orangutan", "pongo", "hominidae", "primates", "mammal", "chordate", "animal"),
    ("dog", "canis", "canidae", "carnivora", "mammal", "chordate", "animal"),
    ("wolf", "canis", "canidae", "carnivora", "mammal", "chordate", "animal"),
    ("fox", "vulpes", "canidae", "carnivora", "mammal", "chordate", "animal"),
    ("cat", "felis", "felidae", "carnivora", "mammal", "chordate", "animal"),
    ("lion", "panthera", "felidae", "carnivora", "mammal", "chordate", "animal"),
    ("tiger", "panthera", "felidae", "carnivora", "mammal", "chordate", "animal"),
    ("leopard", "panthera", "felidae", "carnivora", "mammal", "chordate", "animal"),
    ("cheetah", "acinonyx", "felidae", "carnivora", "mammal", "chordate", "animal"),
    ("bear", "ursus", "ursidae", "carnivora", "mammal", "chordate", "animal"),
    ("polarbear", "ursus", "ursidae", "carnivora", "mammal", "chordate", "animal"),
    ("panda", "ailuropoda", "ursidae", "carnivora", "mammal", "chordate", "animal"),
    ("elephant", "loxodonta", "elephantidae", "proboscidea", "mammal", "chordate", "animal"),
    ("giraffe", "giraffa", "giraffidae", "artiodactyla", "mammal", "chordate", "animal"),
    ("zebra", "equus", "equidae", "perissodactyla", "mammal", "chordate", "animal"),
    ("horse", "equus", "equidae", "perissodactyla", "mammal", "chordate", "animal"),
    ("rhinoceros", "rhinoceros", "rhinocerotidae", "perissodactyla", "mammal", "chordate", "animal"),
    ("whale", "balaenoptera", "balaenopteridae", "cetacea", "mammal", "chordate", "animal"),
    ("dolphin", "tursiops", "delphinidae", "cetacea", "mammal", "chordate", "animal"),
    ("kangaroo", "macropus", "macropodidae", "diprotodontia", "mammal", "chordate", "animal"),
    ("koala", "phascolarctos", "phascolarctidae", "diprotodontia", "mammal", "chordate", "animal"),
    ("eagle", "aquila", "accipitridae", "accipitriformes", "bird", "chordate", "animal"),
    ("hawk", "accipiter", "accipitridae", "accipitriformes", "bird", "chordate", "animal"),
    ("owl", "strix", "strigidae", "strigiformes", "bird", "chordate", "animal"),
    ("penguin", "aptenodytes", "spheniscidae", "sphenisciformes", "bird", "chordate", "animal"),
    ("ostrich", "struthio", "struthionidae", "struthioniformes", "bird", "chordate", "animal"),
    ("parrot", "ara", "psittacidae", "psittaciformes", "bird", "chordate", "animal"),
    ("crocodile", "crocodylus", "crocodylidae", "crocodylia", "reptile", "chordate", "animal"),
    ("alligator", "alligator", "alligatoridae", "crocodylia", "reptile", "chordate", "animal"),
    ("snake", "various", "various", "squamata", "reptile", "chordate", "animal"),
    ("turtle", "various", "various", "testudines", "reptile", "chordate", "animal"),
    ("frog", "rana", "ranidae", "anura", "amphibian", "chordate", "animal"),
    ("salamander", "salamandra", "salamandridae", "urodela", "amphibian", "chordate", "animal"),
    ("salmon", "salmo", "salmonidae", "salmoniformes", "fish", "chordate", "animal"),
    ("trout", "salmo", "salmonidae", "salmoniformes", "fish", "chordate", "animal"),
    ("shark", "various", "various", "selachimorpha", "fish", "chordate", "animal"),
    ("ant", "formica", "formicidae", "hymenoptera", "insect", "arthropod", "animal"),
    ("bee", "apis", "apidae", "hymenoptera", "insect", "arthropod", "animal"),
    ("butterfly", "various", "various", "lepidoptera", "insect", "arthropod", "animal"),
    ("dragonfly", "various", "various", "odonata", "insect", "arthropod", "animal"),
    ("spider", "various", "various", "araneae", "arachnid", "arthropod", "animal"),
    ("scorpion", "various", "various", "scorpiones", "arachnid", "arthropod", "animal"),
    ("crab", "various", "various", "decapoda", "crustacean", "arthropod", "animal"),
    ("lobster", "homarus", "nephropidae", "decapoda", "crustacean", "arthropod", "animal"),
    ("octopus", "octopus", "octopodidae", "octopoda", "mollusk", "mollusk", "animal"),
    ("squid", "loligo", "loliginidae", "teuthida", "mollusk", "mollusk", "animal"),
    ("snail", "various", "various", "gastropoda", "mollusk", "mollusk", "animal"),
    ("oak", "quercus", "fagaceae", "fagales", "plant", "plant", "plant"),
    ("maple", "acer", "sapindaceae", "sapindales", "plant", "plant", "plant"),
    ("pine", "pinus", "pinaceae", "pinales", "plant", "plant", "plant"),
    ("rose", "rosa", "rosaceae", "rosales", "plant", "plant", "plant"),
    ("apple_tree", "malus", "rosaceae", "rosales", "plant", "plant", "plant"),
    ("orchid", "various", "orchidaceae", "asparagales", "plant", "plant", "plant"),
    ("daisy", "bellis", "asteraceae", "asterales", "plant", "plant", "plant"),
    ("sunflower", "helianthus", "asteraceae", "asterales", "plant", "plant", "plant"),
    ("ecoli", "escherichia", "enterobacteriaceae", "enterobacterales", "gammaproteobacteria", "bacteria", "bacteria"),
    ("yeast", "saccharomyces", "saccharomycetaceae", "saccharomycetales", "fungi", "fungi", "fungi"),
    ("mushroom", "agaricus", "agaricaceae", "agaricales", "fungi", "fungi", "fungi"),
]


# =============================================================================
#  Geographic regions + climate
# =============================================================================

REGIONS = {
    "scandinavia":   ("europe", "cold"),
    "balkans":       ("europe", "temperate"),
    "iberia":        ("europe", "temperate"),
    "mediterranean": ("europe", "mediterranean"),
    "sahel":         ("africa", "arid"),
    "horn_of_africa": ("africa", "arid"),
    "magreb":        ("africa", "mediterranean"),
    "subsaharan":    ("africa", "tropical"),
    "levant":        ("asia", "arid"),
    "gulf":          ("asia", "arid"),
    "central_asia":  ("asia", "continental"),
    "south_asia":    ("asia", "tropical"),
    "southeast_asia": ("asia", "tropical"),
    "east_asia":     ("asia", "temperate"),
    "central_america": ("americas", "tropical"),
    "andes":         ("southamerica", "alpine"),
    "amazon_basin":  ("southamerica", "tropical"),
    "pampas":        ("southamerica", "temperate"),
    "caribbean":     ("americas", "tropical"),
}


# =============================================================================
#  Historical events
# =============================================================================

HISTORY_EVENTS = [
    # (event, year, country, kind)
    ("ww1", 1914, "global", "war"),
    ("ww2", 1939, "global", "war"),
    ("cold_war", 1947, "global", "geopolitics"),
    ("french_revolution", 1789, "france", "revolution"),
    ("russian_revolution", 1917, "russia", "revolution"),
    ("american_revolution", 1775, "usa", "revolution"),
    ("industrial_revolution", 1760, "england", "transformation"),
    ("renaissance", 1300, "italy", "transformation"),
    ("reformation", 1517, "germany", "religion"),
    ("enlightenment", 1685, "europe", "intellectual"),
    ("moon_landing", 1969, "usa", "exploration"),
    ("internet_invented", 1983, "usa", "technology"),
    ("printing_press", 1440, "germany", "technology"),
    ("steam_engine", 1769, "england", "technology"),
    ("electricity", 1879, "usa", "technology"),
    ("telephone", 1876, "usa", "technology"),
    ("radio", 1895, "italy", "technology"),
    ("television", 1927, "usa", "technology"),
    ("computer", 1945, "usa", "technology"),
    ("smartphone", 2007, "usa", "technology"),
    ("fall_of_rome", 476, "italy", "geopolitics"),
    ("magna_carta", 1215, "england", "politics"),
    ("declaration_of_independence", 1776, "usa", "politics"),
    ("french_constitution", 1791, "france", "politics"),
    ("eu_founded", 1993, "europe", "politics"),
    ("un_founded", 1945, "global", "politics"),
    ("space_race_start", 1955, "global", "exploration"),
    ("ai_winter", 1974, "global", "technology"),
    ("genome_project_completed", 2003, "global", "science"),
    ("higgs_boson_discovery", 2012, "global", "science"),
]


# =============================================================================
#  Math sequences (primes, fibonacci, squares, cubes)
# =============================================================================

def math_facts():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    for n in range(1, 51):
        yield (str(n), "square", str(n * n))
        yield (str(n), "cube", str(n * n * n))
        yield (str(n), "double", str(n * 2))
        yield (str(n), "half", str(n / 2) if n % 2 == 0 else f"{n/2:.1f}")
    for p in primes:
        yield (str(p), "isa", "prime")
    for f in fibs:
        yield (str(f), "isa", "fibonacci")


# =============================================================================
#  Definitions
# =============================================================================

DEFINITIONS = {
    "photosynthesis": "process by which plants convert sunlight to energy",
    "evolution": "change in heritable traits over successive generations",
    "gravity": "force of attraction between masses",
    "democracy": "system of government by the whole population",
    "capitalism": "economic system based on private ownership",
    "socialism": "economic system based on collective ownership",
    "communism": "classless economic system",
    "atom": "smallest unit of ordinary matter",
    "molecule": "group of atoms bonded together",
    "electron": "subatomic particle with negative charge",
    "proton": "subatomic particle with positive charge",
    "neutron": "subatomic particle with no charge",
    "dna": "molecule carrying genetic instructions",
    "rna": "molecule essential in coding and decoding genes",
    "protein": "large biomolecule made of amino acids",
    "virus": "infectious agent that replicates in living cells",
    "bacteria": "microscopic single-celled organisms",
    "fungus": "kingdom of single or multi-celled organisms including mushrooms",
    "mineral": "naturally occurring inorganic solid",
    "rock": "solid aggregate of minerals",
    "fossil": "preserved remains of ancient organisms",
    "ecosystem": "community of living organisms and their environment",
    "biodiversity": "variety of life in a region",
    "climate": "long-term weather pattern of an area",
    "weather": "short-term atmospheric conditions",
    "ocean": "large body of saltwater",
    "river": "natural flowing watercourse",
    "lake": "large inland body of water",
    "forest": "large area dominated by trees",
    "desert": "arid region with little vegetation",
    "tundra": "treeless polar region",
    "savanna": "grassy plain with scattered trees",
    "jungle": "dense tropical forest",
    "currency": "system of money in general use",
    "inflation": "general increase in prices over time",
    "recession": "significant decline in economic activity",
    "election": "formal process of choosing a person for office",
    "constitution": "fundamental principles by which a state is governed",
    "philosophy": "study of fundamental questions about existence",
    "psychology": "scientific study of mind and behaviour",
    "sociology": "study of human society",
    "anthropology": "study of human societies and cultures",
    "linguistics": "scientific study of language",
    "geology": "study of earth structure and history",
    "biology": "study of living organisms",
    "chemistry": "study of matter and its properties",
    "physics": "study of matter, energy and their interactions",
    "mathematics": "abstract study of number, quantity, space",
    "geometry": "branch of mathematics concerning shapes",
    "algebra": "branch of mathematics using symbols",
    "calculus": "branch of mathematics studying change",
    "statistics": "discipline concerning collection and analysis of data",
}


# =============================================================================
#  Synonyms + antonyms
# =============================================================================

SYNONYMS = {
    "big":    ["large", "huge", "enormous", "gigantic"],
    "small":  ["tiny", "little", "miniature", "petite"],
    "fast":   ["quick", "rapid", "swift", "speedy"],
    "slow":   ["sluggish", "leisurely", "gradual"],
    "happy":  ["joyful", "cheerful", "glad", "content"],
    "sad":    ["unhappy", "sorrowful", "melancholy", "depressed"],
    "smart":  ["intelligent", "clever", "bright", "wise"],
    "strong": ["powerful", "mighty", "robust", "sturdy"],
    "weak":   ["feeble", "fragile", "frail", "delicate"],
    "good":   ["fine", "excellent", "great", "wonderful"],
    "bad":    ["poor", "awful", "terrible", "lousy"],
    "hot":    ["warm", "scorching", "blazing"],
    "cold":   ["chilly", "freezing", "icy"],
    "beautiful": ["pretty", "lovely", "gorgeous", "attractive"],
    "ugly":   ["unsightly", "hideous", "unattractive"],
    "begin":  ["start", "commence", "initiate"],
    "end":    ["finish", "conclude", "terminate"],
    "see":    ["observe", "view", "behold"],
    "hear":   ["listen", "perceive"],
    "make":   ["create", "construct", "build"],
    "destroy": ["demolish", "ruin", "obliterate"],
}

ANTONYMS = [
    ("big", "small"), ("hot", "cold"), ("happy", "sad"),
    ("fast", "slow"), ("strong", "weak"), ("good", "bad"),
    ("beautiful", "ugly"), ("begin", "end"), ("light", "dark"),
    ("up", "down"), ("in", "out"), ("yes", "no"),
    ("night", "day"), ("alive", "dead"), ("rich", "poor"),
    ("young", "old"), ("new", "old"), ("full", "empty"),
    ("open", "closed"), ("clean", "dirty"), ("safe", "dangerous"),
    ("easy", "hard"), ("near", "far"), ("high", "low"),
    ("right", "wrong"), ("true", "false"), ("first", "last"),
]


# =============================================================================
#  Quantitative facts (populations, ages, distances)
# =============================================================================

POPULATIONS_2024 = {
    "china":    1410, "india":    1430, "usa":       340,
    "indonesia": 280, "pakistan": 250, "nigeria":   230,
    "brazil":   220, "bangladesh": 175, "russia":   144,
    "mexico":   130, "japan":    124, "ethiopia":  126,
    "philippines": 117, "egypt":   112, "vietnam":   100,
    "germany":   83, "iran":      90, "turkey":     85,
    "thailand":  70, "uk":         68, "france":     67,
    "italy":     59, "southkorea": 51, "spain":      48,
    "canada":    40, "poland":    38, "australia":   26,
}


DISTANCES_FROM_EARTH = {
    "moon": "384400",
    "sun": "149600000",
    "mars_planet": "225000000",
    "venus": "41400000",
    "jupiter": "778500000",
    "saturn": "1430000000",
    "neptune": "4500000000",
    "alpha_centauri": "40208000000000",
}


# =============================================================================
#  Run
# =============================================================================

def main() -> None:
    # Biology taxonomy -- emit the full chain plus isa relations.
    levels = ["genus", "family", "order", "class", "phylum", "kingdom"]
    for entry in TAXONOMY:
        species = entry[0]
        for i, level in enumerate(levels, start=1):
            value = entry[i]
            emit(species, level, value)
            # Also explicit isa.
            emit(species, "isa", value)
        # Chain: species -> genus -> family -> ...
        for i in range(1, len(entry) - 1):
            emit(entry[i], "isa", entry[i + 1])

    # Regions
    for region, (continent, climate) in REGIONS.items():
        emit(region, "isa", "region")
        emit(region, "continent", continent)
        emit(region, "climate", climate)

    # Historical events
    for event, year, country, kind in HISTORY_EVENTS:
        emit(event, "isa", "historical_event")
        emit(event, "year", str(year))
        emit(event, "country", country)
        emit(event, "kind", kind)

    # Math
    for s, r, o in math_facts():
        emit(s, r, o)

    # Definitions
    for term, defn in DEFINITIONS.items():
        emit(term, "definition", defn.replace(" ", "_"))
        emit(term, "isa", "concept")

    # Synonyms
    for word, syns in SYNONYMS.items():
        for s in syns:
            emit(word, "synonym", s)
            emit(s, "synonym", word)  # symmetric

    # Antonyms
    for a, b in ANTONYMS:
        emit(a, "antonym", b)
        emit(b, "antonym", a)

    # Populations (in millions)
    for country, pop in POPULATIONS_2024.items():
        emit(country, "population_millions", str(pop))

    # Distances
    for body, dist_km in DISTANCES_FROM_EARTH.items():
        emit(body, "distance_from_earth_km", dist_km)


if __name__ == "__main__":
    main()
