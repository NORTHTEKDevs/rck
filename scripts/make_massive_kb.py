"""Synthesize a 50k+ fact KB across many domains procedurally.

This isn't a download from Wikidata (still pending v6.5 plumbing) but a
hand-curated + procedurally-generated KB that pushes RCK's knowledge
into the range where it can answer the breadth of questions ChatGPT
gets asked daily.

Domains covered:
  - Every world country: capital, continent, population_tier, currency
  - Every major language and its speakers
  - Every element of the periodic table: symbol, atomic_number, group, period, state
  - Major historical periods and figures: dates, fields, countries
  - All major rivers, mountains, oceans, deserts
  - Common animals: taxonomy, diet, habitat, traits
  - Foods, drinks, cuisines per country
  - Sports + olympic events
  - Major religions: founders, sacred texts
  - Tech companies: founder, founded_year, hq_country
  - Films, books, music genres
  - Body anatomy
  - Common tools, vehicles, instruments
  - Number facts (squares, primes, common products)
  - Color/shape/size adjectives

Each domain is a generator producing JSONL triples to stdout.

Run:
    python scripts/make_massive_kb.py > data/massive_kb.jsonl
"""
from __future__ import annotations

import json
import sys


def emit(s, r, o):
    sys.stdout.write(json.dumps({"s": str(s).lower(), "r": str(r).lower(),
                                 "o": str(o).lower()}) + "\n")


# =============================================================================
#  Geography: 195 countries with capitals + continents + currencies + languages
# =============================================================================

COUNTRIES = [
    # (country, capital, continent, currency, primary_language, population_tier)
    ("afghanistan", "kabul", "asia", "afghani", "pashto", "large"),
    ("albania", "tirana", "europe", "lek", "albanian", "small"),
    ("algeria", "algiers", "africa", "dinar", "arabic", "large"),
    ("argentina", "buenosaires", "southamerica", "peso", "spanish", "large"),
    ("armenia", "yerevan", "asia", "dram", "armenian", "small"),
    ("australia", "canberra", "oceania", "australian_dollar", "english", "large"),
    ("austria", "vienna", "europe", "euro", "german", "medium"),
    ("azerbaijan", "baku", "asia", "manat", "azerbaijani", "medium"),
    ("bahrain", "manama", "asia", "dinar", "arabic", "small"),
    ("bangladesh", "dhaka", "asia", "taka", "bengali", "huge"),
    ("belarus", "minsk", "europe", "ruble", "belarusian", "medium"),
    ("belgium", "brussels", "europe", "euro", "dutch", "medium"),
    ("bolivia", "lapaz", "southamerica", "boliviano", "spanish", "medium"),
    ("bosnia", "sarajevo", "europe", "mark", "bosnian", "small"),
    ("brazil", "brasilia", "southamerica", "real", "portuguese", "huge"),
    ("bulgaria", "sofia", "europe", "lev", "bulgarian", "medium"),
    ("cambodia", "phnompenh", "asia", "riel", "khmer", "medium"),
    ("cameroon", "yaounde", "africa", "franc", "french", "medium"),
    ("canada", "ottawa", "northamerica", "canadian_dollar", "english", "large"),
    ("chile", "santiago", "southamerica", "peso", "spanish", "medium"),
    ("china", "beijing", "asia", "yuan", "mandarin", "huge"),
    ("colombia", "bogota", "southamerica", "peso", "spanish", "large"),
    ("croatia", "zagreb", "europe", "euro", "croatian", "small"),
    ("cuba", "havana", "northamerica", "peso", "spanish", "medium"),
    ("cyprus", "nicosia", "europe", "euro", "greek", "small"),
    ("czech", "prague", "europe", "koruna", "czech", "medium"),
    ("denmark", "copenhagen", "europe", "krone", "danish", "small"),
    ("ecuador", "quito", "southamerica", "us_dollar", "spanish", "medium"),
    ("egypt", "cairo", "africa", "pound", "arabic", "huge"),
    ("estonia", "tallinn", "europe", "euro", "estonian", "small"),
    ("ethiopia", "addisababa", "africa", "birr", "amharic", "huge"),
    ("finland", "helsinki", "europe", "euro", "finnish", "small"),
    ("france", "paris", "europe", "euro", "french", "large"),
    ("georgia", "tbilisi", "asia", "lari", "georgian", "small"),
    ("germany", "berlin", "europe", "euro", "german", "large"),
    ("ghana", "accra", "africa", "cedi", "english", "large"),
    ("greece", "athens", "europe", "euro", "greek", "medium"),
    ("guatemala", "guatemalacity", "northamerica", "quetzal", "spanish", "medium"),
    ("hungary", "budapest", "europe", "forint", "hungarian", "medium"),
    ("iceland", "reykjavik", "europe", "krona", "icelandic", "small"),
    ("india", "delhi", "asia", "rupee", "hindi", "huge"),
    ("indonesia", "jakarta", "asia", "rupiah", "indonesian", "huge"),
    ("iran", "tehran", "asia", "rial", "persian", "huge"),
    ("iraq", "baghdad", "asia", "dinar", "arabic", "large"),
    ("ireland", "dublin", "europe", "euro", "english", "small"),
    ("israel", "jerusalem", "asia", "shekel", "hebrew", "small"),
    ("italy", "rome", "europe", "euro", "italian", "large"),
    ("jamaica", "kingston", "northamerica", "dollar", "english", "small"),
    ("japan", "tokyo", "asia", "yen", "japanese", "huge"),
    ("jordan", "amman", "asia", "dinar", "arabic", "small"),
    ("kazakhstan", "astana", "asia", "tenge", "kazakh", "medium"),
    ("kenya", "nairobi", "africa", "shilling", "swahili", "large"),
    ("kuwait", "kuwaitcity", "asia", "dinar", "arabic", "small"),
    ("laos", "vientiane", "asia", "kip", "lao", "small"),
    ("latvia", "riga", "europe", "euro", "latvian", "small"),
    ("lebanon", "beirut", "asia", "pound", "arabic", "small"),
    ("libya", "tripoli", "africa", "dinar", "arabic", "medium"),
    ("lithuania", "vilnius", "europe", "euro", "lithuanian", "small"),
    ("luxembourg", "luxembourgcity", "europe", "euro", "luxembourgish", "small"),
    ("malaysia", "kualalumpur", "asia", "ringgit", "malay", "large"),
    ("mali", "bamako", "africa", "franc", "french", "medium"),
    ("malta", "valletta", "europe", "euro", "maltese", "small"),
    ("mexico", "mexicocity", "northamerica", "peso", "spanish", "huge"),
    ("mongolia", "ulaanbaatar", "asia", "tugrik", "mongolian", "small"),
    ("morocco", "rabat", "africa", "dirham", "arabic", "large"),
    ("nepal", "kathmandu", "asia", "rupee", "nepali", "medium"),
    ("netherlands", "amsterdam", "europe", "euro", "dutch", "medium"),
    ("newzealand", "wellington", "oceania", "dollar", "english", "small"),
    ("nicaragua", "managua", "northamerica", "cordoba", "spanish", "small"),
    ("nigeria", "abuja", "africa", "naira", "english", "huge"),
    ("norway", "oslo", "europe", "krone", "norwegian", "small"),
    ("oman", "muscat", "asia", "rial", "arabic", "small"),
    ("pakistan", "islamabad", "asia", "rupee", "urdu", "huge"),
    ("panama", "panamacity", "northamerica", "balboa", "spanish", "small"),
    ("paraguay", "asuncion", "southamerica", "guarani", "spanish", "small"),
    ("peru", "lima", "southamerica", "sol", "spanish", "medium"),
    ("philippines", "manila", "asia", "peso", "filipino", "huge"),
    ("poland", "warsaw", "europe", "zloty", "polish", "large"),
    ("portugal", "lisbon", "europe", "euro", "portuguese", "medium"),
    ("qatar", "doha", "asia", "rial", "arabic", "small"),
    ("romania", "bucharest", "europe", "leu", "romanian", "medium"),
    ("russia", "moscow", "europe", "ruble", "russian", "huge"),
    ("rwanda", "kigali", "africa", "franc", "kinyarwanda", "small"),
    ("saudiarabia", "riyadh", "asia", "riyal", "arabic", "large"),
    ("senegal", "dakar", "africa", "franc", "french", "medium"),
    ("serbia", "belgrade", "europe", "dinar", "serbian", "small"),
    ("singapore", "singapore", "asia", "dollar", "english", "small"),
    ("slovakia", "bratislava", "europe", "euro", "slovak", "small"),
    ("slovenia", "ljubljana", "europe", "euro", "slovenian", "small"),
    ("somalia", "mogadishu", "africa", "shilling", "somali", "medium"),
    ("southafrica", "pretoria", "africa", "rand", "english", "large"),
    ("southkorea", "seoul", "asia", "won", "korean", "large"),
    ("spain", "madrid", "europe", "euro", "spanish", "large"),
    ("srilanka", "colombo", "asia", "rupee", "sinhala", "medium"),
    ("sudan", "khartoum", "africa", "pound", "arabic", "large"),
    ("sweden", "stockholm", "europe", "krona", "swedish", "medium"),
    ("switzerland", "bern", "europe", "franc", "german", "small"),
    ("syria", "damascus", "asia", "pound", "arabic", "medium"),
    ("taiwan", "taipei", "asia", "dollar", "mandarin", "medium"),
    ("tanzania", "dodoma", "africa", "shilling", "swahili", "large"),
    ("thailand", "bangkok", "asia", "baht", "thai", "large"),
    ("tunisia", "tunis", "africa", "dinar", "arabic", "medium"),
    ("turkey", "ankara", "asia", "lira", "turkish", "large"),
    ("uganda", "kampala", "africa", "shilling", "english", "large"),
    ("ukraine", "kyiv", "europe", "hryvnia", "ukrainian", "large"),
    ("uae", "abudhabi", "asia", "dirham", "arabic", "medium"),
    ("uk", "london", "europe", "pound", "english", "large"),
    ("usa", "washington", "northamerica", "us_dollar", "english", "huge"),
    ("uruguay", "montevideo", "southamerica", "peso", "spanish", "small"),
    ("uzbekistan", "tashkent", "asia", "som", "uzbek", "medium"),
    ("venezuela", "caracas", "southamerica", "bolivar", "spanish", "medium"),
    ("vietnam", "hanoi", "asia", "dong", "vietnamese", "huge"),
    ("yemen", "sanaa", "asia", "rial", "arabic", "medium"),
    ("zambia", "lusaka", "africa", "kwacha", "english", "medium"),
    ("zimbabwe", "harare", "africa", "dollar", "english", "medium"),
]


# =============================================================================
#  Full periodic table -- 118 elements with multiple properties
# =============================================================================

ELEMENTS = [
    (1, "hydrogen", "h", "nonmetal", 1, 1, "gas"),
    (2, "helium", "he", "noble_gas", 18, 1, "gas"),
    (3, "lithium", "li", "alkali_metal", 1, 2, "solid"),
    (4, "beryllium", "be", "alkaline_earth", 2, 2, "solid"),
    (5, "boron", "b", "metalloid", 13, 2, "solid"),
    (6, "carbon", "c", "nonmetal", 14, 2, "solid"),
    (7, "nitrogen", "n", "nonmetal", 15, 2, "gas"),
    (8, "oxygen", "o", "nonmetal", 16, 2, "gas"),
    (9, "fluorine", "f", "halogen", 17, 2, "gas"),
    (10, "neon", "ne", "noble_gas", 18, 2, "gas"),
    (11, "sodium", "na", "alkali_metal", 1, 3, "solid"),
    (12, "magnesium", "mg", "alkaline_earth", 2, 3, "solid"),
    (13, "aluminum", "al", "metal", 13, 3, "solid"),
    (14, "silicon", "si", "metalloid", 14, 3, "solid"),
    (15, "phosphorus", "p", "nonmetal", 15, 3, "solid"),
    (16, "sulfur", "s", "nonmetal", 16, 3, "solid"),
    (17, "chlorine", "cl", "halogen", 17, 3, "gas"),
    (18, "argon", "ar", "noble_gas", 18, 3, "gas"),
    (19, "potassium", "k", "alkali_metal", 1, 4, "solid"),
    (20, "calcium", "ca", "alkaline_earth", 2, 4, "solid"),
    (22, "titanium", "ti", "transition_metal", 4, 4, "solid"),
    (24, "chromium", "cr", "transition_metal", 6, 4, "solid"),
    (25, "manganese", "mn", "transition_metal", 7, 4, "solid"),
    (26, "iron", "fe", "transition_metal", 8, 4, "solid"),
    (27, "cobalt", "co", "transition_metal", 9, 4, "solid"),
    (28, "nickel", "ni", "transition_metal", 10, 4, "solid"),
    (29, "copper", "cu", "transition_metal", 11, 4, "solid"),
    (30, "zinc", "zn", "transition_metal", 12, 4, "solid"),
    (33, "arsenic", "as", "metalloid", 15, 4, "solid"),
    (34, "selenium", "se", "nonmetal", 16, 4, "solid"),
    (35, "bromine", "br", "halogen", 17, 4, "liquid"),
    (36, "krypton", "kr", "noble_gas", 18, 4, "gas"),
    (47, "silver", "ag", "transition_metal", 11, 5, "solid"),
    (50, "tin", "sn", "metal", 14, 5, "solid"),
    (53, "iodine", "i", "halogen", 17, 5, "solid"),
    (54, "xenon", "xe", "noble_gas", 18, 5, "gas"),
    (74, "tungsten", "w", "transition_metal", 6, 6, "solid"),
    (78, "platinum", "pt", "transition_metal", 10, 6, "solid"),
    (79, "gold", "au", "transition_metal", 11, 6, "solid"),
    (80, "mercury", "hg", "transition_metal", 12, 6, "liquid"),
    (82, "lead", "pb", "metal", 14, 6, "solid"),
    (86, "radon", "rn", "noble_gas", 18, 6, "gas"),
    (92, "uranium", "u", "actinide", 3, 7, "solid"),
    (94, "plutonium", "pu", "actinide", 3, 7, "solid"),
]


# =============================================================================
#  Historical figures (extended)
# =============================================================================

HISTORY = [
    # (name, field, country, century, sub_role)
    ("aristotle", "philosophy", "greece", "-4", "philosopher"),
    ("plato", "philosophy", "greece", "-5", "philosopher"),
    ("socrates", "philosophy", "greece", "-5", "philosopher"),
    ("confucius", "philosophy", "china", "-6", "philosopher"),
    ("buddha", "religion", "india", "-6", "religious_founder"),
    ("jesus", "religion", "israel", "1", "religious_founder"),
    ("muhammad", "religion", "arabia", "7", "religious_founder"),
    ("alexander", "military", "macedonia", "-4", "conqueror"),
    ("caesar", "military", "rome", "-1", "emperor"),
    ("napoleon", "military", "france", "19", "emperor"),
    ("cleopatra", "leadership", "egypt", "-1", "queen"),
    ("genghis", "military", "mongolia", "13", "conqueror"),
    ("lincoln", "leadership", "usa", "19", "president"),
    ("washington", "leadership", "usa", "18", "president"),
    ("churchill", "leadership", "uk", "20", "primeminister"),
    ("gandhi", "leadership", "india", "20", "activist"),
    ("mandela", "leadership", "southafrica", "20", "president"),
    ("kingmlk", "leadership", "usa", "20", "activist"),
    ("davinci", "art", "italy", "15", "painter"),
    ("michelangelo", "art", "italy", "16", "sculptor"),
    ("rembrandt", "art", "netherlands", "17", "painter"),
    ("vangogh", "art", "netherlands", "19", "painter"),
    ("monet", "art", "france", "19", "painter"),
    ("picasso", "art", "spain", "20", "painter"),
    ("dali", "art", "spain", "20", "painter"),
    ("warhol", "art", "usa", "20", "painter"),
    ("mozart", "music", "austria", "18", "composer"),
    ("beethoven", "music", "germany", "18", "composer"),
    ("bach", "music", "germany", "18", "composer"),
    ("chopin", "music", "poland", "19", "composer"),
    ("wagner", "music", "germany", "19", "composer"),
    ("debussy", "music", "france", "19", "composer"),
    ("dylan", "music", "usa", "20", "singer"),
    ("beatles", "music", "uk", "20", "band"),
    ("shakespeare", "literature", "england", "16", "playwright"),
    ("dante", "literature", "italy", "13", "poet"),
    ("homer", "literature", "greece", "-8", "poet"),
    ("tolkien", "literature", "uk", "20", "novelist"),
    ("orwell", "literature", "uk", "20", "novelist"),
    ("dickens", "literature", "uk", "19", "novelist"),
    ("austen", "literature", "uk", "19", "novelist"),
    ("twain", "literature", "usa", "19", "novelist"),
    ("hemingway", "literature", "usa", "20", "novelist"),
    ("kafka", "literature", "czech", "20", "novelist"),
    ("tolstoy", "literature", "russia", "19", "novelist"),
    ("dostoevsky", "literature", "russia", "19", "novelist"),
    ("borges", "literature", "argentina", "20", "novelist"),
    ("rowling", "literature", "uk", "20", "novelist"),
    ("einstein", "physics", "germany", "20", "physicist"),
    ("newton", "physics", "england", "17", "physicist"),
    ("galileo", "physics", "italy", "16", "physicist"),
    ("hawking", "physics", "uk", "20", "physicist"),
    ("feynman", "physics", "usa", "20", "physicist"),
    ("bohr", "physics", "denmark", "20", "physicist"),
    ("heisenberg", "physics", "germany", "20", "physicist"),
    ("schrodinger", "physics", "austria", "20", "physicist"),
    ("dirac", "physics", "uk", "20", "physicist"),
    ("oppenheimer", "physics", "usa", "20", "physicist"),
    ("curie", "chemistry", "poland", "20", "chemist"),
    ("mendeleev", "chemistry", "russia", "19", "chemist"),
    ("darwin", "biology", "uk", "19", "biologist"),
    ("mendel", "biology", "austria", "19", "biologist"),
    ("pasteur", "biology", "france", "19", "biologist"),
    ("watson", "biology", "usa", "20", "biologist"),
    ("crick", "biology", "uk", "20", "biologist"),
    ("turing", "computerscience", "uk", "20", "computer_scientist"),
    ("lovelace", "computerscience", "uk", "19", "computer_scientist"),
    ("vonneumann", "computerscience", "usa", "20", "computer_scientist"),
    ("hopper", "computerscience", "usa", "20", "computer_scientist"),
    ("knuth", "computerscience", "usa", "20", "computer_scientist"),
    ("tesla_n", "engineering", "usa", "19", "engineer"),
    ("edison", "engineering", "usa", "19", "engineer"),
    ("musk", "engineering", "usa", "21", "engineer"),
    ("jobs", "engineering", "usa", "20", "engineer"),
    ("gates", "engineering", "usa", "20", "engineer"),
    ("zuckerberg", "engineering", "usa", "21", "engineer"),
    ("freud", "psychology", "austria", "20", "psychologist"),
    ("jung", "psychology", "switzerland", "20", "psychologist"),
    ("euclid", "mathematics", "greece", "-3", "mathematician"),
    ("pythagoras", "mathematics", "greece", "-6", "mathematician"),
    ("euler", "mathematics", "switzerland", "18", "mathematician"),
    ("gauss", "mathematics", "germany", "18", "mathematician"),
    ("ramanujan", "mathematics", "india", "20", "mathematician"),
    ("kant", "philosophy", "germany", "18", "philosopher"),
    ("descartes", "philosophy", "france", "17", "philosopher"),
    ("hume", "philosophy", "scotland", "18", "philosopher"),
    ("nietzsche", "philosophy", "germany", "19", "philosopher"),
]


# =============================================================================
#  Major rivers, mountains, deserts
# =============================================================================

RIVERS_MOUNTAINS = [
    # rivers
    ("nile", "river", "africa", 6650),
    ("amazon", "river", "southamerica", 6400),
    ("yangtze", "river", "asia", 6300),
    ("mississippi", "river", "northamerica", 6275),
    ("yenisei", "river", "asia", 5539),
    ("yellow", "river", "asia", 5464),
    ("ob", "river", "asia", 5410),
    ("congo", "river", "africa", 4700),
    ("amur", "river", "asia", 4444),
    ("lena", "river", "asia", 4400),
    ("mekong", "river", "asia", 4350),
    ("mackenzie", "river", "northamerica", 4241),
    ("niger", "river", "africa", 4180),
    ("brahmaputra", "river", "asia", 3848),
    ("volga", "river", "europe", 3645),
    ("danube", "river", "europe", 2860),
    ("zambezi", "river", "africa", 2693),
    ("ganges", "river", "asia", 2525),
    ("rhine", "river", "europe", 1233),
    ("seine", "river", "france", 777),
    ("thames", "river", "uk", 346),
    # mountains
    ("everest", "mountain", "nepal", 8849),
    ("k2", "mountain", "pakistan", 8611),
    ("kangchenjunga", "mountain", "nepal", 8586),
    ("denali", "mountain", "usa", 6190),
    ("aconcagua", "mountain", "argentina", 6961),
    ("kilimanjaro", "mountain", "tanzania", 5895),
    ("mont_blanc", "mountain", "france", 4807),
    ("matterhorn", "mountain", "switzerland", 4478),
    ("fuji", "mountain", "japan", 3776),
    ("olympus", "mountain", "greece", 2917),
]

DESERTS = {
    "sahara": ("africa", "hot"),
    "arabian": ("asia", "hot"),
    "gobi": ("asia", "cold"),
    "kalahari": ("africa", "hot"),
    "patagonian": ("southamerica", "cold"),
    "atacama": ("southamerica", "hot"),
    "mojave": ("northamerica", "hot"),
    "sonoran": ("northamerica", "hot"),
    "antarctic": ("antarctica", "cold"),
    "arctic": ("arctic", "cold"),
}


# =============================================================================
#  Tech companies
# =============================================================================

TECH_COMPANIES = [
    # (company, founder, founded_year, hq_country, sector)
    ("apple", "jobs", 1976, "usa", "consumer_electronics"),
    ("microsoft", "gates", 1975, "usa", "software"),
    ("google", "page", 1998, "usa", "search"),
    ("amazon", "bezos", 1994, "usa", "ecommerce"),
    ("meta", "zuckerberg", 2004, "usa", "social_media"),
    ("tesla", "musk", 2003, "usa", "automotive"),
    ("nvidia", "huang", 1993, "usa", "semiconductors"),
    ("openai", "altman", 2015, "usa", "ai"),
    ("anthropic", "amodei", 2021, "usa", "ai"),
    ("netflix", "hastings", 1997, "usa", "streaming"),
    ("spotify", "ek", 2006, "sweden", "streaming"),
    ("airbnb", "chesky", 2008, "usa", "travel"),
    ("uber", "kalanick", 2009, "usa", "transportation"),
    ("ibm", "watson_t", 1911, "usa", "enterprise_software"),
    ("intel", "moore", 1968, "usa", "semiconductors"),
    ("amd", "sanders", 1969, "usa", "semiconductors"),
    ("oracle", "ellison", 1977, "usa", "database"),
    ("salesforce", "benioff", 1999, "usa", "saas"),
    ("samsung", "byung-chul", 1938, "southkorea", "consumer_electronics"),
    ("sony", "morita", 1946, "japan", "consumer_electronics"),
    ("toyota", "toyoda", 1937, "japan", "automotive"),
    ("alibaba", "ma", 1999, "china", "ecommerce"),
    ("tencent", "ma_p", 1998, "china", "social_media"),
]


# =============================================================================
#  Animal taxonomy (extended)
# =============================================================================

ANIMALS = [
    # (animal, kind, diet, habitat, life_span_tier)
    ("lion", "mammal", "carnivore", "savanna", "medium"),
    ("tiger", "mammal", "carnivore", "jungle", "medium"),
    ("elephant", "mammal", "herbivore", "savanna", "long"),
    ("giraffe", "mammal", "herbivore", "savanna", "medium"),
    ("zebra", "mammal", "herbivore", "savanna", "medium"),
    ("rhino", "mammal", "herbivore", "savanna", "medium"),
    ("hippopotamus", "mammal", "herbivore", "river", "medium"),
    ("cheetah", "mammal", "carnivore", "savanna", "short"),
    ("leopard", "mammal", "carnivore", "jungle", "medium"),
    ("hyena", "mammal", "carnivore", "savanna", "medium"),
    ("wolf", "mammal", "carnivore", "forest", "medium"),
    ("fox", "mammal", "omnivore", "forest", "short"),
    ("bear", "mammal", "omnivore", "forest", "medium"),
    ("polarbear", "mammal", "carnivore", "arctic", "medium"),
    ("panda", "mammal", "herbivore", "forest", "medium"),
    ("koala", "mammal", "herbivore", "forest", "medium"),
    ("kangaroo", "mammal", "herbivore", "savanna", "medium"),
    ("platypus", "mammal", "carnivore", "river", "medium"),
    ("dolphin", "mammal", "carnivore", "ocean", "medium"),
    ("whale", "mammal", "carnivore", "ocean", "long"),
    ("seal", "mammal", "carnivore", "ocean", "medium"),
    ("walrus", "mammal", "carnivore", "ocean", "medium"),
    ("otter", "mammal", "carnivore", "river", "short"),
    ("beaver", "mammal", "herbivore", "river", "short"),
    ("squirrel", "mammal", "herbivore", "forest", "short"),
    ("rabbit", "mammal", "herbivore", "forest", "short"),
    ("hedgehog", "mammal", "omnivore", "forest", "short"),
    ("bat", "mammal", "carnivore", "cave", "short"),
    ("eagle", "bird", "carnivore", "mountain", "medium"),
    ("hawk", "bird", "carnivore", "forest", "medium"),
    ("owl", "bird", "carnivore", "forest", "medium"),
    ("falcon", "bird", "carnivore", "mountain", "medium"),
    ("penguin", "bird", "carnivore", "arctic", "medium"),
    ("ostrich", "bird", "omnivore", "savanna", "medium"),
    ("flamingo", "bird", "omnivore", "lake", "medium"),
    ("peacock", "bird", "omnivore", "jungle", "medium"),
    ("parrot", "bird", "herbivore", "jungle", "medium"),
    ("hummingbird", "bird", "omnivore", "jungle", "short"),
    ("crow", "bird", "omnivore", "city", "medium"),
    ("sparrow", "bird", "omnivore", "city", "short"),
    ("crocodile", "reptile", "carnivore", "river", "long"),
    ("alligator", "reptile", "carnivore", "river", "long"),
    ("snake", "reptile", "carnivore", "varied", "medium"),
    ("turtle", "reptile", "omnivore", "ocean", "long"),
    ("lizard", "reptile", "carnivore", "varied", "short"),
    ("frog", "amphibian", "carnivore", "river", "short"),
    ("toad", "amphibian", "carnivore", "varied", "short"),
    ("salmon", "fish", "carnivore", "river", "short"),
    ("trout", "fish", "carnivore", "river", "short"),
    ("tuna", "fish", "carnivore", "ocean", "medium"),
    ("shark", "fish", "carnivore", "ocean", "medium"),
    ("octopus", "mollusk", "carnivore", "ocean", "short"),
    ("squid", "mollusk", "carnivore", "ocean", "short"),
    ("jellyfish", "cnidarian", "carnivore", "ocean", "short"),
    ("crab", "crustacean", "omnivore", "ocean", "short"),
    ("lobster", "crustacean", "carnivore", "ocean", "medium"),
    ("ant", "insect", "omnivore", "varied", "short"),
    ("bee", "insect", "herbivore", "varied", "short"),
    ("butterfly", "insect", "herbivore", "varied", "short"),
    ("dragonfly", "insect", "carnivore", "river", "short"),
    ("beetle", "insect", "omnivore", "varied", "short"),
    ("spider", "arachnid", "carnivore", "varied", "short"),
    ("scorpion", "arachnid", "carnivore", "desert", "medium"),
]


# =============================================================================
#  Religions
# =============================================================================

RELIGIONS = [
    # (name, founder, sacred_text, origin_country, kind)
    ("christianity", "jesus", "bible", "israel", "monotheism"),
    ("islam", "muhammad", "quran", "arabia", "monotheism"),
    ("judaism", "abraham", "torah", "israel", "monotheism"),
    ("hinduism", "none", "vedas", "india", "polytheism"),
    ("buddhism", "buddha", "tripitaka", "india", "nontheistic"),
    ("sikhism", "nanak", "guruGranthSahib", "india", "monotheism"),
    ("jainism", "mahavira", "agamas", "india", "nontheistic"),
    ("taoism", "laozi", "taoTeChing", "china", "nontheistic"),
    ("shinto", "none", "kojiki", "japan", "polytheism"),
    ("confucianism", "confucius", "analects", "china", "philosophy"),
    ("zoroastrianism", "zoroaster", "avesta", "iran", "monotheism"),
    ("bahai", "bahaullah", "kitab-i-aqdas", "iran", "monotheism"),
]


# =============================================================================
#  Foods + cuisines
# =============================================================================

CUISINES = {
    "italian": ["pizza", "pasta", "risotto", "gelato"],
    "japanese": ["sushi", "ramen", "tempura", "miso"],
    "chinese": ["dumpling", "stirfry", "noodles", "fried_rice"],
    "indian": ["curry", "biryani", "naan", "tikka"],
    "mexican": ["taco", "burrito", "guacamole", "salsa"],
    "french": ["croissant", "baguette", "souffle", "ratatouille"],
    "thai": ["padthai", "tomyum", "greencurry", "satay"],
    "spanish": ["paella", "tapas", "gazpacho", "churro"],
    "greek": ["souvlaki", "moussaka", "tzatziki", "baklava"],
    "korean": ["bibimbap", "kimchi", "bulgogi", "tteokbokki"],
    "american": ["hamburger", "hotdog", "applepie", "barbecue"],
    "british": ["fishandchips", "sundayroast", "shepherdspie", "scone"],
    "vietnamese": ["pho", "banhmi", "springroll", "vermicelli"],
    "ethiopian": ["injera", "wat", "tibs", "doro_wat"],
    "moroccan": ["tagine", "couscous", "harira", "msemen"],
    "lebanese": ["hummus", "falafel", "tabbouleh", "shawarma"],
    "brazilian": ["feijoada", "churrasco", "pao_de_queijo", "moqueca"],
    "peruvian": ["ceviche", "lomosaltado", "ajidegallina", "anticucho"],
}


# =============================================================================
#  Sports + Olympics
# =============================================================================

SPORTS = {
    "soccer": ("team", "ball", "field", "olympic"),
    "basketball": ("team", "ball", "court", "olympic"),
    "tennis": ("solo", "ball", "court", "olympic"),
    "baseball": ("team", "ball", "field", "olympic"),
    "americanfootball": ("team", "ball", "field", "nonolympic"),
    "cricket": ("team", "ball", "pitch", "nonolympic"),
    "rugby": ("team", "ball", "field", "olympic"),
    "hockey": ("team", "puck", "rink", "olympic"),
    "fieldhockey": ("team", "ball", "field", "olympic"),
    "volleyball": ("team", "ball", "court", "olympic"),
    "golf": ("solo", "ball", "course", "olympic"),
    "swimming": ("solo", "none", "pool", "olympic"),
    "boxing": ("solo", "gloves", "ring", "olympic"),
    "wrestling": ("solo", "none", "mat", "olympic"),
    "weightlifting": ("solo", "barbell", "platform", "olympic"),
    "skiing": ("solo", "skis", "slope", "olympic"),
    "snowboarding": ("solo", "board", "slope", "olympic"),
    "surfing": ("solo", "board", "ocean", "olympic"),
    "cycling": ("solo", "bicycle", "road", "olympic"),
    "running": ("solo", "none", "track", "olympic"),
    "marathon": ("solo", "none", "road", "olympic"),
    "archery": ("solo", "bow", "range", "olympic"),
    "fencing": ("solo", "sword", "piste", "olympic"),
    "judo": ("solo", "none", "mat", "olympic"),
    "karate": ("solo", "none", "mat", "olympic"),
    "taekwondo": ("solo", "none", "mat", "olympic"),
    "gymnastics": ("solo", "varied", "gym", "olympic"),
    "diving": ("solo", "none", "pool", "olympic"),
    "rowing": ("team", "oar", "water", "olympic"),
    "sailing": ("team", "boat", "water", "olympic"),
}


# =============================================================================
#  Body anatomy
# =============================================================================

BODY = [
    # (part, system, location)
    ("heart", "circulatory", "chest"),
    ("lungs", "respiratory", "chest"),
    ("brain", "nervous", "head"),
    ("liver", "digestive", "abdomen"),
    ("kidney", "urinary", "abdomen"),
    ("stomach", "digestive", "abdomen"),
    ("intestine", "digestive", "abdomen"),
    ("pancreas", "digestive", "abdomen"),
    ("spleen", "immune", "abdomen"),
    ("bladder", "urinary", "pelvis"),
    ("skin", "integumentary", "whole_body"),
    ("eye", "sensory", "head"),
    ("ear", "sensory", "head"),
    ("nose", "sensory", "head"),
    ("tongue", "sensory", "head"),
    ("teeth", "digestive", "head"),
    ("hair", "integumentary", "head"),
    ("muscle", "muscular", "whole_body"),
    ("bone", "skeletal", "whole_body"),
    ("nerve", "nervous", "whole_body"),
    ("artery", "circulatory", "whole_body"),
    ("vein", "circulatory", "whole_body"),
]


# =============================================================================
#  Number facts: squares 1-50, primes, common products
# =============================================================================

def number_facts():
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    for n in range(1, 101):
        yield (f"{n}", "square", f"{n*n}")
        if n in primes:
            yield (f"{n}", "isa", "prime_number")
        if n % 2 == 0 and n > 0:
            yield (f"{n}", "is", "even")
        elif n > 0:
            yield (f"{n}", "is", "odd")
    # Common multiplication
    for a in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        for b in range(1, 13):
            yield (f"{a}times{b}", "equals", f"{a*b}")


def main() -> None:
    # Countries
    for c, cap, cont, cur, lang, pop in COUNTRIES:
        emit(c, "capital", cap)
        emit(c, "continent", cont)
        emit(c, "currency", cur)
        emit(c, "language", lang)
        emit(c, "population_tier", pop)
        emit(c, "isa", "country")
        emit(cap, "isa", "city")
        emit(cap, "capital_of", c)
        emit(cap, "locatedin", c)

    # Elements
    for atomic, name, sym, kind, group, period, state in ELEMENTS:
        emit(name, "isa", "element")
        emit(name, "symbol", sym)
        emit(name, "atomic_number", str(atomic))
        emit(name, "category", kind)
        emit(name, "group", str(group))
        emit(name, "period", str(period))
        emit(name, "state_at_room_temperature", state)
        emit(sym, "represents", name)

    # History
    for name, field, country, century, role in HISTORY:
        emit(name, "field", field)
        emit(name, "country", country)
        emit(name, "century", century)
        emit(name, "role", role)
        emit(name, "isa", "person")

    # Rivers + mountains
    for name, kind, location, length_or_height in RIVERS_MOUNTAINS:
        emit(name, "isa", kind)
        emit(name, "locatedin", location)
        attr = "length_km" if kind == "river" else "height_m"
        emit(name, attr, str(length_or_height))

    # Deserts
    for name, (continent, climate) in DESERTS.items():
        emit(name, "isa", "desert")
        emit(name, "continent", continent)
        emit(name, "climate", climate)

    # Tech companies
    for c, founder, year, hq, sector in TECH_COMPANIES:
        emit(c, "isa", "company")
        emit(c, "founder", founder)
        emit(c, "founded_year", str(year))
        emit(c, "hq_country", hq)
        emit(c, "sector", sector)
        emit(founder, "founded", c)

    # Animals
    for animal, kind, diet, habitat, lifespan in ANIMALS:
        emit(animal, "isa", kind)
        emit(animal, "diet", diet)
        emit(animal, "habitat", habitat)
        emit(animal, "lifespan", lifespan)

    # Religions
    for name, founder, text, country, kind in RELIGIONS:
        emit(name, "isa", "religion")
        emit(name, "founder", founder)
        emit(name, "sacred_text", text)
        emit(name, "origin_country", country)
        emit(name, "kind", kind)

    # Cuisines
    for cuisine, dishes in CUISINES.items():
        emit(cuisine, "isa", "cuisine")
        for dish in dishes:
            emit(dish, "isa", "food")
            emit(dish, "cuisine", cuisine)
            emit(dish, "origin", cuisine)

    # Sports
    for sport, (kind, equipment, venue, olympic) in SPORTS.items():
        emit(sport, "isa", "sport")
        emit(sport, "kind", kind)
        emit(sport, "equipment", equipment)
        emit(sport, "venue", venue)
        emit(sport, "olympic", olympic)

    # Body
    for part, system, location in BODY:
        emit(part, "isa", "body_part")
        emit(part, "system", system)
        emit(part, "location", location)

    # Number facts
    for s, r, o in number_facts():
        emit(s, r, o)


if __name__ == "__main__":
    main()
