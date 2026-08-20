"""CLUTRR-style kinship inductive-reasoning study for RCK.

RCK's headline claim (see docs/design/v13-chain-depth-finding.md and
docs/guide/03-reasoning.md) is compositional multi-hop reasoning that stays
correct to 50+ hops. Every existing study in scripts/ tests that claim with
probes RCK's own authors constructed (synthetic `next`-chains, hand-built
commonsense KBs). None of it is drawn from, or modeled on, a benchmark this
project did not build itself.

CLUTRR (Sinha et al., EMNLP 2019, https://github.com/facebookresearch/clutrr)
is the standard external benchmark for exactly RCK's claimed capability:
given a chain of kinship facts, infer the relation between the two endpoints,
and do it at chain lengths longer than anything seen in isolation -- i.e.
systematic generalisation over compositional structure. That is the right
outside check for RCK.

*** DATA PROVENANCE -- READ BEFORE INTERPRETING RESULTS ***
This script does NOT use the official CLUTRR dataset. Network access to
GitHub/the CLUTRR data release was not available in this environment (see
the run log). Every example below is CLUTRR-STYLE and LOCALLY GENERATED:
algorithmically produced from a family tree plus the kinship composition
table defined in this file. It is legitimate and fully reproducible (this
is literally how CLUTRR itself is generated -- procedurally, from a kinship
composition table over a synthetic family tree), but:
  - it is NOT the official CLUTRR data or split,
  - scores here are NOT comparable to published CLUTRR leaderboard numbers,
  - "CLUTRR" in variable/function names below always means "CLUTRR-style,
    locally generated" -- never the official benchmark.
Every report and printout from this script says so explicitly.

--------------------------------------------------------------------------
DESIGN

1. A synthetic family tree (`build_family_tree`) with up to 6 generations,
   built from 8 PRIMITIVE kinship relations: mother, father, son, daughter,
   brother, sister, husband, wife. Every fact told to RCK is a real edge
   read directly off this tree -- never invented.

2. Ground truth for "what is B to A" is computed two independent ways that
   MUST agree:
     a) `resolve_updown` walks the ACTUAL tree via lowest-common-ancestor
        (LCA) search -- authoritative, does not look at edge labels at all.
     b) The symbolic baseline (`symbolic_infer`) walks ONLY the revealed
        k edges (exactly what RCK also receives) and recomputes the answer
        by forward-composing relation labels through `term_from_up_down`,
        the same "composition table" cited in the task brief
        (mother of mother = grandmother, etc.), reproduced below as
        closed-form formulas instead of a giant hand-typed table, plus 8
        explicit in-law rules for 2-hop marriage links.
   If the generator and the table are correct, (a) and (b) must always
   match on every example, since edges are built to contain exactly the
   up/down hops the tree's LCA computation says they should. The symbolic
   baseline's accuracy is therefore a harness sanity check, not a
   meaningful "system" being evaluated -- see the printed report.

3. RCK is evaluated BLIND: it receives ONLY the k edge triples via
   `agent.tell(...)`. It never sees the tree, the up/down counts, or the
   composition table's internals -- it is handed the SAME externally
   supplied composition table (`term_from_up_down` + the in-law rules) as
   the symbolic baseline, and must supply the relation PATH itself via
   `agent.discover(start, end, max_depth=k)`, RCK's BFS chain-discovery
   API (docs/guide/03-reasoning.md, "Chain discovery"). Composing whatever
   path RCK reports through the shared table isolates exactly the variable
   RCK claims to be good at: whether its HRR-based multi-hop retrieval
   finds the right path at all as chain length grows -- not whether it
   knows kinship semantics (nothing in RCK's public API claims that).

   `agent.reason(start, relations)` is NOT used for the primary metric:
   it requires being TOLD the relation sequence in advance and only
   confirms entity reachability, which is not "inferring the endpoint
   relation" -- it is retrieval given the answer's key ingredient. It is
   run as a secondary, clearly-labeled, easier ablation for comparison
   (see `evaluate_reason_ablation`).

SCOPE (deliberate, to keep the composition table small and provably
correct -- see docs strings on `term_from_up_down`):
  - In-law relations (mother-in-law, brother-in-law, son-in-law, ...) only
    ever appear as k=2 chains (a single marriage link plus a single blood
    link). Longer in-law chains (e.g. "grandmother-in-law") have no common
    English term and are excluded, not mislabeled.
  - Collateral relations where ascent != descent and both >= 2 (e.g.
    "first cousin once removed") are excluded for the same reason.
  - All marriages in the synthetic tree are opposite-gender (a modelling
    simplification of the synthetic tree, not a claim about real families).
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.conscious_agent import ConsciousAgent

SEED = 20260819
K_VALUES = (2, 3, 4, 5, 6)
MAX_PER_PATTERN = 8
GENERATIONS = 7          # gen 0 (root) .. gen 6 -> max ancestor distance 6
CHILDREN_PER_COUPLE = 2

MALE_NAMES = [
    "james", "robert", "john", "michael", "david", "william", "richard",
    "thomas", "charles", "daniel", "matthew", "anthony", "mark", "paul",
    "steven", "andrew", "kenneth", "joshua", "kevin", "brian",
]
FEMALE_NAMES = [
    "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara",
    "susan", "jessica", "sarah", "karen", "nancy", "lisa", "margaret",
    "betty", "sandra", "ashley", "kimberly", "emily", "donna", "michelle",
]

BLOOD_PRIMITIVES = {"mother", "father", "son", "daughter", "brother", "sister"}
SPOUSE_PRIMITIVES = {"husband", "wife"}
ALL_PRIMITIVES = BLOOD_PRIMITIVES | SPOUSE_PRIMITIVES


# ===========================================================================
# 1. Family tree
# ===========================================================================

@dataclass
class Person:
    id: str
    gender: str  # "M" or "F"
    mother: str | None = None
    father: str | None = None
    spouse: str | None = None
    children: list[str] = field(default_factory=list)


def build_family_tree(rng: random.Random, generations: int = GENERATIONS,
                       children_per_couple: int = CHILDREN_PER_COUPLE
                       ) -> tuple[dict[str, Person], str]:
    """Build a synthetic multi-generation family tree.

    Every married-in spouse is a fresh person with no parents in the tree
    (so the mother/father graph stays a proper forest -- no marriage loops,
    no shared ancestors through in-laws, which is what makes LCA search
    below unambiguous).
    """
    tree: dict[str, Person] = {}
    counters = {"M": 0, "F": 0}

    def new_person(gender: str) -> str:
        counters[gender] += 1
        pool = MALE_NAMES if gender == "M" else FEMALE_NAMES
        base = pool[(counters[gender] - 1) % len(pool)]
        suffix = (counters[gender] - 1) // len(pool)
        pid = f"{base}{counters['M'] + counters['F']}{'x' + str(suffix) if suffix else ''}"
        tree[pid] = Person(id=pid, gender=gender)
        return pid

    def marry(pid: str) -> str:
        p = tree[pid]
        if p.spouse:
            return p.spouse
        sgender = "F" if p.gender == "M" else "M"
        sid = new_person(sgender)
        tree[sid].spouse = pid
        p.spouse = sid
        return sid

    root_gender = rng.choice(["M", "F"])
    root = new_person(root_gender)
    marry(root)

    frontier = [root]
    for _gen in range(generations - 1):
        next_frontier: list[str] = []
        for pid in frontier:
            p = tree[pid]
            sid = p.spouse
            mother_id = pid if p.gender == "F" else sid
            father_id = pid if p.gender == "M" else sid
            for _ in range(children_per_couple):
                cgender = rng.choice(["M", "F"])
                cid = new_person(cgender)
                tree[cid].mother = mother_id
                tree[cid].father = father_id
                tree[mother_id].children.append(cid)
                tree[father_id].children.append(cid)
                marry(cid)
                next_frontier.append(cid)
        frontier = next_frontier
    return tree, root


def ancestors_with_dist(tree: dict[str, Person], pid: str) -> dict[str, int]:
    """BFS up the mother/father graph. Returns {ancestor_id: generations_up}."""
    dist = {pid: 0}
    frontier = [pid]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for x in frontier:
            p = tree[x]
            for par in (p.mother, p.father):
                if par and par not in dist:
                    dist[par] = d
                    nxt.append(par)
        frontier = nxt
    return dist


def resolve_updown(tree: dict[str, Person], a: str, b: str
                    ) -> tuple[int, int] | None:
    """Authoritative ground truth: LCA search over the ACTUAL tree.
    Does not look at any relation label. Returns (up, down) -- generations
    from a up to the lowest common ancestor, and down from there to b."""
    da = ancestors_with_dist(tree, a)
    db = ancestors_with_dist(tree, b)
    common = set(da) & set(db)
    if not common:
        return None
    # Tie-break on the node id, not on set iteration order. Without the
    # `n` term, two common ancestors at equal total distance are separated
    # by str hashing, which PYTHONHASHSEED randomises per process -- so a
    # fixed `random.Random(seed)` still produced a DIFFERENT dataset in
    # every process (measured: 344/348/350 edges across three runs).
    best = min(common, key=lambda n: (da[n] + db[n], n))
    return da[best], db[best]


def path_up_to(tree: dict[str, Person], start: str, target: str, dist: int
               ) -> list[str] | None:
    """Reconstruct the unique `dist`-hop parent chain from start to target."""
    if dist == 0:
        return [start] if start == target else None
    chain = [start]
    cur = start
    remaining = dist
    while remaining > 0:
        p = tree[cur]
        next_node = None
        for c in (p.mother, p.father):
            if c is None:
                continue
            dc = ancestors_with_dist(tree, c)
            if dc.get(target) == remaining - 1:
                next_node = c
                break
        if next_node is None:
            return None
        chain.append(next_node)
        cur = next_node
        remaining -= 1
    return chain


def siblings_of(tree: dict[str, Person], pid: str) -> list[str]:
    p = tree[pid]
    if p.mother is None or p.father is None:
        return []
    return [x for x, person in tree.items()
            if x != pid and person.mother == p.mother and person.father == p.father]


# ===========================================================================
# 2. Composition table (shared by ground truth AND the symbolic baseline)
# ===========================================================================
#
# Closed-form instead of a giant hand-typed table, but it IS exactly the
# pairwise idea the task describes (mother-of-mother = grandmother,
# father-of-brother = uncle, mother-of-son = brother): every rule below
# reduces to composing (generation delta, gender) the same way those
# examples do, just generalised to arbitrary depth via formulas so the
# table stays small enough to read and verify in one sitting.

def ancestor_term(up: int, gender: str) -> str:
    if up == 1:
        return "father" if gender == "M" else "mother"
    if up == 2:
        return "grandfather" if gender == "M" else "grandmother"
    return "great-" * (up - 2) + ("grandfather" if gender == "M" else "grandmother")


def descendant_term(down: int, gender: str) -> str:
    if down == 1:
        return "son" if gender == "M" else "daughter"
    if down == 2:
        return "grandson" if gender == "M" else "granddaughter"
    return "great-" * (down - 2) + ("grandson" if gender == "M" else "granddaughter")


def auncle_term(up: int, gender: str) -> str:
    base = "uncle" if gender == "M" else "aunt"
    return "great-" * (up - 2) + base


def nibling_term(down: int, gender: str) -> str:
    base = "nephew" if gender == "M" else "niece"
    return "great-" * (down - 2) + base


def cousin_term(up: int) -> str:
    degree = up - 1
    if degree == 1:
        return "cousin"
    ordinal = {2: "second", 3: "third", 4: "fourth"}.get(degree, f"{degree}th")
    return f"{ordinal} cousin"


def term_from_up_down(up: int, down: int, gender: str) -> str | None:
    """THE composition table for pure-blood relations. (up, down) is the
    generation distance to/from the lowest common ancestor; `gender` is
    the gender of the endpoint entity being described. Returns None for
    shapes with no common English term (deliberately excluded, see the
    module docstring's SCOPE section)."""
    if up == 0 and down == 0:
        return None
    if down == 0:
        return ancestor_term(up, gender)
    if up == 0:
        return descendant_term(down, gender)
    if up == 1 and down == 1:
        return "brother" if gender == "M" else "sister"
    if down == 1 and up >= 2:
        return auncle_term(up, gender)
    if up == 1 and down >= 2:
        return nibling_term(down, gender)
    if up == down and up >= 2:
        return cousin_term(up)
    return None  # e.g. "cousin once removed" shapes -- excluded, not guessed


# In-law rules: exactly 2 hops, one spouse primitive + one blood primitive.
# (first_relation, second_relation) -> relation name keyed by final gender.
INLAW_PARENT = {"F": "mother-in-law", "M": "father-in-law"}
INLAW_SIBLING = {"M": "brother-in-law", "F": "sister-in-law"}


def symbolic_infer(edges: list[tuple[str, str, str]]) -> str | None:
    """The control: forward-compose the SAME table above over ONLY the
    revealed edge list (no tree access). Must reproduce the ground truth
    on every generated example, or the generator/table has a bug."""
    rels = [r for _, r, _ in edges]
    if len(rels) == 2 and (rels[0] in SPOUSE_PRIMITIVES or rels[1] in SPOUSE_PRIMITIVES):
        final_gender = _gender_of_primitive(rels[-1])
        if rels[0] in SPOUSE_PRIMITIVES and rels[1] in {"mother", "father"}:
            return INLAW_PARENT[final_gender]
        if rels[0] in SPOUSE_PRIMITIVES and rels[1] in {"brother", "sister"}:
            return INLAW_SIBLING[final_gender]
        if rels[0] == "son" and rels[1] in SPOUSE_PRIMITIVES:
            return "daughter-in-law"
        if rels[0] == "daughter" and rels[1] in SPOUSE_PRIMITIVES:
            return "son-in-law"
        if rels[0] == "brother" and rels[1] in SPOUSE_PRIMITIVES:
            return "sister-in-law"
        if rels[0] == "sister" and rels[1] in SPOUSE_PRIMITIVES:
            return "brother-in-law"
        return None
    up = down = 0
    for r in rels:
        if r not in BLOOD_PRIMITIVES:
            return None  # unexpected primitive in a "pure blood" chain
        if r in ("mother", "father"):
            up += 1
        elif r in ("son", "daughter"):
            down += 1
        else:  # brother / sister should not appear in pure-blood generated chains
            return None
    final_gender = _gender_of_primitive(rels[-1])
    return term_from_up_down(up, down, final_gender)


def _gender_of_primitive(rel: str) -> str:
    return {"mother": "F", "father": "M", "son": "M", "daughter": "F",
            "brother": "M", "sister": "F", "husband": "M", "wife": "F"}[rel]


# ===========================================================================
# 3. Example generation
# ===========================================================================

@dataclass
class Example:
    id: str
    k: int
    pattern: str
    start: str
    end: str
    edges: list[tuple[str, str, str]]
    true_relation: str


def build_blood_edges(tree: dict[str, Person], a: str, b: str,
                       up: int, down: int) -> list[tuple[str, str, str]] | None:
    da = ancestors_with_dist(tree, a)
    db = ancestors_with_dist(tree, b)
    common = set(da) & set(db)
    # Same deterministic tie-break as resolve_updown -- these two must
    # agree on the LCA or the edges will not match the resolved (up, down).
    lca = min(common, key=lambda n: (da[n] + db[n], n))
    up_chain = path_up_to(tree, a, lca, up)
    down_chain = path_up_to(tree, b, lca, down)
    if up_chain is None or down_chain is None:
        return None
    down_chain = list(reversed(down_chain))  # lca -> ... -> b
    edges: list[tuple[str, str, str]] = []
    for i in range(up):
        s, o = up_chain[i], up_chain[i + 1]
        r = "mother" if tree[o].gender == "F" else "father"
        edges.append((s, r, o))
    for i in range(down):
        s, o = down_chain[i], down_chain[i + 1]
        r = "son" if tree[o].gender == "M" else "daughter"
        edges.append((s, r, o))
    return edges


def generate_blood_candidates(tree: dict[str, Person], rng: random.Random
                               ) -> dict[tuple[int, str], list[Example]]:
    buckets: dict[tuple[int, str], list[Example]] = defaultdict(list)
    ids = list(tree.keys())
    rng.shuffle(ids)
    for a in ids:
        for b in ids:
            if a == b:
                continue
            res = resolve_updown(tree, a, b)
            if res is None:
                continue
            up, down = res
            k = up + down
            if k not in K_VALUES or k == 0:
                continue
            gender = tree[b].gender
            term = term_from_up_down(up, down, gender)
            if term is None:
                continue
            key = (k, term)
            if len(buckets[key]) >= MAX_PER_PATTERN * 3:
                continue  # cheap cap before final subsample, keeps this loop fast
            edges = build_blood_edges(tree, a, b, up, down)
            if edges is None or len(edges) != k:
                continue
            buckets[key].append(Example(
                id=f"blood-{a}-{b}", k=k, pattern=term,
                start=a, end=b, edges=edges, true_relation=term,
            ))
    return buckets


def generate_inlaw_candidates(tree: dict[str, Person]
                               ) -> dict[tuple[int, str], list[Example]]:
    buckets: dict[tuple[int, str], list[Example]] = defaultdict(list)

    def add(start: str, end: str, edges: list[tuple[str, str, str]], label: str) -> None:
        buckets[(2, label)].append(Example(
            id=f"inlaw-{start}-{end}", k=2, pattern=label,
            start=start, end=end, edges=edges, true_relation=label,
        ))

    for pid, p in tree.items():
        if p.spouse:
            s = tree[p.spouse]
            r1 = "husband" if s.gender == "M" else "wife"
            for par_id in (s.mother, s.father):
                if par_id:
                    par = tree[par_id]
                    r2 = "mother" if par.gender == "F" else "father"
                    label = INLAW_PARENT[par.gender]
                    add(pid, par_id, [(pid, r1, p.spouse), (p.spouse, r2, par_id)], label)
            for sib_id in siblings_of(tree, p.spouse):
                sib = tree[sib_id]
                r2 = "brother" if sib.gender == "M" else "sister"
                label = INLAW_SIBLING[sib.gender]
                add(pid, sib_id, [(pid, r1, p.spouse), (p.spouse, r2, sib_id)], label)
        for cid in p.children:
            c = tree[cid]
            if c.spouse:
                r1 = "son" if c.gender == "M" else "daughter"
                cs = tree[c.spouse]
                r2 = "wife" if cs.gender == "F" else "husband"
                label = "daughter-in-law" if cs.gender == "F" else "son-in-law"
                add(pid, c.spouse, [(pid, r1, cid), (cid, r2, c.spouse)], label)
        for sib_id in siblings_of(tree, pid):
            sib = tree[sib_id]
            if sib.spouse:
                r1 = "brother" if sib.gender == "M" else "sister"
                ss = tree[sib.spouse]
                r2 = "wife" if ss.gender == "F" else "husband"
                # The endpoint IS ss (sibling's spouse) -- label by ss's own
                # gender, same convention as every other in-law rule (e.g.
                # "brother's wife" = sister-in-law, keyed on the wife being
                # female). A prior version flipped this key and was caught
                # by the symbolic baseline scoring < 100%.
                label = INLAW_SIBLING[ss.gender]
                add(pid, sib.spouse, [(pid, r1, sib_id), (sib_id, r2, sib.spouse)], label)
    return buckets


def generate_dataset(seed: int = SEED) -> list[Example]:
    rng = random.Random(seed)
    tree, _root = build_family_tree(rng)
    blood = generate_blood_candidates(tree, rng)
    inlaw = generate_inlaw_candidates(tree)
    all_buckets: dict[tuple[int, str], list[Example]] = defaultdict(list)
    for key, exs in blood.items():
        all_buckets[key].extend(exs)
    for key, exs in inlaw.items():
        all_buckets[key].extend(exs)

    dataset: list[Example] = []
    for key in sorted(all_buckets):
        pool = all_buckets[key]
        rng.shuffle(pool)
        chosen = pool[:MAX_PER_PATTERN]
        for i, ex in enumerate(chosen):
            ex.id = f"{ex.pattern}_k{ex.k}_{i}"
            dataset.append(ex)
    rng.shuffle(dataset)
    return dataset


# ===========================================================================
# 4. Evaluation
# ===========================================================================

def evaluate_rck_discover(example: Example) -> dict:
    """Primary metric. Blind: RCK receives only the edge triples via
    `tell()`, then must find the connecting relation path itself via
    `discover()` and we compose it through the SAME external table."""
    agent = ConsciousAgent(install_self=False)
    for s, r, o in example.edges:
        agent.tell(s, r, o)
    spec = agent.discover(example.start, example.end, max_depth=example.k,
                           allow_reverse=False)
    if spec is None:
        return {"predicted": None, "correct": False, "found_path": False,
                "path_matches_edges": False}
    # Compose RCK's discovered relation path through the shared table.
    rels = spec["relations"]
    if len(rels) == 2 and (rels[0] in SPOUSE_PRIMITIVES or rels[1] in SPOUSE_PRIMITIVES):
        predicted = symbolic_infer([("_", r, "_") for r in rels])
    elif rels and all(r in BLOOD_PRIMITIVES for r in rels):
        predicted = symbolic_infer([("_", r, "_") for r in rels])
    else:
        predicted = None
    true_edge_rels = [r for _, r, _ in example.edges]
    return {
        "predicted": predicted,
        "correct": predicted == example.true_relation,
        "found_path": True,
        "discovered_relations": rels,
        "path_matches_edges": rels == true_edge_rels,
        "discovery_confidence": spec.get("confidence"),
    }


def evaluate_reason_ablation(example: Example) -> dict:
    """Secondary, easier ablation: RCK IS TOLD the true relation sequence
    and must only retrieve the endpoint entity via `reason()`. This is
    NOT "inferring the endpoint relation" (the relations are the input),
    so it is reported separately, not as the headline number."""
    agent = ConsciousAgent(install_self=False)
    for s, r, o in example.edges:
        agent.tell(s, r, o)
    true_rels = [r for _, r, _ in example.edges]
    res = agent.reason(example.start, true_rels)
    return {
        "answer": res["answer"],
        "correct_entity": res["answer"] == example.end,
        "confidence": res["confidence"],
        "hedge": res["hedge"],
    }


def evaluate_symbolic(example: Example) -> dict:
    predicted = symbolic_infer(example.edges)
    return {"predicted": predicted, "correct": predicted == example.true_relation}


# ===========================================================================
# 5. Report
# ===========================================================================

def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print(" CLUTRR-STYLE KINSHIP STUDY (locally generated -- NOT official CLUTRR)")
    print("=" * 78)
    print(f"seed={SEED}  k_values={K_VALUES}  max_per_pattern={MAX_PER_PATTERN}")

    dataset = generate_dataset(SEED)
    print(f"\nGenerated {len(dataset)} examples.")
    by_k_count: dict[int, int] = defaultdict(int)
    for ex in dataset:
        by_k_count[ex.k] += 1
    print("Examples per k:", dict(sorted(by_k_count.items())))

    rows = []
    for i, ex in enumerate(dataset):
        sym = evaluate_symbolic(ex)
        rck = evaluate_rck_discover(ex)
        reason_ab = evaluate_reason_ablation(ex)
        rows.append({
            "id": ex.id, "k": ex.k, "pattern": ex.pattern,
            "start": ex.start, "end": ex.end,
            "edges": [list(e) for e in ex.edges],
            "true_relation": ex.true_relation,
            "symbolic": sym,
            "rck_discover": rck,
            "rck_reason_ablation": reason_ab,
        })
        if (i + 1) % 50 == 0:
            print(f"  ... evaluated {i + 1}/{len(dataset)}")

    # Aggregate.
    def acc_by_k(key_getter) -> dict[int, dict]:
        agg: dict[int, list[bool]] = defaultdict(list)
        for r in rows:
            agg[r["k"]].append(bool(key_getter(r)))
        return {k: {"n": len(v), "accuracy": sum(v) / len(v) if v else 0.0}
                for k, v in sorted(agg.items())}

    sym_by_k = acc_by_k(lambda r: r["symbolic"]["correct"])
    rck_by_k = acc_by_k(lambda r: r["rck_discover"]["correct"])
    rck_path_match_by_k = acc_by_k(lambda r: r["rck_discover"]["path_matches_edges"])
    rck_found_by_k = acc_by_k(lambda r: r["rck_discover"]["found_path"])
    reason_by_k = acc_by_k(lambda r: r["rck_reason_ablation"]["correct_entity"])

    sym_total = sum(1 for r in rows if r["symbolic"]["correct"])
    rck_total = sum(1 for r in rows if r["rck_discover"]["correct"])
    reason_total = sum(1 for r in rows if r["rck_reason_ablation"]["correct_entity"])
    n = len(rows)

    print("\n" + "-" * 78)
    print(" PER-k ACCURACY  (endpoint-relation inference)")
    print("-" * 78)
    header = f"{'k':>3}  {'n':>4}  {'symbolic control':>18}  {'RCK discover()':>16}  " \
             f"{'RCK path==edges':>16}  {'RCK reason() entity':>20}"
    print(header)
    for k in K_VALUES:
        s = sym_by_k.get(k, {"n": 0, "accuracy": 0.0})
        r = rck_by_k.get(k, {"n": 0, "accuracy": 0.0})
        pm = rck_path_match_by_k.get(k, {"n": 0, "accuracy": 0.0})
        ra = reason_by_k.get(k, {"n": 0, "accuracy": 0.0})
        print(f"{k:>3}  {s['n']:>4}  {s['accuracy'] * 100:>16.1f}%  "
              f"{r['accuracy'] * 100:>14.1f}%  {pm['accuracy'] * 100:>14.1f}%  "
              f"{ra['accuracy'] * 100:>18.1f}%")

    print("-" * 78)
    print(f"OVERALL  symbolic control: {sym_total}/{n} = {sym_total / n * 100:.1f}%"
          f"   (harness sanity check)")
    print(f"OVERALL  RCK discover()+compose: {rck_total}/{n} = {rck_total / n * 100:.1f}%"
          f"   (primary metric)")
    print(f"OVERALL  RCK reason() entity-retrieval (given true relations, ablation): "
          f"{reason_total}/{n} = {reason_total / n * 100:.1f}%")
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    out = {
        "provenance": "CLUTRR-style, locally generated. NOT the official "
                       "CLUTRR dataset. Scores are NOT comparable to "
                       "published CLUTRR numbers.",
        "seed": SEED,
        "k_values": list(K_VALUES),
        "max_per_pattern": MAX_PER_PATTERN,
        "n_examples": n,
        "examples_per_k": dict(sorted(by_k_count.items())),
        "symbolic_control_by_k": sym_by_k,
        "symbolic_control_overall": sym_total / n if n else 0.0,
        "rck_discover_by_k": rck_by_k,
        "rck_discover_overall": rck_total / n if n else 0.0,
        "rck_path_matches_true_edges_by_k": rck_path_match_by_k,
        "rck_discover_found_any_path_by_k": rck_found_by_k,
        "rck_reason_ablation_entity_by_k": reason_by_k,
        "rck_reason_ablation_overall": reason_total / n if n else 0.0,
        "rows": rows,
    }
    out_path = Path("data/clutrr_style_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
