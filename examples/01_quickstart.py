"""01 - Quickstart: the shortest path to a useful RCK agent.

Five minutes from `pip install rck` to seeing the agent
(1) answer a known question, (2) say "I don't know", (3) reason
across multiple hops, (4) explain why it knows what it knows.

Run:
    python examples/01_quickstart.py
"""
from rck.conscious_agent import ConsciousAgent


def main() -> None:
    agent = ConsciousAgent(expected_facts=100, install_self=False)

    # ---- 1. Tell the agent some facts ----
    print("Telling the agent some facts...")
    agent.tell("paris", "capital_of", "france")
    agent.tell("france", "locatedin", "europe")
    agent.tell("europe", "isa", "continent")
    agent.tell("dog", "isa", "mammal")
    agent.tell("mammal", "isa", "animal")
    agent.tell("dog", "has", "fur")

    # ---- 2. Ask known questions ----
    print("\n== Direct questions ==")
    res = agent.ask_with_idk({"S": "paris", "R": "capital_of"}, "O")
    print(f"capital_of paris -> {res.verbalize()}")

    res = agent.ask_with_idk({"S": "dog", "R": "has"}, "O")
    print(f"dog has -> {res.verbalize()}")

    # ---- 3. Ask something the agent doesn't know ----
    print("\n== \"I don't know\" ==")
    res = agent.ask_with_idk({"S": "narwhal", "R": "capital"}, "O")
    print(f"capital of narwhal? -> {res.verbalize()}")

    # ---- 4. Multi-hop reasoning ----
    print("\n== Multi-hop chain reasoning ==")
    spec = agent.discover("paris", "continent", max_depth=4)
    if spec:
        print(f"discovered chain: {' -> '.join(spec['relations'])}")
        res = agent.reason("paris", spec["relations"],
                           directions=spec["directions"])
        print(f"  answer: {res['answer']}  ({res['hedge']})")

    # ---- 5. Induce a new direct fact ----
    print("\n== Learn by deriving ==")
    induced = agent.induce("dog", "animal")
    if induced is not None and induced.verified:
        print(f"derived: ({induced.subject}, {induced.relation}, "
              f"{induced.obj})")

    # ---- 6. Explain WHY the agent knows something ----
    print("\n== Explainability ==")
    if induced is not None and induced.verified:
        node = agent.explain_why(induced.subject, induced.relation,
                                  induced.obj)
        print(node.verbalize())

    # ---- 7. Negate a fact ----
    print("\n== Denial: positive certainty about non-membership ==")
    agent.deny("dog", "isa", "fish")
    candidates = agent.knowledge.query(
        {"S": "dog", "R": "isa"}, "O", top_k=5,
    )
    filtered = agent.filter_negatives("dog", "isa", candidates)
    print(f"after filtering: top candidates for 'dog isa ?' "
          f"-> {[str(s) for s, _ in filtered[:3]]}")

    # ---- 8. Status snapshot ----
    print("\n== Status ==")
    print(agent.status_report())


if __name__ == "__main__":
    main()
