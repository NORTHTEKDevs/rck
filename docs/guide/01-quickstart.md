# 1. Quickstart

The shortest path to a useful RCK agent.

## Install

```bash
pip install git+https://github.com/NORTHTEKDevs/rck
```

(The `rck` name on PyPI belongs to an unrelated bioinformatics
project; the distribution name is `rck-kernel`, the import name is
`rck`.)

Or for development from source:

```bash
git clone https://github.com/NORTHTEKDevs/rck
cd rck
pip install -e ".[dev]"
```

## Five lines that prove it works

```python
from rck.conscious_agent import ConsciousAgent

agent = ConsciousAgent()
agent.tell("paris", "capital_of", "france")
ans, _ = agent.knowledge.answer({"S": "paris", "R": "capital_of"}, "O")
print(ans)   # -> france
```

## Telling vs asking

RCK uses subject-relation-object triples. Everything you tell it is a
fact in that shape.

```python
agent.tell("france", "isa", "country")
agent.tell("france", "capital", "paris")
agent.tell("paris", "locatedin", "europe")
```

There are several ways to ask:

```python
# Direct top-1 answer.
ans, score = agent.knowledge.answer({"S": "france", "R": "capital"}, "O")

# With explicit IDK detection.
res = agent.ask_with_idk({"S": "france", "R": "capital"}, "O")
print(res.state.value, res.top_symbol)  # 'known' 'paris'

# Top-K candidates with calibrated scores.
candidates = agent.knowledge.query(
    {"S": "france", "R": "capital"}, "O", top_k=3,
)
```

## Multi-hop reasoning

When the answer isn't a single hop away, ask RCK to discover the chain.

```python
# Already told: france capital paris, paris locatedin europe.
spec = agent.discover("france", "europe", max_depth=3)
# spec is a dict with 'relations', 'directions', 'confidence', 'trace'

# Then walk it (also auto-registers as a skill the agent has learned).
res = agent.reason("france", spec["relations"], directions=spec["directions"])
print(res["answer"], res["confidence"], res["hedge"])
```

## "I don't know"

RCK won't make things up.

```python
res = agent.ask_with_idk({"S": "narwhal", "R": "capital"}, "O")
print(res.state.value)  # -> 'idk'
print(res.verbalize())  # -> "I don't know. ..."
```

This is the single most important behavioural difference from an LLM.

## Where to next

- [02-tell-and-ask](02-tell-and-ask.md) — fact shapes, relations,
  symmetrisation.
- [03-reasoning](03-reasoning.md) — chains, induction, rules.
- [04-explainability](04-explainability.md) — provenance and
  `explain_why`.
- [05-multi-agent](05-multi-agent.md) — merge, consensus, diff.
- [06-maintenance](06-maintenance.md) — `agent.maintain()`,
  persistence, pruning.
- [07-faq](07-faq.md) — what RCK is *not*, and other practicalities.
