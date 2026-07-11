# 7. FAQ

The honest answers about what RCK is and isn't.

## Is this a language model?

No. RCK doesn't generate freeform text the way GPT does. It stores
discrete (s, r, o) facts in a vector substrate and reasons over them
explicitly. There IS a small optional `rck.polisher` transformer that
renders structured answers as English sentences, but it's a thin
surface layer — the brains are in the substrate, not the LM.

## Can it answer "open" questions like "tell me a story about a dragon"?

Not well. RCK is good at structured knowledge and reasoning. For
creative or narrative text, you want an LLM. For "does X imply Y in
my knowledge base?", RCK beats every LLM at correctness and cost.

## How big can the knowledge base get?

Auto-sharding (`recommend_shards`) keeps each shard under the capacity
cliff (~80 facts/shard at D=4096). At 1024 shards you have ~80k facts;
at 4096 shards ~320k. Beyond that, you'd want a larger D (8192 or
16384) for capacity headroom. The bundled benchmarks tested up to
~4000 facts; the math says millions are reachable on a laptop and
billions with sharding across machines, but those haven't been
publicly benchmarked.

## Does it hallucinate?

It cannot hallucinate POSITIVELY in the LLM sense — there's no
generative model fabricating facts. But the HRR substrate has
crosstalk noise, so the cosine-similarity cleanup can RANK an answer
above the true answer when shards are overloaded. The capacity-cliff
filters and the `ask_with_idk` thresholds catch most of this. The
filter stack on chain induction (inverse-pair / non-transitive / lifting /
cycle) catches the rest. The agent says "I don't know" before it
makes things up.

## How do I feed it the internet?

You don't, out of the box. The bundled Open IE extractor is
rule-based and limited. To feed it Wikipedia or larger corpora, plug
in a better triple extractor (could be a small local LLM you use ONCE
per document, then never again). Ingestion is then a one-time pass.
After that the agent runs on a laptop forever, cost ~$0.

## Cost vs LLMs?

* **Build cost**: ~$200 (training a small polisher on a few rented
  GPU hours) to ~$3000 (paying an existing LLM API to do the triple
  extraction for a Wikipedia-scale ingestion).
* **Run cost**: ~$0. Single laptop CPU.

Compare to GPT-4-class training: ~$100M. The cost gap is 4-5 orders
of magnitude.

## Why CPU only?

The HRR substrate is dominated by O(D) elementwise operations on
small (~10s of MB) tensors. There's no benefit to GPU for the
typical workload. The optional `rck.polisher` transformer DOES use
PyTorch and can use a GPU at training time.

## Is this neuro-symbolic AI?

Yes. The vectors (neural-ish) provide fuzzy matching and approximate
retrieval; the relational structure (symbolic) provides discrete
facts you can edit, audit, and reason over. The combination is the
whole point.

## How does it compare to NARS?

Both NARS and RCK are non-axiomatic, evidence-based reasoners with
explicit confidence. NARS uses a fixed term-logic syntax; RCK uses
HRR + an open-vocabulary triple format. NARS has a more mature
inference rule set; RCK's chain walker + filter stack is younger but
empirically tested. They could probably learn from each other.

## How does it compare to neural-symbolic systems like DeepProbLog?

Those embed neural perception into logical programs. RCK does the
opposite: it embeds logic INTO a neural-shaped substrate. Different
trade-offs. RCK is simpler to operate (no Prolog interpreter to wire
up), DeepProbLog is more expressive for complex logical programs.

## How does it compare to retrieval-augmented LLMs (RAG)?

RAG retrieves documents and feeds them to a generator that may still
hallucinate. RCK retrieves DISCRETE FACTS and reasons over them
explicitly, with provenance. No hallucination layer. RAG produces
text; RCK produces auditable triples (then optionally polishes them
to text via the small polisher).

## I want to use this commercially. Can I?

Yes — MIT license. No restrictions. Credit appreciated but not
required.

## I found a bug. Where do I file it?

`https://github.com/NORTHTEKDevs/rck/issues`. Failing pytest cases
are the best bug reports.

## I want to contribute. Where do I start?

See `CONTRIBUTING.md`. The codebase is ~18.7k lines of plain numpy
Python across small single-purpose modules; the core substrate
(`knowledge_base.py`, `relational.py`, `codebook.py`) is an
afternoon's read, the full stack a weekend's. The filter stack on
derivation is load-bearing — keep it intact.

## Why is it called "Resonant Cognitive Kernel"?

The "resonant" alludes to the HRR substrate (vectors that resonate
when bound; cleanup as resonance to the nearest atom). "Cognitive"
because the agent has self-model + introspection. "Kernel" in the
mathematical / OS sense — a small core that other systems can build
on. Don't read too much into it. The name is older than the current
architecture.
