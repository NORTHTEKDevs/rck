from rck.bigram import BigramMemory
from rck.codebook import Codebook


def test_bigram_recovers_simple_pair():
    cb = Codebook(dim=2048, seed=0)
    bm = BigramMemory(dim=2048, order=1)
    seq = list("the quick brown fox the lazy dog the rain in spain")
    for i in range(len(seq) - 1):
        bm.observe(cb, seq[i], seq[i + 1])
    bm.reset_context()
    bm.observe(cb, "t", None)
    top = [s for s, _ in bm.query(cb, top_k=3)]
    assert "h" in top


def test_bigram_handles_unigram_top():
    cb = Codebook(dim=1024, seed=1)
    bm = BigramMemory(dim=1024, order=1)
    for c in "aaaaabb":
        bm.observe(cb, c, c)
    top = bm.unigram_top(k=2)
    assert top[0] == "a"


def test_bigram_order_2_uses_context():
    cb = Codebook(dim=2048, seed=2)
    bm = BigramMemory(dim=2048, order=2)
    seq = list("the cat sat on the mat " * 4)
    for i in range(len(seq) - 1):
        bm.observe(cb, seq[i], seq[i + 1])
    bm.reset_context()
    bm.observe(cb, "t", None)
    bm.observe(cb, "h", None)
    top = [s for s, _ in bm.query(cb, top_k=3)]
    assert "e" in top
