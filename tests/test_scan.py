from examples.scan_lite import BigramBaseline, RCKCompositional, make_split


def test_rck_compositional_perfect_on_scan_lite_split():
    train, test = make_split()
    rck = RCKCompositional(dim=4096, seed=0)
    rck.fit(train)
    correct = sum(1 for cmd, gold in test if rck.predict(cmd) == gold)
    assert correct == len(test), f"RCK got {correct}/{len(test)} -- compositional gen broke"


def test_bigram_baseline_fails_compositional_split():
    """The bigram baseline serves as a 'simplest LLM' control. It should NOT
    solve the compositional split, otherwise the benchmark is too easy."""
    train, test = make_split()
    bg = BigramBaseline()
    bg.fit(train)
    correct = sum(1 for cmd, gold in test if bg.predict(cmd) == gold)
    assert correct < len(test) * 0.25, (
        f"bigram baseline solved {correct}/{len(test)} -- the SCAN split is not hard enough"
    )
