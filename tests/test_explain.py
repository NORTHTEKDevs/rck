from rck.agent import RCKAgent
from rck.compose import CompositionalReasoner
from rck.explain import explain_step, explain_composition, explain_generate


def test_explain_step_includes_emission_and_module():
    agent = RCKAgent(hv_dim=256, n_columns=2, reservoir_dim=16, n_clauses=4,
                     vocab_size=16, fep_rank=8, bigram_order=1, seed=0)
    agent.observe("hello", learn=True)
    tr = agent.step("h", learn=False)
    text = explain_step(tr)
    assert "I emitted" in text
    assert str(tr.emitted_symbol) in text


def test_explain_composition_lists_each_slot():
    cr = CompositionalReasoner(dim=2048, seed=0)
    cr.teach_pair("color", "red", "R")
    cr.teach_pair("shape", "ball", "o")
    text = explain_composition(cr, {"color": "red", "shape": "ball"})
    assert "color" in text and "red" in text and "R" in text
    assert "shape" in text and "ball" in text and "o" in text
    assert "Composition trace" in text


def test_explain_generate_returns_string():
    agent = RCKAgent(hv_dim=256, n_columns=2, reservoir_dim=16, n_clauses=4,
                     vocab_size=16, fep_rank=8, bigram_order=1, seed=0)
    agent.observe("abc", learn=True)
    text = explain_generate(agent, "a", max_new=3)
    assert "Prompt:" in text
    assert "Emitted:" in text
