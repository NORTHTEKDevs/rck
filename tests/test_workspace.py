import numpy as np

from rck.vsa import random_hv
from rck.workspace import GlobalWorkspace


def test_winner_takes_all():
    ws = GlobalWorkspace(dim=512)
    rng = np.random.default_rng(0)
    a = random_hv(512, rng)
    b = random_hv(512, rng)
    ws.step({"a": (a, 0.1), "b": (b, 0.9)})
    name, score = ws.last_winner()
    assert name == "b"
    assert score == 0.9


def test_workspace_accumulates_winners():
    ws = GlobalWorkspace(dim=512, decay=0.5)
    rng = np.random.default_rng(1)
    a = random_hv(512, rng)
    ws.step({"a": (a, 1.0)})
    state_after_one = ws.state().copy()
    ws.step({"a": (a, 1.0)})
    state_after_two = ws.state().copy()
    # Workspace state should be close to `a` direction after repeated wins.
    # Use cosine via float cast.
    s1 = state_after_one.astype(np.float32)
    s2 = state_after_two.astype(np.float32)
    af = a.astype(np.float32)
    cos1 = float(np.dot(s1, af) / (np.linalg.norm(s1) * np.linalg.norm(af)))
    cos2 = float(np.dot(s2, af) / (np.linalg.norm(s2) * np.linalg.norm(af)))
    assert cos1 > 0.5
    assert cos2 > 0.5
