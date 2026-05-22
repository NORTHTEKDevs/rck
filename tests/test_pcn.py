import numpy as np

from rck.pcn import PCNEncoder
from rck.vsa import cosine


def test_pcn_output_is_bipolar():
    pcn = PCNEncoder(input_dim=10, hidden_dims=(32, 32), hv_dim=512, seed=0)
    x = np.zeros(10, dtype=np.float32)
    x[3] = 1.0
    hv = pcn.encode(x, learn=False)
    assert set(np.unique(hv).tolist()) <= {-1, 1}
    assert hv.shape == (512,)


def test_pcn_repeated_input_settles_similar():
    pcn = PCNEncoder(input_dim=10, hidden_dims=(32, 32), hv_dim=1024, seed=1)
    x = np.zeros(10, dtype=np.float32)
    x[5] = 1.0
    h1 = pcn.encode(x, learn=False)
    h2 = pcn.encode(x, learn=False)
    # Without learning, deterministic input must give identical output.
    assert np.array_equal(h1, h2)


def test_pcn_different_inputs_diverge():
    pcn = PCNEncoder(input_dim=10, hidden_dims=(32, 32), hv_dim=2048, seed=2)
    xa = np.zeros(10, dtype=np.float32); xa[0] = 1
    xb = np.zeros(10, dtype=np.float32); xb[7] = 1
    ha = pcn.encode(xa, learn=False)
    hb = pcn.encode(xb, learn=False)
    assert cosine(ha, hb) < 0.9
