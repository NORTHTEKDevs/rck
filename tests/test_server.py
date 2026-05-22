import json
import threading
import time
import urllib.request

from rck.agent import RCKAgent
from rck.server import RCKHandler, _AGENT_LOCK, serve
from http.server import ThreadingHTTPServer


def _start_server(agent, port=17861):
    import rck.server as srv
    srv._AGENT = agent
    server = ThreadingHTTPServer(("127.0.0.1", port), RCKHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    return server


def _req(port, path, body=None):
    if body is None:
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    else:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_server_lifecycle_index_and_state_and_generate_and_reset():
    agent = RCKAgent(hv_dim=128, n_columns=2, reservoir_dim=16, n_clauses=4,
                     vocab_size=16, fep_rank=8, bigram_order=1, seed=0)
    agent.observe("abc", learn=True)

    port = 17861
    server = _start_server(agent, port=port)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            html = r.read().decode()
            assert "RCK" in html and "Resonant" in html

        state = _req(port, "/api/state")
        assert state["codebook_size"] >= 3
        assert state["hv_dim"] == 128

        gen = _req(port, "/api/generate", {"prompt": "a", "max_new": 4, "temperature": 0.0})
        assert "emitted" in gen
        assert isinstance(gen["emitted"], str)

        obs = _req(port, "/api/observe", {"text": "xyz"})
        assert obs["steps"] == 3

        reset = _req(port, "/api/reset", {})
        assert reset["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
