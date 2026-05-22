import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from rck.generative import GenerativeRCK
import rck.gen_server as srv


def _start(agent, port: int):
    srv._AGENT = agent
    server = ThreadingHTTPServer(("127.0.0.1", port), srv.GenHandler)
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


def test_gen_server_tell_then_ask():
    agent = GenerativeRCK(dim=2048, seed=0)
    server = _start(agent, port=17862)
    try:
        with urllib.request.urlopen("http://127.0.0.1:17862/", timeout=5) as r:
            assert "RCK" in r.read().decode()
        # tell
        res = _req(17862, "/api/tell",
                   {"subject": "sky", "relation": "color", "object": "blue"})
        assert res["fact_count"] >= 1
        # ask
        res = _req(17862, "/api/ask", {"question": "What color is the sky?"})
        assert res["answer"] == "blue"
        assert res["source"].startswith("structured")
    finally:
        server.shutdown(); server.server_close()


def test_gen_server_teach_then_ask():
    agent = GenerativeRCK(dim=2048, seed=0)
    server = _start(agent, port=17863)
    try:
        _req(17863, "/api/teach",
             {"text": "The sky is blue. The grass is green."})
        res = _req(17863, "/api/ask", {"question": "What color is the sky?"})
        assert res["answer"] == "blue"
    finally:
        server.shutdown(); server.server_close()
