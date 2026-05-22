"""HTTP server + web chat UI for GenerativeRCK.

Standalone server (separate from rck/server.py which is the lower-level
char-LM RCKAgent server). Endpoints:

  GET  /              chat UI
  POST /api/tell      {subject, relation, object}
  POST /api/teach     {text}
  POST /api/ask       {question} -> {answer, source, confidence, candidates}
  POST /api/generate  {prompt, max_words, temperature} -> {text}
  POST /api/forget    {subject, relation, object}
  GET  /api/state     model summary

Run:
  python -m rck.gen_server --port 7860 [--bootstrap]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rck.generative import GenerativeRCK


_AGENT: GenerativeRCK | None = None
_LOCK = threading.Lock()


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>RCK Generative</title>
<style>
:root { --bg:#0b0d10; --panel:#15181d; --border:#232830; --text:#e6e8ec;
        --muted:#8b919b; --accent:#0a84ff; --ok:#34d399; --warn:#f5a524; }
*{box-sizing:border-box}
body{font-family:-apple-system,"Inter",sans-serif;background:var(--bg);
     color:var(--text);margin:0;padding:16px}
h1{font-size:16px;font-weight:600;margin:0 0 12px}
h2{font-size:11px;font-weight:600;margin:0 0 8px;text-transform:uppercase;
   letter-spacing:.08em;color:var(--muted)}
.layout{display:grid;grid-template-columns:1.5fr 1fr;gap:16px;
        height:calc(100vh - 60px)}
.col{display:flex;flex-direction:column;gap:12px;min-height:0}
.panel{background:var(--panel);border:1px solid var(--border);
       border-radius:8px;padding:12px}
.chat{flex:1;overflow-y:auto;font-family:ui-monospace,monospace;
      font-size:13px;min-height:0}
.row{padding:8px 0;border-bottom:1px solid var(--border)}
.role{color:var(--muted);font-size:10px;text-transform:uppercase;
      letter-spacing:.08em;margin-right:8px}
.you .role{color:var(--accent)}
.rck .role{color:var(--warn)}
.sys .role{color:var(--ok)}
.src{font-size:10px;color:var(--muted);margin-top:2px;
     font-family:ui-monospace,monospace}
input[type=text],textarea{background:var(--bg);border:1px solid var(--border);
       border-radius:6px;color:var(--text);padding:8px 10px;
       font-family:ui-monospace,monospace;font-size:13px;width:100%;outline:none}
input[type=text]:focus,textarea:focus{border-color:var(--accent)}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;
       padding:8px 14px;font-weight:600;font-size:13px;cursor:pointer}
button.ghost{background:transparent;color:var(--text);
             border:1px solid var(--border)}
.tabbar{display:flex;gap:6px;margin-bottom:8px}
.tab{padding:6px 12px;border-radius:6px;border:1px solid var(--border);
     cursor:pointer;font-size:12px}
.tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 12px;
    font-size:12px;font-family:ui-monospace,monospace}
.kv .k{color:var(--muted)}
.cand{padding:4px 0;font-family:ui-monospace,monospace;font-size:11px;
      color:#b9c0ca}
.cand .score{color:var(--muted);margin-left:8px}
.row-actions{display:flex;gap:6px;margin-top:6px}
.example{font-size:11px;color:var(--muted);margin-top:4px;font-style:italic}
</style></head><body>
<h1>RCK &middot; Generative AI</h1>
<div class="layout">
  <div class="col">
    <div class="panel chat" id="chat"></div>
    <div class="panel">
      <div class="tabbar">
        <div class="tab active" data-mode="ask">Ask</div>
        <div class="tab" data-mode="teach">Teach</div>
        <div class="tab" data-mode="tell">Tell (S,R,O)</div>
        <div class="tab" data-mode="generate">Generate</div>
      </div>
      <input id="prompt" type="text" placeholder="ask RCK a question and hit enter" autofocus>
      <div class="example" id="example">e.g. "What color is the sky?"</div>
      <div class="row-actions">
        <button id="send">Send</button>
        <button class="ghost" id="clear-chat">clear chat</button>
        <span class="src" id="hint"></span>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="panel">
      <h2>State</h2>
      <div class="kv" id="state"></div>
    </div>
    <div class="panel" style="flex:1;overflow-y:auto;min-height:0">
      <h2>Last answer breakdown</h2>
      <div id="breakdown"></div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let mode = "ask";

const PLACEHOLDERS = {
  ask:      'ask a question, e.g. "What color is the sky?"',
  teach:    'paste a sentence to learn from, e.g. "The sky is blue."',
  tell:     'subject relation object   e.g. "sky color blue"',
  generate: 'a prompt to continue freely, e.g. "the king is"',
};
const EXAMPLES = {
  ask:      'e.g. "What is the capital of France?"',
  teach:    'e.g. "Shakespeare wrote Hamlet."',
  tell:     'e.g. "fox color orange"',
  generate: 'free-form character continuation (best after lots of training)',
};

document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    mode = t.dataset.mode;
    $('prompt').placeholder = PLACEHOLDERS[mode];
    $('example').textContent = EXAMPLES[mode];
  });
});

async function api(path, body) {
  const opts = body ? {method:'POST',headers:{'Content-Type':'application/json'},
                       body:JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  return r.json();
}

function append(role, content, src) {
  const div = document.createElement('div');
  div.className = 'row ' + role;
  let html = `<span class="role">${role}</span>${escapeHtml(content)}`;
  if (src) html += `<div class="src">${escapeHtml(src)}</div>`;
  div.innerHTML = html;
  $('chat').appendChild(div);
  $('chat').scrollTop = $('chat').scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

async function send() {
  const text = $('prompt').value.trim();
  if (!text) return;
  $('prompt').value = '';

  if (mode === "tell") {
    const parts = text.split(/\s+/);
    if (parts.length < 3) { append('sys', 'usage: subject relation object'); return; }
    const [s, r] = [parts[0], parts[1]];
    const o = parts.slice(2).join(' ');
    append('you', `tell ${s} ${r} ${o}`);
    const res = await api('/api/tell', {subject:s, relation:r, object:o});
    append('sys', `stored. memory has ${res.fact_count} facts.`);
    refreshState();
    return;
  }
  if (mode === "teach") {
    append('you', text, '(teach)');
    const res = await api('/api/teach', {text});
    append('sys', `learned. tokens_ingested=${res.tokens_ingested} new_facts=${res.new_facts}`);
    refreshState();
    return;
  }
  if (mode === "generate") {
    append('you', text, '(generate)');
    const res = await api('/api/generate',
                          {prompt:text, max_words: 20, temperature: 0.4});
    append('rck', res.text);
    refreshState();
    return;
  }
  // ask
  append('you', text);
  const res = await api('/api/ask', {question:text});
  append('rck', res.answer, `source=${res.source}, conf=${res.confidence.toFixed(2)}`);
  renderBreakdown(res);
  refreshState();
}

function renderBreakdown(res) {
  const box = $('breakdown');
  box.innerHTML = '';
  const meta = document.createElement('div');
  meta.className = 'kv';
  meta.innerHTML = `
    <span class="k">answer</span><span>${escapeHtml(res.answer)}</span>
    <span class="k">source</span><span>${escapeHtml(res.source)}</span>
    <span class="k">confidence</span><span>${res.confidence.toFixed(3)}</span>
    <span class="k">parsed</span><span>${escapeHtml(JSON.stringify(res.parsed || {}))}</span>
  `;
  box.appendChild(meta);
  if (res.candidates && res.candidates.length) {
    const hdr = document.createElement('h2');
    hdr.textContent = "Candidates";
    hdr.style.marginTop = "12px";
    box.appendChild(hdr);
    for (const [sym, score] of res.candidates.slice(0, 8)) {
      const d = document.createElement('div');
      d.className = 'cand';
      d.innerHTML = `${escapeHtml(sym)}<span class="score">cos=${score.toFixed(3)}</span>`;
      box.appendChild(d);
    }
  }
}

async function refreshState() {
  const s = await api('/api/state');
  $('state').innerHTML = `
    <span class="k">version</span><span>${s.version}</span>
    <span class="k">facts</span><span>${s.fact_count}</span>
    <span class="k">tokens</span><span>${s.tokens_ingested}</span>
    <span class="k">codebook</span><span>${s.codebook_size}</span>
    <span class="k">memory</span><span>${s.memory_facts}</span>
  `;
}

$('send').onclick = send;
$('prompt').addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
$('clear-chat').onclick = () => { $('chat').innerHTML = ''; $('breakdown').innerHTML = ''; };
refreshState();
</script>
</body></html>
"""


def _json(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    n = int(handler.headers.get("Content-Length") or 0)
    if not n:
        return {}
    return json.loads(handler.rfile.read(n))


class GenHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            page = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif path == "/api/state":
            with _LOCK:
                _json(self, 200, _AGENT.state())
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        body = _read_json(self)
        with _LOCK:
            try:
                if path == "/api/tell":
                    _AGENT.tell(body["subject"], body["relation"], body["object"])
                    _json(self, 200, {"ok": True, **_AGENT.state()})
                elif path == "/api/teach":
                    info = _AGENT.ingest(body.get("text", ""))
                    _json(self, 200, info)
                elif path == "/api/ask":
                    res = _AGENT.ask(body.get("question", ""))
                    _json(self, 200, res)
                elif path == "/api/generate":
                    txt = _AGENT.generate(
                        body.get("prompt", ""),
                        max_words=int(body.get("max_words", 20)),
                        temperature=float(body.get("temperature", 0.4)),
                    )
                    _json(self, 200, {"text": txt})
                elif path == "/api/forget":
                    _AGENT.memory.forget(_AGENT.codebook, {
                        "S": body["subject"], "R": body["relation"], "O": body["object"],
                    })
                    _json(self, 200, {"ok": True})
                else:
                    _json(self, 404, {"error": "not found"})
            except Exception as exc:
                _json(self, 500, {"error": str(exc)})


def serve(agent: GenerativeRCK, host: str = "127.0.0.1", port: int = 7860) -> None:
    global _AGENT
    _AGENT = agent
    s = ThreadingHTTPServer((host, port), GenHandler)
    print(f"RCK Generative serving on http://{host}:{port}")
    print(f"  facts={agent.memory.size()}  codebook={agent.codebook.size()}")
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        s.server_close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rck.gen_server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--bootstrap", action="store_true",
                   help="preload data/world_knowledge.txt if present")
    p.add_argument("--hv-dim", type=int, default=4096)
    args = p.parse_args(argv)
    agent = GenerativeRCK(dim=args.hv_dim, seed=0)
    if args.bootstrap:
        path = Path("data/world_knowledge.txt")
        if path.exists():
            info = agent.ingest(path.read_text(encoding="utf-8"))
            print(f"[bootstrap] {info}")
    serve(agent, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
