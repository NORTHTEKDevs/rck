"""Dependency-free HTTP server for RCK.

Exposes a tiny REST API + a chat web UI. No Flask, no FastAPI -- pure
stdlib so RCK ships without extra dependencies.

Endpoints:
  GET  /            -- chat web UI
  POST /api/observe -- {"text": "..."} -> learn from text
  POST /api/generate -- {"prompt": "...", "max_new": 40, "temperature": 0.5}
                        -> {"emitted": [...], "trace": {...}}
  POST /api/reset   -- clear temporal state
  GET  /api/state   -- model summary (codebook size, position, etc.)
  POST /api/save    -- {"path": "..."} -> persist
  POST /api/load    -- {"path": "..."} -> swap in new agent

Run with:
  python -m rck.server --port 7860 [--load checkpoints/rck_100k]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rck.agent import RCKAgent
from rck.persist import load as persist_load, save as persist_save


# Global mutable handle to the current agent; the request handler reads it.
_AGENT: RCKAgent | None = None
_AGENT_LOCK = threading.Lock()


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RCK -- Resonant Cognitive Kernel</title>
<style>
:root {
  --bg: #0b0d10;
  --panel: #15181d;
  --border: #232830;
  --text: #e6e8ec;
  --muted: #8b919b;
  --accent: #0A84FF;
  --rev: #1f2730;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, "Inter", sans-serif; background: var(--bg);
       color: var(--text); margin: 0; padding: 16px; }
h1 { font-size: 16px; font-weight: 600; margin: 0 0 12px 0; letter-spacing: -0.01em; }
h2 { font-size: 11px; font-weight: 600; margin: 0 0 8px 0; text-transform: uppercase;
     letter-spacing: 0.08em; color: var(--muted); }
.layout { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; height: calc(100vh - 60px); }
.col { display: flex; flex-direction: column; gap: 12px; min-height: 0; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
.chat { flex: 1; overflow-y: auto; font-family: ui-monospace, "JetBrains Mono", monospace; font-size: 13px;
        min-height: 0; }
.row { padding: 6px 0; border-bottom: 1px solid var(--border); }
.role { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
        margin-right: 8px; }
.you .role { color: var(--accent); }
.rck .role { color: #f5a524; }
.input-row { display: flex; gap: 8px; }
input[type=text], textarea {
  background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
  color: var(--text); padding: 8px 10px; font-family: ui-monospace, monospace; font-size: 13px;
  width: 100%; outline: none;
}
input[type=text]:focus, textarea:focus { border-color: var(--accent); }
button {
  background: var(--accent); color: white; border: 0; border-radius: 6px;
  padding: 8px 14px; font-weight: 600; font-size: 13px; cursor: pointer;
}
button:hover { filter: brightness(1.1); }
button.ghost { background: transparent; color: var(--text); border: 1px solid var(--border); }
.controls { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; font-size: 12px; font-family: ui-monospace, monospace; }
.kv .k { color: var(--muted); }
.clause { font-family: ui-monospace, monospace; font-size: 11px; padding: 3px 0; color: #b9c0ca; }
.clause .pos { color: #34d399; }
.clause .neg { color: #f87171; }
.flex-grow { flex: 1; min-height: 0; overflow-y: auto; }
label { color: var(--muted); font-size: 11px; margin-right: 4px; }
.warn { color: #f5a524; font-size: 11px; }
.dim { color: var(--muted); }
</style>
</head>
<body>
<h1>RCK &middot; Resonant Cognitive Kernel</h1>
<div class="layout">
  <div class="col">
    <div class="panel chat" id="chat"></div>
    <div class="panel">
      <div class="input-row">
        <input id="prompt" type="text" placeholder="say something to RCK and hit enter" autofocus>
        <button id="send">Send</button>
      </div>
      <div class="controls" style="margin-top:10px">
        <label>max_new</label>
        <input id="max_new" type="text" value="60" style="width:60px">
        <label>T</label>
        <input id="temp" type="text" value="0.0" style="width:60px">
        <button class="ghost" id="why">why</button>
        <button class="ghost" id="reset">reset state</button>
        <button class="ghost" id="teach">teach</button>
        <span class="dim" id="state-summary"></span>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="panel">
      <h2>workspace + signals</h2>
      <div class="kv" id="signals"></div>
    </div>
    <div class="panel flex-grow">
      <h2>Tsetlin reasoning clauses (most recent step)</h2>
      <div id="clauses"></div>
    </div>
    <div class="panel">
      <h2>model</h2>
      <div class="kv" id="model"></div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let lastTrace = null;

async function api(path, body) {
  const opts = body ? { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

function append(role, text) {
  const div = document.createElement('div');
  div.className = 'row ' + role;
  div.innerHTML = `<span class="role">${role}</span>${escapeHtml(text)}`;
  $('chat').appendChild(div);
  $('chat').scrollTop = $('chat').scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function renderTrace(tr) {
  lastTrace = tr;
  $('signals').innerHTML = `
    <span class="k">winner</span><span>${tr.winner ?? '—'}</span>
    <span class="k">winner_score</span><span>${(tr.winner_score ?? 0).toFixed(3)}</span>
    <span class="k">column_unc</span><span>${(tr.uncertainty ?? 0).toFixed(4)}</span>
    <span class="k">tsetlin_score</span><span>${(tr.tsetlin_score ?? 0).toFixed(2)}</span>
    <span class="k">pred_err</span><span>${(tr.pred_err ?? 0).toFixed(3)}</span>
    <span class="k">bigram_top</span><span>${(tr.bigram_top || []).map(x=>x[0]).join(' ')}</span>
  `;
  const clausesEl = $('clauses');
  clausesEl.innerHTML = '';
  for (const c of (tr.clauses || [])) {
    const el = document.createElement('div');
    el.className = 'clause';
    const polarity = c.startsWith('(+)') ? 'pos' : 'neg';
    el.innerHTML = `<span class="${polarity}">${c[1]}</span> ${escapeHtml(c.slice(4))}`;
    clausesEl.appendChild(el);
  }
  if ((tr.clauses || []).length === 0) {
    clausesEl.innerHTML = '<div class="dim">no clauses fired yet (need more training).</div>';
  }
}

async function send() {
  const prompt = $('prompt').value;
  if (!prompt) return;
  append('you', prompt);
  $('prompt').value = '';
  const max_new = parseInt($('max_new').value) || 40;
  const temp = parseFloat($('temp').value) || 0;
  try {
    const res = await api('/api/generate', { prompt, max_new, temperature: temp });
    append('rck', res.emitted);
    renderTrace(res.trace);
    refreshState();
  } catch(e) {
    append('rck', '[error] ' + e.message);
  }
}

async function refreshState() {
  try {
    const s = await api('/api/state');
    $('model').innerHTML = `
      <span class="k">vocab</span><span>${s.codebook_size}</span>
      <span class="k">hv_dim</span><span>${s.hv_dim}</span>
      <span class="k">columns</span><span>${s.n_columns}</span>
      <span class="k">reservoir</span><span>${s.reservoir_dim}</span>
      <span class="k">clauses</span><span>${s.n_clauses}</span>
      <span class="k">fep_rank</span><span>${s.fep_rank}</span>
      <span class="k">position</span><span>${s.position}</span>
    `;
    $('state-summary').textContent =
      `${s.codebook_size} symbols, position ${s.position}`;
  } catch(e) {}
}

$('send').onclick = send;
$('prompt').addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

$('why').onclick = () => {
  if (!lastTrace) { append('rck', '(nothing to explain yet)'); return; }
  append('rck', `winner=${lastTrace.winner} score=${(lastTrace.winner_score ?? 0).toFixed(3)} `
                + `unc=${(lastTrace.uncertainty ?? 0).toFixed(4)} `
                + `clauses=${(lastTrace.clauses || []).length}`);
};

$('reset').onclick = async () => {
  await api('/api/reset', {});
  append('rck', '[temporal state cleared]');
};

$('teach').onclick = async () => {
  const t = prompt('Teach RCK (text to learn from):');
  if (!t) return;
  const res = await api('/api/observe', { text: t });
  append('rck', `[learned ${res.steps} chars, codebook now ${res.codebook_size}]`);
  refreshState();
};

refreshState();
</script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if not length:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw)


class RCKHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Quiet by default.
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            page = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif path == "/api/state":
            with _AGENT_LOCK:
                a = _AGENT
                body = {
                    "codebook_size": a.codebook.size(),
                    "hv_dim": a.hv_dim,
                    "n_columns": a.n_columns,
                    "reservoir_dim": a.reservoir_dim,
                    "n_clauses": a.n_clauses,
                    "fep_rank": a.fep_rank,
                    "position": a._position,
                    "version": "1.0.0",
                }
            _json_response(self, 200, body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        global _AGENT
        path = urlparse(self.path).path
        body = _read_json(self)
        with _AGENT_LOCK:
            a = _AGENT
            try:
                if path == "/api/generate":
                    prompt = body.get("prompt", "")
                    max_new = int(body.get("max_new", 40))
                    temperature = float(body.get("temperature", 0.0))
                    a.stochastic_decode = temperature > 1e-4
                    a.fep.temperature = max(temperature, 1e-3)
                    out, traces = a.generate(list(prompt), max_new=max_new)
                    text = "".join(str(c) for c in out)
                    tr = traces[-1] if traces else None
                    trace_body = {
                        "winner": getattr(tr, "workspace_winner", None),
                        "winner_score": getattr(tr, "workspace_score", 0.0),
                        "uncertainty": getattr(tr, "column_uncertainty", 0.0),
                        "tsetlin_score": getattr(tr, "tsetlin_score", 0.0),
                        "clauses": list(getattr(tr, "tsetlin_clauses", []) or []),
                        "pred_err": getattr(tr, "pred_err", 0.0),
                        "bigram_top": list(getattr(tr, "bigram_top", []) or []),
                    } if tr else {}
                    _json_response(self, 200, {"emitted": text, "trace": trace_body})

                elif path == "/api/observe":
                    text = body.get("text", "")
                    if not text:
                        _json_response(self, 400, {"error": "missing text"}); return
                    a.observe(list(text), learn=True)
                    _json_response(self, 200, {"steps": len(text), "codebook_size": a.codebook.size()})

                elif path == "/api/reset":
                    a.reset_temporal()
                    _json_response(self, 200, {"ok": True})

                elif path == "/api/save":
                    p = body.get("path")
                    if not p:
                        _json_response(self, 400, {"error": "missing path"}); return
                    persist_save(a, p)
                    _json_response(self, 200, {"ok": True, "path": p})

                elif path == "/api/load":
                    p = body.get("path")
                    if not p:
                        _json_response(self, 400, {"error": "missing path"}); return
                    _AGENT = persist_load(p)
                    _json_response(self, 200, {"ok": True, "path": p})

                else:
                    _json_response(self, 404, {"error": "not found"})
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})


def serve(agent: RCKAgent, host: str = "127.0.0.1", port: int = 7860) -> None:
    global _AGENT
    _AGENT = agent
    server = ThreadingHTTPServer((host, port), RCKHandler)
    print(f"RCK serving on http://{host}:{port}")
    print(f"  codebook={agent.codebook.size()}  hv_dim={agent.hv_dim}  cols={agent.n_columns}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rck.server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--load", default=None, help="checkpoint path (without extension)")
    p.add_argument("--hv-dim", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=80)
    p.add_argument("--columns", type=int, default=2)
    p.add_argument("--reservoir", type=int, default=96)
    p.add_argument("--clauses", type=int, default=16)
    p.add_argument("--fep-rank", type=int, default=64)
    p.add_argument("--bigram-order", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.load and Path(args.load).with_suffix(".npz").exists():
        agent = persist_load(args.load)
        print(f"loaded {args.load}")
    else:
        agent = RCKAgent(
            vocab_size=args.vocab, hv_dim=args.hv_dim,
            n_columns=args.columns, reservoir_dim=args.reservoir,
            n_clauses=args.clauses, fep_rank=args.fep_rank,
            bigram_order=args.bigram_order, seed=args.seed,
        )
        print(f"fresh agent (no checkpoint provided)")

    serve(agent, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
