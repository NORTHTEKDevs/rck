import asyncio

import pytest

pytest.importorskip("mcp", reason="MCP tests need the [mcp] extra")

from mcp.server.fastmcp import FastMCP

from rck.agent import RCKAgent
import rck.mcp_server as mcp_mod


@pytest.fixture
def agent():
    a = RCKAgent(hv_dim=128, n_columns=1, reservoir_dim=16, n_clauses=4,
                 vocab_size=16, fep_rank=8, bigram_order=1, seed=0)
    a.observe("abcabcabc", learn=True)
    mcp_mod._AGENT = a
    yield a
    mcp_mod._AGENT = None


def test_mcp_tools_registered(agent):
    s = mcp_mod.build(FastMCP("rck-test"))
    tools = asyncio.run(s.list_tools())
    names = {t.name for t in tools}
    assert {
        "rck_observe", "rck_generate", "rck_reset_temporal",
        "rck_state", "rck_one_shot", "rck_explain",
        "rck_save", "rck_load",
    } <= names


def test_mcp_state_tool_returns_dict(agent):
    s = mcp_mod.build(FastMCP("rck-test"))
    result = asyncio.run(s.call_tool("rck_state", {}))
    # FastMCP returns (content_blocks, structured_dict) for newer versions,
    # or just content_blocks for older. Accept both.
    state = None
    if isinstance(result, tuple) and len(result) == 2:
        _, state = result
    else:
        # Extract from content blocks.
        import json
        for block in result:
            text = getattr(block, "text", None)
            if text:
                try:
                    state = json.loads(text)
                    break
                except Exception:
                    pass
    assert state is not None
    assert "codebook_size" in state
    assert state["version"] == "1.0.0"


def test_mcp_one_shot_grows_codebook(agent):
    s = mcp_mod.build(FastMCP("rck-test"))
    before = agent.codebook.size()
    asyncio.run(s.call_tool("rck_one_shot", {"symbol": "Z"}))
    assert agent.codebook.size() == before + 1
