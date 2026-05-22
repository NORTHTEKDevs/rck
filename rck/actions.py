"""Tool / action registry.

RCK doesn't need to LEARN to use tools -- it has explicit hooks. A tool
is a Python callable with a name + JSON-schema-ish args definition.
When the user asks something the KB can't answer, the agent tries
matching the question to a registered tool.

Bundled tools:
  - `calculator`: arithmetic via rck.numbers.evaluate_arithmetic.
  - `time_now`: current time (deterministic, no IO).
  - `length_of`: character-count of a string.

External tools can be registered by user code.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    pattern: re.Pattern[str] | None = None   # optional regex to detect invocation


@dataclass
class ActionRegistry:
    """Holds tools the agent can invoke."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, name: str, description: str,
                 func: Callable[..., Any],
                 pattern: str | None = None) -> None:
        self.tools[name] = Tool(
            name=name, description=description, func=func,
            pattern=re.compile(pattern, re.IGNORECASE) if pattern else None,
        )

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description}
                for t in self.tools.values()]

    def match(self, text: str) -> tuple[Tool, dict] | None:
        """Find a tool whose pattern matches `text`. Returns (tool, groupdict)."""
        for tool in self.tools.values():
            if tool.pattern is None:
                continue
            m = tool.pattern.match(text)
            if m:
                return tool, m.groupdict()
        return None

    def invoke(self, name: str, *args, **kwargs) -> Any:
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        return self.tools[name].func(*args, **kwargs)


# ---------------------------------------------------------------------------
#  Bundled tools
# ---------------------------------------------------------------------------

def _tool_calculator(expression: str) -> dict:
    """Evaluate a simple arithmetic expression."""
    from rck.numbers import evaluate_arithmetic
    res = evaluate_arithmetic(expression)
    if res is None:
        return {"ok": False, "error": "could not parse expression"}
    return {"ok": True, **res}


def _tool_time_now() -> dict:
    """Current Unix epoch + ISO timestamp."""
    t = time.time()
    return {"epoch": t, "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))}


def _tool_length(s: str) -> dict:
    return {"length": len(s), "verbal": f"The string is {len(s)} characters long."}


def make_default_registry() -> ActionRegistry:
    r = ActionRegistry()
    r.register(
        "calculator",
        "Evaluate arithmetic expressions like '5 + 3' or 'what is 12 * 4'",
        _tool_calculator,
        pattern=r"^(?:what\s+is\s+)?-?\d+(?:\.\d+)?\s*[+\-*/]\s*-?\d+",
    )
    r.register(
        "time_now",
        "Return the current time as a Unix epoch + ISO string",
        _tool_time_now,
        pattern=r"^what\s+time\s+is\s+it\b",
    )
    r.register(
        "length_of",
        "Return the character length of a string",
        _tool_length,
        pattern=r"^how\s+long\s+is\s+(?P<s>.+)$",
    )
    return r
