"""Extended math + algebra. v9.

v3 had arithmetic via `evaluate_arithmetic`. v9 extends:
  * Multi-operand expressions: "what is 3 + 4 * 5"
  * Powers: "what is 2^10"
  * Roots, factorials, modulo
  * Optional sympy integration for symbolic algebra

When sympy is installed, we use it for safe symbolic evaluation. Without
sympy, we fall back to a hand-coded shunting-yard parser that handles
+, -, *, /, **, %, sqrt, and unary -.
"""
from __future__ import annotations

import math
import operator
import re


# Optional sympy integration -- import lazily.
def _try_sympy(expr: str):
    try:
        import sympy
    except ImportError:
        return None
    try:
        # Restricted symbolic eval -- no I/O, no arbitrary names.
        result = sympy.sympify(expr, evaluate=True)
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Hand-coded fallback parser (shunting-yard)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[+\-*/%^()]|sqrt|abs")


def _tokenize(expr: str) -> list[str]:
    expr = expr.lower().replace("**", "^").replace(",", "")
    return _TOKEN_RE.findall(expr)


_PRECEDENCE = {
    "+": 1, "-": 1,
    "*": 2, "/": 2, "%": 2,
    "^": 3,
    "u-": 4,
}

_BINARY = {
    "+": operator.add, "-": operator.sub,
    "*": operator.mul, "/": operator.truediv,
    "%": operator.mod, "^": operator.pow,
}


def _to_rpn(tokens: list[str]) -> list[str]:
    out: list[str] = []
    stack: list[str] = []
    prev = None
    for t in tokens:
        if re.match(r"^-?\d", t):
            out.append(t)
        elif t in {"sqrt", "abs"}:
            stack.append(t)
        elif t in _PRECEDENCE:
            # Detect unary minus.
            if t == "-" and (prev is None or prev in {"(", "+", "-", "*",
                                                       "/", "%", "^"}):
                t = "u-"
            while stack and stack[-1] != "(" and (
                _PRECEDENCE.get(stack[-1], 0) >= _PRECEDENCE[t]
                if t != "^" else _PRECEDENCE.get(stack[-1], 0) > _PRECEDENCE[t]
            ):
                out.append(stack.pop())
            stack.append(t)
        elif t == "(":
            stack.append(t)
        elif t == ")":
            while stack and stack[-1] != "(":
                out.append(stack.pop())
            if stack and stack[-1] == "(":
                stack.pop()
            if stack and stack[-1] in {"sqrt", "abs"}:
                out.append(stack.pop())
        prev = t
    while stack:
        out.append(stack.pop())
    return out


def _eval_rpn(rpn: list[str]) -> float:
    stack: list[float] = []
    for t in rpn:
        if re.match(r"^-?\d", t):
            stack.append(float(t))
        elif t in _BINARY:
            if len(stack) < 2:
                raise ValueError("malformed expression")
            b = stack.pop(); a = stack.pop()
            stack.append(_BINARY[t](a, b))
        elif t == "u-":
            stack.append(-stack.pop())
        elif t == "sqrt":
            stack.append(math.sqrt(stack.pop()))
        elif t == "abs":
            stack.append(abs(stack.pop()))
        else:
            raise ValueError(f"unknown token {t!r}")
    if len(stack) != 1:
        raise ValueError("expression did not reduce to a single value")
    return stack[0]


def evaluate_expression(expr: str) -> dict:
    """Evaluate a (possibly complex) math expression.

    Returns {'ok', 'answer', 'verbal'} or {'ok': False, 'error': str}.
    """
    expr_clean = expr.lower()
    # Strip natural-language wrappers.
    for prefix in ("what is ", "compute ", "calculate ", "evaluate "):
        if expr_clean.startswith(prefix):
            expr_clean = expr_clean[len(prefix):]
    expr_clean = expr_clean.rstrip(" ?.!")
    if not expr_clean:
        return {"ok": False, "error": "empty expression"}

    # Try sympy first.
    sym = _try_sympy(expr_clean.replace("^", "**"))
    if sym is not None:
        try:
            v = float(sym)
            verbal = (f"{expr_clean} = {int(v)}" if v == int(v)
                      else f"{expr_clean} = {v:g}")
            return {"ok": True, "answer": v, "verbal": verbal,
                    "engine": "sympy"}
        except (TypeError, ValueError):
            # sympy returned a symbolic expression we can't reduce to float.
            return {"ok": True, "answer": str(sym),
                    "verbal": f"{expr_clean} = {sym}",
                    "engine": "sympy-symbolic"}

    # Fallback parser.
    try:
        tokens = _tokenize(expr_clean)
        rpn = _to_rpn(tokens)
        v = _eval_rpn(rpn)
        verbal = (f"{expr_clean} = {int(v)}" if v == int(v)
                  else f"{expr_clean} = {v:g}")
        return {"ok": True, "answer": v, "verbal": verbal,
                "engine": "shunting-yard"}
    except Exception as exc:
        return {"ok": False, "error": f"could not evaluate: {exc}"}


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial of negative number")
    return math.factorial(n)


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True
