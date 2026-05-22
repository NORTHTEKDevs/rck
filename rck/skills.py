"""Skill / procedure discovery.

When the agent successfully chains N reasoning steps to answer a
question, that chain is a REUSABLE skill. We name it, store it, and
re-apply it on future queries that match the same pattern.

This builds a library of LEARNED procedures over time -- something
LLMs have implicitly (through training) but RCK gets explicitly
(through recording successes).

Each skill is:
  * a name (auto-generated from the pattern)
  * a pattern (sequence of (relation, role-played) edges)
  * a count of how many times it succeeded
  * a count of how many times it was tried-and-failed
  * a confidence score

Skills are stored in a `SkillLibrary` that can be queried by pattern
match, and the highest-scoring matching skill is preferred on new
queries.
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Skill:
    name: str
    pattern: list[tuple[str, str]]  # list of (role, relation) edges
    success_count: int = 0
    failure_count: int = 0
    last_used: float = field(default_factory=time.time)

    @property
    def confidence(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def record(self, succeeded: bool) -> None:
        if succeeded:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_used = time.time()


def _pattern_signature(pattern: list[tuple[str, str]]) -> str:
    """Stable hash of a pattern -- used as skill ID."""
    key = "|".join(f"{role}:{rel}" for role, rel in pattern)
    h = hashlib.blake2b(key.encode(), digest_size=4).hexdigest()
    return f"skill_{h}"


@dataclass
class SkillLibrary:
    """Stores learned procedures keyed by pattern signature."""

    skills: dict[str, Skill] = field(default_factory=dict)

    def record_success(self, pattern: list[tuple[str, str]]) -> Skill:
        """A reasoning chain just succeeded. Record / reinforce."""
        sig = _pattern_signature(pattern)
        if sig not in self.skills:
            self.skills[sig] = Skill(
                name=_auto_name_from_pattern(pattern),
                pattern=list(pattern),
            )
        self.skills[sig].record(succeeded=True)
        return self.skills[sig]

    def record_failure(self, pattern: list[tuple[str, str]]) -> Skill | None:
        sig = _pattern_signature(pattern)
        if sig not in self.skills:
            return None
        self.skills[sig].record(succeeded=False)
        return self.skills[sig]

    def find_matching(self, pattern: list[tuple[str, str]]) -> Skill | None:
        """Exact-pattern match."""
        return self.skills.get(_pattern_signature(pattern))

    def find_partial_matching(self, partial_pattern: list[tuple[str, str]],
                              top_k: int = 3) -> list[Skill]:
        """Match patterns whose prefix equals `partial_pattern`."""
        results: list[Skill] = []
        for sk in self.skills.values():
            if len(sk.pattern) < len(partial_pattern):
                continue
            if sk.pattern[:len(partial_pattern)] == partial_pattern:
                results.append(sk)
        results.sort(key=lambda s: -s.confidence)
        return results[:top_k]

    def most_used(self, n: int = 10) -> list[Skill]:
        return sorted(
            self.skills.values(),
            key=lambda s: -(s.success_count + s.failure_count),
        )[:n]

    def by_success_rate(self, min_uses: int = 3, n: int = 10) -> list[Skill]:
        candidates = [
            s for s in self.skills.values()
            if (s.success_count + s.failure_count) >= min_uses
        ]
        candidates.sort(key=lambda s: -s.confidence)
        return candidates[:n]

    def stats(self) -> dict:
        if not self.skills:
            return {"n": 0, "total_uses": 0, "high_conf": 0}
        return {
            "n": len(self.skills),
            "total_uses": sum(s.success_count + s.failure_count
                              for s in self.skills.values()),
            "high_conf": sum(1 for s in self.skills.values()
                              if s.confidence >= 0.8
                              and (s.success_count + s.failure_count) >= 3),
        }

    def save(self, path) -> int:
        """Persist the library to a JSONL file. One skill per line."""
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for sig, skill in self.skills.items():
                rec = {
                    "name": skill.name,
                    "pattern": [list(p) for p in skill.pattern],
                    "success_count": skill.success_count,
                    "failure_count": skill.failure_count,
                    "last_used": skill.last_used,
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
        return n

    def load(self, path, *, replace: bool = False) -> int:
        """Restore skills from a JSONL file produced by save()."""
        import json
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return 0
        if replace:
            self.skills.clear()
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pattern = [tuple(p) for p in rec.get("pattern", [])]
                sig = _pattern_signature(pattern)
                self.skills[sig] = Skill(
                    name=rec.get("name", "loaded"),
                    pattern=pattern,
                    success_count=int(rec.get("success_count", 0)),
                    failure_count=int(rec.get("failure_count", 0)),
                    last_used=float(rec.get("last_used", 0.0)),
                )
                n += 1
        return n


def _auto_name_from_pattern(pattern: list[tuple[str, str]]) -> str:
    if not pattern:
        return "empty_skill"
    parts = [rel for role, rel in pattern]
    name = "_then_".join(parts[:3])
    if len(parts) > 3:
        name += f"_plus{len(parts) - 3}"
    return name
