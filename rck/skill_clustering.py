"""Cluster SkillLibrary patterns by shared prefixes.

Skills are recorded as ordered (role, relation) edges. Many skills
in practice share long prefixes: `[(O, partof), (O, locatedin),
(O, isa)]` and `[(O, partof), (O, locatedin), (O, continent)]`
diverge only at the last step. Grouping them by common prefix
surfaces "skill families" -- e.g. "all chains that start
partof->locatedin".

Output is a `SkillFamily` per shared-prefix group, sorted by total
success_count across members.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rck.skills import Skill, SkillLibrary


@dataclass
class SkillFamily:
    prefix: tuple[tuple[str, str], ...]   # the shared (role, relation) prefix
    members: list[Skill] = field(default_factory=list)

    @property
    def total_success(self) -> int:
        return sum(s.success_count for s in self.members)

    @property
    def total_failure(self) -> int:
        return sum(s.failure_count for s in self.members)

    @property
    def total_uses(self) -> int:
        return self.total_success + self.total_failure

    @property
    def confidence(self) -> float:
        t = self.total_uses
        return self.total_success / t if t else 0.0

    def verbalize(self) -> str:
        prefix_str = " -> ".join(r for _role, r in self.prefix)
        n = len(self.members)
        return (f"family[{prefix_str}]  members={n}  "
                f"successes={self.total_success}  conf={self.confidence:.2f}")


def cluster_skills_by_prefix(skills: SkillLibrary,
                              *, prefix_length: int = 2,
                              min_members: int = 2) -> list[SkillFamily]:
    """Group skills by the first `prefix_length` edges of their pattern.

    Returns SkillFamily objects sorted by total_success descending.
    Families with fewer than `min_members` skills are dropped.
    """
    by_prefix: dict[tuple[tuple[str, str], ...], list[Skill]] = defaultdict(list)
    for skill in skills.skills.values():
        if len(skill.pattern) < prefix_length:
            continue
        prefix = tuple(tuple(p) for p in skill.pattern[:prefix_length])
        by_prefix[prefix].append(skill)
    families: list[SkillFamily] = []
    for prefix, members in by_prefix.items():
        if len(members) < min_members:
            continue
        families.append(SkillFamily(prefix=prefix, members=list(members)))
    families.sort(key=lambda f: -f.total_success)
    return families


def family_summary(families: list[SkillFamily], *,
                    top_k: int = 10) -> list[dict]:
    """Compact JSON-friendly summary of families for logging."""
    return [
        {
            "prefix": [list(p) for p in f.prefix],
            "n_members": len(f.members),
            "total_success": f.total_success,
            "total_failure": f.total_failure,
            "confidence": f.confidence,
            "verbal": f.verbalize(),
        }
        for f in families[:top_k]
    ]
