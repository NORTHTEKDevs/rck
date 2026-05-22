"""Persistent identity across sessions.

LLMs reset every conversation. RCK can maintain:

  1. A long-term user model: facts the user has shared, preferences,
     topics they care about, communication style.
  2. The agent's own continuity: history of conversations, personality
     evolution, learned shortcuts.
  3. Relationship state: what the agent knows about each user.

Everything is stored as triples in dedicated KBs (one per user). Wakes
up the same agent every session with full memory of who you are.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class UserProfile:
    """Per-user long-term state."""

    user_id: str
    kb: ShardedKnowledgeBase = field(default=None, init=False)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    interaction_count: int = 0
    topics_of_interest: Counter = field(default_factory=Counter)
    preferences: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kb is None:
            self.kb = ShardedKnowledgeBase(dim=4096, n_shards=32, seed=hash(self.user_id) & 0xFFFF)

    def touch(self) -> None:
        self.last_seen = time.time()
        self.interaction_count += 1

    def remember_fact(self, subject: str, relation: str, obj: str) -> None:
        self.kb.store({"S": subject.lower(), "R": relation.lower(),
                        "O": obj.lower()})

    def note_topic(self, topic: str) -> None:
        self.topics_of_interest[topic] += 1

    def set_preference(self, key: str, value: str) -> None:
        self.preferences[key] = value

    def top_topics(self, n: int = 5) -> list[tuple[str, int]]:
        return self.topics_of_interest.most_common(n)

    def summary(self) -> str:
        topics = ", ".join(t for t, _ in self.top_topics(3)) or "no topics yet"
        return (f"User {self.user_id}: {self.interaction_count} interactions, "
                f"top topics: {topics}, prefs: {self.preferences}")


@dataclass
class IdentityStore:
    """Holds UserProfile objects, persists to disk."""

    storage_dir: Path = field(default_factory=lambda: Path("data/identities"))
    profiles: dict[str, UserProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.storage_dir = Path(self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(self, user_id: str) -> UserProfile:
        if user_id in self.profiles:
            return self.profiles[user_id]
        profile = UserProfile(user_id=user_id)
        self.profiles[user_id] = profile
        return profile

    def save(self, user_id: str) -> Path:
        profile = self.profiles.get(user_id)
        if profile is None:
            raise KeyError(user_id)
        path = self.storage_dir / f"{user_id}.json"
        # We don't persist the KB tensor here (use rck.session for that).
        # Just persist the lightweight metadata.
        data = {
            "user_id": profile.user_id,
            "first_seen": profile.first_seen,
            "last_seen": profile.last_seen,
            "interaction_count": profile.interaction_count,
            "topics_of_interest": dict(profile.topics_of_interest),
            "preferences": dict(profile.preferences),
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    def load(self, user_id: str) -> UserProfile:
        path = self.storage_dir / f"{user_id}.json"
        if not path.exists():
            return self.get_or_create(user_id)
        data = json.loads(path.read_text())
        profile = UserProfile(user_id=data["user_id"])
        profile.first_seen = float(data.get("first_seen", time.time()))
        profile.last_seen = float(data.get("last_seen", time.time()))
        profile.interaction_count = int(data.get("interaction_count", 0))
        profile.topics_of_interest = Counter(data.get("topics_of_interest", {}))
        profile.preferences = dict(data.get("preferences", {}))
        self.profiles[user_id] = profile
        return profile

    def all_users(self) -> list[str]:
        # List both in-memory and on-disk profiles.
        in_mem = set(self.profiles)
        on_disk = {p.stem for p in self.storage_dir.glob("*.json")}
        return sorted(in_mem | on_disk)
