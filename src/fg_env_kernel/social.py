"""Social graph & content model -- social media dynamics for simulations.

Provides:
- Social graph: follow/friend/block/mute relationships
- Content creation: posts, replies, shares, reactions
- Feed generation: per-agent content feeds
- Reputation system: public reputation from actions and content
- Viral spread: information/content propagation models
"""
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SocialRelationType(Enum):
    FOLLOW = "follow"
    FRIEND = "friend"
    BLOCK = "block"
    MUTE = "mute"


class ContentType(Enum):
    POST = "post"
    REPLY = "reply"
    SHARE = "share"
    REACTION = "reaction"


class ContentVisibility(Enum):
    PUBLIC = "public"
    FOLLOWERS_ONLY = "followers_only"
    PRIVATE = "private"
    GROUP = "group"


# ---------------------------------------------------------------------------
# Content Item
# ---------------------------------------------------------------------------

@dataclass
class ContentItem:
    """A piece of social content (post, reply, share, reaction)."""
    id: str = ""
    author_id: str = ""
    content_type: ContentType = ContentType.POST
    text: str = ""
    parent_id: Optional[str] = None  # For replies/shares
    visibility: ContentVisibility = ContentVisibility.PUBLIC
    tags: List[str] = field(default_factory=list)
    round_created: int = 0
    reactions: Dict[str, int] = field(default_factory=dict)  # reaction_type -> count
    share_count: int = 0
    reach: Set[str] = field(default_factory=set)  # entity_ids who saw it

    def __post_init__(self):
        if not self.id:
            self.id = f"content_{uuid.uuid4().hex[:8]}"

    def total_engagement(self) -> int:
        """Total engagement (reactions + shares)."""
        return sum(self.reactions.values()) + self.share_count

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "content_type": self.content_type.value,
            "text": self.text,
            "parent_id": self.parent_id,
            "visibility": self.visibility.value,
            "tags": list(self.tags),
            "round_created": self.round_created,
            "reactions": dict(self.reactions),
            "share_count": self.share_count,
            "reach": list(self.reach),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentItem":
        item = cls(
            id=data.get("id", ""),
            author_id=data.get("author_id", ""),
            content_type=ContentType(data.get("content_type", "post")),
            text=data.get("text", ""),
            parent_id=data.get("parent_id"),
            visibility=ContentVisibility(data.get("visibility", "public")),
            tags=data.get("tags", []),
            round_created=data.get("round_created", 0),
            reactions=data.get("reactions", {}),
            share_count=data.get("share_count", 0),
        )
        item.reach = set(data.get("reach", []))
        return item


# ---------------------------------------------------------------------------
# Social Graph
# ---------------------------------------------------------------------------

class SocialGraph:
    """Directed social graph with follow/friend/block/mute relationships."""

    def __init__(self):
        # {entity_id: {rel_type: set(target_ids)}}
        self._outgoing: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    def follow(self, follower_id: str, target_id: str):
        """A follows B."""
        self._outgoing[follower_id]["follow"].add(target_id)

    def unfollow(self, follower_id: str, target_id: str):
        """A unfollows B."""
        self._outgoing[follower_id]["follow"].discard(target_id)

    def friend(self, a: str, b: str):
        """Mutual friendship (bidirectional)."""
        self._outgoing[a]["friend"].add(b)
        self._outgoing[b]["friend"].add(a)

    def unfriend(self, a: str, b: str):
        """Remove mutual friendship."""
        self._outgoing[a]["friend"].discard(b)
        self._outgoing[b]["friend"].discard(a)

    def block(self, blocker_id: str, target_id: str):
        """A blocks B."""
        self._outgoing[blocker_id]["block"].add(target_id)

    def unblock(self, blocker_id: str, target_id: str):
        """A unblocks B."""
        self._outgoing[blocker_id]["block"].discard(target_id)

    def mute(self, muter_id: str, target_id: str):
        """A mutes B."""
        self._outgoing[muter_id]["mute"].add(target_id)

    def unmute(self, muter_id: str, target_id: str):
        """A unmutes B."""
        self._outgoing[muter_id]["mute"].discard(target_id)

    def get_following(self, entity_id: str) -> Set[str]:
        """Get all entities this entity follows."""
        return set(self._outgoing[entity_id].get("follow", set()))

    def get_followers(self, entity_id: str) -> Set[str]:
        """Get all entities that follow this entity."""
        followers = set()
        for eid, rels in self._outgoing.items():
            if entity_id in rels.get("follow", set()):
                followers.add(eid)
        return followers

    def get_friends(self, entity_id: str) -> Set[str]:
        """Get all friends of this entity."""
        return set(self._outgoing[entity_id].get("friend", set()))

    def get_blocked(self, entity_id: str) -> Set[str]:
        """Get all entities blocked by this entity."""
        return set(self._outgoing[entity_id].get("block", set()))

    def get_muted(self, entity_id: str) -> Set[str]:
        """Get all entities muted by this entity."""
        return set(self._outgoing[entity_id].get("mute", set()))

    def is_following(self, follower_id: str, target_id: str) -> bool:
        return target_id in self._outgoing[follower_id].get("follow", set())

    def is_blocked(self, blocker_id: str, target_id: str) -> bool:
        return target_id in self._outgoing[blocker_id].get("block", set())

    def get_influence_score(self, entity_id: str) -> float:
        """Simple influence score based on follower count and friend count."""
        followers = len(self.get_followers(entity_id))
        friends = len(self.get_friends(entity_id))
        return followers + friends * 0.5

    def get_echo_chamber_score(self, entity_id: str) -> float:
        """How insular is this agent's network? 0.0 = diverse, 1.0 = echo chamber.

        Measures what fraction of an agent's connections also follow each other.
        """
        following = self.get_following(entity_id)
        friends = self.get_friends(entity_id)
        connections = following | friends
        if len(connections) < 2:
            return 0.0

        # Count interconnections among connections
        interconnections = 0
        max_possible = len(connections) * (len(connections) - 1)
        for c in connections:
            c_connections = self.get_following(c) | self.get_friends(c)
            interconnections += len(c_connections & connections)

        return interconnections / max_possible if max_possible > 0 else 0.0

    def remove_entity(self, entity_id: str):
        """Remove all social connections for an entity."""
        self._outgoing.pop(entity_id, None)
        for eid in list(self._outgoing.keys()):
            for rel_type in list(self._outgoing[eid].keys()):
                self._outgoing[eid][rel_type].discard(entity_id)

    def to_dict(self) -> dict:
        result = {}
        for eid, rels in self._outgoing.items():
            result[eid] = {
                rel_type: list(targets) for rel_type, targets in rels.items() if targets
            }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SocialGraph":
        graph = cls()
        for eid, rels in data.items():
            for rel_type, targets in rels.items():
                for t in targets:
                    graph._outgoing[eid][rel_type].add(t)
        return graph


# ---------------------------------------------------------------------------
# Reputation System
# ---------------------------------------------------------------------------

class ReputationSystem:
    """Per-agent public reputation (0.0-1.0 scale)."""

    def __init__(self):
        self._scores: Dict[str, float] = {}  # entity_id -> reputation

    def get_reputation(self, entity_id: str) -> float:
        """Get reputation score (default 0.5)."""
        return self._scores.get(entity_id, 0.5)

    def set_reputation(self, entity_id: str, value: float):
        """Set reputation score (clamped 0-1)."""
        self._scores[entity_id] = max(0.0, min(1.0, value))

    def update_from_action(self, entity_id: str, action_name: str, success: bool,
                           observers: Optional[List[str]] = None):
        """Update reputation based on action outcome."""
        current = self.get_reputation(entity_id)
        delta = 0.02 if success else -0.01
        self.set_reputation(entity_id, current + delta)

    def update_from_content(self, entity_id: str, content: ContentItem):
        """Update reputation based on content engagement."""
        current = self.get_reputation(entity_id)
        engagement = content.total_engagement()
        if engagement > 0:
            delta = min(0.05, engagement * 0.005)
            self.set_reputation(entity_id, current + delta)

    def get_reputation_modifiers(self, entity_id: str) -> Dict[str, float]:
        """Get trust bonus/penalty for interactions."""
        rep = self.get_reputation(entity_id)
        return {
            "trust_bonus": max(0, (rep - 0.5) * 2),  # 0-1 bonus
            "persuasion_modifier": rep,
        }

    def remove(self, entity_id: str):
        """Remove reputation for an entity."""
        self._scores.pop(entity_id, None)

    def to_dict(self) -> dict:
        return {"scores": dict(self._scores)}

    @classmethod
    def from_dict(cls, data: dict) -> "ReputationSystem":
        rep = cls()
        rep._scores = data.get("scores", {})
        return rep


# ---------------------------------------------------------------------------
# Viral Spread Model
# ---------------------------------------------------------------------------

class ViralSpreadModel:
    """Information/content propagation models."""

    def __init__(self, model_type: str = "simple_cascade", spread_probability: float = 0.3):
        self.model_type = model_type
        self.spread_probability = spread_probability

    def calculate_spread(
        self,
        content: ContentItem,
        social_graph: SocialGraph,
        already_reached: Set[str],
        rng: Optional["random.Random"] = None,
    ) -> List[str]:
        """Calculate who sees this content next. Returns new entity_ids reached.

        ``rng`` should be the engine's seeded ``random.Random`` so spread is
        reproducible. When omitted, falls back to the global module RNG.
        """
        if self.model_type == "threshold_model":
            return self._threshold_model(content, social_graph, already_reached)
        return self._simple_cascade(content, social_graph, already_reached, rng)

    def _simple_cascade(self, content: ContentItem, social_graph: SocialGraph,
                        already_reached: Set[str],
                        rng: Optional["random.Random"] = None) -> List[str]:
        """Each reached entity spreads to followers with probability p."""
        roll = rng.random if rng is not None else random.random
        new_reached: List[str] = []
        # Sort the reached set: set iteration order is not stable across runs,
        # and dedup against ``new_reached`` makes outcomes order-dependent.
        for entity_id in sorted(already_reached):
            followers = social_graph.get_followers(entity_id)
            for follower in sorted(followers):
                if follower not in already_reached and follower not in new_reached:
                    if roll() < self.spread_probability:
                        new_reached.append(follower)
        return new_reached

    def _threshold_model(self, content: ContentItem, social_graph: SocialGraph,
                         already_reached: Set[str]) -> List[str]:
        """Entity adopts if fraction of connections have adopted exceeds threshold."""
        new_reached = []
        threshold = 0.3  # 30% of connections must have seen it
        all_entities = set()
        for eid in already_reached:
            all_entities.update(social_graph.get_followers(eid))
            all_entities.update(social_graph.get_following(eid))
        for entity_id in sorted(all_entities):
            if entity_id in already_reached:
                continue
            connections = social_graph.get_following(entity_id) | social_graph.get_friends(entity_id)
            if not connections:
                continue
            fraction_reached = len(connections & already_reached) / len(connections)
            if fraction_reached >= threshold:
                new_reached.append(entity_id)
        return new_reached

    def apply_persuasion(self, source_reputation: float) -> float:
        """Calculate persuasion probability based on source reputation."""
        return min(1.0, self.spread_probability * (0.5 + source_reputation))

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "spread_probability": self.spread_probability,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ViralSpreadModel":
        return cls(
            model_type=data.get("model_type", "simple_cascade"),
            spread_probability=data.get("spread_probability", 0.3),
        )


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

class Feed:
    """Per-agent content feed."""

    def __init__(self, max_items: int = 50):
        self._items: List[ContentItem] = []
        self._max_items = max_items

    def add_item(self, item: ContentItem):
        """Add an item to the feed."""
        self._items.append(item)
        if len(self._items) > self._max_items:
            self._items = self._items[-self._max_items:]

    def get_feed(self, limit: int = 10) -> List[ContentItem]:
        """Get most recent feed items."""
        return list(reversed(self._items[-limit:]))

    def get_count(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# Social Platform Manager
# ---------------------------------------------------------------------------

class SocialPlatformManager:
    """Orchestrates all social subsystems for a simulation.

    Manages: social graph, content, feeds, reputation, viral spread.
    """

    def __init__(
        self,
        viral_model: Optional[ViralSpreadModel] = None,
        reputation_enabled: bool = True,
    ):
        self.social_graph = SocialGraph()
        self.reputation = ReputationSystem() if reputation_enabled else None
        self.viral_model = viral_model or ViralSpreadModel()
        self._content: Dict[str, ContentItem] = {}  # content_id -> ContentItem
        self._feeds: Dict[str, Feed] = {}  # entity_id -> Feed
        self._pending_content: List[str] = []  # content_ids to process this round

    def create_content(
        self,
        author_id: str,
        text: str,
        content_type: ContentType = ContentType.POST,
        visibility: ContentVisibility = ContentVisibility.PUBLIC,
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        round_number: int = 0,
    ) -> ContentItem:
        """Create a new content item."""
        item = ContentItem(
            author_id=author_id,
            content_type=content_type,
            text=text,
            parent_id=parent_id,
            visibility=visibility,
            tags=tags or [],
            round_created=round_number,
        )
        item.reach.add(author_id)
        self._content[item.id] = item
        self._pending_content.append(item.id)
        self._get_feed(author_id).add_item(item)
        return item

    def react_to_content(self, entity_id: str, content_id: str, reaction_type: str = "like"):
        """React to a piece of content."""
        content = self._content.get(content_id)
        if content:
            content.reactions[reaction_type] = content.reactions.get(reaction_type, 0) + 1
            content.reach.add(entity_id)

    def share_content(self, entity_id: str, content_id: str, round_number: int = 0) -> Optional[ContentItem]:
        """Share/repost existing content."""
        original = self._content.get(content_id)
        if not original:
            return None
        original.share_count += 1
        share = self.create_content(
            author_id=entity_id,
            text=original.text,
            content_type=ContentType.SHARE,
            parent_id=content_id,
            tags=original.tags,
            round_number=round_number,
        )
        return share

    def get_content(self, content_id: str) -> Optional[ContentItem]:
        """Get a content item by ID."""
        return self._content.get(content_id)

    def get_feed(self, entity_id: str, limit: int = 5) -> List[ContentItem]:
        """Get an agent's content feed."""
        feed = self._feeds.get(entity_id)
        if feed:
            return feed.get_feed(limit)
        return []

    def get_trending(self, limit: int = 5) -> List[ContentItem]:
        """Get trending content sorted by engagement."""
        all_content = sorted(
            self._content.values(),
            key=lambda c: c.total_engagement(),
            reverse=True,
        )
        return all_content[:limit]

    def tick(self, state: Any, round_number: int,
             rng: Optional["random.Random"] = None):
        """Process pending content: calculate spread, update feeds, update reputation.

        ``rng`` is the engine's seeded RNG, threaded through so viral spread is
        reproducible run-to-run.
        """
        # Process viral spread for pending content
        for content_id in self._pending_content:
            content = self._content.get(content_id)
            if not content:
                continue

            # Calculate viral spread
            new_reached = self.viral_model.calculate_spread(
                content, self.social_graph, content.reach, rng,
            )
            for entity_id in new_reached:
                # Check if blocked
                if self.social_graph.is_blocked(entity_id, content.author_id):
                    continue
                content.reach.add(entity_id)
                self._get_feed(entity_id).add_item(content)

            # Update author reputation
            if self.reputation:
                self.reputation.update_from_content(content.author_id, content)

        self._pending_content.clear()

    def get_perception_data(self, entity_id: str) -> Dict[str, Any]:
        """Get social data for an agent's perception."""
        feed_items = self.get_feed(entity_id, limit=5)
        followers = self.social_graph.get_followers(entity_id)
        following = self.social_graph.get_following(entity_id)

        data = {
            "social_feed": [
                {
                    "author": item.author_id,
                    "text": item.text[:100],
                    "type": item.content_type.value,
                    "engagement": item.total_engagement(),
                }
                for item in feed_items
            ],
            "follower_count": len(followers),
            "following_count": len(following),
            "influence_score": self.social_graph.get_influence_score(entity_id),
        }

        if self.reputation:
            data["reputation"] = round(self.reputation.get_reputation(entity_id), 2)

        return data

    def remove_entity(self, entity_id: str):
        """Clean up all social data for a despawned entity."""
        self.social_graph.remove_entity(entity_id)
        if self.reputation:
            self.reputation.remove(entity_id)
        self._feeds.pop(entity_id, None)

    def _get_feed(self, entity_id: str) -> Feed:
        """Get or create feed for an entity."""
        if entity_id not in self._feeds:
            self._feeds[entity_id] = Feed()
        return self._feeds[entity_id]

    def to_dict(self) -> dict:
        return {
            "social_graph": self.social_graph.to_dict(),
            "reputation": self.reputation.to_dict() if self.reputation else None,
            "viral_model": self.viral_model.to_dict(),
            "content": {cid: c.to_dict() for cid, c in self._content.items()},
            "pending_content": list(self._pending_content),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SocialPlatformManager":
        viral = ViralSpreadModel.from_dict(data.get("viral_model", {}))
        mgr = cls(
            viral_model=viral,
            reputation_enabled=data.get("reputation") is not None,
        )
        mgr.social_graph = SocialGraph.from_dict(data.get("social_graph", {}))
        if data.get("reputation"):
            mgr.reputation = ReputationSystem.from_dict(data["reputation"])
        for cid, cdata in data.get("content", {}).items():
            mgr._content[cid] = ContentItem.from_dict(cdata)
        mgr._pending_content = data.get("pending_content", [])
        return mgr
