"""Real voting — ballot collection, quorum, tally.

Distinct from `resolution.VotingResolution`, which models a SINGLE-actor
skill check against an abstract threshold. This module is for multi-
actor ballots: each eligible voter submits one vote, and a tally
produces a winner under a configurable rule.

Use cases:
  - Mafia accusation phase (everyone votes whom to eliminate)
  - Jury verdict (guilty / not guilty / hung)
  - Elections, council decisions
  - Group consensus checkpoints

Polls are FIRST-CLASS state objects so they persist across rounds and
snapshot cleanly. A typical flow inside a domain module is:

    poll = state.polls.open_poll(
        poll_id="day1_accusation",
        eligible_voters=[p.id for p in alive_players],
        options=[p.id for p in alive_players],
        rule="plurality",
    )
    # ... agents cast votes during a phase ...
    result = state.polls.tally(poll.poll_id)
    if result.winner:
        ...eliminate result.winner...
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Poll:
    poll_id: str
    eligible_voters: List[str]
    options: List[str]                                  # what can be voted for
    rule: str = "plurality"                             # plurality | majority | unanimity | supermajority
    threshold: float = 0.5                              # used by majority/supermajority (fraction)
    allow_abstain: bool = True
    description: str = ""
    is_open: bool = True
    votes: Dict[str, str] = field(default_factory=dict)  # voter_id -> option
    round_opened: int = 0
    round_closed: Optional[int] = None


@dataclass
class PollResult:
    poll_id: str
    rule: str
    tally: Dict[str, int]
    total_votes: int
    abstentions: int
    winner: Optional[str]    # None on tie / no quorum / rule unmet
    tied: List[str] = field(default_factory=list)
    quorum_met: bool = True


class PollManager:
    """Owns the open polls and serves the tally."""

    def __init__(self) -> None:
        self._polls: Dict[str, Poll] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open_poll(
        self,
        poll_id: str,
        eligible_voters: List[str],
        options: List[str],
        *,
        rule: str = "plurality",
        threshold: float = 0.5,
        allow_abstain: bool = True,
        description: str = "",
        round_opened: int = 0,
    ) -> Poll:
        if poll_id in self._polls:
            raise ValueError(f"Poll '{poll_id}' is already open.")
        if rule not in {"plurality", "majority", "unanimity", "supermajority"}:
            raise ValueError(f"Unknown poll rule '{rule}'.")
        poll = Poll(
            poll_id=poll_id,
            eligible_voters=list(eligible_voters),
            options=list(options),
            rule=rule,
            threshold=threshold,
            allow_abstain=allow_abstain,
            description=description,
            round_opened=round_opened,
        )
        self._polls[poll_id] = poll
        return poll

    def get(self, poll_id: str) -> Optional[Poll]:
        return self._polls.get(poll_id)

    def list_open(self) -> List[Poll]:
        return [p for p in self._polls.values() if p.is_open]

    def list_for_voter(self, voter_id: str) -> List[Poll]:
        return [p for p in self._polls.values() if p.is_open and voter_id in p.eligible_voters]

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def cast_vote(self, poll_id: str, voter_id: str, option: str) -> bool:
        """Cast / overwrite a vote. Returns False on invalid attempt."""
        poll = self._polls.get(poll_id)
        if poll is None or not poll.is_open:
            return False
        if voter_id not in poll.eligible_voters:
            return False
        if option == "_abstain":
            if not poll.allow_abstain:
                return False
            poll.votes[voter_id] = option
            return True
        if option not in poll.options:
            return False
        poll.votes[voter_id] = option
        return True

    def retract_vote(self, poll_id: str, voter_id: str) -> bool:
        poll = self._polls.get(poll_id)
        if poll is None or not poll.is_open:
            return False
        return poll.votes.pop(voter_id, None) is not None

    # ------------------------------------------------------------------
    # Tally
    # ------------------------------------------------------------------

    def tally(self, poll_id: str) -> Optional[PollResult]:
        poll = self._polls.get(poll_id)
        if poll is None:
            return None

        non_abstain = {v: o for v, o in poll.votes.items() if o != "_abstain"}
        abstentions = sum(1 for o in poll.votes.values() if o == "_abstain")
        counts = Counter(non_abstain.values())
        total_votes = sum(counts.values())

        winner: Optional[str] = None
        tied: List[str] = []

        if counts:
            top = counts.most_common()
            top_count = top[0][1]
            top_options = [opt for opt, c in counts.items() if c == top_count]

            if poll.rule == "plurality":
                if len(top_options) == 1:
                    winner = top_options[0]
                else:
                    tied = sorted(top_options)
            elif poll.rule == "majority":
                if top_count > len(poll.eligible_voters) * poll.threshold:
                    winner = top_options[0] if len(top_options) == 1 else None
                    if winner is None:
                        tied = sorted(top_options)
            elif poll.rule == "supermajority":
                # Like majority but threshold typically 2/3 or 3/4
                if top_count >= len(poll.eligible_voters) * poll.threshold:
                    winner = top_options[0] if len(top_options) == 1 else None
                    if winner is None:
                        tied = sorted(top_options)
            elif poll.rule == "unanimity":
                if len(counts) == 1 and top_count == len(poll.eligible_voters):
                    winner = top_options[0]

        quorum_met = (total_votes + abstentions) > 0
        return PollResult(
            poll_id=poll.poll_id,
            rule=poll.rule,
            tally=dict(counts),
            total_votes=total_votes,
            abstentions=abstentions,
            winner=winner,
            tied=tied,
            quorum_met=quorum_met,
        )

    def close_poll(self, poll_id: str, round_closed: int = 0) -> Optional[PollResult]:
        """Close the poll and return the final tally."""
        poll = self._polls.get(poll_id)
        if poll is None:
            return None
        result = self.tally(poll_id)
        poll.is_open = False
        poll.round_closed = round_closed
        return result

    def discard(self, poll_id: str) -> None:
        self._polls.pop(poll_id, None)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "polls": {
                pid: {
                    "poll_id": p.poll_id,
                    "eligible_voters": list(p.eligible_voters),
                    "options": list(p.options),
                    "rule": p.rule,
                    "threshold": p.threshold,
                    "allow_abstain": p.allow_abstain,
                    "description": p.description,
                    "is_open": p.is_open,
                    "votes": dict(p.votes),
                    "round_opened": p.round_opened,
                    "round_closed": p.round_closed,
                }
                for pid, p in self._polls.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PollManager":
        """Restore from a to_dict snapshot."""
        mgr = cls()
        for pid, pd in (data.get("polls") or {}).items():
            mgr._polls[pid] = Poll(
                poll_id=pd["poll_id"],
                eligible_voters=list(pd.get("eligible_voters", [])),
                options=list(pd.get("options", [])),
                rule=pd.get("rule", "plurality"),
                threshold=pd.get("threshold"),
                allow_abstain=pd.get("allow_abstain", False),
                description=pd.get("description", ""),
                is_open=pd.get("is_open", True),
                votes=dict(pd.get("votes", {})),
                round_opened=pd.get("round_opened", 0),
                round_closed=pd.get("round_closed"),
            )
        return mgr
