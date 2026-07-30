"""Structured inter-agent communication: negotiations, agreements, and auctions.

Agents can propose structured deals (give X, receive Y), counter-offer,
accept, or reject. Successful negotiations create binding agreements with
optional enforcement effects for violations. Auctions support ascending,
descending, and sealed-bid protocols.
"""
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class NegotiationState(Enum):
    PROPOSED = "proposed"
    COUNTERED = "countered"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Proposal:
    """A structured offer in a negotiation."""
    proposer_id: str
    terms: Dict[str, Any]           # {"give": {"gold": 50}, "receive": {"iron": 10}}
    description: str = ""
    round_proposed: int = 0

    def to_dict(self) -> dict:
        return {
            "proposer_id": self.proposer_id,
            "terms": self.terms,
            "description": self.description,
            "round_proposed": self.round_proposed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Proposal":
        return cls(
            proposer_id=data["proposer_id"],
            terms=data.get("terms", {}),
            description=data.get("description", ""),
            round_proposed=data.get("round_proposed", 0),
        )


@dataclass
class Negotiation:
    """A negotiation between 2+ parties with a state machine lifecycle."""
    id: str
    initiator_id: str
    participant_ids: List[str]                          # All parties including initiator
    state: NegotiationState = NegotiationState.PROPOSED
    proposals: List[Proposal] = field(default_factory=list)
    current_proposal: Optional[Proposal] = None
    round_started: int = 0
    max_rounds: int = 3                                 # Expires after this many rounds
    negotiation_type: str = "bilateral"                 # "bilateral" | "multilateral"
    accepted_by: List[str] = field(default_factory=list)  # For multilateral: who accepted so far

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "initiator_id": self.initiator_id,
            "participant_ids": self.participant_ids,
            "state": self.state.value,
            "proposals": [p.to_dict() for p in self.proposals],
            "current_proposal": self.current_proposal.to_dict() if self.current_proposal else None,
            "round_started": self.round_started,
            "max_rounds": self.max_rounds,
            "negotiation_type": self.negotiation_type,
            "accepted_by": list(self.accepted_by),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Negotiation":
        neg = cls(
            id=data["id"],
            initiator_id=data["initiator_id"],
            participant_ids=data.get("participant_ids", []),
            state=NegotiationState(data.get("state", "proposed")),
            round_started=data.get("round_started", 0),
            max_rounds=data.get("max_rounds", 3),
            negotiation_type=data.get("negotiation_type", "bilateral"),
            accepted_by=data.get("accepted_by", []),
        )
        neg.proposals = [Proposal.from_dict(p) for p in data.get("proposals", [])]
        cp = data.get("current_proposal")
        neg.current_proposal = Proposal.from_dict(cp) if cp else None
        return neg


@dataclass
class Agreement:
    """A binding agreement resulting from a successful negotiation."""
    id: str
    negotiation_id: str
    parties: List[str]
    terms: Dict[str, Any]
    round_made: int
    duration: int = 0                   # 0 = permanent, >0 = expires after N rounds
    enforcement_effects: List[Dict[str, Any]] = field(default_factory=list)
    violated: bool = False
    violated_by: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "negotiation_id": self.negotiation_id,
            "parties": self.parties,
            "terms": self.terms,
            "round_made": self.round_made,
            "duration": self.duration,
            "enforcement_effects": self.enforcement_effects,
            "violated": self.violated,
            "violated_by": self.violated_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agreement":
        return cls(
            id=data["id"],
            negotiation_id=data.get("negotiation_id", ""),
            parties=data.get("parties", []),
            terms=data.get("terms", {}),
            round_made=data.get("round_made", 0),
            duration=data.get("duration", 0),
            enforcement_effects=data.get("enforcement_effects", []),
            violated=data.get("violated", False),
            violated_by=data.get("violated_by"),
        )


@dataclass
class AuctionState:
    """State for an auction protocol."""
    id: str
    auctioneer_id: str
    item_description: str
    auction_type: str = "ascending"     # "ascending" | "descending" | "sealed_bid"
    resource: str = ""                  # Resource used for bidding
    bids: Dict[str, float] = field(default_factory=dict)  # bidder_id -> amount
    min_bid: float = 0.0
    current_price: float = 0.0
    round_started: int = 0
    max_rounds: int = 3
    state: str = "open"                 # "open" | "closed" | "awarded"
    winner_id: Optional[str] = None
    winning_bid: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "auctioneer_id": self.auctioneer_id,
            "item_description": self.item_description,
            "auction_type": self.auction_type,
            "resource": self.resource,
            "bids": dict(self.bids),
            "min_bid": self.min_bid,
            "current_price": self.current_price,
            "round_started": self.round_started,
            "max_rounds": self.max_rounds,
            "state": self.state,
            "winner_id": self.winner_id,
            "winning_bid": self.winning_bid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuctionState":
        return cls(
            id=data["id"],
            auctioneer_id=data.get("auctioneer_id", ""),
            item_description=data.get("item_description", ""),
            auction_type=data.get("auction_type", "ascending"),
            resource=data.get("resource", ""),
            bids=data.get("bids", {}),
            min_bid=data.get("min_bid", 0.0),
            current_price=data.get("current_price", 0.0),
            round_started=data.get("round_started", 0),
            max_rounds=data.get("max_rounds", 3),
            state=data.get("state", "open"),
            winner_id=data.get("winner_id"),
            winning_bid=data.get("winning_bid", 0.0),
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class NegotiationManager:
    """Manages all active negotiations, agreements, and auctions."""

    def __init__(self):
        self._negotiations: Dict[str, Negotiation] = {}
        self._agreements: Dict[str, Agreement] = {}
        self._auctions: Dict[str, AuctionState] = {}
        self._pending_agreements: List[str] = []  # Agreement IDs awaiting execution

    # -- Negotiation lifecycle --

    def propose(
        self,
        initiator_id: str,
        target_ids: List[str],
        terms: Dict[str, Any],
        round_num: int,
        max_rounds: int = 3,
        description: str = "",
    ) -> Negotiation:
        """Create a new negotiation proposal."""
        neg_id = f"neg_{uuid.uuid4().hex[:8]}"
        all_parties = [initiator_id] + [t for t in target_ids if t != initiator_id]

        proposal = Proposal(
            proposer_id=initiator_id,
            terms=terms,
            description=description,
            round_proposed=round_num,
        )

        neg_type = "bilateral" if len(all_parties) == 2 else "multilateral"
        negotiation = Negotiation(
            id=neg_id,
            initiator_id=initiator_id,
            participant_ids=all_parties,
            state=NegotiationState.PROPOSED,
            proposals=[proposal],
            current_proposal=proposal,
            round_started=round_num,
            max_rounds=max_rounds,
            negotiation_type=neg_type,
        )

        self._negotiations[neg_id] = negotiation
        return negotiation

    def counter(
        self,
        negotiation_id: str,
        proposer_id: str,
        new_terms: Dict[str, Any],
        round_num: int,
        description: str = "",
    ) -> bool:
        """Counter-propose in an existing negotiation. Returns True if successful."""
        neg = self._negotiations.get(negotiation_id)
        if not neg or neg.state not in (NegotiationState.PROPOSED, NegotiationState.COUNTERED):
            return False
        if proposer_id not in neg.participant_ids:
            return False

        proposal = Proposal(
            proposer_id=proposer_id,
            terms=new_terms,
            description=description,
            round_proposed=round_num,
        )
        neg.proposals.append(proposal)
        neg.current_proposal = proposal
        neg.state = NegotiationState.COUNTERED
        neg.accepted_by = []  # Reset acceptances on counter
        return True

    def accept(self, negotiation_id: str, acceptor_id: str, round_num: int = 0) -> Optional[Agreement]:
        """Accept the current proposal. Returns Agreement if all parties accepted."""
        neg = self._negotiations.get(negotiation_id)
        if not neg or neg.state not in (NegotiationState.PROPOSED, NegotiationState.COUNTERED):
            return None
        if acceptor_id not in neg.participant_ids:
            return None
        if acceptor_id == neg.current_proposal.proposer_id:
            return None  # Proposer can't accept their own offer

        if acceptor_id not in neg.accepted_by:
            neg.accepted_by.append(acceptor_id)

        # Check if all non-proposer parties have accepted
        required = [p for p in neg.participant_ids if p != neg.current_proposal.proposer_id]
        if set(required).issubset(set(neg.accepted_by)):
            neg.state = NegotiationState.ACCEPTED
            agreement = self._create_agreement(neg, round_num)
            return agreement

        return None  # Still waiting for other parties

    def reject(self, negotiation_id: str, rejector_id: str) -> bool:
        """Reject a negotiation. Returns True if successful."""
        neg = self._negotiations.get(negotiation_id)
        if not neg or neg.state not in (NegotiationState.PROPOSED, NegotiationState.COUNTERED):
            return False
        if rejector_id not in neg.participant_ids:
            return False

        neg.state = NegotiationState.REJECTED
        return True

    def _create_agreement(self, neg: Negotiation, round_num: int) -> Agreement:
        """Create a binding agreement from a completed negotiation."""
        agreement = Agreement(
            id=f"agr_{uuid.uuid4().hex[:8]}",
            negotiation_id=neg.id,
            parties=list(neg.participant_ids),
            terms=neg.current_proposal.terms if neg.current_proposal else {},
            round_made=round_num,
        )
        self._agreements[agreement.id] = agreement
        self._pending_agreements.append(agreement.id)
        return agreement

    def flush_pending_agreements(self, state: Any) -> List[Dict[str, Any]]:
        """Execute all pending agreement resource transfers. Called by engine each round."""
        all_events = []
        for agr_id in list(self._pending_agreements):
            agr = self._agreements.get(agr_id)
            if agr:
                events = self.execute_agreement(agr, state)
                all_events.extend(events)
        self._pending_agreements.clear()
        return all_events

    def execute_agreement(self, agreement: Agreement, state: Any) -> List[Dict[str, Any]]:
        """Execute resource transfers specified in agreement terms.

        Terms format: {"give": {"resource": amount}, "receive": {"resource": amount}}
        For bilateral: initiator gives what's in "give", receives what's in "receive".
        The other party does the reverse.

        Returns list of transfer event dicts.
        """
        events = []
        terms = agreement.terms
        if not terms:
            return events

        parties = agreement.parties
        if len(parties) < 2:
            return events

        # Bilateral: party[0] = initiator, party[1] = responder
        initiator = parties[0]
        responder = parties[1]

        # Initiator gives -> responder receives
        give_terms = terms.get("give", {})
        for resource_name, amount in give_terms.items():
            pool = state.resources.get(resource_name)
            if pool and amount > 0:
                success = pool.transfer(initiator, responder, float(amount))
                if success:
                    events.append({
                        "type": "agreement_transfer",
                        "agreement_id": agreement.id,
                        "resource": resource_name,
                        "from": initiator,
                        "to": responder,
                        "amount": amount,
                        "narrative": f"Agreement fulfilled: {initiator} gives {amount} {resource_name} to {responder}.",
                    })

        # Responder gives (initiator receives) — from "receive" terms
        receive_terms = terms.get("receive", {})
        for resource_name, amount in receive_terms.items():
            pool = state.resources.get(resource_name)
            if pool and amount > 0:
                success = pool.transfer(responder, initiator, float(amount))
                if success:
                    events.append({
                        "type": "agreement_transfer",
                        "agreement_id": agreement.id,
                        "resource": resource_name,
                        "from": responder,
                        "to": initiator,
                        "amount": amount,
                        "narrative": f"Agreement fulfilled: {responder} gives {amount} {resource_name} to {initiator}.",
                    })

        return events

    def execute_auction_award(self, auction: AuctionState, state: Any) -> List[Dict[str, Any]]:
        """Execute resource transfer for an awarded auction.

        Winner pays the winning bid amount to the auctioneer.
        Returns list of transfer event dicts.
        """
        events = []
        if not auction.winner_id or auction.state != "awarded":
            return events

        # Transfer bid resources from winner to auctioneer
        pool = state.resources.get(auction.resource)
        if pool and auction.winning_bid > 0:
            success = pool.transfer(auction.winner_id, auction.auctioneer_id, float(auction.winning_bid))
            if success:
                events.append({
                    "type": "auction_payment",
                    "auction_id": auction.id,
                    "resource": auction.resource,
                    "from": auction.winner_id,
                    "to": auction.auctioneer_id,
                    "amount": auction.winning_bid,
                    "narrative": (
                        f"Auction payment: {auction.winner_id} pays {auction.winning_bid} "
                        f"{auction.resource} to {auction.auctioneer_id} for {auction.item_description}."
                    ),
                })

        return events

    # -- Auction lifecycle --

    def start_auction(
        self,
        auctioneer_id: str,
        item_description: str,
        auction_type: str,
        resource: str,
        min_bid: float,
        round_num: int,
        max_rounds: int = 3,
    ) -> AuctionState:
        """Start a new auction."""
        auction_id = f"auc_{uuid.uuid4().hex[:8]}"
        auction = AuctionState(
            id=auction_id,
            auctioneer_id=auctioneer_id,
            item_description=item_description,
            auction_type=auction_type,
            resource=resource,
            min_bid=min_bid,
            current_price=min_bid,
            round_started=round_num,
            max_rounds=max_rounds,
        )
        self._auctions[auction_id] = auction
        return auction

    def place_bid(self, auction_id: str, bidder_id: str, amount: float) -> bool:
        """Place a bid on an auction. Returns True if bid was accepted."""
        auction = self._auctions.get(auction_id)
        if not auction or auction.state != "open":
            return False
        if bidder_id == auction.auctioneer_id:
            return False  # Can't bid on own auction

        if auction.auction_type == "ascending":
            if amount <= auction.current_price:
                return False  # Must exceed current price
            auction.bids[bidder_id] = amount
            auction.current_price = amount
            return True

        elif auction.auction_type == "sealed_bid":
            if bidder_id in auction.bids:
                return False  # Already bid
            if amount < auction.min_bid:
                return False
            auction.bids[bidder_id] = amount
            return True

        elif auction.auction_type == "descending":
            # First person to accept the current (descending) price wins
            auction.bids[bidder_id] = auction.current_price
            return True

        return False

    def close_auction(self, auction_id: str) -> Optional[Dict[str, Any]]:
        """Close an auction and determine the winner. Returns result dict or None."""
        auction = self._auctions.get(auction_id)
        if not auction or auction.state != "open":
            return None

        auction.state = "closed"

        if not auction.bids:
            return {"auction_id": auction_id, "winner_id": None, "winning_bid": 0, "status": "no_bids"}

        # Determine winner (highest bid). Tie-break on bidder id so the
        # outcome is reproducible — plain max() would resolve ties by
        # bid-insertion order, which depends on LLM completion timing.
        winner_id = min(auction.bids, key=lambda bid_id: (-auction.bids[bid_id], bid_id))
        winning_bid = auction.bids[winner_id]

        auction.winner_id = winner_id
        auction.winning_bid = winning_bid
        auction.state = "awarded"

        return {
            "auction_id": auction_id,
            "winner_id": winner_id,
            "winning_bid": winning_bid,
            "item": auction.item_description,
            "status": "awarded",
        }

    # -- Per-round processing --

    def tick(self, round_num: int) -> List[Dict[str, Any]]:
        """Expire old negotiations, close timed-out auctions. Returns events."""
        events = []

        # Expire negotiations
        for neg_id, neg in list(self._negotiations.items()):
            if neg.state in (NegotiationState.PROPOSED, NegotiationState.COUNTERED):
                if round_num - neg.round_started >= neg.max_rounds:
                    neg.state = NegotiationState.EXPIRED
                    events.append({
                        "type": "negotiation_expired",
                        "negotiation_id": neg_id,
                        "participants": neg.participant_ids,
                        "narrative": f"Negotiation between {', '.join(neg.participant_ids)} expired.",
                    })

        # Close timed-out auctions
        for auc_id, auction in list(self._auctions.items()):
            if auction.state == "open":
                if round_num - auction.round_started >= auction.max_rounds:
                    result = self.close_auction(auc_id)
                    if result:
                        events.append({
                            "type": "auction_closed",
                            **result,
                            "narrative": (
                                f"Auction for {auction.item_description} closed. "
                                f"{'Winner: ' + result['winner_id'] if result.get('winner_id') else 'No bids.'}"
                            ),
                        })

        # Check agreement expiry
        for agr_id, agr in list(self._agreements.items()):
            if agr.duration > 0 and not agr.violated:
                if round_num - agr.round_made >= agr.duration:
                    events.append({
                        "type": "agreement_expired",
                        "agreement_id": agr_id,
                        "parties": agr.parties,
                        "narrative": f"Agreement between {', '.join(agr.parties)} has expired.",
                    })
                    del self._agreements[agr_id]

        return events

    # -- Agreement enforcement --

    def check_agreement_violations(self, state: Any, round_num: int) -> List[Dict[str, Any]]:
        """Check if any agreement terms have been violated.

        For bilateral agreements:
        - The initiator (parties[0]) committed to "give" terms
        - The responder (parties[1]) committed to "receive" terms (which they give)

        Only the party who committed to give is checked for violations.
        """
        violations = []
        for agr_id, agr in self._agreements.items():
            if agr.violated:
                continue

            parties = agr.parties
            if len(parties) < 2:
                continue

            initiator = parties[0]
            responder = parties[1]

            # Check initiator's "give" commitments
            give_terms = agr.terms.get("give", {})
            for resource_name, amount in give_terms.items():
                pool = state.resources.get(resource_name)
                if pool:
                    holding = pool.get(initiator)
                    if holding < amount * 0.5:
                        agr.violated = True
                        agr.violated_by = initiator
                        violations.append({
                            "type": "agreement_violated",
                            "agreement_id": agr_id,
                            "violator": initiator,
                            "resource": resource_name,
                            "narrative": f"{initiator} violated agreement: insufficient {resource_name}.",
                        })
                        break

            if agr.violated:
                continue

            # Check responder's "receive" commitments (what responder gives back)
            receive_terms = agr.terms.get("receive", {})
            for resource_name, amount in receive_terms.items():
                pool = state.resources.get(resource_name)
                if pool:
                    holding = pool.get(responder)
                    if holding < amount * 0.5:
                        agr.violated = True
                        agr.violated_by = responder
                        violations.append({
                            "type": "agreement_violated",
                            "agreement_id": agr_id,
                            "violator": responder,
                            "resource": resource_name,
                            "narrative": f"{responder} violated agreement: insufficient {resource_name}.",
                        })
                        break

        return violations

    # -- Queries --

    def get_active_for_entity(self, entity_id: str, current_round: int = 0) -> Dict[str, Any]:
        """Get all active negotiations, agreements, and auctions for an entity."""
        result = {"negotiations": [], "agreements": [], "auctions": []}

        for neg in self._negotiations.values():
            if entity_id in neg.participant_ids and neg.state in (
                NegotiationState.PROPOSED, NegotiationState.COUNTERED
            ):
                result["negotiations"].append({
                    "id": neg.id,
                    "participants": neg.participant_ids,
                    "state": neg.state.value,
                    "current_terms": neg.current_proposal.terms if neg.current_proposal else {},
                    "proposer": neg.current_proposal.proposer_id if neg.current_proposal else "",
                    "round_started": neg.round_started,
                    "rounds_remaining": max(0, (neg.round_started + neg.max_rounds) - current_round),
                })

        for agr in self._agreements.values():
            if entity_id in agr.parties and not agr.violated:
                result["agreements"].append({
                    "id": agr.id,
                    "parties": agr.parties,
                    "terms": agr.terms,
                    "round_made": agr.round_made,
                })

        for auction in self._auctions.values():
            if auction.state == "open":
                result["auctions"].append({
                    "id": auction.id,
                    "item": auction.item_description,
                    "type": auction.auction_type,
                    "current_price": auction.current_price,
                    "my_bid": auction.bids.get(entity_id),
                    "round_started": auction.round_started,
                })

        return result

    def get_negotiation(self, negotiation_id: str) -> Optional[Negotiation]:
        return self._negotiations.get(negotiation_id)

    def remove_entity(self, entity_id: str) -> None:
        """Expire any active negotiations involving this entity."""
        for neg in self._negotiations.values():
            if entity_id in neg.participant_ids and neg.state in (
                NegotiationState.PROPOSED,
                NegotiationState.COUNTERED,
            ):
                neg.state = NegotiationState.EXPIRED

    def get_agreement(self, agreement_id: str) -> Optional[Agreement]:
        return self._agreements.get(agreement_id)

    def get_auction(self, auction_id: str) -> Optional[AuctionState]:
        return self._auctions.get(auction_id)

    # -- Serialization --

    def to_dict(self) -> dict:
        return {
            "negotiations": {nid: n.to_dict() for nid, n in self._negotiations.items()},
            "agreements": {aid: a.to_dict() for aid, a in self._agreements.items()},
            "auctions": {aid: a.to_dict() for aid, a in self._auctions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NegotiationManager":
        mgr = cls()
        for nid, ndata in data.get("negotiations", {}).items():
            mgr._negotiations[nid] = Negotiation.from_dict(ndata)
        for aid, adata in data.get("agreements", {}).items():
            mgr._agreements[aid] = Agreement.from_dict(adata)
        for aid, adata in data.get("auctions", {}).items():
            mgr._auctions[aid] = AuctionState.from_dict(adata)
        return mgr
