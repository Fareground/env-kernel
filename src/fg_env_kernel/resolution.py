"""Resolution archetypes -- how action outcomes are determined."""
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _safe_float(value: Any, default: float = 50.0) -> float:
    """Safely convert a property value to float. Non-numeric values get the default."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


@dataclass
class ResolutionResult:
    """Outcome of resolving an action."""
    success: bool
    magnitude: float = 1.0
    narrative: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    # Partial success support: graduated outcomes between full success and failure
    partial: bool = False
    success_degree: float = 1.0  # 0.0 (total failure) to 1.0 (total success)


class ResolutionArchetype(ABC):
    """Base class for all resolution mechanics."""

    @abstractmethod
    def resolve(
        self,
        actor_properties: Dict[str, Any],
        target_properties: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        action_params: Dict[str, Any],
        rng: Optional[random.Random] = None,
    ) -> ResolutionResult:
        """Resolve an action attempt. Returns success/failure + magnitude."""
        ...


class DeterministicResolution(ResolutionArchetype):
    """Always succeeds. Used for unconditional actions like 'move' or 'speak'."""

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        return ResolutionResult(
            success=True,
            magnitude=1.0,
            narrative="Action succeeds.",
        )


class ProbabilisticSkillCheck(ResolutionArchetype):
    """
    Roll against a skill property with a difficulty threshold.

    Params:
      skill_property: str  -- actor property to check (e.g., "charisma")
      difficulty: float    -- 0.0 (trivial) to 1.0 (impossible)

    Mechanic:
      roll = random(0, 1)
      effective = roll * 0.5 + skill * 0.5  (50% luck, 50% skill)
      success = effective > difficulty
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        skill_prop = params.get("skill_property", "skill")
        difficulty = params.get("difficulty", 0.5)

        skill_value = _safe_float(actor_properties.get(skill_prop, 0.5), 0.5)
        # Normalize to 0-1 if needed
        skill_normalized = max(0.0, min(1.0, skill_value / 100.0 if skill_value > 1.0 else skill_value))

        _rng = rng or random.Random()
        roll = _rng.random()
        total = roll * 0.5 + skill_normalized * 0.5
        margin = abs(total - difficulty)

        # Graduated success: success_degree = how well the check was passed
        success_degree = min(1.0, total / max(difficulty, 0.01))
        success = success_degree >= 0.7  # Full success at 70%+ of difficulty
        partial = not success and success_degree >= 0.4  # Partial at 40-70%

        if success:
            outcome_str = "Success"
        elif partial:
            outcome_str = f"Partial success ({success_degree:.0%})"
        else:
            outcome_str = "Failure"

        return ResolutionResult(
            success=success,
            partial=partial,
            success_degree=round(success_degree, 3),
            magnitude=min(1.0, margin * 2),
            narrative=f"Skill check: roll {roll:.2f}, skill {skill_normalized:.2f}, total {total:.2f} vs difficulty {difficulty:.2f}. {outcome_str}.",
            details={"roll": roll, "skill": skill_normalized, "total": total, "difficulty": difficulty, "margin": margin, "success_degree": round(success_degree, 3)},
        )


class ContestOpposed(ResolutionArchetype):
    """
    Two entities compete. Higher roll + stat wins.

    Params:
      attacker_property: str
      defender_property: str
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        att_prop = params.get("attacker_property", "strength")
        def_prop = params.get("defender_property", "defense")

        if target_properties is None:
            return ResolutionResult(success=True, magnitude=1.0, narrative="No defender present.", success_degree=1.0)

        att_val = _safe_float(actor_properties.get(att_prop, 50))
        def_val = _safe_float(target_properties.get(def_prop, 50))

        _rng = rng or random.Random()
        att_roll = _rng.random() * 100 + att_val
        def_roll = _rng.random() * 100 + def_val

        margin = abs(att_roll - def_roll)
        # success_degree: how dominant the attacker was (0.5 = dead even)
        total = att_roll + def_roll
        success_degree = att_roll / max(total, 0.01)
        success = success_degree > 0.50  # Attacker must be strictly dominant (no tie bias)
        partial = not success and success_degree >= 0.45  # Close contest — partial success

        if success:
            outcome_str = "Attacker wins"
        elif partial:
            outcome_str = f"Narrow contest ({success_degree:.0%} dominance)"
        else:
            outcome_str = "Defender wins"

        return ResolutionResult(
            success=success,
            partial=partial,
            success_degree=round(success_degree, 3),
            magnitude=min(1.0, margin / 100.0),
            narrative=f"Contest: attacker {att_roll:.1f} vs defender {def_roll:.1f}. {outcome_str}.",
            details={"attacker_roll": att_roll, "defender_roll": def_roll, "margin": margin, "success_degree": round(success_degree, 3)},
        )


class VotingResolution(ResolutionArchetype):
    """
    Governance / group-decision mechanic. Simulates a vote among a virtual pool.

    Params:
      voting_property: str  -- actor property influencing persuasion (0-1)
      threshold: float      -- required vote share (0.5=majority, 0.67=supermajority)

    Mechanic:
      base_share = voting_property_value * 0.6 + random * 0.4
      success = base_share > threshold
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        vote_prop = params.get("voting_property", "persuasion")
        threshold = params.get("threshold", 0.5)

        prop_value = _safe_float(actor_properties.get(vote_prop, 0.5), 0.5)
        # Normalize to 0-1 if on 0-100 scale
        prop_normalized = max(0.0, min(1.0, prop_value / 100.0 if prop_value > 1.0 else prop_value))

        _rng = rng or random.Random()
        noise = _rng.random()
        vote_share = prop_normalized * 0.6 + noise * 0.4
        total_votes = 100
        votes_for = int(vote_share * total_votes)
        votes_against = total_votes - votes_for

        # Graduated success: how close to threshold
        success_degree = min(1.0, vote_share / max(threshold, 0.01))
        success = vote_share > threshold
        partial = not success and success_degree >= 0.7  # Got close to passing

        if success:
            outcome_str = "Motion carries"
        elif partial:
            outcome_str = f"Motion narrowly fails ({success_degree:.0%} of threshold)"
        else:
            outcome_str = "Motion fails"

        return ResolutionResult(
            success=success,
            partial=partial,
            success_degree=round(success_degree, 3),
            magnitude=vote_share,
            narrative=(
                f"Vote: {votes_for} for, {votes_against} against "
                f"(share {vote_share:.2f} vs threshold {threshold:.2f}). "
                f"{outcome_str}."
            ),
            details={
                "votes_for": votes_for,
                "votes_against": votes_against,
                "total_votes": total_votes,
                "vote_share": round(vote_share, 4),
                "success_degree": round(success_degree, 3),
            },
        )


class EvidenceChainResolution(ResolutionArchetype):
    """
    Investigation / deduction mechanic. Accumulates evidence weight against a proof threshold.

    Params:
      evidence_property: str    -- actor property like "investigation" (0-1)
      proof_threshold: float    -- threshold for conclusive evidence (0.0-1.0)

    Mechanic:
      evidence_weight = evidence_property * 0.6 + random * 0.4
      success = evidence_weight > proof_threshold
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        evidence_prop = params.get("evidence_property", "investigation")
        threshold = params.get("proof_threshold", 0.5)

        prop_value = _safe_float(actor_properties.get(evidence_prop, 0.5), 0.5)
        prop_normalized = max(0.0, min(1.0, prop_value / 100.0 if prop_value > 1.0 else prop_value))

        _rng = rng or random.Random()
        noise = _rng.random()
        evidence_weight = prop_normalized * 0.6 + noise * 0.4
        margin = evidence_weight - threshold

        # Graduated success
        success_degree = min(1.0, evidence_weight / max(threshold, 0.01))
        success = evidence_weight > threshold
        partial = not success and success_degree >= 0.6  # Partial evidence found

        if success:
            outcome_str = "Conclusive evidence found"
        elif partial:
            outcome_str = f"Partial evidence ({success_degree:.0%} of proof threshold)"
        else:
            outcome_str = "Insufficient evidence"

        return ResolutionResult(
            success=success,
            partial=partial,
            success_degree=round(success_degree, 3),
            magnitude=abs(margin),
            narrative=(
                f"Evidence weight {evidence_weight:.2f} vs threshold {threshold:.2f} "
                f"(margin {margin:+.2f}). "
                f"{outcome_str}."
            ),
            details={
                "evidence_weight": round(evidence_weight, 4),
                "threshold": threshold,
                "margin": round(margin, 4),
                "success_degree": round(success_degree, 3),
            },
        )


class OrderBookResolution(ResolutionArchetype):
    """
    Market / auction mechanic. Simulates matching a buy/sell order.

    Params:
      resource: str         -- the resource being traded
      price_property: str   -- actor property influencing price competitiveness (0-100)

    Mechanic:
      fill_rate = clamp(price_property / 100 + random * 0.3, 0, 1)
      success = fill_rate > 0.5
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        price_prop = params.get("price_property", "bargaining")
        resource = params.get("resource", "goods")

        price_value = _safe_float(actor_properties.get(price_prop, 50))
        # Normalize to 0-1
        price_normalized = max(0.0, min(1.0, price_value / 100.0 if price_value > 1.0 else price_value))

        _rng = rng or random.Random()
        noise = _rng.random() * 0.3
        fill_rate = max(0.0, min(1.0, price_normalized + noise))

        # Graduated success: fill_rate IS the success_degree
        success_degree = fill_rate
        success = fill_rate > 0.6  # Full fill
        partial = not success and fill_rate > 0.3  # Partial fill

        if success:
            outcome_str = "Order filled"
        elif partial:
            outcome_str = f"Partially filled ({fill_rate:.0%})"
        else:
            outcome_str = "Order unfilled"

        return ResolutionResult(
            success=success,
            partial=partial,
            success_degree=round(success_degree, 3),
            magnitude=fill_rate,
            narrative=(
                f"Order for {resource}: fill rate {fill_rate:.2f} "
                f"(price competitiveness {price_normalized:.2f}). "
                f"{outcome_str}."
            ),
            details={
                "fill_rate": round(fill_rate, 4),
                "price_offered": round(price_normalized, 4),
                "success_degree": round(success_degree, 3),
            },
        )


class CPMMResolution(ResolutionArchetype):
    """
    Constant Product Market Maker for prediction markets.

    Implements the x*y=k invariant used by Polymarket/Uniswap.
    Two liquidity pools (yes_pool, no_pool). Price = no_pool / (yes_pool + no_pool).
    Buying YES deposits cash into yes_pool and receives shares proportional to
    the change in no_pool.

    Params (from domain module state):
      yes_pool: float   -- current YES liquidity pool
      no_pool: float    -- current NO liquidity pool

    Action params (from agent decision):
      direction: str    -- "yes" or "no"
      amount: float     -- cash to spend

    Returns:
      success: True if amount > 0 and pools remain positive
      magnitude: shares received
      details: {price_before, price_after, shares, slippage, direction, amount}
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        yes_pool = _safe_float(params.get("yes_pool", 1000), 1000)
        no_pool = _safe_float(params.get("no_pool", 1000), 1000)
        trading_fee = _safe_float(params.get("trading_fee", 0), 0)
        # Infer direction from action name if not explicitly provided
        # This handles LLM agents that don't include direction in parameters
        action_name = params.get("action_name", "")
        direction = action_params.get("direction", None)
        if direction is None:
            if "no" in action_name:
                direction = "no"
            else:
                direction = "yes"
        amount = _safe_float(action_params.get("amount", 0), 0)

        # Auto-size: if agent didn't specify amount, use ~10-20% of their budget
        if amount <= 0:
            budget = _safe_float(actor_properties.get("budget", 0), 0)
            if budget > 0:
                rng_local = rng or __import__("random").Random()
                fraction = rng_local.uniform(0.08, 0.25)
                amount = round(budget * fraction, 2)

        # Apply trading fee (e.g. 2% on Polymarket) — fee is taken from the
        # trade amount before it enters the pool, reducing effective purchasing power
        fee_amount = 0.0
        if trading_fee > 0 and amount > 0:
            fee_amount = round(amount * trading_fee, 4)
            amount = round(amount - fee_amount, 4)

        if amount <= 0 or yes_pool <= 0 or no_pool <= 0:
            return ResolutionResult(
                success=False,
                magnitude=0.0,
                narrative="Trade failed: invalid amount or depleted pool.",
                details={"direction": direction, "amount": amount},
            )

        # Per-fill slippage cap. Without this, one trade against a small
        # pool can move price by tens of pp in a single print
        # (e.g. a $200 buy against a $500/$500 pool jumps price 0.50 →
        # 0.66). Real prediction markets break large orders into smaller
        # fills against deeper liquidity — we approximate that by
        # limiting the effective amount that hits the pool per resolution
        # to MAX_TRADE_FRAC_OF_POOL. Anything above that is silently
        # refunded so the agent doesn't get penalized for over-sizing.
        MAX_TRADE_FRAC_OF_POOL = 0.05  # ≈ 2.4pp max move at price=0.5
        affected_pool = yes_pool if direction == "yes" else no_pool
        max_amount = affected_pool * MAX_TRADE_FRAC_OF_POOL
        if amount > max_amount:
            refunded = round(amount - max_amount, 4)
            amount = round(max_amount, 4)
        else:
            refunded = 0.0

        k = yes_pool * no_pool
        price_before = yes_pool / (yes_pool + no_pool)

        if direction == "yes":
            # Buying YES: deposit `amount` into yes_pool, receive shares from no_pool
            new_yes = yes_pool + amount
            new_no = k / new_yes
            shares = no_pool - new_no
        else:
            # Buying NO: deposit `amount` into no_pool, receive shares from yes_pool
            new_no = no_pool + amount
            new_yes = k / new_no
            shares = yes_pool - new_yes

        if shares <= 0 or new_yes <= 0 or new_no <= 0:
            return ResolutionResult(
                success=False,
                magnitude=0.0,
                narrative="Trade failed: insufficient liquidity.",
                details={"direction": direction, "amount": amount},
            )

        price_after = new_yes / (new_yes + new_no)
        slippage = abs(price_after - price_before)

        total_cost = amount + fee_amount  # Total deducted from agent budget — refunded portion is NOT charged
        fee_note = f" (fee: {fee_amount:.2f})" if fee_amount > 0 else ""
        cap_note = f" (capped: refunded ${refunded:.2f} for slippage protection)" if refunded > 0 else ""
        return ResolutionResult(
            success=True,
            magnitude=round(shares, 4),
            narrative=(
                f"{'Bought YES' if direction == 'yes' else 'Bought NO'}: "
                f"spent {total_cost:.2f}{fee_note}, received {shares:.2f} shares. "
                f"Price moved {price_before:.4f} → {price_after:.4f}.{cap_note}"
            ),
            details={
                "direction": direction,
                "amount": total_cost,
                "shares": round(shares, 4),
                "price_before": round(price_before, 4),
                "price_after": round(price_after, 4),
                "slippage": round(slippage, 4),
                "fee": round(fee_amount, 4),
                "refunded": round(refunded, 4),
                "new_yes_pool": round(new_yes, 4),
                "new_no_pool": round(new_no, 4),
            },
        )


class MarketImpactResolution(ResolutionArchetype):
    """
    Securities trading resolution using a market impact model.

    Models price movement from buy/sell orders: buys push the price up,
    sells push the price down.  The magnitude of movement depends on
    trade size relative to market liquidity.

    Params (injected by SecuritiesTradingModule via modify_resolution):
      current_price: float   -- current asset price in quote currency
      liquidity: float       -- virtual liquidity depth (higher = less slippage)
      ticker: str            -- asset ticker symbol (informational)

    Action params (from agent decision):
      side: str              -- "buy" or "sell"
      amount: float          -- cash amount for buys, share count for sells

    Returns:
      success: True if trade executed
      magnitude: number of shares traded
      details: {side, amount, shares, execution_price, price_before, price_after,
                slippage_pct, new_price}
    """

    def resolve(self, actor_properties, target_properties, params,
                action_params, rng=None) -> ResolutionResult:
        current_price = _safe_float(params.get("current_price", 100), 100)
        liquidity = _safe_float(params.get("liquidity", 10000), 10000)
        # Determine the trade side. Prefer an explicit ``side`` param, but
        # fall back to the action name (the engine passes it in params) —
        # otherwise an agent that picks the ``sell`` action without setting
        # ``parameters={"side": "sell"}`` (e.g. the trait-conditioned or random
        # strategy) would be silently executed as a BUY, making every action
        # raise the price and the market one-directional.
        side = action_params.get("side")
        if side not in ("buy", "sell"):
            action_name = str(params.get("action_name", "")).lower()
            side = "sell" if action_name == "sell" else "buy"
        amount = _safe_float(action_params.get("amount", 0), 0)

        # Auto-size: if agent didn't specify, use ~5-15% of available cash (buy)
        # or ~10-30% of shares (sell)
        if amount <= 0:
            if side == "buy":
                cash = _safe_float(actor_properties.get("cash", 0), 0)
                if cash > 0:
                    _rng = rng or random.Random()
                    amount = round(cash * _rng.uniform(0.05, 0.15), 2)
            else:
                shares = _safe_float(actor_properties.get("shares", 0), 0)
                _rng = rng or random.Random()
                if shares > 0:
                    amount = round(shares * _rng.uniform(0.1, 0.3), 6)
                elif params.get("allow_short"):
                    # Short sale: the agent holds no inventory but shorting is
                    # enabled, so size the sale from its cash budget (in share
                    # terms), like a buy. Without this, short sells auto-size to
                    # zero and have no price impact — the market stays one-sided.
                    cash = _safe_float(actor_properties.get("cash", 0), 0)
                    if cash > 0 and current_price > 0:
                        amount = round((cash * _rng.uniform(0.05, 0.15)) / current_price, 6)

        if amount <= 0 or current_price <= 0:
            return ResolutionResult(
                success=False, magnitude=0.0,
                narrative="Trade failed: invalid amount or price.",
                details={"side": side, "amount": amount},
            )

        # Market impact: concave square-root model with a hard cap.
        #
        # The old model was strictly linear: impact = notional / liquidity.
        # That blew up the moment any single agent's trade approached the
        # liquidity figure — a $5K trade against $10K liquidity moved
        # price 50%, and a few rounds of compounding (1+0.5)^N sent the
        # price from $267 to $5987 in three rounds (observed in prod).
        #
        # Real markets have CONCAVE impact (Kyle-lambda / sqrt of
        # notional traded), and individual prints can't move the price
        # by more than a few percent without triggering circuit
        # breakers. We mimic that here:
        #   1. Use sqrt(notional / liquidity) so large trades have
        #      diminishing returns per dollar.
        #   2. Hard-cap impact at MAX_IMPACT_PER_TRADE (2.5%) so no
        #      single agent can spike the tape.
        # Compounded across many agents the price still moves, but at a
        # realistic pace.
        import math as _math
        MAX_IMPACT_PER_TRADE = 0.025  # 2.5% per fill
        IMPACT_K = 0.08               # coefficient on sqrt term

        if side == "buy":
            notional = amount
        else:
            notional = amount * current_price

        raw_ratio = max(0.0, notional / liquidity) if liquidity > 0 else 0.0
        impact = min(MAX_IMPACT_PER_TRADE, IMPACT_K * _math.sqrt(raw_ratio))
        price_before = current_price

        if side == "buy":
            # Execution price is above current (you pay slippage)
            exec_price = current_price * (1.0 + impact * 0.5)
            shares = amount / exec_price
            new_price = current_price * (1.0 + impact)
        else:
            # Check sufficient shares — unless shorting is allowed, in which
            # case the agent may sell beyond its inventory (go short). Without
            # this carve-out a short sale is rejected here and has no price
            # impact, leaving the market one-sided.
            held = _safe_float(actor_properties.get("shares", 0), 0)
            if not params.get("allow_short") and amount > held + 0.0001:
                return ResolutionResult(
                    success=False, magnitude=0.0,
                    narrative="Trade failed: insufficient shares to sell.",
                    details={"side": side, "amount": amount, "held": held},
                )
            # Execution price is below current (you receive less)
            exec_price = current_price * (1.0 - impact * 0.5)
            if exec_price <= 0:
                exec_price = current_price * 0.01
            shares = amount  # shares sold
            cash_received = amount * exec_price
            new_price = current_price * (1.0 - impact)
            if new_price <= 0:
                new_price = current_price * 0.01

        slippage_pct = abs(exec_price - current_price) / current_price * 100

        details = {
            "side": side,
            "amount": round(amount, 6),
            "shares": round(shares, 6),
            "execution_price": round(exec_price, 4),
            "price_before": round(price_before, 4),
            "price_after": round(new_price, 4),
            "slippage_pct": round(slippage_pct, 4),
            "new_price": round(new_price, 4),
        }
        if side == "sell":
            details["cash_received"] = round(cash_received, 4)

        ticker = params.get("ticker", "ASSET")
        return ResolutionResult(
            success=True,
            magnitude=round(shares, 6),
            narrative=(
                f"{'Bought' if side == 'buy' else 'Sold'} {shares:.4f} {ticker} "
                f"@ ${exec_price:,.2f}. "
                f"Price moved ${price_before:,.2f} → ${new_price:,.2f} "
                f"({slippage_pct:.2f}% slippage)."
            ),
            details=details,
        )


class DeterministicMath(ResolutionArchetype):
    """
    Pure math resolution -- always succeeds, magnitude computed from parameters.
    Used for economic actions, crafting, etc.
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        amount = action_params.get("amount", params.get("amount", 1))
        safe_amount = _safe_float(amount, 1.0)
        return ResolutionResult(
            success=True,
            magnitude=safe_amount,
            narrative=f"Deterministic: amount {amount}.",
            details={"amount": amount},
        )


class AuctionSealedFirstPrice(ResolutionArchetype):
    """Sealed-bid first-price auction. Each agent submits a `bid`
    parameter; highest bid wins; winner pays their bid.

    Used inside a `resolution_mode: simultaneous` phase so all bids
    submit against the same perception. Caller (action effect) is
    responsible for transferring the winning bid as money — the
    resolution archetype just identifies the winner + price.
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        # In simultaneous mode the engine collects all bids and passes
        # them in via `params["all_bids"]` (dict of actor_id → bid).
        all_bids = params.get("all_bids") or {}
        my_bid = float(action_params.get("bid", params.get("bid", 0)))
        if not all_bids:
            return ResolutionResult(success=True, magnitude=my_bid,
                                    narrative=f"Sealed bid: {my_bid}", details={"bid": my_bid})
        # Top bid wins
        winner_id, winner_bid = max(all_bids.items(), key=lambda kv: float(kv[1]))
        actor_id = params.get("_actor_id") or ""
        won = actor_id == winner_id
        return ResolutionResult(
            success=won,
            magnitude=winner_bid if won else 0,
            narrative=("Won the auction" if won else f"Bid {my_bid}, lost to {winner_id}"),
            details={"winner_id": winner_id, "winning_bid": winner_bid,
                     "my_bid": my_bid, "all_bids": dict(all_bids)},
        )


class AuctionSealedSecondPrice(ResolutionArchetype):
    """Sealed-bid second-price (Vickrey) auction. Highest bid wins
    but pays the SECOND-highest bid. Encourages truthful bidding."""

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        all_bids = params.get("all_bids") or {}
        my_bid = action_params.get("bid", params.get("bid", 0))
        if not all_bids:
            return ResolutionResult(success=True, magnitude=0, narrative="Single bid",
                                    details={"bid": my_bid})
        sorted_bids = sorted(all_bids.items(), key=lambda kv: float(kv[1]), reverse=True)
        winner_id, _ = sorted_bids[0]
        price = float(sorted_bids[1][1]) if len(sorted_bids) > 1 else float(sorted_bids[0][1])
        actor_id = params.get("_actor_id") or ""
        won = actor_id == winner_id
        return ResolutionResult(
            success=won,
            magnitude=price if won else 0,
            narrative=("Won at second-price " + str(price)) if won else "Lost the auction",
            details={"winner_id": winner_id, "price": price,
                     "my_bid": my_bid, "all_bids": dict(all_bids)},
        )


class AuctionEnglish(ResolutionArchetype):
    """English open-outcry — bids ascend over rounds; resolves when
    only one bidder remains active. Stateless archetype that just
    reports the current high-bid state; the state_machine module
    drives the round structure."""

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        bid = float(action_params.get("bid", params.get("bid", 0)))
        high_bid = float(params.get("current_high_bid", 0))
        ok = bid > high_bid
        return ResolutionResult(
            success=ok,
            magnitude=bid if ok else high_bid,
            narrative=("Raised to " + str(bid)) if ok else f"Bid {bid} <= high {high_bid}",
            details={"bid": bid, "current_high_bid": high_bid, "valid": ok},
        )


class AuctionDutch(ResolutionArchetype):
    """Dutch (descending-price) auction. Price drops each round; first
    accept claim wins at the current price. `params['current_price']`
    is set by the state machine; `bid=accept` resolves success."""

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        current_price = float(params.get("current_price", 0))
        accept = bool(action_params.get("accept", params.get("accept", False))
                      or action_params.get("bid") == "accept")
        return ResolutionResult(
            success=accept,
            magnitude=current_price if accept else 0,
            narrative=f"{'Accepted' if accept else 'Passed'} at {current_price}",
            details={"price": current_price, "accepted": accept},
        )


# Registry of all resolution archetypes
RESOLUTION_REGISTRY: Dict[str, ResolutionArchetype] = {
    "deterministic": DeterministicResolution(),
    "skill_check": ProbabilisticSkillCheck(),
    "contest": ContestOpposed(),
    "deterministic_math": DeterministicMath(),
    "voting": VotingResolution(),
    "evidence_chain": EvidenceChainResolution(),
    "order_book": OrderBookResolution(),
    "cpmm": CPMMResolution(),
    "market_impact": MarketImpactResolution(),
    # Tier 5c — auction family
    "auction_sealed_first": AuctionSealedFirstPrice(),
    "auction_sealed_second": AuctionSealedSecondPrice(),
    "auction_english": AuctionEnglish(),
    "auction_dutch": AuctionDutch(),
}


def get_resolution(name: str) -> ResolutionArchetype:
    """Get a resolution archetype by name."""
    if name not in RESOLUTION_REGISTRY:
        raise ValueError(f"Unknown resolution archetype: '{name}'. Available: {list(RESOLUTION_REGISTRY.keys())}")
    return RESOLUTION_REGISTRY[name]
