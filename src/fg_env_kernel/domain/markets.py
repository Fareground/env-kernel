"""Market DomainModules — full-featured price discovery + position tracking.

Two concrete market implementations available via the registry:
  - PredictionMarketModule  — CPMM-based binary-outcome prediction markets
  - SecuritiesTradingModule — market-impact model for crypto/stock-style trading

Both auto-register via DomainModuleRegistry on import.
"""
import logging
import random as _rng
from typing import Any, Dict, List, Optional

from .base import DomainConstraint, DomainModule

class PredictionMarketModule(DomainModule):
    """Prediction market domain: CPMM price discovery, position tracking, belief updating.

    Manages a binary outcome market where agents buy/sell YES and NO shares.
    Price = no_pool / (yes_pool + no_pool), representing the implied probability.

    Params:
      - initial_liquidity: float (default 1000) — total starting liquidity
      - initial_price: float (default 0.5) — starting YES price (0-1).
        Pools are split to match: yes_pool = liq*(1-price), no_pool = liq*price
        so that price = no_pool / (yes_pool + no_pool) = initial_price.
      - question: str (default "") — the market question, injected into agent perception
    """

    def __init__(self, name: str = "prediction_market", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)
        liq = float(self._params.get("initial_liquidity", 1000))
        initial_price = float(self._params.get("initial_price", 0.5))
        logging.getLogger(__name__).info(
            "PredictionMarketModule init: initial_price=%.4f, liquidity=%.0f, params=%s",
            initial_price, liq, self._params,
        )
        # Set pools so that price = yes_pool / (yes_pool + no_pool) = initial_price
        # Derivation: yes_pool = liq * price, no_pool = liq * (1 - price)
        initial_price = max(0.01, min(0.99, initial_price))
        self._yes_pool: float = liq * initial_price
        self._no_pool: float = liq * (1 - initial_price)
        self._positions: Dict[str, Dict[str, float]] = {}  # entity_id -> {yes_shares, no_shares}
        self._price_history: List[Dict[str, Any]] = []
        self._round_volume: float = 0.0
        self._round_trades: int = 0
        self._total_volume: float = 0.0
        self._positions_initialized: bool = False  # Track whether initial positions have been seeded
        self._trading_fee: float = float(self._params.get("trading_fee", 0.0))
        # Pinned to the sim seed by the engine via reseed(); see below.
        self._rng = _rng.Random(42)

    def reseed(self, rng) -> None:
        """Pin initial-position assignment to the sim seed (engine startup)."""
        self._rng = rng

        # Seed price history from real market data (negative round numbers = pre-sim)
        history_seed = self._params.get("price_history_seed", [])
        if history_seed:
            for i, price in enumerate(history_seed):
                p = float(price) if not isinstance(price, (int, float)) else price
                self._price_history.append({
                    "round": -(len(history_seed) - i),
                    "price": round(p, 4),
                    "yes_pool": 0, "no_pool": 0,
                    "volume": 0, "trades": 0,
                })

    @property
    def description(self) -> str:
        return "Prediction market domain: binary outcome betting with CPMM price discovery"

    @property
    def required_properties(self) -> List[str]:
        return ["budget", "belief", "risk_tolerance"]

    @property
    def custom_actions(self) -> List[str]:
        return ["buy_yes", "buy_no", "sell_yes", "sell_no", "hold", "argue_for", "argue_against"]

    @property
    def current_price(self) -> float:
        """Current YES price (implied probability).

        Standard CPMM: price = YES_pool / (YES_pool + NO_pool).
        Buying YES increases YES_pool → price goes UP (more demand = higher price).
        """
        total = self._yes_pool + self._no_pool
        return self._yes_pool / total if total > 0 else 0.5

    def _seed_initial_positions(self, state: Any) -> None:
        """Seed pre-existing positions to simulate market continuation.

        When ``initial_positions`` param is provided, a fraction of agents start
        with YES or NO shares based on their belief vs. market price.  This
        prevents the "cold start" problem where all agents rush to trade in
        round 1 from zero.

        Params (from ``initial_positions`` dict):
          ratio: float (0-1) — fraction of agents that start with positions
          avg_size: float — average position size in shares
        """
        import math

        config = self._params.get("initial_positions")
        if not config or not isinstance(config, dict):
            self._positions_initialized = True
            return

        ratio = float(config.get("ratio", 0.5))
        avg_size = float(config.get("avg_size", 50))
        price = self.current_price

        agents = list(state.get_agent_entities()) if hasattr(state, 'get_agent_entities') else []
        if not agents:
            self._positions_initialized = True
            return

        # Sort by id so sampling is stable regardless of entity insertion
        # order (which can differ after a snapshot/fork rebuild), and draw
        # from the sim-seeded RNG.
        agents = sorted(agents, key=lambda a: a.id)
        num_with_positions = max(1, int(len(agents) * ratio))
        selected = self._rng.sample(agents, min(num_with_positions, len(agents)))

        for agent in selected:
            belief = float(agent.properties.get("belief", 0.5))
            budget = float(agent.properties.get("budget", 0) or 0)
            if budget <= 0:
                continue

            # Position size: log-normal around avg_size (few large, many small)
            size = max(1.0, self._rng.lognormvariate(math.log(avg_size), 0.8))
            # Cost = size * price (approximate — real CPMM is nonlinear but this
            # is for initialization, not an actual trade through the AMM)
            direction = "yes" if belief > price else "no"
            cost_per_share = price if direction == "yes" else (1 - price)
            cost = size * cost_per_share

            # Don't spend more than 60% of budget on initial position
            max_spend = budget * 0.6
            if cost > max_spend:
                size = max_spend / max(cost_per_share, 0.01)
                cost = size * cost_per_share

            if size < 0.5:
                continue

            pos = self._positions.setdefault(agent.id, {"yes_shares": 0.0, "no_shares": 0.0})
            if direction == "yes":
                pos["yes_shares"] += round(size, 2)
            else:
                pos["no_shares"] += round(size, 2)

            # Deduct cost from budget (they "already bought in")
            agent.properties["budget"] = round(budget - cost, 2)

        self._positions_initialized = True

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        """Record price snapshot and reset per-round counters."""
        # Seed initial positions on first tick (when entities are available)
        if not self._positions_initialized:
            self._seed_initial_positions(state)

        snapshot = {
            "round": round_number,
            "price": round(self.current_price, 4),
            "yes_pool": round(self._yes_pool, 2),
            "no_pool": round(self._no_pool, 2),
            "volume": round(self._round_volume, 2),
            "trades": self._round_trades,
        }
        self._price_history.append(snapshot)
        self._total_volume += self._round_volume
        self._round_volume = 0.0
        self._round_trades = 0
        return [{"type": "market_tick", **snapshot}]

    def validate_action(self, action_name: str, actor: Any, target: Any,
                         state: Any) -> Optional[str]:
        """Check the agent has budget for the trade."""
        if action_name in ("buy_yes", "buy_no"):
            budget = 0.0
            if hasattr(actor, 'get'):
                budget = float(actor.get("budget", 0) or 0)
            elif hasattr(actor, 'properties'):
                budget = float(actor.properties.get("budget", 0) or 0)
            if budget <= 0:
                return f"Agent has no budget to trade (budget={budget})"
        return None

    def modify_resolution(
        self,
        actor_props: Dict[str, Any],
        target_props: Optional[Dict[str, Any]],
        action_def: Any,
        state: Any,
    ) -> tuple:
        """Inject current pool state and trading fee into resolution params."""
        if hasattr(action_def, 'resolution_archetype') and action_def.resolution_archetype == "cpmm":
            actor_props = dict(actor_props)
            actor_props["_yes_pool"] = self._yes_pool
            actor_props["_no_pool"] = self._no_pool
            actor_props["_trading_fee"] = self._trading_fee
        return actor_props, target_props

    def post_resolution(
        self,
        actor_id: str,
        action_name: str,
        success: bool,
        result: Any,
        state: Any,
    ) -> List[Dict[str, Any]]:
        """Update pools and positions after a trade."""
        changes: List[Dict[str, Any]] = []
        if not success:
            return changes

        details = result.details if hasattr(result, 'details') else {}

        if action_name in ("buy_yes", "buy_no"):
            # Update pools from resolution result
            new_yes = details.get("new_yes_pool")
            new_no = details.get("new_no_pool")
            if new_yes is not None and new_no is not None:
                self._yes_pool = new_yes
                self._no_pool = new_no

            # Update positions
            shares = details.get("shares", 0)
            direction = details.get("direction", "yes")
            amount = details.get("amount", 0)
            pos = self._positions.setdefault(actor_id, {"yes_shares": 0.0, "no_shares": 0.0})
            if direction == "yes":
                pos["yes_shares"] += shares
            else:
                pos["no_shares"] += shares

            # Track volume
            self._round_volume += amount
            self._round_trades += 1

            # Deduct budget from actor
            if hasattr(state, 'entities'):
                entity = state.entities.get(actor_id)
                if entity and hasattr(entity, 'properties'):
                    old_budget = float(entity.properties.get("budget", 0) or 0)
                    entity.properties["budget"] = round(old_budget - amount, 2)

            changes.append({
                "type": "market_trade",
                "actor": actor_id,
                "direction": direction,
                "amount": amount,
                "shares": shares,
                "price_after": details.get("price_after", self.current_price),
            })

        return changes

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        """Inject rich market state into agent perception.

        Provides comprehensive market context so agents can make informed
        trading decisions — similar to what a real trader would see on
        a prediction market dashboard.
        """
        pos = self._positions.get(entity_id, {"yes_shares": 0.0, "no_shares": 0.0})

        # Extended price history (last 50 rounds for trend analysis)
        history_window = self._price_history[-50:]
        recent_prices = [round(p["price"], 4) for p in history_window]
        recent_volumes = [p.get("volume", 0) for p in history_window]

        # Multi-timeframe sentiment
        def compute_trend(prices: list) -> str:
            if len(prices) < 2:
                return "neutral"
            change = prices[-1] - prices[0]
            return "bullish" if change > 0.02 else ("bearish" if change < -0.02 else "neutral")

        sentiment_short = compute_trend(recent_prices[-5:])   # last 5 rounds
        sentiment_medium = compute_trend(recent_prices[-15:])  # last 15 rounds
        sentiment_long = compute_trend(recent_prices)          # full window

        # Price statistics
        if recent_prices:
            price_high = max(recent_prices)
            price_low = min(recent_prices)
            price_avg = sum(recent_prices) / len(recent_prices)
            # Volatility (standard deviation)
            if len(recent_prices) > 1:
                variance = sum((p - price_avg) ** 2 for p in recent_prices) / len(recent_prices)
                volatility = variance ** 0.5
            else:
                volatility = 0.0
        else:
            price_high = price_low = price_avg = self.current_price
            volatility = 0.0

        # Aggregate market stats (anonymous — no individual trader data)
        total_yes = sum(p.get("yes_shares", 0) for p in self._positions.values())
        total_no = sum(p.get("no_shares", 0) for p in self._positions.values())
        active_traders = sum(1 for p in self._positions.values()
                            if p.get("yes_shares", 0) > 0 or p.get("no_shares", 0) > 0)

        # Recent market arguments/messages from event log (social feed)
        recent_arguments: list = []
        try:
            if hasattr(state, 'event_log'):
                all_events = state.event_log.get_all()
                # Get last 20 argue_for / argue_against events
                arg_events = [
                    e for e in all_events
                    if getattr(e, 'action_name', None) in ('argue_for', 'argue_against')
                       and getattr(e, 'event_type', '') == 'action_resolved'
                ]
                for e in arg_events[-20:]:
                    narrative = getattr(e, 'narrative', '')
                    if narrative:
                        direction = 'FOR' if e.action_name == 'argue_for' else 'AGAINST'
                        recent_arguments.append(f"[{direction}] {narrative[:200]}")
        except Exception:
            pass

        # Compute your P&L estimate
        yes_value = pos["yes_shares"] * self.current_price
        no_value = pos["no_shares"] * (1 - self.current_price)
        portfolio_value = round(yes_value + no_value, 2)

        # Betting odds (decimal format: payout per $1 risked)
        yes_price = self.current_price
        no_price = 1.0 - yes_price
        yes_odds = round(1.0 / max(yes_price, 0.01), 2)   # e.g. price=0.60 → 1.67x
        no_odds = round(1.0 / max(no_price, 0.01), 2)      # e.g. price=0.60 → 2.50x
        yes_payout = round(1.0 - yes_price, 4)               # Profit per share if YES wins
        no_payout = round(1.0 - no_price, 4)                 # Profit per share if NO wins

        data = {
            # Market identity
            "market_question": self._params.get("question", ""),
            # Current state
            "current_price": round(self.current_price, 4),
            "current_price_pct": f"{self.current_price * 100:.1f}%",
            # Betting odds
            "yes_odds": yes_odds,
            "no_odds": no_odds,
            "yes_cost": round(yes_price, 4),    # Cost to buy 1 YES share
            "no_cost": round(no_price, 4),      # Cost to buy 1 NO share
            "yes_payout_if_win": yes_payout,    # Profit per YES share if event happens
            "no_payout_if_win": no_payout,      # Profit per NO share if event doesn't happen
            # Price history (last 50 rounds)
            "price_history": recent_prices,
            "volume_history": recent_volumes,
            # Price stats
            "price_high": round(price_high, 4),
            "price_low": round(price_low, 4),
            "price_average": round(price_avg, 4),
            "volatility": round(volatility, 4),
            # Multi-timeframe sentiment
            "sentiment": {
                "short_term": sentiment_short,
                "medium_term": sentiment_medium,
                "long_term": sentiment_long,
            },
            # Your position (private)
            "your_position": {
                "yes_shares": round(pos["yes_shares"], 2),
                "no_shares": round(pos["no_shares"], 2),
                "estimated_portfolio_value": portfolio_value,
                "yes_shares_payout_if_win": round(pos["yes_shares"] * 1.0, 2),  # Each share pays $1
                "no_shares_payout_if_win": round(pos["no_shares"] * 1.0, 2),
            },
            # Market depth (public, anonymous)
            "market_depth": {
                "total_yes_interest": round(total_yes, 2),
                "total_no_interest": round(total_no, 2),
                "active_participants": active_traders,
                "buy_sell_ratio": round(total_yes / max(total_no, 0.01), 2),
            },
            "total_volume": round(self._total_volume, 2),
            "liquidity": {
                "yes_pool": round(self._yes_pool, 2),
                "no_pool": round(self._no_pool, 2),
            },
            # Social feed — recent arguments from other traders
            "recent_arguments": recent_arguments[-10:],
            # Trading guide — think in ODDS and EXPECTED VALUE, not just price
            "trading_guide": (
                f"THIS IS A BETTING MARKET. Each share pays $1 if the outcome occurs, $0 otherwise.\n"
                f"YES share costs {yes_price:.2f}¢ → odds {yes_odds:.2f}x → "
                f"profit {yes_payout:.2f}¢ per share if YES wins.\n"
                f"NO share costs {no_price:.2f}¢ → odds {no_odds:.2f}x → "
                f"profit {no_payout:.2f}¢ per share if NO wins.\n"
                f"DECISION RULE: Compare your BELIEF to the MARKET PRICE.\n"
                f"- If you believe YES probability > {yes_price * 100:.0f}%, buy YES "
                f"(expected value = belief × ${1:.2f} - cost ${yes_price:.2f} > 0).\n"
                f"- If you believe YES probability < {yes_price * 100:.0f}%, buy NO "
                f"(the NO side is underpriced at {no_odds:.2f}x odds).\n"
                f"- Bet SIZE should be proportional to your edge "
                f"(belief - market price). Bigger edge = bigger bet."
            ),
        }
        return data

    def get_visibility_overrides(self) -> Dict[str, Any]:
        """Override visibility rules for prediction market simulations.

        In prediction markets, agents are anonymous. They should NOT see
        other agents' identities, positions, or properties. They only
        observe aggregate market data (injected via get_perception_data).
        """
        return {
            "hide_agent_identities": True,
            "hide_agent_properties": True,
        }

    def get_constraints(self) -> List[DomainConstraint]:
        return [
            DomainConstraint(
                name="pool_positivity",
                description="Liquidity pools must remain positive",
                check_type="custom",
                params={"yes_pool_min": 0.01, "no_pool_min": 0.01},
                severity="error",
            ),
        ]

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["state"] = {
            "yes_pool": self._yes_pool,
            "no_pool": self._no_pool,
            "positions": dict(self._positions),
            "price_history": list(self._price_history),
            "total_volume": self._total_volume,
        }
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "PredictionMarketModule":
        module = cls(name=data.get("name", "prediction_market"), params=data.get("params", {}))
        market_state = data.get("state", {})
        module._yes_pool = market_state.get("yes_pool", module._yes_pool)
        module._no_pool = market_state.get("no_pool", module._no_pool)
        module._positions = market_state.get("positions", {})
        module._price_history = market_state.get("price_history", [])
        module._total_volume = market_state.get("total_volume", 0.0)
        return module


class SecuritiesTradingModule(DomainModule):
    """Securities trading domain: price discovery via market impact, position tracking, P&L.

    Models buying and selling a single security (crypto, stock, etc.) where:
    - Price starts at real market price and moves based on buy/sell pressure
    - Each buy pushes price up, each sell pushes price down
    - Impact magnitude depends on trade size relative to liquidity

    Params:
      - initial_price: float (required) — starting price in quote currency (e.g. USD)
      - liquidity: float (default 10000) — virtual liquidity depth
      - ticker: str (default "ASSET") — asset symbol (e.g. "BTC", "ETH")
      - quote_currency: str (default "USD") — quote currency symbol
    """

    def __init__(self, name: str = "securities_trading", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)
        self._current_price: float = float(self._params.get("initial_price", 100.0))
        self._liquidity: float = float(self._params.get("liquidity", 10000.0))
        self._ticker: str = str(self._params.get("ticker", "ASSET"))
        self._quote_currency: str = str(self._params.get("quote_currency", "USD"))
        # When True, agents may sell without holding shares (go short → shares
        # negative). Default False preserves the long-only Simulation product;
        # the predictor enables it so the market is genuinely two-sided and the
        # simulated price isn't structurally biased upward.
        self._allow_short: bool = bool(self._params.get("allow_short", False))
        # Per-round baseline drift (the asset's expected return per round).
        # Default 0 → price moves only from order flow (Simulation product
        # behavior). The predictor's optimizer tunes this to match an asset's
        # historical up/down base rate — a directional calibration knob.
        self._drift_per_round: float = float(self._params.get("drift_per_round", 0.0))
        self._positions: Dict[str, Dict[str, float]] = {}  # entity_id -> {shares, avg_entry, realized_pnl}
        self._price_history: List[Dict[str, Any]] = []
        self._round_volume: float = 0.0
        self._round_trades: int = 0
        self._round_buy_volume: float = 0.0
        self._round_sell_volume: float = 0.0
        self._total_volume: float = 0.0

        # Seed price history from real market data so the chart shows
        # the run-up to "now" instead of a flat line, and so agents'
        # perception sees a realistic trend. Negative round numbers
        # tag pre-sim history. The final seeded close becomes the
        # current_price so trading continues from the last known price.
        history_seed = self._params.get("price_history_seed", [])
        if isinstance(history_seed, list) and history_seed:
            for i, price in enumerate(history_seed):
                p = float(price) if isinstance(price, (int, float)) else None
                if p is None or p <= 0:
                    continue
                self._price_history.append({
                    "round": -(len(history_seed) - i),
                    "price": round(p, 4),
                    "volume": 0.0,
                    "trades": 0,
                    "buy_volume": 0.0,
                    "sell_volume": 0.0,
                })
            # Anchor the live tape to the last seeded close — otherwise
            # `initial_price` (passed separately) and the last seed bar
            # can disagree, producing a visual gap on the chart.
            last_seed = self._price_history[-1]["price"]
            if last_seed > 0:
                self._current_price = last_seed

        logging.getLogger(__name__).info(
            "SecuritiesTradingModule init: ticker=%s, price=%.4f, liquidity=%.0f, seeded=%d bars",
            self._ticker, self._current_price, self._liquidity, len(self._price_history),
        )

    @property
    def description(self) -> str:
        return f"Securities trading: {self._ticker} price discovery via market impact model"

    @property
    def required_properties(self) -> List[str]:
        return ["cash", "shares", "risk_tolerance"]

    @property
    def custom_actions(self) -> List[str]:
        return ["buy", "sell", "hold"]

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        """Record price snapshot and reset per-round counters."""
        # Apply the calibrated baseline drift before snapshotting (no-op when
        # drift_per_round is 0, i.e. the Simulation product).
        if self._drift_per_round:
            self._current_price = max(0.01, self._current_price * (1.0 + self._drift_per_round))
        snapshot = {
            "round": round_number,
            "price": round(self._current_price, 4),
            "volume": round(self._round_volume, 2),
            "trades": self._round_trades,
            "buy_volume": round(self._round_buy_volume, 2),
            "sell_volume": round(self._round_sell_volume, 2),
        }
        self._price_history.append(snapshot)
        self._round_volume = 0.0
        self._round_trades = 0
        self._round_buy_volume = 0.0
        self._round_sell_volume = 0.0
        return [{"type": "market_tick", **snapshot, "ticker": self._ticker}]

    def validate_action(self, action_name: str, actor: Any, target: Any,
                        state: Any) -> Optional[str]:
        if action_name == "buy":
            cash = getattr(actor, 'properties', {}).get("cash", 0) if hasattr(actor, 'properties') else 0
            if cash <= 0:
                return "Cannot buy: no cash available"
        elif action_name == "sell":
            if self._allow_short:
                return None  # shorting allowed — no inventory requirement
            entity_id = getattr(actor, 'id', str(actor))
            pos = self._positions.get(entity_id, {})
            shares = pos.get("shares", 0)
            if shares <= 0:
                return f"Cannot sell: no {self._ticker} shares held"
        return None

    def modify_resolution(
        self,
        actor_props: Dict[str, Any],
        target_props: Optional[Dict[str, Any]],
        action_def: Any,
        state: Any,
    ) -> tuple:
        archetype = getattr(action_def, 'resolution_archetype', '') if hasattr(action_def, 'resolution_archetype') else ''
        if archetype == "market_impact":
            actor_props = dict(actor_props)
            actor_props["_current_price"] = self._current_price
            actor_props["_liquidity"] = self._liquidity
            actor_props["_ticker"] = self._ticker
            actor_props["_allow_short"] = self._allow_short
            # Inject held shares for sell validation
            entity_id = actor_props.get("_entity_id", "")
            pos = self._positions.get(entity_id, {})
            actor_props["shares"] = pos.get("shares", 0)
        return actor_props, target_props

    def post_resolution(
        self,
        actor_id: str,
        action_name: str,
        success: bool,
        result: Any,
        state: Any,
    ) -> List[Dict[str, Any]]:
        if not success or action_name not in ("buy", "sell"):
            return []

        details = getattr(result, 'details', {}) if hasattr(result, 'details') else {}
        side = details.get("side", action_name)
        shares_traded = float(details.get("shares", 0))
        exec_price = float(details.get("execution_price", self._current_price))
        new_price = float(details.get("new_price", self._current_price))

        if shares_traded <= 0:
            return []

        # Update current price
        self._current_price = new_price

        # Update position
        pos = self._positions.setdefault(actor_id, {"shares": 0.0, "avg_entry": 0.0, "realized_pnl": 0.0})

        # Update agent's entity properties
        entity = None
        if hasattr(state, 'entities'):
            entity = state.entities.get(actor_id)

        if side == "buy":
            cost = float(details.get("amount", shares_traded * exec_price))
            old_shares = pos["shares"]
            new_total = old_shares + shares_traded
            if old_shares < 0:
                # Buying back (covering) a short: realize P&L on the covered
                # portion — a short profits when it covers BELOW its entry.
                # Mirror of the sell-side long-close split.
                covered = min(shares_traded, -old_shares)
                pos["realized_pnl"] = pos.get("realized_pnl", 0) + (
                    (pos["avg_entry"] - exec_price) * covered
                )
                if new_total > 0:
                    # Crossed through flat into a net long — basis is this fill.
                    pos["avg_entry"] = exec_price
                elif abs(new_total) < 1e-9:
                    pos["avg_entry"] = 0
                # else still net short → keep the short entry basis
            elif new_total > 0:
                # Adding to / opening a long — weighted-average entry price.
                old_cost = old_shares * pos["avg_entry"]
                pos["avg_entry"] = (old_cost + cost) / new_total
            pos["shares"] = new_total
            # Sync to entity properties
            if entity and hasattr(entity, 'properties'):
                entity.properties["cash"] = max(0, entity.properties.get("cash", 0) - cost)
                entity.properties["shares"] = pos["shares"]
            self._round_buy_volume += cost
            self._round_volume += cost
            self._total_volume += cost
        else:  # sell
            cash_received = float(details.get("cash_received", shares_traded * exec_price))
            old_shares = pos["shares"]
            if self._allow_short:
                # Realize P&L only on long inventory actually being closed —
                # NOT on the short portion (opening a short books no profit;
                # its basis is the entry price). Without this split, selling
                # from a flat position would mint spurious P&L.
                closed_long = max(0.0, min(old_shares, shares_traded))
                if closed_long > 0:
                    pos["realized_pnl"] = pos.get("realized_pnl", 0) + (
                        (exec_price - pos["avg_entry"]) * closed_long
                    )
                new_shares = old_shares - shares_traded
                if new_shares < 0:
                    # Net short — basis becomes the short entry price so that
                    # position value (shares * price, negative) reflects the
                    # liability and cash + position stays conserved.
                    pos["avg_entry"] = exec_price
                elif abs(new_shares) < 1e-9:
                    new_shares = 0.0
                    pos["avg_entry"] = 0
                pos["shares"] = new_shares
            else:
                realized = (exec_price - pos["avg_entry"]) * shares_traded
                pos["realized_pnl"] = pos.get("realized_pnl", 0) + realized
                pos["shares"] = max(0, old_shares - shares_traded)
                if pos["shares"] < 0.0001:
                    pos["shares"] = 0
                    pos["avg_entry"] = 0
            # Add cash and sync shares
            if entity and hasattr(entity, 'properties'):
                entity.properties["cash"] = entity.properties.get("cash", 0) + cash_received
                entity.properties["shares"] = pos["shares"]
            self._round_sell_volume += cash_received
            self._round_volume += cash_received
            self._total_volume += cash_received

        self._round_trades += 1

        return [{
            "type": "trade_executed",
            "ticker": self._ticker,
            "actor_id": actor_id,
            "side": side,
            "shares": round(shares_traded, 6),
            "execution_price": round(exec_price, 4),
            "new_price": round(new_price, 4),
            "position_shares": round(pos["shares"], 6),
        }]

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        pos = self._positions.get(entity_id, {"shares": 0, "avg_entry": 0, "realized_pnl": 0})
        shares = pos.get("shares", 0)
        avg_entry = pos.get("avg_entry", 0)
        # Works for both long (shares>0) and short (shares<0): a short loses
        # when price rises above its entry. Flat (0) → no exposure.
        unrealized_pnl = (self._current_price - avg_entry) * shares if shares else 0
        portfolio_value = shares * self._current_price

        # Price trend analysis
        prices = [h["price"] for h in self._price_history if h["price"] > 0]
        volatility = 0.0
        if len(prices) >= 2:
            import math
            returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices)) if prices[i - 1] > 0]
            if returns:
                mean_ret = sum(returns) / len(returns)
                variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                volatility = math.sqrt(variance)

        # Sentiment from recent price movement
        sentiment = "neutral"
        if len(prices) >= 3:
            short = prices[-1] - prices[-min(3, len(prices))]
            if short > 0:
                sentiment = "bullish"
            elif short < 0:
                sentiment = "bearish"

        # Buy/sell pressure
        recent_history = self._price_history[-5:] if self._price_history else []
        total_buy = sum(h.get("buy_volume", 0) for h in recent_history)
        total_sell = sum(h.get("sell_volume", 0) for h in recent_history)

        return {
            "ticker": self._ticker,
            "quote_currency": self._quote_currency,
            "current_price": round(self._current_price, 4),
            "price_history": [
                {"round": h["round"], "price": h["price"]}
                for h in self._price_history[-30:]
            ],
            "volatility": round(volatility, 6),
            "sentiment": sentiment,
            "buy_sell_pressure": {
                "recent_buy_volume": round(total_buy, 2),
                "recent_sell_volume": round(total_sell, 2),
                "ratio": round(total_buy / total_sell, 2) if total_sell > 0 else 999,
            },
            "your_position": {
                "shares": round(shares, 6),
                "avg_entry_price": round(avg_entry, 4),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "realized_pnl": round(pos.get("realized_pnl", 0), 2),
                "portfolio_value": round(portfolio_value, 2),
            },
            "total_volume": round(self._total_volume, 2),
            "liquidity": round(self._liquidity, 2),
        }

    def get_visibility_overrides(self) -> Dict[str, Any]:
        return {
            "hide_agent_identities": True,
            "hide_agent_properties": True,
        }

    def get_constraints(self) -> List[DomainConstraint]:
        return [
            DomainConstraint(
                name="price_positivity",
                description="Asset price must remain positive",
                check_type="custom",
                params={"min_price": 0.001},
                severity="error",
            )
        ]

    def to_dict(self) -> dict:
        return {
            "type": "securities_trading",
            "params": self._params,
            "market_state": {
                "current_price": self._current_price,
                "liquidity": self._liquidity,
                "ticker": self._ticker,
                "positions": self._positions,
                "price_history": self._price_history,
                "total_volume": self._total_volume,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SecuritiesTradingModule":
        module = cls(params=data.get("params", {}))
        ms = data.get("market_state", {})
        module._current_price = ms.get("current_price", module._current_price)
        module._liquidity = ms.get("liquidity", module._liquidity)
        module._ticker = ms.get("ticker", module._ticker)
        module._positions = ms.get("positions", {})
        module._price_history = ms.get("price_history", [])
        module._total_volume = ms.get("total_volume", 0.0)
        return module


