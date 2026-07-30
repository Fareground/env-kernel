"""Additional resolution archetype tests."""
from fg_env_kernel.resolution import (
    ProbabilisticSkillCheck, ContestOpposed,
    VotingResolution, EvidenceChainResolution, OrderBookResolution,
)
import random


class TestSkillCheckNormalization:
    """Test that skill values > 1.0 are normalized to 0-1 range."""

    def test_skill_100_scale(self):
        """Skills on 0-100 scale should be normalized to 0-1."""
        r = ProbabilisticSkillCheck()
        random.seed(42)
        result = r.resolve(
            actor_properties={"combat": 85},  # 0-100 scale
            target_properties=None,
            params={"skill_property": "combat", "difficulty": 0.3},
            action_params={},
        )
        # skill 85 -> normalized to 0.85
        assert result.details["skill"] == 0.85

    def test_skill_01_scale(self):
        """Skills on 0-1 scale should not be re-normalized."""
        r = ProbabilisticSkillCheck()
        random.seed(42)
        result = r.resolve(
            actor_properties={"combat": 0.85},
            target_properties=None,
            params={"skill_property": "combat", "difficulty": 0.3},
            action_params={},
        )
        assert result.details["skill"] == 0.85


class TestContestFairness:
    """Test that contest resolution is fair when stats are equal."""

    def test_equal_stats_roughly_50_50(self):
        r = ContestOpposed()
        wins = 0
        trials = 1000
        for _ in range(trials):
            result = r.resolve(
                actor_properties={"str": 50},
                target_properties={"def": 50},
                params={"attacker_property": "str", "defender_property": "def"},
                action_params={},
            )
            if result.success:
                wins += 1
        # Should be roughly 50/50 (+/- 10%)
        assert 350 < wins < 650, f"Expected ~500 wins, got {wins}"


# -----------------------------------------------------------------------
# Voting Resolution
# -----------------------------------------------------------------------

class TestVotingResolution:
    """Tests for the VotingResolution archetype."""

    def test_high_persuasion_succeeds(self):
        """High persuasion with low threshold should usually succeed."""
        r = VotingResolution()
        random.seed(42)
        result = r.resolve(
            actor_properties={"persuasion": 0.9},
            target_properties=None,
            params={"voting_property": "persuasion", "threshold": 0.3},
            action_params={},
        )
        # 0.9 * 0.6 = 0.54, + noise * 0.4 -> very likely > 0.3
        assert result.success is True
        assert "votes_for" in result.details
        assert "vote_share" in result.details

    def test_low_persuasion_fails(self):
        """Low persuasion with high threshold should usually fail."""
        r = VotingResolution()
        random.seed(0)
        result = r.resolve(
            actor_properties={"persuasion": 0.1},
            target_properties=None,
            params={"voting_property": "persuasion", "threshold": 0.9},
            action_params={},
        )
        # 0.1 * 0.6 = 0.06, + noise * 0.4 -> very unlikely > 0.9
        assert result.success is False
        assert result.details["votes_for"] + result.details["votes_against"] == 100

    def test_voting_fairness(self):
        """With 50% persuasion and 50% threshold, should be roughly 50/50."""
        r = VotingResolution()
        wins = 0
        trials = 1000
        for _ in range(trials):
            result = r.resolve(
                actor_properties={"persuasion": 0.5},
                target_properties=None,
                params={"voting_property": "persuasion", "threshold": 0.5},
                action_params={},
            )
            if result.success:
                wins += 1
        # With p=0.5 * 0.6 + U(0,1)*0.4 vs 0.5, success when result > 0.5
        # Expected to be around 50% +/- variance
        assert 200 < wins < 800, f"Expected ~500 wins, got {wins}"


# -----------------------------------------------------------------------
# Evidence Chain Resolution
# -----------------------------------------------------------------------

class TestEvidenceChainResolution:
    """Tests for the EvidenceChainResolution archetype."""

    def test_high_skill_succeeds(self):
        """High investigation skill with low threshold succeeds."""
        r = EvidenceChainResolution()
        random.seed(42)
        result = r.resolve(
            actor_properties={"investigation": 0.9},
            target_properties=None,
            params={"evidence_property": "investigation", "proof_threshold": 0.3},
            action_params={},
        )
        assert result.success is True
        assert "evidence_weight" in result.details
        assert "margin" in result.details

    def test_low_skill_fails(self):
        """Low investigation skill with high threshold fails."""
        r = EvidenceChainResolution()
        random.seed(0)
        result = r.resolve(
            actor_properties={"investigation": 0.1},
            target_properties=None,
            params={"evidence_property": "investigation", "proof_threshold": 0.9},
            action_params={},
        )
        assert result.success is False
        assert result.details["margin"] < 0

    def test_evidence_normalizes_100_scale(self):
        """Skills on 0-100 scale are normalized to 0-1."""
        r = EvidenceChainResolution()
        random.seed(42)
        result = r.resolve(
            actor_properties={"investigation": 80},
            target_properties=None,
            params={"evidence_property": "investigation", "proof_threshold": 0.3},
            action_params={},
        )
        # 80 -> 0.80 normalized
        assert result.success is True


# -----------------------------------------------------------------------
# Order Book Resolution
# -----------------------------------------------------------------------

class TestOrderBookResolution:
    """Tests for the OrderBookResolution archetype."""

    def test_high_price_succeeds(self):
        """High price competitiveness should fill the order."""
        r = OrderBookResolution()
        random.seed(42)
        result = r.resolve(
            actor_properties={"bargaining": 80},
            target_properties=None,
            params={"price_property": "bargaining", "resource": "wheat"},
            action_params={},
        )
        # 80/100 = 0.8 + noise*0.3 -> very likely > 0.5
        assert result.success is True
        assert "fill_rate" in result.details
        assert "price_offered" in result.details

    def test_low_price_fails(self):
        """Low price competitiveness should not fill the order."""
        r = OrderBookResolution()
        random.seed(0)
        result = r.resolve(
            actor_properties={"bargaining": 5},
            target_properties=None,
            params={"price_property": "bargaining", "resource": "wheat"},
            action_params={},
        )
        # 5/100 = 0.05 + noise*0.3 -> max ~0.35, unlikely > 0.5
        assert result.success is False

    def test_fill_rate_clamped(self):
        """Fill rate should always be between 0 and 1."""
        r = OrderBookResolution()
        for seed in range(100):
            random.seed(seed)
            result = r.resolve(
                actor_properties={"bargaining": 95},
                target_properties=None,
                params={"price_property": "bargaining", "resource": "gold"},
                action_params={},
            )
            assert 0.0 <= result.details["fill_rate"] <= 1.0
