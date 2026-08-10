"""Cognitive architecture -- emotions, biases, bounded rationality.

Models human-like cognitive limitations for simulation agents:
- Emotional states that decay and influence decisions
- Cognitive biases that nudge behavior
- Bounded rationality that limits information processing
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Emotional State
# ---------------------------------------------------------------------------

EMOTIONS = ["joy", "fear", "anger", "trust", "disgust", "surprise"]


@dataclass
class EmotionalState:
    """An agent's emotional state. Each emotion is 0.0-1.0."""
    joy: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    trust: float = 0.5  # Default neutral trust
    disgust: float = 0.0
    surprise: float = 0.0
    stress: float = 0.0    # Aggregate stress level
    fatigue: float = 0.0   # Cumulative fatigue

    def trigger(self, emotion: str, amount: float, cause: str = ""):
        """Spike an emotion by a given amount (clamped to 0-1)."""
        if emotion in EMOTIONS:
            current = getattr(self, emotion)
            setattr(self, emotion, min(1.0, max(0.0, current + amount)))
        # Strong emotions increase stress
        if abs(amount) > 0.3:
            self.stress = min(1.0, self.stress + abs(amount) * 0.2)

    def decay(self, rate: float = 0.1):
        """Decay all emotions toward baseline each round."""
        for emotion in EMOTIONS:
            current = getattr(self, emotion)
            baseline = 0.5 if emotion == "trust" else 0.0
            diff = current - baseline
            new_val = current - diff * rate
            setattr(self, emotion, max(0.0, min(1.0, new_val)))
        # Stress and fatigue decay slower
        self.stress = max(0.0, self.stress - rate * 0.5)
        self.fatigue = min(1.0, self.fatigue + 0.02)  # Fatigue slowly builds

    def dominant_emotion(self) -> str:
        """Return the emotion with highest intensity."""
        best = "trust"
        best_val = 0.0
        for emotion in EMOTIONS:
            val = getattr(self, emotion)
            # For trust, measure distance from neutral
            effective = abs(val - 0.5) * 2 if emotion == "trust" else val
            if effective > best_val:
                best_val = effective
                best = emotion
        return best

    def intensity(self) -> float:
        """Overall emotional intensity (average of all emotions)."""
        total = sum(abs(getattr(self, e) - (0.5 if e == "trust" else 0.0)) for e in EMOTIONS)
        return total / len(EMOTIONS)

    def to_dict(self) -> dict:
        return {e: round(getattr(self, e), 3) for e in EMOTIONS + ["stress", "fatigue"]}

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalState":
        return cls(**{k: data.get(k, 0.0) for k in EMOTIONS + ["stress", "fatigue"]})


# ---------------------------------------------------------------------------
# Cognitive Biases
# ---------------------------------------------------------------------------

@dataclass
class CognitiveBias:
    """A cognitive bias that nudges agent behavior."""
    name: str
    description: str = ""
    trigger_condition: str = ""  # When this bias activates (e.g., "loss_scenario")
    effect_description: str = ""  # How it affects decisions
    strength: float = 0.5  # 0.0-1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "trigger_condition": self.trigger_condition,
            "effect_description": self.effect_description,
            "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CognitiveBias":
        return cls(**data)


# Built-in bias definitions
BUILTIN_BIASES = {
    "confirmation_bias": CognitiveBias(
        name="confirmation_bias",
        description="Tendency to favor information that confirms existing beliefs",
        trigger_condition="information_evaluation",
        effect_description="You tend to interpret new information as confirming what you already believe",
        strength=0.5,
    ),
    "loss_aversion": CognitiveBias(
        name="loss_aversion",
        description="Losses feel roughly twice as painful as equivalent gains feel good",
        trigger_condition="risk_decision",
        effect_description="You strongly prefer avoiding losses over acquiring equivalent gains",
        strength=0.6,
    ),
    "anchoring": CognitiveBias(
        name="anchoring",
        description="Over-reliance on the first piece of information encountered",
        trigger_condition="negotiation",
        effect_description="You anchor heavily on initial values or offers",
        strength=0.4,
    ),
    "bandwagon": CognitiveBias(
        name="bandwagon",
        description="Tendency to follow the crowd",
        trigger_condition="group_decision",
        effect_description="You are influenced by what the majority is doing",
        strength=0.4,
    ),
    "sunk_cost": CognitiveBias(
        name="sunk_cost",
        description="Continuing an endeavor due to previously invested resources",
        trigger_condition="investment_decision",
        effect_description="You find it hard to abandon projects you have already invested in",
        strength=0.5,
    ),
    "authority_bias": CognitiveBias(
        name="authority_bias",
        description="Tendency to attribute greater accuracy to authority figures",
        trigger_condition="social_influence",
        effect_description="You give more weight to opinions from perceived authority figures",
        strength=0.4,
    ),
    "in_group_favoritism": CognitiveBias(
        name="in_group_favoritism",
        description="Preferring members of one's own group",
        trigger_condition="social_interaction",
        effect_description="You favor those in your group and are suspicious of outsiders",
        strength=0.5,
    ),
    "status_quo_bias": CognitiveBias(
        name="status_quo_bias",
        description="Preference for the current state of affairs",
        trigger_condition="change_decision",
        effect_description="You prefer things to stay the same and resist change",
        strength=0.4,
    ),
    "recency_bias": CognitiveBias(
        name="recency_bias",
        description="Giving more weight to recent events",
        trigger_condition="information_evaluation",
        effect_description="You give disproportionate weight to the most recent events",
        strength=0.4,
    ),
    "negativity_bias": CognitiveBias(
        name="negativity_bias",
        description="Negative events have a greater impact than positive ones",
        trigger_condition="information_evaluation",
        effect_description="You focus more on negative information and threats",
        strength=0.5,
    ),
}


# ---------------------------------------------------------------------------
# Bounded Rationality
# ---------------------------------------------------------------------------

@dataclass
class BoundedRationalityConfig:
    """Configuration for cognitive limitations."""
    max_options_considered: int = 4       # Max actions to seriously evaluate
    attention_span: int = 6               # Max entities to process in perception
    information_decay: float = 0.1        # How much old info degrades per round
    stress_threshold: float = 0.7         # Above this, decision quality drops
    stress_option_reduction: int = 2      # Options lost when stressed


# ---------------------------------------------------------------------------
# Cognitive Profile (per-agent)
# ---------------------------------------------------------------------------

# Personality trait -> emotional/cognitive mappings
PERSONALITY_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "aggressive": {"anger_sensitivity": 1.5, "fear_sensitivity": 0.5, "biases": ["negativity_bias"]},
    "cautious": {"fear_sensitivity": 1.5, "anger_sensitivity": 0.5, "biases": ["loss_aversion", "status_quo_bias"]},
    "charismatic": {"trust_sensitivity": 1.5, "biases": ["bandwagon"]},
    "analytical": {"max_options_bonus": 2, "emotion_dampening": 0.5, "biases": []},
    "impulsive": {"max_options_penalty": 2, "emotion_amplification": 1.5, "biases": ["recency_bias"]},
    "stubborn": {"biases": ["sunk_cost", "confirmation_bias", "status_quo_bias"]},
    "trusting": {"trust_sensitivity": 2.0, "biases": ["authority_bias"]},
    "suspicious": {"trust_sensitivity": 0.3, "biases": ["negativity_bias", "in_group_favoritism"]},
}


@dataclass
class CognitiveProfile:
    """Per-agent cognitive profile: emotions, biases, rationality bounds."""
    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    active_biases: List[CognitiveBias] = field(default_factory=list)
    rationality: BoundedRationalityConfig = field(default_factory=BoundedRationalityConfig)
    personality_modifiers: Dict[str, float] = field(default_factory=dict)
    _emotion_decay_rate: float = 0.1

    def tick(self, round_number: int):
        """Per-round update: decay emotions, update stress."""
        self.emotional_state.decay(self._emotion_decay_rate)

    def trigger_emotion(self, emotion: str, amount: float, cause: str = ""):
        """Trigger an emotion with personality modifiers applied."""
        sensitivity_key = f"{emotion}_sensitivity"
        multiplier = self.personality_modifiers.get(sensitivity_key, 1.0)
        dampening = self.personality_modifiers.get("emotion_dampening", 1.0)
        amplification = self.personality_modifiers.get("emotion_amplification", 1.0)
        adjusted = amount * multiplier * dampening * amplification
        self.emotional_state.trigger(emotion, adjusted, cause)

    def process_perception(self, perception: dict) -> dict:
        """Apply bounded rationality to filter perception.
        
        - Truncates visible entities based on attention_span
        - Reduces action options under stress
        Returns modified perception dict.
        """
        filtered = dict(perception)
        
        # Limit visible entities
        if "visible_entities" in filtered:
            max_entities = self.rationality.attention_span
            entities = filtered["visible_entities"]
            if len(entities) > max_entities:
                filtered["visible_entities"] = entities[:max_entities]
        
        return filtered

    def get_effective_max_options(self) -> int:
        """Get the effective max options considering stress and personality."""
        base = self.rationality.max_options_considered
        bonus = int(self.personality_modifiers.get("max_options_bonus", 0))
        penalty = int(self.personality_modifiers.get("max_options_penalty", 0))
        effective = base + bonus - penalty
        
        # Stress reduces options
        if self.emotional_state.stress > self.rationality.stress_threshold:
            effective -= self.rationality.stress_option_reduction
        
        return max(2, effective)  # Always at least 2 options

    def get_bias_effects(self) -> List[str]:
        """Return list of active bias effect descriptions."""
        effects = []
        for bias in self.active_biases:
            if bias.strength > 0.3:  # Only include meaningful biases
                effects.append(f"- {bias.name}: {bias.effect_description} (strength: {bias.strength:.1f})")
        return effects

    def get_prompt_overlay(self) -> str:
        """Generate formatted text for LLM prompt injection."""
        lines = []
        
        # Emotional state (3 lines max)
        es = self.emotional_state
        dominant = es.dominant_emotion()
        intensity = es.intensity()
        if intensity > 0.1:
            lines.append(f"Dominant emotion: {dominant} (intensity: {intensity:.1f})")
            if es.stress > 0.3:
                lines.append(f"Stress level: {'high' if es.stress > 0.7 else 'moderate'} ({es.stress:.1f})")
            if es.fatigue > 0.5:
                lines.append(f"Fatigue: {'high' if es.fatigue > 0.7 else 'moderate'}")
        
        # Bias nudges (3 lines max)
        bias_effects = self.get_bias_effects()
        if bias_effects:
            lines.append("Active cognitive tendencies:")
            for be in bias_effects[:3]:
                lines.append(be)
        
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "emotional_state": self.emotional_state.to_dict(),
            "active_biases": [b.to_dict() for b in self.active_biases],
            "rationality": {
                "max_options_considered": self.rationality.max_options_considered,
                "attention_span": self.rationality.attention_span,
                "information_decay": self.rationality.information_decay,
                "stress_threshold": self.rationality.stress_threshold,
                "stress_option_reduction": self.rationality.stress_option_reduction,
            },
            "personality_modifiers": dict(self.personality_modifiers),
            "emotion_decay_rate": self._emotion_decay_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CognitiveProfile":
        profile = cls(
            emotional_state=EmotionalState.from_dict(data.get("emotional_state", {})),
            active_biases=[CognitiveBias.from_dict(b) for b in data.get("active_biases", [])],
            personality_modifiers=data.get("personality_modifiers", {}),
        )
        rat_data = data.get("rationality", {})
        if rat_data:
            profile.rationality = BoundedRationalityConfig(
                max_options_considered=rat_data.get("max_options_considered", 4),
                attention_span=rat_data.get("attention_span", 6),
                information_decay=rat_data.get("information_decay", 0.1),
                stress_threshold=rat_data.get("stress_threshold", 0.7),
                stress_option_reduction=rat_data.get("stress_option_reduction", 2),
            )
        profile._emotion_decay_rate = data.get("emotion_decay_rate", 0.1)
        return profile

    @classmethod
    def from_personality_traits(cls, traits: List[str], biases: Optional[List[str]] = None) -> "CognitiveProfile":
        """Create a cognitive profile from personality trait names."""
        profile = cls()
        all_bias_names = set()
        
        for trait in traits:
            mapping = PERSONALITY_MAPPINGS.get(trait.lower(), {})
            for key, value in mapping.items():
                if key == "biases":
                    all_bias_names.update(value)
                elif key in ("max_options_bonus", "max_options_penalty",
                           "emotion_dampening", "emotion_amplification"):
                    profile.personality_modifiers[key] = value
                else:
                    profile.personality_modifiers[key] = value
        
        # Add explicitly requested biases
        if biases:
            all_bias_names.update(biases)
        
        # Resolve bias objects. Sort the set first: active_biases is later
        # truncated to the first 3 for the LLM prompt overlay, so an
        # unordered set would feed different biases into the prompt across
        # processes (set order varies with PYTHONHASHSEED) — nondeterministic
        # decisions even at a fixed sim seed.
        for bias_name in sorted(all_bias_names):
            if bias_name in BUILTIN_BIASES:
                profile.active_biases.append(BUILTIN_BIASES[bias_name])
        
        return profile


# ---------------------------------------------------------------------------
# Cognition Manager
# ---------------------------------------------------------------------------

class CognitionManager:
    """Per-simulation manager for all agent cognitive profiles."""

    def __init__(self):
        self._profiles: Dict[str, CognitiveProfile] = {}

    def register(self, entity_id: str, profile: CognitiveProfile):
        """Register a cognitive profile for an entity."""
        self._profiles[entity_id] = profile

    def get(self, entity_id: str) -> Optional[CognitiveProfile]:
        """Get a cognitive profile by entity ID."""
        return self._profiles.get(entity_id)

    def remove(self, entity_id: str) -> Optional[CognitiveProfile]:
        """Remove and return a cognitive profile."""
        return self._profiles.pop(entity_id, None)

    def tick_all(self, round_number: int):
        """Tick all cognitive profiles (emotional decay, stress updates)."""
        for profile in self._profiles.values():
            profile.tick(round_number)

    def trigger_emotion_for(self, entity_id: str, emotion: str, amount: float, cause: str = ""):
        """Trigger an emotion for a specific entity."""
        profile = self._profiles.get(entity_id)
        if profile:
            profile.trigger_emotion(emotion, amount, cause)

    def process_perception_for(self, entity_id: str, perception: dict) -> dict:
        """Apply bounded rationality filters to perception for an entity."""
        profile = self._profiles.get(entity_id)
        if profile:
            return profile.process_perception(perception)
        return perception

    def get_prompt_overlay(self, entity_id: str) -> str:
        """Get the prompt overlay for an entity."""
        profile = self._profiles.get(entity_id)
        if profile:
            return profile.get_prompt_overlay()
        return ""

    def list_entities(self) -> List[str]:
        """List all entity IDs with cognitive profiles."""
        return list(self._profiles.keys())

    def to_dict(self) -> dict:
        return {
            "profiles": {eid: p.to_dict() for eid, p in self._profiles.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CognitionManager":
        mgr = cls()
        for eid, pdata in data.get("profiles", {}).items():
            mgr._profiles[eid] = CognitiveProfile.from_dict(pdata)
        return mgr
