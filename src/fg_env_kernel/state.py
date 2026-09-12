"""Complete world state -- the runtime state graph."""
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .entity import EntityType, Entity
from .resource import ResourceType, ResourcePool
from .spatial import SpatialModel, NoSpace
from .temporal import TemporalModel
from .relations import RelationType, RelationGraph
from .action import ActionDefinition, Operator
from .visibility import VisibilityRule
from .event import EventLog
from .status_effects import StatusEffectDefinition, StatusEffectTracker
from .factions import Faction, FactionManager
from .sequences import SequenceTracker
from .messaging import MessageBoard
from .location_properties import LocationPropertyManager
from .inventory import InventoryManager
from .goals import GoalTracker
from .skills import SkillTracker, SkillDefinition
from .crafting import RecipeManager, Recipe
from .world_model import AgentWorldModel
from .property_dynamics import PropertyDynamicsEngine
from .physics import PhysicsModel
from .negotiation import NegotiationManager
from .planning import PlanManager
from .roles import RoleRegistry
from .polls import PollManager
from .connectors import ConnectorManager
from .domain_module import DomainModuleManager
from .sim_controller import SimController
from .cognition import CognitionManager
from .social import SocialPlatformManager


@dataclass(frozen=True)
class ActionRecord:
    """A record of an action taken by an entity."""
    action_name: str
    success: bool
    round_number: int


class ActionHistory:
    """Tracks what actions each entity has taken, for action chain validation."""

    _EMPTY_DIGEST = hashlib.sha256(b'fg-action-history-v1').digest()

    def __init__(self):
        self._history: Dict[str, List[ActionRecord]] = {}   # entity_id -> records
        self._unlocked: Dict[str, Set[str]] = {}             # entity_id -> unlocked action names
        self._locked: Dict[str, Set[str]] = {}               # entity_id -> locked action names
        self._cooldowns: Dict[Tuple[str, str], int] = {}     # (entity_id, action_name) -> cooldown_until_round
        # Records are append-only. Prefix fingerprints and membership indexes
        # update once per action, never by rescanning an accumulated history.
        self._digests: Dict[str, bytes] = {}
        self._counts: Dict[str, int] = {}
        self._performed: Dict[str, Set[str]] = {}
        self._succeeded: Dict[str, Set[str]] = {}

    def record(self, entity_id: str, action_name: str, success: bool, round_num: int):
        """Record an action taken by an entity."""
        if entity_id not in self._history:
            self._history[entity_id] = []
        self._history[entity_id].append(ActionRecord(action_name, success, round_num))
        encoded = json.dumps([action_name, success, round_num], ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
        self._digests[entity_id] = hashlib.sha256(
            self._digests.get(entity_id, self._EMPTY_DIGEST) + encoded).digest()
        self._counts[entity_id] = self._counts.get(entity_id, 0) + 1
        self._performed.setdefault(entity_id, set()).add(action_name)
        if success:
            self._succeeded.setdefault(entity_id, set()).add(action_name)

    def is_available(
        self,
        entity_id: str,
        action_name: str,
        action_def: ActionDefinition,
        round_num: int,
    ) -> bool:
        """Check if action is available (respects chains, locks, cooldowns)."""
        # Check if action is locked
        locked = self._locked.get(entity_id, set())
        if action_name in locked:
            return False

        # Check cooldown
        cd_key = (entity_id, action_name)
        if cd_key in self._cooldowns:
            if round_num < self._cooldowns[cd_key]:
                return False

        # Check requires_action
        if action_def.requires_action:
            actions = self._succeeded if action_def.requires_action_success else self._performed
            found = action_def.requires_action in actions.get(entity_id, set())
            if not found:
                # Check if it's been dynamically unlocked
                unlocked = self._unlocked.get(entity_id, set())
                if action_name not in unlocked:
                    return False

        return True

    def unlock(self, entity_id: str, action_names: List[str]):
        """Dynamically unlock actions for an entity."""
        if entity_id not in self._unlocked:
            self._unlocked[entity_id] = set()
        self._unlocked[entity_id].update(action_names)

    def lock(self, entity_id: str, action_names: List[str]):
        """Dynamically lock actions for an entity."""
        if entity_id not in self._locked:
            self._locked[entity_id] = set()
        self._locked[entity_id].update(action_names)

    def get_recent(self, entity_id: str, n: int = 3) -> List[ActionRecord]:
        """Get the N most recent action records for an entity."""
        history = self._history.get(entity_id, [])
        return history[-n:]

    def set_cooldown(self, entity_id: str, action_name: str, until_round: int):
        """Set a cooldown for an action."""
        self._cooldowns[(entity_id, action_name)] = until_round

    def reference(self) -> dict:
        """A detached, verifiable prefix manifest, O(actors), not O(actions).

        The host stores the records once, indexed by these exact prefix counts.
        Private history storage must not be edited; use record/from_dict.
        """
        entities = {}
        for eid, records in self._history.items():
            count = self._counts.get(eid, 0)
            if len(records) != count:
                raise ValueError('Action history changed outside its append-only interface')
            entities[eid] = {'count': count, 'digest': self._digests.get(eid, self._EMPTY_DIGEST).hex()}
        return {'format': 'fg-action-history-v1', 'entities': entities}

    def verify_reference(self, reference) -> None:
        """Reject malformed manifests as well as mismatched history contents."""
        if (not isinstance(reference, dict) or set(reference) != {'format', 'entities'}
                or reference['format'] != 'fg-action-history-v1'
                or not isinstance(reference['entities'], dict)):
            raise ValueError('Invalid action history prefix manifest')
        for eid, row in reference['entities'].items():
            if (not isinstance(eid, str) or not isinstance(row, dict)
                    or set(row) != {'count', 'digest'} or type(row['count']) is not int
                    or row['count'] < 0 or not isinstance(row['digest'], str)
                    or len(row['digest']) != 64
                    or any(c not in '0123456789abcdef' for c in row['digest'])):
                raise ValueError('Invalid action history prefix manifest')
        if self.reference() != reference:
            raise ValueError('checkpoint action history does not match its prefix fingerprint')

    def to_dict(self, *, include_records=True) -> dict:
        """Serialize for snapshots."""
        return {
            "history": {
                eid: [{"action": r.action_name, "success": r.success, "round": r.round_number}
                      for r in records]
                for eid, records in self._history.items()
            } if include_records else None,
            "unlocked": {eid: sorted(actions) for eid, actions in self._unlocked.items()},
            "locked": {eid: sorted(actions) for eid, actions in self._locked.items()},
            "cooldowns": [
                [eid, action, until]
                for (eid, action), until in self._cooldowns.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionHistory":
        ah = cls()
        for eid, records in data.get("history", {}).items():
            ah._history[eid] = []
            for row in records:
                ah.record(eid, row['action'], row['success'], row['round'])
        for eid, actions in data.get("unlocked", {}).items():
            ah._unlocked[eid] = set(actions)
        for eid, actions in data.get("locked", {}).items():
            ah._locked[eid] = set(actions)
        for row in data.get("cooldowns", []):
            if len(row) == 3:
                ah._cooldowns[(row[0], row[1])] = row[2]
        return ah


@dataclass
class WorldState:
    """
    The complete simulation state. This is the state graph.

    All simulation logic reads and writes through this object.
    The kernel/engine.py drives the tick loop; WorldState is the data.
    """
    # Schema (type definitions -- immutable during simulation)
    entity_types: Dict[str, EntityType] = field(default_factory=dict)
    resource_types: Dict[str, ResourceType] = field(default_factory=dict)
    action_definitions: Dict[str, ActionDefinition] = field(default_factory=dict)
    visibility_rules: List[VisibilityRule] = field(default_factory=list)

    # Mutable state
    entities: Dict[str, Entity] = field(default_factory=dict)
    resources: Dict[str, ResourcePool] = field(default_factory=dict)
    relations: RelationGraph = field(default_factory=RelationGraph)

    # Spatial model
    spatial: SpatialModel = field(default_factory=NoSpace)

    # Temporal model
    temporal: TemporalModel = field(default_factory=TemporalModel)

    # Event log
    event_log: EventLog = field(default_factory=EventLog)

    # Status effects
    status_effects: StatusEffectTracker = field(default_factory=StatusEffectTracker)
    status_effect_defs: Dict[str, "StatusEffectDefinition"] = field(default_factory=dict)

    # Action history (for chains / cooldowns)
    action_history: ActionHistory = field(default_factory=ActionHistory)

    # Factions
    factions: FactionManager = field(default_factory=FactionManager)

    # Action sequences
    sequences: SequenceTracker = field(default_factory=SequenceTracker)

    # Message board
    messages: MessageBoard = field(default_factory=MessageBoard)

    # Schema-level lookup tables. Read by `$lookup(name, key)` expressions
    # in EffectDSL. Lets games declare rent tables, payout schedules,
    # blind ladders, status modifiers, etc. as JSON without code.
    tables: Dict[str, Any] = field(default_factory=dict)

    # Spatial adjacency graph (location_id -> adjacent location_ids)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)

    # Spatial index  (location_id -> set of entity_ids)
    locations: Dict[str, str] = field(default_factory=dict)         # entity_id -> location_id
    spatial_index: Dict[str, Set[str]] = field(default_factory=dict)  # location_id -> entity_ids

    # Location properties
    location_properties: LocationPropertyManager = field(default_factory=LocationPropertyManager)

    # World-scoped properties — global facts domain modules record (e.g.
    # verdict_recorded, market_halted) that no single entity owns. Read by
    # scope:"world" termination checks and outcome extraction.
    properties: Dict[str, Any] = field(default_factory=dict)

    # Inventory system
    inventory: InventoryManager = field(default_factory=InventoryManager)

    # Goal tracking
    goals: GoalTracker = field(default_factory=GoalTracker)

    # Skill system
    skills: SkillTracker = field(default_factory=SkillTracker)

    # Crafting/recipe system
    recipes: RecipeManager = field(default_factory=RecipeManager)

    # Per-agent persistent world models (spatial memory, stale observations)
    world_models: Dict[str, AgentWorldModel] = field(default_factory=dict)

    # Autonomous environment dynamics (property drift, spawning, cascades)
    property_dynamics: Optional[PropertyDynamicsEngine] = None

    # Continuous coupled-dynamics integrator ("code is physics"): a dt-aware
    # system of ODEs advanced between agent turns (predator/prey, SIR, price
    # discovery). None when the env declares no physics block.
    physics: Optional[PhysicsModel] = None

    # Negotiation manager (proposals, agreements, auctions)
    negotiations: NegotiationManager = field(default_factory=NegotiationManager)

    # Role registry (hidden / asymmetric roles for social deduction games)
    roles: RoleRegistry = field(default_factory=RoleRegistry)

    # Poll/ballot subsystem (real voting — distinct from VotingResolution
    # which is a skill check). Used for elections, jury verdicts, mafia
    # accusations, etc.
    polls: PollManager = field(default_factory=PollManager)

    # Planning & theory of mind
    plans: PlanManager = field(default_factory=PlanManager)

    # Data connectors (external data integration)
    connectors: Optional[ConnectorManager] = None

    # Domain modules (pluggable domain-specific physics)
    domain_modules: Optional[DomainModuleManager] = None

    # Mid-simulation controller (event injection, breakpoints, takeover, directives)
    controller: Optional[SimController] = None

    # Cognitive architecture (emotions, biases, bounded rationality)
    cognition: Optional[CognitionManager] = None

    # Multi-resolution crowd agents (CrowdAgentManager from agents.crowd_agent)
    crowd_agents: Optional[Any] = None  # Type is agents.crowd_agent.CrowdAgentManager

    # Social platform (social graph, content, feeds, reputation, viral spread)
    social: Optional[SocialPlatformManager] = None

    # Plugin module registry. Every named subsystem above is auto-
    # registered here in __post_init__ so the engine can dispatch
    # lifecycle hooks (on_entity_despawn, to_dict, ...) without
    # hardcoding manager names. Custom games register their own
    # KernelModule instances via ``state.register_module(name, mod)``.
    modules: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-register built-in subsystems into ``self.modules`` so
        custom modules can participate in the same lifecycle dispatch."""
        # The named fields remain the source of truth for direct access
        # (state.factions, state.inventory). The modules dict holds the
        # same instances under stable string keys for plugin dispatch.
        self._wire_builtin_modules()

    def _wire_builtin_modules(self) -> None:
        """Populate ``self.modules`` with the named managers. Re-runnable
        — called from __post_init__ and after from_dict restores."""
        named = {
            "factions": self.factions,
            "sequences": self.sequences,
            "messages": self.messages,
            "status_effects": self.status_effects,
            "action_history": self.action_history,
            "location_properties": self.location_properties,
            "inventory": self.inventory,
            "goals": self.goals,
            "skills": self.skills,
            "recipes": self.recipes,
            "negotiations": self.negotiations,
            "roles": self.roles,
            "polls": self.polls,
            "plans": self.plans,
            "relations": self.relations,
        }
        for k, v in named.items():
            if v is not None:
                self.modules.setdefault(k, v)
        # Optional managers — only register when constructed.
        for k, v in [
            ("property_dynamics", self.property_dynamics),
            ("physics", self.physics),
            ("connectors", self.connectors),
            ("domain_modules", self.domain_modules),
            ("controller", self.controller),
            ("cognition", self.cognition),
            ("crowd_agents", self.crowd_agents),
            ("social", self.social),
        ]:
            if v is not None:
                self.modules.setdefault(k, v)

    def register_module(self, name: str, module: Any, *, replace: bool = False) -> None:
        """Register a KernelModule under ``name`` so the engine can
        dispatch lifecycle hooks (``on_entity_despawn``, ``to_dict``,
        etc.) to it. Custom games use this to plug in new subsystems
        without modifying WorldState.

        Raises ``ValueError`` if the name is already taken unless
        ``replace=True``. Built-in subsystems are reserved names and
        cannot be overwritten without ``replace=True``."""
        if not replace and name in self.modules:
            raise ValueError(
                f"WorldState.register_module: '{name}' already registered. "
                "Pass replace=True to override."
            )
        self.modules[name] = module

    def unregister_module(self, name: str) -> Optional[Any]:
        """Remove a module by name. Returns the removed instance or
        None. Note: built-in subsystems remain accessible via their
        named-field attributes (``state.factions`` etc.) regardless."""
        return self.modules.pop(name, None)

    def get_module(self, name: str) -> Optional[Any]:
        """Look up a module by name. Returns None if not registered."""
        return self.modules.get(name)

    # -- Helpers --

    def get_or_create_world_model(self, entity_id: str) -> AgentWorldModel:
        """Get or lazily create a world model for an agent."""
        if entity_id not in self.world_models:
            self.world_models[entity_id] = AgentWorldModel(owner_id=entity_id)
        return self.world_models[entity_id]

    # -- Schema Registration --

    def register_entity_type(self, entity_type: EntityType):
        """Register an entity type definition."""
        self.entity_types[entity_type.name] = entity_type

    def register_resource_type(self, resource_type: ResourceType):
        """Register a resource type and create its pool."""
        self.resource_types[resource_type.name] = resource_type
        self.resources[resource_type.name] = ResourcePool(resource_type=resource_type)

    def register_action(self, action_def: ActionDefinition):
        """Register an action definition."""
        self.action_definitions[action_def.name] = action_def

    def register_relation_type(self, rel_type: RelationType):
        """Register a relation type."""
        self.relations.register_relation_type(rel_type)

    def register_status_effect(self, effect_def: "StatusEffectDefinition"):
        """Register a status effect definition (for APPLY_STATUS effects)."""
        self.status_effect_defs[effect_def.name] = effect_def

    def register_faction(self, faction: Faction):
        """Register a faction."""
        self.factions.register(faction)

    def register_skill_definition(self, skill_def: SkillDefinition):
        """Register a skill type definition."""
        self.skills.register_skill(skill_def)

    def register_recipe(self, recipe: Recipe):
        """Register a crafting recipe."""
        self.recipes.register(recipe)

    # -- Entity Management --

    def spawn_entity(self, entity: Entity) -> Entity:
        """Add an entity to the world.

        Populates ``locations`` and ``spatial_index`` if the entity
        declares a ``location_id`` — without this, spatial queries
        like ``$adjacent_entities`` and ``$within_range`` would
        silently return empty results for declaratively-placed entities.
        """
        self.entities[entity.id] = entity
        loc = getattr(entity, "location_id", None)
        if loc:
            self.locations[entity.id] = loc
            self.spatial_index.setdefault(loc, set()).add(entity.id)
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a given type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_agent_entities(self) -> List[Entity]:
        """Return all entities whose type has role='agent' and are alive."""
        return [
            e for e in self.entities.values()
            if e.alive
            and e.entity_type in self.entity_types
            and self.entity_types[e.entity_type].role == "agent"
        ]

    # -- Action Validation --

    def get_valid_actions(self, entity_id: str) -> List[str]:
        """Return names of all actions currently valid for an entity."""
        entity = self.entities.get(entity_id)
        if not entity or not entity.alive:
            return []

        # Get blocked actions from status effects
        blocked = set(self.status_effects.get_blocked_actions(entity_id))
        round_num = self.temporal.current_round

        valid = []
        for action_name, action_def in self.action_definitions.items():
            if action_def.actor_type != entity.entity_type:
                continue
            if action_name in blocked:
                continue
            if not self.action_history.is_available(entity_id, action_name, action_def, round_num):
                continue
            if self._check_actor_preconditions(entity, action_def):
                valid.append(action_name)
        return valid

    def get_prioritized_actions(self, entity_id: str, max_actions: int = 10) -> List[Tuple[str, float, Optional[str]]]:
        """
        Return actions scored by goal relevance, urgency, and novelty.

        Returns list of (action_name, score, goal_hint) sorted by score descending.
        goal_hint is a short string like 'advances "Accumulate gold"' if applicable.

        Scoring:
        - Goal alignment (+0.5): does the action's effects advance an active goal?
        - Urgency (+0.3): is a goal condition close to completion?
        - Novelty (-0.2): action used recently → penalize repetition
        """
        valid = self.get_valid_actions(entity_id)
        if not valid:
            return []

        active_goals = self.goals.get_active_goals(entity_id)

        scored: List[Tuple[str, float, Optional[str]]] = []
        for action_name in valid:
            action_def = self.action_definitions.get(action_name)
            if not action_def:
                scored.append((action_name, 0.5, None))
                continue

            score = 0.5  # Base score
            goal_hint = None

            # Goal alignment: check if effects_on_success advance any goal
            if active_goals:
                alignment, hint = self._score_action_goal_alignment(
                    entity_id, action_name, action_def, active_goals,
                )
                score += alignment * 0.5  # up to +0.5
                goal_hint = hint

            # Urgency: boost if a goal is close to completion
            if active_goals:
                urgency = self._score_goal_urgency(entity_id, active_goals)
                score += urgency * 0.3  # up to +0.3

            # Novelty penalty: recently used actions get penalized
            recent = self.action_history.get_recent(entity_id, 3)
            recent_names = [r.action_name for r in recent]
            if action_name in recent_names:
                count = recent_names.count(action_name)
                score -= 0.2 * count  # Penalize repetition

            scored.append((action_name, round(max(0.0, score), 3), goal_hint))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_actions]

    def _score_action_goal_alignment(
        self, entity_id: str, action_name: str, action_def: ActionDefinition,
        active_goals: list,
    ) -> Tuple[float, Optional[str]]:
        """Score how well an action's effects align with active goals. Returns (score, hint)."""
        best_score = 0.0
        best_hint = None

        for goal in active_goals:
            for cond in goal.conditions:
                # Check if any effect_on_success modifies the relevant field/resource
                for effect in action_def.effects_on_success:
                    match = False
                    if cond.check_type in ("property_gte", "property_lte") and effect.field == cond.field:
                        match = True
                    elif cond.check_type == "has_resource" and effect.resource == cond.field:
                        match = True
                    elif cond.check_type == "at_location" and effect.operation.value == "move_to":
                        match = True
                    elif cond.check_type == "relation_gte" and effect.relation_type == cond.field:
                        match = True

                    if match:
                        # Weight by goal priority
                        alignment = min(1.0, goal.priority / 3.0)
                        if alignment > best_score:
                            best_score = alignment
                            best_hint = f'advances "{goal.description[:40]}"'

        return best_score, best_hint

    def _score_goal_urgency(self, entity_id: str, active_goals: list) -> float:
        """Score urgency: how close is the closest goal to completion?"""
        # Simple heuristic: check how many conditions are already met
        best_urgency = 0.0
        for goal in active_goals:
            if not goal.conditions:
                continue
            met = sum(
                1 for c in goal.conditions
                if self.goals._check_condition(c, entity_id, self)
            )
            progress = met / len(goal.conditions)
            # High progress → high urgency (almost there!)
            urgency = progress * (goal.priority / 3.0)
            best_urgency = max(best_urgency, min(1.0, urgency))
        return best_urgency

    def _check_actor_preconditions(self, actor: Entity, action_def: ActionDefinition) -> bool:
        """Check actor-side preconditions only."""
        for pc in action_def.preconditions:
            # Modern: full-expression form. One grammar for every guard.
            expr = getattr(pc, "expr", None)
            if expr:
                from .predicates import evaluate as _predicate_eval, requires_decision_context
                if requires_decision_context(expr):
                    # Target/parameter guards cannot be evaluated before the
                    # decision. Resolution rechecks every guard with its inputs.
                    continue
                if not _predicate_eval(expr, actor=actor, state=self):
                    return False
                continue
            if pc.subject != "actor":
                continue
            if pc.operator == Operator.IS_ALIVE:
                if not actor.alive:
                    return False
            elif pc.operator in (Operator.GTE, Operator.GT, Operator.LTE, Operator.LT) and pc.field:
                import math
                val = actor.get(pc.field)
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    return False
                if ((pc.operator == Operator.GTE and val < pc.value)
                    or (pc.operator == Operator.GT and val <= pc.value)
                    or (pc.operator == Operator.LTE and val > pc.value)
                    or (pc.operator == Operator.LT and val >= pc.value)):
                    return False
            elif pc.operator == Operator.EQ and pc.field:
                if actor.get(pc.field) != pc.value:
                    return False
            elif pc.operator == Operator.NEQ and pc.field:
                if actor.get(pc.field) == pc.value:
                    return False
            elif pc.operator == Operator.HAS_RESOURCE and pc.resource:
                pool = self.resources.get(pc.resource)
                if not pool or pool.get(actor.id) < (pc.value or 0):
                    return False
            elif pc.operator == Operator.AT_LOCATION:
                actor_loc = self.locations.get(actor.id) or actor.location_id
                if actor_loc != pc.value:
                    return False
            elif pc.operator == Operator.HAS_ITEM:
                if not self.inventory.has_item(actor.id, pc.value):
                    return False
            elif pc.operator == Operator.HAS_ITEM_TYPE:
                if not self.inventory.has_item_type(actor.id, pc.value):
                    return False
            elif pc.operator == Operator.INVENTORY_NOT_FULL:
                cap = self.inventory.get_capacity(actor.id)
                if cap is not None and self.inventory.count_items(actor.id) >= cap:
                    return False
            elif pc.operator == Operator.SKILL_GTE:
                skill_level = self.skills.get_level(actor.id, pc.field or "")
                if skill_level < (pc.value or 0):
                    return False
            elif pc.operator == Operator.HAS_RECIPE:
                if not self.recipes.can_craft(actor.id, pc.value, self):
                    return False
        return True

    # -- Entity Lifecycle --

    def despawn_entity(self, entity_id: str) -> Optional["Entity"]:
        """Remove an entity from the world entirely. Returns the removed entity."""
        entity = self.entities.pop(entity_id, None)
        if entity:
            # Capture last known location BEFORE any cleanup that clears it.
            last_loc = self.locations.get(entity_id) or getattr(entity, 'location_id', None)
            # Clean up from factions
            self.factions.remove_member(entity_id)
            # Clean up from sequences
            self.sequences.cancel(entity_id)
            # Clean up from spatial index
            if entity_id in self.locations:
                loc = self.locations[entity_id]
                if loc in self.spatial_index:
                    self.spatial_index[loc].discard(entity_id)
                    if not self.spatial_index[loc]:
                        del self.spatial_index[loc]
                del self.locations[entity_id]
            # Clean up relations (all edges involving this entity)
            self.relations.remove_entity(entity_id)
            # Clean up status effects
            self.status_effects.clear_entity(entity_id)
            # Clean up goals
            self.goals.remove_entity(entity_id)
            # Clean up skills
            self.skills.remove_entity(entity_id)
            # Clean up world models (remove this entity's model AND references in others)
            self.world_models.pop(entity_id, None)
            # Clean up inventory (drop items at entity's captured last location)
            inv_items = self.inventory.get_inventory(entity_id)
            for item in inv_items:
                if last_loc:
                    self.inventory.drop_item(entity_id, item.id, last_loc)
                else:
                    self.inventory.remove_item(entity_id, item.id)
            # Clean up negotiations (expire any active negotiations involving this entity)
            self.negotiations.remove_entity(entity_id)
            # Clean up plans (remove plans owned by or targeting this entity)
            self.plans.remove_entity(entity_id)
            # Clean up controller takeovers
            if self.controller:
                self.controller.end_takeover(entity_id)
            # Clean up cognitive profile
            if self.cognition:
                self.cognition.remove(entity_id)
            # Clean up crowd agent tracking
            if self.crowd_agents:
                self.crowd_agents.remove(entity_id)
            # Clean up social connections and content
            if self.social:
                self.social.remove_entity(entity_id)
            # Dispatch to any custom plugin modules that registered an
            # on_entity_despawn hook. Built-ins were already cleaned up
            # via their explicit named calls above — the dispatcher
            # skips them because their hooks (remove_member, cancel,
            # clear_entity, remove_entity, etc.) live under names the
            # built-in cleanup already covered. Plugin modules get
            # their generic on_entity_despawn called here.
            from .kernel_module import dispatch_despawn
            _builtin_names = {
                "factions", "sequences", "relations", "status_effects",
                "goals", "skills", "inventory", "negotiations", "plans",
                "social", "crowd_agents", "cognition", "controller",
            }
            plugin_modules = {
                k: v for k, v in self.modules.items()
                if k not in _builtin_names
            }
            dispatch_despawn(plugin_modules, entity_id)
        return entity

    def spawn_entity_from_template(
        self,
        template_name: str,
        entity_id: str,
        name: str,
        location: Optional[str] = None,
        property_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional["Entity"]:
        """Spawn a new entity using an entity_type as template."""
        from .entity import Entity  # local import to avoid circular

        entity_type = self.entity_types.get(template_name)
        if not entity_type:
            return None
        # Build properties from type defaults + overrides
        props: Dict[str, Any] = {}
        for pschema in entity_type.properties:
            props[pschema.name] = pschema.default if pschema.default is not None else 0
        if property_overrides:
            props.update(property_overrides)
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=template_name,
            properties=props,
            location_id=location,
        )
        self.entities[entity_id] = entity
        if location:
            self.locations[entity_id] = location
            if location not in self.spatial_index:
                self.spatial_index[location] = set()
            self.spatial_index[location].add(entity_id)
        return entity

    # -- Snapshot --

    def to_dict(self, *, include_action_history=True) -> dict:
        """Serialize the full state to a dictionary."""
        derived = getattr(self, "_derived_rules", None)
        return copy.deepcopy({
            "snapshot_version": 2,
            "resource_definitions": {name: asdict(rt) for name, rt in self.resource_types.items()},
            "tables": self.tables,
            "derived_rules": derived.to_dict() if derived is not None else None,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "resources": {rn: rp.to_dict() for rn, rp in self.resources.items()},
            "relations": self.relations.to_dict(),
            "temporal": self.temporal.to_dict(),
            "entity_types": list(self.entity_types.keys()),
            "action_definitions": list(self.action_definitions.keys()),
            "status_effects": self.status_effects.to_dict(),
            "action_history": self.action_history.to_dict(include_records=include_action_history),
            "factions": self.factions.to_dict(),
            "sequences": self.sequences.to_dict(),
            "messages": self.messages.to_dict(),
            "adjacency": dict(self.adjacency),
            "locations": dict(self.locations),
            "spatial_index": {loc: sorted(eids) for loc, eids in self.spatial_index.items()},
            "properties": dict(self.properties),
            "location_properties": self.location_properties.to_dict(),
            "inventory": self.inventory.to_dict(),
            "goals": self.goals.to_dict(),
            "skills": self.skills.to_dict(),
            "recipes": self.recipes.to_dict(),
            "world_models": {
                eid: wm.to_dict() for eid, wm in self.world_models.items()
            },
            "property_dynamics": self.property_dynamics.to_dict() if self.property_dynamics else None,
            "physics": self.physics.to_dict() if self.physics else None,
            "negotiations": self.negotiations.to_dict(),
            "plans": self.plans.to_dict(),
            "roles": self.roles.to_dict(),
            "polls": self.polls.to_dict(),
            "connectors": self.connectors.to_dict() if self.connectors else None,
            "domain_modules": self.domain_modules.to_dict() if self.domain_modules else None,
            "domain_module_aliases": {
                alias: name for alias, obj in self.modules.items()
                for name, module in (self.domain_modules._modules.items() if self.domain_modules else [])
                if obj is module
            },
            "controller": self.controller.to_dict() if self.controller else None,
            "cognition": self.cognition.to_dict() if self.cognition else None,
            "crowd_agents": self.crowd_agents.to_dict() if self.crowd_agents else None,
            "social": self.social.to_dict() if self.social else None,
            # Plugin modules registered via state.register_module().
            # Built-in subsystems are already serialized under their
            # named keys above; this captures the rest.
            "plugin_modules": self._plugin_modules_to_dict(),
        })

    # -- Restore --

    def apply_snapshot(self, data: Dict[str, Any]) -> None:
        """Atomically replace supplied snapshot sections, preserving schema registrations.

        Missing sections support legacy partial snapshots. Invalid or unrestorable
        sections raise SnapshotRestoreError without changing the existing world.
        """
        from .snapshot import apply_snapshot
        apply_snapshot(self, data)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        schema_provider: Optional[Any] = None,
    ) -> "WorldState":
        """Construct a fresh WorldState from a snapshot dict.

        Schema (entity_types, action_definitions, visibility rules)
        cannot be reconstructed from a snapshot alone — the snapshot
        only stores their names. Provide a ``schema_provider`` whose
        attributes (``entity_types``, ``action_definitions``,
        ``visibility_rules``, ``status_effect_defs``, ``resource_types``)
        will be copied onto the new state before mutable state is
        overlaid.

        Without a schema_provider, the returned state has empty
        schema dicts — useful for inspecting raw snapshot data but
        the engine won't be able to run new actions on it.
        """
        state = cls()
        # Copy schema from provider if available
        if schema_provider is not None:
            for attr in ("entity_types", "resource_types", "action_definitions",
                         "visibility_rules", "status_effect_defs", "tables"):
                src = getattr(schema_provider, attr, None)
                if src is not None:
                    if isinstance(src, dict):
                        setattr(state, attr, dict(src))
                    elif isinstance(src, list):
                        setattr(state, attr, list(src))
        # Re-create resource pools from declared resource_types so
        # apply_snapshot's holdings restore has somewhere to land.
        from .resource import ResourcePool as _ResourcePool
        for rname, rtype in state.resource_types.items():
            if rname not in state.resources:
                state.resources[rname] = _ResourcePool(resource_type=rtype)

        if schema_provider is not None:
            state.temporal = copy.deepcopy(schema_provider.temporal)
            state.temporal.current_round = 0
            state.temporal.current_phase_index = 0
            state.temporal.current_turn_index = 0
            state.temporal.turn_order = []
            state.relations.relation_types = copy.deepcopy(schema_provider.relations.relation_types)
            if data.get("domain_modules") is not None:
                state.domain_modules = getattr(schema_provider, "domain_modules", None)
            if data.get("crowd_agents") is not None:
                state.crowd_agents = getattr(schema_provider, "crowd_agents", None)
            # Registered instances provide classes for reconstruction, never live
            # state shared with the provider in the returned world.
            templates = getattr(schema_provider, "modules", {})
            for name in data.get("plugin_modules", {}):
                if name in templates:
                    state.modules[name] = templates[name]
            if state.domain_modules is not None:
                domain_ids = {id(module) for module in state.domain_modules._modules.values()}
                state.modules.update({name: module for name, module in templates.items() if id(module) in domain_ids})
        state.apply_snapshot(data)
        return state

    def _plugin_modules_to_dict(self) -> Dict[str, Any]:
        """Return ``{name: to_dict()}`` for every module in self.modules
        that isn't already serialized under a top-level named key."""
        from .kernel_module import collect_snapshots
        _builtin_keys = {
            "factions", "sequences", "messages", "status_effects",
            "action_history", "location_properties", "inventory",
            "goals", "skills", "recipes", "negotiations", "roles",
            "polls", "plans", "relations", "property_dynamics",
            "connectors", "domain_modules", "controller", "cognition",
            "crowd_agents", "social", "physics",
        }
        domain_ids = {id(mod) for mod in self.domain_modules._modules.values()} if self.domain_modules else set()
        plugin_only = {
            k: v for k, v in self.modules.items()
            if k not in _builtin_keys and id(v) not in domain_ids
        }
        return collect_snapshots(plugin_only)
