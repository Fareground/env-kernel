"""Role registry — first-class hidden/asymmetric roles for agents.

Social deduction games (Mafia, Werewolf, Coup, Among Us) and any game
with asymmetric private information need a way to attach a `role` to
each agent that:

  - other agents normally cannot see (visibility filtered),
  - some roles CAN see (e.g. mafia know each other),
  - groups by team for win conditions / private chat / shared night
    actions.

This module stays small and unopinionated: it's a lookup table plus a
visibility predicate. Domain modules wire it into their own game logic
(who wins when, who chats with whom, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Role:
    """A role assignable to an agent.

    `team` groups roles into win-condition cohorts (e.g. all 'mafia',
    all 'town'). `sees_teammates` lets members of the same team see
    each other's roles in perception — the typical "mafia knows mafia"
    pattern. `extra_visible_roles` is for asymmetric peeks (e.g. the
    Seer sees one role per night — that's not this; that's an action).
    """
    name: str
    team: str = ""
    sees_teammates: bool = True
    description: str = ""
    extra_visible_roles: List[str] = field(default_factory=list)


@dataclass
class RoleRegistry:
    """Tracks role assignments per entity.

    Populated by domain modules at hand/round start. Persists for the
    whole match unless explicitly cleared.
    """
    roles: Dict[str, Role] = field(default_factory=dict)            # role_name -> Role
    assignments: Dict[str, str] = field(default_factory=dict)       # entity_id -> role_name

    def define_role(self, role: Role) -> None:
        self.roles[role.name] = role

    def assign(self, entity_id: str, role_name: str) -> None:
        if role_name not in self.roles:
            raise ValueError(f"Unknown role '{role_name}' — define it before assigning.")
        self.assignments[entity_id] = role_name

    def unassign(self, entity_id: str) -> None:
        self.assignments.pop(entity_id, None)

    def get_role(self, entity_id: str) -> Optional[Role]:
        name = self.assignments.get(entity_id)
        return self.roles.get(name) if name else None

    def get_role_name(self, entity_id: str) -> Optional[str]:
        return self.assignments.get(entity_id)

    def entities_with_role(self, role_name: str) -> List[str]:
        return [eid for eid, rn in self.assignments.items() if rn == role_name]

    def entities_on_team(self, team: str) -> List[str]:
        return [
            eid for eid, rn in self.assignments.items()
            if (self.roles.get(rn) and self.roles[rn].team == team)
        ]

    def can_see_role(self, observer_id: str, target_id: str) -> bool:
        """Should `observer_id` see `target_id`'s role in perception?"""
        if observer_id == target_id:
            return True  # always know your own role
        own = self.get_role(observer_id)
        their = self.get_role(target_id)
        if own is None or their is None:
            return False
        if own.sees_teammates and own.team and own.team == their.team:
            return True
        if their.name in own.extra_visible_roles:
            return True
        return False

    def visible_role_map(self, observer_id: str) -> Dict[str, str]:
        """Return {entity_id: role_name} for every role the observer can see."""
        out: Dict[str, str] = {}
        for eid in self.assignments:
            if self.can_see_role(observer_id, eid):
                out[eid] = self.assignments[eid]
        return out

    def teammates(self, entity_id: str) -> Set[str]:
        """Visible teammates only — returns empty if the observer's role
        doesn't permit seeing teammates. Mafia know mafia; town roles
        don't know each other unless their Role explicitly enables it.
        """
        own = self.get_role(entity_id)
        if not own or not own.team or not own.sees_teammates:
            return set()
        return set(self.entities_on_team(own.team)) - {entity_id}

    def to_dict(self) -> dict:
        return {
            "roles": {
                name: {
                    "name": r.name,
                    "team": r.team,
                    "sees_teammates": r.sees_teammates,
                    "description": r.description,
                    "extra_visible_roles": list(r.extra_visible_roles),
                }
                for name, r in self.roles.items()
            },
            "assignments": dict(self.assignments),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoleRegistry":
        """Restore from a to_dict snapshot."""
        reg = cls()
        for name, r in (data.get("roles") or {}).items():
            reg.roles[name] = Role(
                name=r["name"],
                team=r.get("team"),
                sees_teammates=r.get("sees_teammates", False),
                description=r.get("description", ""),
                extra_visible_roles=list(r.get("extra_visible_roles", [])),
            )
        reg.assignments = dict(data.get("assignments") or {})
        return reg
