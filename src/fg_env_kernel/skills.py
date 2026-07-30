"""Skill growth and learning -- entities develop skills through practice.

Skills gate actions via preconditions, modify resolution outcomes, and
enable RPG-style progression, specialization, and emergent expertise.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillDefinition:
    """Blueprint for a skill type."""
    name: str                          # "swordsmanship", "alchemy", "stealth"
    description: str = ""
    max_level: float = 100.0
    default_level: float = 0.0


@dataclass
class SkillEntry:
    """An entity's current level in a skill."""
    skill_name: str
    level: float = 0.0
    xp: float = 0.0                    # Experience points accumulated


class SkillTracker:
    """Manages entity skill levels and XP progression."""

    def __init__(self):
        self._skills: Dict[str, Dict[str, SkillEntry]] = {}   # entity_id -> {skill_name -> SkillEntry}
        self._definitions: Dict[str, SkillDefinition] = {}      # skill_name -> definition

    def register_skill(self, definition: SkillDefinition):
        """Register a skill definition."""
        self._definitions[definition.name] = definition

    def get_definition(self, skill_name: str) -> Optional[SkillDefinition]:
        """Get a skill definition by name."""
        return self._definitions.get(skill_name)

    def set_level(self, entity_id: str, skill_name: str, level: float):
        """Set an entity's level in a skill directly."""
        if entity_id not in self._skills:
            self._skills[entity_id] = {}
        defn = self._definitions.get(skill_name)
        max_level = defn.max_level if defn else 100.0
        clamped = min(level, max_level)
        if skill_name in self._skills[entity_id]:
            self._skills[entity_id][skill_name].level = clamped
        else:
            self._skills[entity_id][skill_name] = SkillEntry(
                skill_name=skill_name,
                level=clamped,
                xp=0.0,
            )

    def get_level(self, entity_id: str, skill_name: str) -> float:
        """Get an entity's level in a skill. Returns 0.0 if unset."""
        skills = self._skills.get(entity_id, {})
        entry = skills.get(skill_name)
        return entry.level if entry else 0.0

    def award_xp(self, entity_id: str, skill_name: str, amount: float) -> Optional[dict]:
        """Award XP to an entity's skill. Returns level-up info or None.

        XP formula: level up when xp >= max(current_level * 10, 10).
        On level up: level += 1, xp -= threshold, capped at max_level.
        Returns {"leveled_up": True, "new_level": N, "skill": name} on level-up.
        """
        if entity_id not in self._skills:
            self._skills[entity_id] = {}
        if skill_name not in self._skills[entity_id]:
            defn = self._definitions.get(skill_name)
            default_level = defn.default_level if defn else 0.0
            self._skills[entity_id][skill_name] = SkillEntry(
                skill_name=skill_name,
                level=default_level,
                xp=0.0,
            )

        entry = self._skills[entity_id][skill_name]
        defn = self._definitions.get(skill_name)
        max_level = defn.max_level if defn else 100.0

        # Already at max level — no more XP needed
        if entry.level >= max_level:
            return None

        entry.xp += amount

        # Check for level-up(s)
        leveled_up = False
        while entry.level < max_level:
            threshold = max(entry.level * 10, 10)
            if entry.xp >= threshold:
                entry.xp -= threshold
                entry.level += 1
                entry.level = min(entry.level, max_level)
                leveled_up = True
            else:
                break

        if leveled_up:
            return {
                "leveled_up": True,
                "new_level": entry.level,
                "skill": skill_name,
                "entity_id": entity_id,
            }
        return None

    def get_all_skills(self, entity_id: str) -> Dict[str, SkillEntry]:
        """Get all skill entries for an entity."""
        return dict(self._skills.get(entity_id, {}))

    def remove_entity(self, entity_id: str):
        """Remove all skills for an entity (for despawn cleanup)."""
        self._skills.pop(entity_id, None)

    def to_dict(self) -> dict:
        """Serialize for snapshots. Shape:
        ``{entity_id: {skill_name: {level, xp}}}``.

        Skill definitions are part of the schema, not the runtime
        state, and live separately on the WorldState — they don't go
        through this snapshot."""
        result = {}
        for entity_id, skills in self._skills.items():
            result[entity_id] = {
                skill_name: {"level": entry.level, "xp": entry.xp}
                for skill_name, entry in skills.items()
            }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SkillTracker":
        """Restore from a to_dict snapshot."""
        tracker = cls()
        for entity_id, skills in (data or {}).items():
            tracker._skills[entity_id] = {
                name: SkillEntry(level=s.get("level", 0.0), xp=s.get("xp", 0.0))
                for name, s in skills.items()
            }
        return tracker
