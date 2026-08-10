"""Quickstart — load a template, plug in an agent, run, read the outcome.

    PYTHONPATH=src python3 examples/quickstart.py
"""
import json
import random
from pathlib import Path

from fg_env_kernel import ActionInstance, Kernel

template = json.loads(
    (Path(__file__).parent / "tic_tac_toe" / "template.json").read_text()
)

rng = random.Random(7)


def decision_fn(entity_id, perception, valid_actions):
    """Toy agent: pick a random cell. Swap in an LLM call here."""
    if "place_mark" not in valid_actions:
        return None
    return ActionInstance(
        action_name="place_mark",
        actor_id=entity_id,
        parameters={"cell": rng.randint(1, 9)},
    )


kernel = Kernel(seed=42)
world = kernel.load(template, decision_fn=decision_fn)

while not world.finished:
    world.step()

print(f"Rounds played: {world.current_round}")
print(f"Terminated by: {world.terminated_by or 'round budget'}")
print(f"Final event:   {world.events[-1].narrative}")
