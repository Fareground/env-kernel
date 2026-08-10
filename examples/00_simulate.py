"""The one-liner — simulate a template with the built-in random agent.

    PYTHONPATH=src python3 examples/00_simulate.py
"""
from pathlib import Path

from fg_env_kernel import simulate

world = simulate(Path(__file__).parent / "tic_tac_toe" / "template.json", seed=7)

print(world)              # <World 'Tic-Tac-Toe': round 4/9, finished, ...>
print(world.summary())
