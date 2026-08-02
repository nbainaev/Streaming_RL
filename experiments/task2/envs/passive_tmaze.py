"""Native JAX port of Memory-RL's Passive T-Maze (M_passive: O=S, L=T-1)."""

from __future__ import annotations

from experiments.task2.envs.tmaze_base import TMazeBase, TMazeParams, TMazeState

# Kept as aliases for backwards compatibility with existing imports/type hints.
PassiveTMazeParams = TMazeParams
PassiveTMazeState = TMazeState


class PassiveTMaze(TMazeBase):
    """Passive T-Maze with a one-step oracle and a fixed episode horizon.

    Actions follow the reference implementation:
      0 = right, 1 = up, 2 = left, 3 = down.
    """

    def __init__(self, corridor_length: int, expose_goal: bool = False):
        super().__init__(
            corridor_length=corridor_length,
            oracle_length=0,
            expose_goal=expose_goal,
        )
