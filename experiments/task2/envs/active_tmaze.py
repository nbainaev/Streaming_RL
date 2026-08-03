"""Native JAX port of Memory-RL's Active T-Maze (Bakker, 2001).

Same as the Passive T-Maze except the agent starts one cell past the oracle
and must actively move onto it to see the goal before the corridor
pace-check penalty kicks in (``oracle_length=1``).
"""

from __future__ import annotations

from experiments.task2.envs.tmaze_base import TMazeBase, TMazeParams, TMazeState

ActiveTMazeParams = TMazeParams
ActiveTMazeState = TMazeState


class ActiveTMaze(TMazeBase):
    """Active T-Maze: actions follow 0=right, 1=up, 2=left, 3=down."""

    def __init__(self, corridor_length: int, expose_goal: bool = False):
        super().__init__(
            corridor_length=corridor_length,
            oracle_length=1,
            expose_goal=expose_goal,
        )
