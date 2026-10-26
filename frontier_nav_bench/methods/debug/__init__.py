"""Ground-truth attachment for explicitly unscored debugging cells.

Only debug-stop-now is distributed. Official test campaigns reject this channel.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from frontier_nav_bench.interfaces import Pose2D
from frontier_nav_bench.oracle import MultiFloorGrid, TruthGrid


@dataclass(frozen=True)
class DebugTruth:
    """Simulation-world ground truth exposed only to debugging methods."""

    grid: Optional[TruthGrid]  # Free truth grid required by oracle_walker, unused by the other two
    # methods
    truth_now: Callable[[], Pose2D]  # Current world-frame robot truth pose, equal to
    # SimEnv.truth_pose
    goal_world: Pose2D  # World-frame body pose at goal-image capture
    multifloor: Optional[MultiFloorGrid] = None  # Multistory env3 truth for the stair SRU debugging
    # method
    truth_z_now: Optional[Callable[[], float]] = None  # Current world body height for cross-floor
    # arrival checks


class DebugMethodBase:
    """Shared debugging base with method_id, truth attachment and a no-op close."""

    uses_truth = True

    def __init__(self, method_id: str) -> None:
        self.method_id = method_id
        self.truth: Optional[DebugTruth] = None

    def attach_truth(self, truth: DebugTruth) -> None:
        self.truth = truth

    def close(self) -> None:
        pass
