"""Timestamp lookup and planar trajectories used by report rendering."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

FRAME_STAMP_TOLERANCE_S = 1e-3


def index_at(times: np.ndarray, t: float, tol: float = 0.0) -> int:
    """Index at."""
    return int(np.searchsorted(times, t + tol, side="right")) - 1


@dataclass(frozen=True)
class FrameTrack:
    """Frametrack."""

    stamps: np.ndarray
    paths: Tuple[str, ...]

    def index_at(self, t: float) -> int:
        """Index at."""
        return index_at(self.stamps, t, FRAME_STAMP_TOLERANCE_S)

    def image_at(self, t: float) -> Optional[np.ndarray]:
        """Image at."""
        idx = max(self.index_at(t), 0)
        return cv2.imread(self.paths[idx])


@dataclass(frozen=True)
class PoseTrack:
    """Posetrack."""

    stamps: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray

    @classmethod
    def from_samples(
        cls, stamps: Sequence[float], xy: Sequence[Tuple[float, float]], yaw: Sequence[float]
    ) -> "PoseTrack":
        """From samples."""
        if len(stamps) == 0:
            return cls(
                stamps=np.empty((0,), dtype=np.float64),
                xy=np.empty((0, 2), dtype=np.float64),
                yaw=np.empty((0,), dtype=np.float64),
            )
        stamps_arr = np.array(stamps, dtype=np.float64)
        order = np.argsort(stamps_arr, kind="stable")
        xy_arr = np.array(xy, dtype=np.float64)[order]
        yaw_arr = np.unwrap(np.array(yaw, dtype=np.float64)[order])
        return cls(stamps=stamps_arr[order], xy=xy_arr, yaw=yaw_arr)

    def at(self, t: float) -> Tuple[float, float, float]:
        """At."""
        x = float(np.interp(t, self.stamps, self.xy[:, 0]))
        y = float(np.interp(t, self.stamps, self.xy[:, 1]))
        yaw = float(np.interp(t, self.stamps, self.yaw))
        return x, y, yaw

    def path_until(self, t: float) -> np.ndarray:
        """Path until."""
        idx = int(np.searchsorted(self.stamps, t, side="right"))
        x, y, _yaw = self.at(t)
        tail = np.array([[x, y]], dtype=np.float64)
        return np.vstack([self.xy[:idx], tail])

    def distance_until(self, t: float) -> float:
        """Distance until."""
        pts = self.path_until(t)
        if len(pts) < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
