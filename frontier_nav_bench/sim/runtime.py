"""Go2-only Isaac runtime helpers for lockstep evaluation.

Simulation uses 200 Hz physics and 10 Hz RGB-D. Image timestamps account for one
render of annotator latency. Isaac imports stay inside functions until the
SimulationApp exists; this module can be imported by offline evaluation."""

# Adapted from nav_sim_isaac/scripts/sim_sensor_publisher.py.
# Source SHA256: 18ea3a70d64720fb7ddd2b39a5937347b89bdc4bdb855cd369fe86195509d2da.
# Adapted from nav_sim_isaac/nav_sim_isaac/rendered_frame_time.py.
# Source SHA256: c9a68b974f05a9bba44c46ac33eb513d2c129ff62200dbd619ff93d89351df99.
# Copyright (c) 2026 Jianhao Jiao. MIT license; see LICENSE.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from frontier_nav_bench.sim import camera, frames, walking
from frontier_nav_bench.sim import codecs as codecs
from frontier_nav_bench.sim import protocol as protocol
from frontier_nav_bench.sim.motion import PartMotionDriver as PartMotionDriver
from frontier_nav_bench.sim.motion import robot_body_size as robot_body_size

NEAR_CLIP_M = 0.05
FAR_CLIP_M = 1000.0
WARMUP_RENDER_FRAMES = 12
RAY_START_Z_M = 50.0
RAY_LENGTH_M = 100.0
ROBOT_PRIM = "/World/Robot"
THIRD_PERSON_PRIM = "/World/ThirdPersonCamera"
THIRD_PERSON_JPEG_QUALITY = 85


def find_ground(xy, near_z: float | None = None) -> float | None:
    """Raycast vertically and choose a standing surface using the frozen rules."""
    from omni.physx import get_physx_scene_query_interface

    hits: list[tuple[float, float]] = []

    def on_hit(hit) -> bool:
        hits.append((float(hit.position[2]), float(hit.normal[2])))
        return True

    get_physx_scene_query_interface().raycast_all(
        (float(xy[0]), float(xy[1]), RAY_START_Z_M), (0.0, 0.0, -1.0), RAY_LENGTH_M, on_hit
    )
    return walking.pick_ground(hits, near_z)


def apply_follow_pose(third_camera, third_usd_camera, pose, np) -> None:
    """Apply tracking-camera pose and its occlusion-dependent clipping distance."""
    from pxr import Gf

    third_camera.set_world_pose(
        position=np.array(pose.position),
        orientation=np.array(pose.orientation_wxyz),
        camera_axes="world",
    )
    third_usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(pose.near_clip_m, camera.FAR_CLIP_M))


def raycast_skip_robot(origin, direction, max_distance):
    """Return the nearest scene ray hit while excluding the robot geometry."""
    from omni.physx import get_physx_scene_query_interface

    hits: list[tuple[float, str, str]] = []

    def on_hit(hit) -> bool:
        rigid_body = str(getattr(hit, "rigid_body", ""))
        collision = str(getattr(hit, "collision", ""))
        hits.append((float(hit.distance), rigid_body, collision))
        return True

    get_physx_scene_query_interface().raycast_all(
        (float(origin[0]), float(origin[1]), float(origin[2])),
        (float(direction[0]), float(direction[1]), float(direction[2])),
        float(max_distance),
        on_hit,
    )
    others = [
        distance
        for (distance, rigid_body, collision) in hits
        if not rigid_body.startswith(ROBOT_PRIM) and (not collision.startswith(ROBOT_PRIM))
    ]
    return min(others) if others else None


def isaac_pose(position, orientation_wxyz):
    """Convert Isaac wxyz pose into a homogeneous transform."""
    (w, x, y, z) = (float(v) for v in orientation_wxyz)
    return frames.pose_to_transform([float(v) for v in position], (x, y, z, w))


def intersect_ranges(a, b) -> tuple[tuple[float, float], ...]:
    """Intersect protocol and walking-policy velocity bounds axis by axis."""
    return tuple((max(lo1, lo2), min(hi1, hi2)) for ((lo1, hi1), (lo2, hi2)) in zip(a, b))


def limit_depenetration(stage, root_path: str) -> None:
    """Use the frozen 1 m/s robot depenetration limit."""
    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(root_path) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            api.CreateMaxDepenetrationVelocityAttr().Set(walking.MAX_DEPENETRATION_MPS)


def set_translate(xformable, translation) -> None:
    """Reuse an existing USD translation operation when positioning a reference."""
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(translation)
            return
    xformable.AddTranslateOp().Set(translation)


ANNOTATOR_LAG_RENDERS = 1


class RenderedFrameTimes:
    """Account for the measured one-render Isaac camera annotator delay."""

    def __init__(self, lag_renders: int = ANNOTATOR_LAG_RENDERS) -> None:
        if lag_renders < 1:
            raise ValueError(f"lag_renders 要 >= 1,收到 {lag_renders}")
        self._lag = lag_renders
        self._history: deque[float] = deque(maxlen=lag_renders + 1)

    def on_render(self, t_sim: float) -> float | None:
        """Return the timestamp of the image currently available from the annotator."""
        self._history.append(t_sim)
        if len(self._history) <= self._lag:
            return None
        return self._history[0]


@dataclass(frozen=True)
class SensorRates:
    """Frozen benchmark physics and rendering frequencies, in hertz."""

    physics_hz: int = 200
    camera_hz: float = 10.0


RATES = SensorRates()
PROTOCOL_COMMAND_RANGE = ((-1.5, 1.5), (-0.6, 0.6), (-2.0, 2.0))
AMBIENT_DOME_INTENSITY = 1000.0


def needs_ambient_light(light_types):
    """Add fallback ambient light only if the stage has no global light."""
    return not {"DomeLight", "DistantLight"}.intersection(light_types)


def yaw_quat_wxyz(yaw_deg: float):
    """Convert a scene heading in degrees into Isaac's wxyz quaternion."""
    import math

    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))
