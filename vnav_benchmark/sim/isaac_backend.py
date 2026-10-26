"""Isaac backend for Python 3.11, with shared logic using Python 3.8 syntax.

Use the package-native Go2 runtime for calibrated frames, walking inference,
and scene elevator motion. There is no robot workspace import or publisher shim.

Physics advances only in step(), for hold_s times 200 physics ticks. Commands are
refreshed on every tick, preventing the driver's 0.5 s expiry. Idle periods do
not render or advance physics. An extra render at each hold boundary compensates
for annotator latency; FrameDeduper timestamps and deduplicates the 10 Hz frames
using their actual rendered times.

Teleportation resets only the robot: set_default_state, post_reset, a new driver
and tracking-camera reset. Avoid world.reset(), which resets camera mounts,
driver gains and moving parts. TELEPORT_RESET_ALLOWED gates teleport resets;
scored episodes start a fresh Isaac instance at their spawn pose.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vnav_benchmark import paths, pins
from vnav_benchmark.sim import runtime as sim_runtime
from vnav_benchmark.sim.ctrl_protocol import (
    E_BAD_REQUEST,
    E_POSE_INVALID,
    E_RENDER_EMPTY,
    E_TELEPORT_UNSTABLE,
    CtrlError,
    StepReply,
    TraceSample,
    WireObs,
    WirePose,
)

TELEPORT_RESET_ALLOWED = False
"Allow teleport resets only after proving equivalence to a fresh simulator spawn."
SPAWN_MATCH_M = 0.01
SPAWN_MATCH_RAD = math.radians(0.5)
TELEPORT_MAX_DRIFT_M = 0.05
TELEPORT_MIN_UP_Z = 0.95
RESET_WARM_RENDERS = 10
RENDER_RETRIES = 3
UP_Z_FALLEN = 0.5


class FrameDeduper:
    """Deduplicate frames by rendered time, accounting for annotator latency."""

    def __init__(self) -> None:
        self.last_t: Optional[float] = None

    def accept(self, image_time: Optional[float]) -> bool:
        if image_time is None:
            return False
        if self.last_t is not None and image_time <= self.last_t + 1e-09:
            return False
        self.last_t = image_time
        return True


def hold_schedule(hold_s: float, physics_hz: float, camera_hz: float) -> Tuple[int, int]:
    """Return physics ticks and render interval; hold must divide both exactly."""
    steps = int(round(hold_s * physics_hz))
    per_frame = int(round(physics_hz / camera_hz))
    if steps <= 0 or steps % per_frame or abs(steps - hold_s * physics_hz) > 1e-06:
        raise CtrlError(E_BAD_REQUEST, f"hold {hold_s} s 不是渲染周期 {1 / camera_hz} s 的整数倍")
    return (steps, per_frame)


def yaw_close(a: float, b: float, tol: float) -> bool:
    return abs(math.remainder(a - b, 2.0 * math.pi)) <= tol


ROBOTS = ("go2",)
"Supported simulation robot backends."


class IsaacBackend:
    """Start Isaac, load the scene, spawn and settle the robot at construction."""

    def __init__(
        self,
        scene: str,
        start_xy: Tuple[float, float],
        start_yaw: float,
        third_person: bool = False,
        gui: bool = False,
        robot: str = "go2",
    ) -> None:
        if robot not in ROBOTS:
            raise ValueError(f"不支持的机器人 {robot!r},只支持 {ROBOTS}")
        self.robot_name = robot
        self.rt = rt = sim_runtime
        self.rates = rates = rt.RATES
        self.scene_name = scene
        self.third_person = third_person
        scene_dir = Path(paths.setting("VNAV_SCENE_DIR", str(paths.asset_root() / "scenes")))
        scene_path = scene_dir / (scene + ".usd")
        robot_path = Path(
            paths.setting(
                "VNAV_GO2_USD",
                str(paths.asset_root() / "robots/go2/go2_odin.usd"),
            )
        )
        if not scene_path.is_file() or not robot_path.is_file():
            raise FileNotFoundError(
                "Scene/robot USD assets are missing; configure VNAV_ASSET_DIR and "
                "VNAV_GO2_USD before starting Isaac"
            )
        self.scene_path = Path(scene_path)
        self.frames = frames = rt.frames.GO2
        from isaacsim import SimulationApp

        self.sim_app = SimulationApp({"headless": not gui})
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom, UsdLux

        self.np = np
        world = World(
            physics_dt=1.0 / rates.physics_hz,
            rendering_dt=1.0 / rates.camera_hz,
            stage_units_in_meters=1.0,
        )
        self.world = world
        add_reference_to_stage(usd_path=str(scene_path), prim_path="/World/Scene")
        stage = get_current_stage()
        light_types = [p.GetTypeName() for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
        if rt.needs_ambient_light(light_types):
            dome = UsdLux.DomeLight.Define(stage, "/World/FallbackDomeLight")
            dome.CreateIntensityAttr(rt.AMBIENT_DOME_INTENSITY)
        world.reset()
        self.parts = rt.PartMotionDriver.discover(stage, "/World/Scene", Path(scene_path).stem)
        if self.parts is not None:
            self.parts.register(world)
            self.parts.set_robot_pose_fn(
                self._robot_pose_for_parts, body_size_m=rt.robot_body_size(self.robot_name)
            )
        ground_z = rt.find_ground(start_xy, None)
        self.ground_z = 0.0 if ground_z is None else float(ground_z)
        prim = add_reference_to_stage(usd_path=str(robot_path), prim_path=rt.ROBOT_PRIM)
        rt.set_translate(
            UsdGeom.Xformable(prim),
            Gf.Vec3d(start_xy[0], start_xy[1], self.ground_z + rt.walking.SPAWN_CLEAR_M),
        )
        rt.limit_depenetration(stage, rt.ROBOT_PRIM)
        root_path = f"{rt.ROBOT_PRIM}/{frames.urdf_root}"
        trunk_path = f"{rt.ROBOT_PRIM}/{frames.mount_link}"
        camera_path = f"{trunk_path}/odin_camera"
        self.camera = Camera(
            prim_path=camera_path,
            resolution=(rt.camera.REAL_RGBD_WIDTH, rt.camera.REAL_RGBD_HEIGHT),
        )
        tpc = rt.camera
        self.third_camera = (
            Camera(prim_path=rt.THIRD_PERSON_PRIM, resolution=(tpc.WIDTH, tpc.HEIGHT))
            if third_person
            else None
        )
        self.robot = world.scene.add(SingleArticulation(prim_path=root_path, name=robot))
        self.mount_body = None
        if trunk_path != root_path:
            from isaacsim.core.prims import SingleRigidPrim

            self.mount_body = world.scene.add(
                SingleRigidPrim(prim_path=trunk_path, name="mount_link")
            )
        world.reset()
        self.joint_names = list(self.robot.dof_names)
        self.stance = np.array(rt.walking.default_stance(self.joint_names))
        self._apply_default_state(start_xy[0], start_xy[1], start_yaw, self.ground_z)
        world.reset()
        self.driver = self._new_driver()
        if self.parts is not None:
            self.parts.after_reset()
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        (mount_xyz, mount_xyzw) = rt.frames.transform_to_pose(frames.t_mount_camera)
        self.camera.set_local_pose(
            translation=np.array(mount_xyz),
            orientation=np.array([mount_xyzw[3], *mount_xyzw[:3]]),
            camera_axes="ros",
        )
        usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        lens = rt.camera.real_robot_lens()
        usd_camera.CreateProjectionAttr().Set("perspective")
        usd_camera.CreateFocalLengthAttr().Set(lens["focalLength"])
        usd_camera.CreateHorizontalApertureAttr().Set(lens["horizontalAperture"])
        usd_camera.CreateVerticalApertureAttr().Set(lens["verticalAperture"])
        usd_camera.CreateHorizontalApertureOffsetAttr().Set(lens["horizontalApertureOffset"])
        usd_camera.CreateVerticalApertureOffsetAttr().Set(lens["verticalApertureOffset"])
        usd_camera.CreateClippingRangeAttr().Set(Gf.Vec2f(rt.NEAR_CLIP_M, rt.FAR_CLIP_M))
        self.third_usd_camera = None
        if self.third_camera is not None:
            self.third_camera.initialize()
            third_lens = tpc.lens()
            self.third_usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(rt.THIRD_PERSON_PRIM))
            self.third_usd_camera.CreateProjectionAttr().Set("perspective")
            self.third_usd_camera.CreateFocalLengthAttr().Set(third_lens["focalLength"])
            self.third_usd_camera.CreateHorizontalApertureAttr().Set(
                third_lens["horizontalAperture"]
            )
            self.third_usd_camera.CreateVerticalApertureAttr().Set(third_lens["verticalAperture"])
            self.third_usd_camera.CreateClippingRangeAttr().Set(
                Gf.Vec2f(third_lens["near"], third_lens["far"])
            )
        self.K = (
            rt.camera.REAL_RGBD_FX,
            0.0,
            rt.camera.REAL_RGBD_CX,
            0.0,
            rt.camera.REAL_RGBD_FY,
            rt.camera.REAL_RGBD_CY,
            0.0,
            0.0,
            1.0,
        )
        self.t_world_sim = rt.frames.scene_to_sim_world(self.ground_z)
        self.depth_encoding = rt.protocol.DEPTH_ENC_PNG
        self.follow = tpc.FollowCamera(raycast=rt.raycast_skip_robot) if third_person else None
        self.spawn = (float(start_xy[0]), float(start_xy[1]), float(start_yaw))
        self.fresh = True
        self.sim_time = 0.0
        self.step_index = 0
        self.previous_odom: Any = None
        self._settle(rt.walking.SETTLE_S)
        for index in range(rt.WARMUP_RENDER_FRAMES):
            self._drive(0.0, index)
            world.step(render=False)
            world.render()
        self._begin_episode_clock()

    def _apply_default_state(self, x: float, y: float, yaw: float, ground_z: float) -> None:
        (np, rt) = (self.np, self.rt)
        root_z = ground_z + self.frames.stand_mount_height_m + rt.walking.STAND_SPAWN_GAP_M
        self.robot.set_joints_default_state(
            positions=self.stance, velocities=np.zeros(len(self.stance))
        )
        self.robot.set_default_state(
            position=np.array([x, y, root_z]),
            orientation=np.array(rt.yaw_quat_wxyz(math.degrees(yaw))),
        )

    def _new_driver(self) -> Any:
        rt = self.rt
        policy = Path(
            paths.setting(
                "VNAV_WALK_POLICY",
                str(paths.asset_root() / "weights/go2/go2_moe_cts_176k_0.6984.pt"),
            )
        )
        expected = pins.ASSET_PINS["weights/go2/go2_moe_cts_176k_0.6984.pt"]
        if not policy.is_file() or pins.file_sha256(policy) != expected:
            raise pins.PinError("Missing or mismatched Go2 walking policy; run download_data.sh")
        return rt.walking.Go2Driver(self.robot, self.stance, policy, self.rates.physics_hz, self.np)

    def _drive(self, now: float, step: int) -> None:
        self.driver.step(now, step)
        if self.parts is not None:
            self.parts.step(now, self._robot_xy)

    def _robot_xy(self) -> Tuple[float, float]:
        position = self.robot.get_world_pose()[0]
        return (float(position[0]), float(position[1]))

    def _robot_pose_for_parts(self) -> Optional[Tuple[float, float, float, float]]:
        """Return (x, y, z, yaw_deg) to PartMotionDriver, or None before spawning."""
        if getattr(self, "robot", None) is None:
            return None
        position = self.robot.get_world_pose()[0]
        pose = self._pose(None)
        yaw_rad = math.atan2(
            2.0 * (pose.quat_xyzw[3] * pose.quat_xyzw[2] + pose.quat_xyzw[0] * pose.quat_xyzw[1]),
            1.0 - 2.0 * (pose.quat_xyzw[1] ** 2 + pose.quat_xyzw[2] ** 2),
        )
        return (float(position[0]), float(position[1]), float(position[2]), math.degrees(yaw_rad))

    def _settle(self, seconds: float) -> None:
        for index in range(round(seconds * self.rates.physics_hz)):
            self._drive(0.0, index)
            self.world.step(render=False)

    def _t_world_trunk(self) -> Any:
        body = self.mount_body if self.mount_body is not None else self.robot
        return self.t_world_sim @ self.rt.isaac_pose(*body.get_world_pose())

    def _render(self, t_render: float) -> Optional[float]:
        """Render and return the captured frame time, or None without history."""
        if self.follow is not None:
            trunk = self._t_world_trunk()
            pose = self.follow.update(
                dt=1.0 / self.rates.camera_hz,
                robot_xy=(float(trunk[0, 3]), float(trunk[1, 3])),
                ground_z=self.ground_z,
                yaw=float(self.np.arctan2(trunk[1, 0], trunk[0, 0])),
            )
            self.rt.apply_follow_pose(self.third_camera, self.third_usd_camera, pose, self.np)
        self.world.render()
        return self.frame_times.on_render(t_render)

    def _grab_rgb(self, camera: Any, quality: int) -> Optional[bytes]:
        rgba = camera.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None
        bgr = self.np.asarray(rgba)[:, :, :3][:, :, ::-1].astype("uint8")
        return bytes(self.rt.codecs.encode_rgb(bgr, quality=quality))

    def _pose(self, dt: Optional[float]) -> WirePose:
        """Return world truth in scene XY; difference the last 10 Hz velocity sample."""
        t_world_trunk = self._t_world_trunk()
        (odom, _, _) = self.frames.odom_poses(t_world_trunk)
        if self.previous_odom is not None and dt:
            (linear, angular) = self.rt.frames.planar_twist(self.previous_odom, odom, dt)
        else:
            (linear, angular) = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.previous_odom = odom
        (xyz, xyzw) = self.rt.frames.transform_to_pose(odom)
        return WirePose(
            position_xyz=tuple(float(v) for v in xyz),
            quat_xyzw=tuple(float(v) for v in xyzw),
            v_body=tuple(float(v) for v in linear),
            w_body=tuple(float(v) for v in angular),
            up_z=float(t_world_trunk[2, 2]),
        )

    def _obs_now(self, pose: Optional[WirePose] = None) -> WireObs:
        """Render once more to capture hold-end RGB-D despite annotator latency."""
        rt = self.rt
        for _ in range(RENDER_RETRIES + 1):
            self._render(self.sim_time)
            rgb = self._grab_rgb(self.camera, rt.codecs.JPEG_QUALITY)
            depth = self.camera.get_depth()
            if rgb is not None and depth is not None and (getattr(depth, "size", 0) > 0):
                break
        else:
            raise CtrlError(E_RENDER_EMPTY, "补渲多次仍取不到图")
        (data, enc) = rt.codecs.encode_depth(self.np.asarray(depth), encoding=self.depth_encoding)
        return WireObs(
            t_sim=round(self.sim_time, 6),
            rgb_jpeg=rgb,
            depth=bytes(data),
            depth_encoding=enc,
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
            K=self.K,
            pose=pose or self._pose(None),
        )

    def _begin_episode_clock(self) -> None:
        self.sim_time = 0.0
        self.step_index = 0
        self.previous_odom = None
        self.frame_times = self.rt.RenderedFrameTimes()
        self.frame_times.on_render(0.0)
        self.dedup = FrameDeduper()
        self.dedup.accept(0.0)

    def _teleport(self, x: float, y: float, yaw: float, settle_s: float) -> None:
        """Reset the robot/driver/tracking camera without calling world.reset()."""
        ground = self.rt.find_ground((x, y), None)
        if ground is None:
            raise CtrlError(E_POSE_INVALID, f"({x:.2f}, {y:.2f}) 处找不到地面")
        self._apply_default_state(x, y, yaw, float(ground))
        self.robot.post_reset()
        self.driver = self._new_driver()
        self.previous_odom = None
        self._settle(settle_s)
        if self.follow is not None:
            trunk = self._t_world_trunk()
            pose = self.follow.reset(
                (float(trunk[0, 3]), float(trunk[1, 3])),
                self.ground_z,
                float(self.np.arctan2(trunk[1, 0], trunk[0, 0])),
            )
            self.rt.apply_follow_pose(self.third_camera, self.third_usd_camera, pose, self.np)
        pose = self._pose(None)
        drift = math.hypot(pose.position_xyz[0] - x, pose.position_xyz[1] - y)
        if drift >= TELEPORT_MAX_DRIFT_M or pose.up_z <= TELEPORT_MIN_UP_Z:
            raise CtrlError(E_TELEPORT_UNSTABLE, f"整定后漂 {drift:.3f} m、up_z {pose.up_z:.3f}")
        self.fresh = False

    def info(self) -> Dict[str, Any]:
        return {
            "backend": "isaac",
            "scene": self.scene_name,
            "scene_usd": str(self.scene_path),
            "width": int(self.rt.camera.REAL_RGBD_WIDTH),
            "height": int(self.rt.camera.REAL_RGBD_HEIGHT),
            "K": list(self.K),
            "physics_hz": self.rates.physics_hz,
            "camera_hz": self.rates.camera_hz,
            "spawn": list(self.spawn),
            "ground_z": self.ground_z,
            "teleport_reset_allowed": TELEPORT_RESET_ALLOWED,
            "robot": self.robot_name,
        }

    def reset(self, x: float, y: float, yaw: float) -> WireObs:
        at_spawn = math.hypot(x - self.spawn[0], y - self.spawn[1]) <= SPAWN_MATCH_M and yaw_close(
            yaw, self.spawn[2], SPAWN_MATCH_RAD
        )
        if not (self.fresh and at_spawn):
            if not TELEPORT_RESET_ALLOWED:
                raise CtrlError(E_POSE_INVALID, "出发位姿与生成位姿不同或已走过:要新起 Isaac")
            self._teleport(x, y, yaw, self.rt.walking.SETTLE_S)
        self.fresh = False
        self._begin_episode_clock()
        return self._obs_now()

    def step(
        self, cmd: Tuple[float, float, float], hold_s: float, want_frames: bool, want_third: bool
    ) -> StepReply:
        (n_steps, per_frame) = hold_schedule(hold_s, self.rates.physics_hz, self.rates.camera_hz)
        dt = 1.0 / self.rates.physics_hz
        trace: List[TraceSample] = []
        frames: List[Tuple[float, bytes]] = []
        third: List[Tuple[float, bytes]] = []
        quality_third = int(self.rt.THIRD_PERSON_JPEG_QUALITY)

        def collect(image_time: Optional[float]) -> None:
            if not self.dedup.accept(image_time):
                return
            if want_frames:
                jpeg = self._grab_rgb(self.camera, self.rt.codecs.JPEG_QUALITY)
                if jpeg is not None:
                    frames.append((round(float(image_time), 6), jpeg))
            if want_third and self.third_camera is not None:
                jpeg = self._grab_rgb(self.third_camera, quality_third)
                if jpeg is not None:
                    third.append((round(float(image_time), 6), jpeg))

        driver_range = getattr(self.driver, "command_range", None)
        limits = (
            self.rt.intersect_ranges(self.rt.PROTOCOL_COMMAND_RANGE, driver_range)
            if driver_range
            else self.rt.PROTOCOL_COMMAND_RANGE
        )
        cmd = tuple(self.rt.walking.clamp_command(cmd, limits))
        for _ in range(n_steps):
            render_now = self.step_index % per_frame == per_frame - 1
            self.driver.on_command(self.sim_time, cmd)
            self._drive(self.sim_time, self.step_index)
            self.world.step(render=False)
            self.sim_time += dt
            self.step_index += 1
            if render_now:
                collect(self._render(self.sim_time))
                trace.append(TraceSample(round(self.sim_time, 6), self._pose(per_frame * dt)))
        self.fresh = False
        last_pose = trace[-1].pose if trace else self._pose(None)
        obs = self._obs_now(last_pose)
        collect(obs.t_sim)
        return StepReply(
            obs=obs,
            trace=trace,
            frames=frames,
            third=third,
            applied_cmd=tuple(float(v) for v in cmd),
            fallen_now=last_pose.up_z < UP_Z_FALLEN,
        )

    def set_pose(self, x: float, y: float, yaw: float, settle_s: float) -> WireObs:
        self._teleport(x, y, yaw, settle_s)
        self._begin_episode_clock()
        for _ in range(RESET_WARM_RENDERS - 1):
            self._render(self.sim_time)
        return self._obs_now()

    def capture(self, warm_renders: int) -> WireObs:
        for _ in range(max(0, warm_renders - 1)):
            self._render(self.sim_time)
        return self._obs_now(self._pose(None))

    def truth(self) -> WirePose:
        return self._pose(None)

    def close(self) -> None:
        if self.parts is not None:
            self.parts.close()
        self.sim_app.close()
