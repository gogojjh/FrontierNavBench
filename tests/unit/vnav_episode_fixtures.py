"""Build synthetic episode directories for episode-video tests without simulation or network.

Cover the twelve v1 configurations in fairness.V1_METHOD_IDS. Image-goal method/vlm/steps.jsonl
and calls.jsonl reuse prompt.build_request and VlmMemory, keeping parts manifests byte-identical
to the real policy instead of duplicating prompt assembly.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from frontier_nav_bench.fairness import parse_method_id
from frontier_nav_bench.interfaces import UsageRecord
from frontier_nav_bench.rundir import TRACE_COLUMNS
from frontier_nav_bench.vlm.client import ImagePart, TextPart, VlmRequest
from frontier_nav_bench.vlm.memory import VlmMemory
from frontier_nav_bench.vlm.prompt import (
    FRAME_SIZE,
    StepInputs,
    SubgoalView,
    build_request,
    prompt_fingerprint,
)
from frontier_nav_bench.vlm.prompt import HOLD_S as IMGGOAL_HOLD_S

PTGOAL_HOLD_S = (
    0.2  # Match ptgoal_sru.HOLD_S without importing the module and its SRU-client dependencies.
)
EGO_SIZE = FRAME_SIZE  # (640, 480)
SCENE = "env0_office_single_floor"
# Verified free, mutually visible lobby-to-lounge segment in the truth grid.
# build_truth_grid gives free endpoints, clear line of sight and an exactly 3.0 m straight path.
START_XY: Tuple[float, float] = (12.5, 2.6)
GOAL_XY: Tuple[float, float] = (15.5, 2.6)
SHORTEST_M = 3.0
N_SUBGOALS = 3  # Synthetic subgoal count for mapped image-goal methods.
ENDPOINT_NAME = {"astra": "astra_flex", "qwen": "qwen"}  # Match imggoal_vlm.DEFAULT_ENDPOINT.


def frame_color_bgr(t_ms: int) -> Tuple[int, int, int]:
    """Return deterministic frame colors from t_ms for EGO-panel mean-color checks."""
    b = 40 + (t_ms // 50) % 180
    g = 220 - (t_ms // 80) % 160
    r = 90 + (t_ms // 130) % 140
    return int(b), int(g), int(r)


def _make_ego_frame(t_ms: int) -> np.ndarray:
    w, h = EGO_SIZE
    img = np.full((h, w, 3), frame_color_bgr(t_ms), dtype=np.uint8)
    cv2.putText(
        img, str(t_ms), (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA
    )
    return img


def _jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "JPEG 编码失败"
    return buf.tobytes()


def _solid_jpeg(size_wh: Tuple[int, int], color_bgr: Tuple[int, int, int]) -> bytes:
    w, h = size_wh
    return _jpeg(np.full((h, w, 3), color_bgr, dtype=np.uint8))


def _make_depth_frame(w: int, h: int) -> np.ndarray:
    """Return uint16 millimeter depth with a horizontal 0.5-6 m gradient and an invalid zero strip.
    """
    row = np.linspace(500, 6000, w, dtype=np.uint16)
    depth = np.tile(row, (h, 1))
    depth[:, : w // 10] = 0
    return depth


def fake_grid() -> Any:
    """Build a duck-typed 12 by 12 m grid at 0.1 m resolution with perimeter walls.

    Inject it through load_episode(grid_fn=...) to avoid expensive truth-grid construction and
    nav_sim_isaac imports.
    """
    cell_m = 0.1
    n = int(round(12.0 / cell_m))
    free = np.ones((n, n), dtype=bool)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    raw_blocked = ~free

    @dataclass
    class _FakeGrid:
        free: np.ndarray
        raw_blocked: np.ndarray
        x0: float
        y0: float
        cell_m: float
        walls: Tuple[Any, ...]

        @property
        def shape(self) -> Tuple[int, int]:
            return self.free.shape  # type: ignore[return-value]

    return _FakeGrid(free=free, raw_blocked=raw_blocked, x0=0.0, y0=0.0, cell_m=cell_m, walls=())


def _manifest_and_save(req: VlmRequest, images_dir: Path) -> List[Dict[str, Any]]:
    """Match the policy manifest: retain text, hash images and save them under images/<sha>.jpg."""
    images_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for part in req.parts:
        if isinstance(part, TextPart):
            items.append({"t": part.text, "bp": part.cache_breakpoint})
        else:
            assert isinstance(part, ImagePart)
            sha = hashlib.sha256(part.jpeg).hexdigest()
            path = images_dir / f"{sha}.jpg"
            if not path.exists():
                path.write_bytes(part.jpeg)
            items.append({"img": sha, "w": part.width, "h": part.height, "detail": part.detail})
    return items


def _lerp_xy(frac: float) -> Tuple[float, float]:
    frac = min(1.0, max(0.0, frac))
    x = START_XY[0] + (GOAL_XY[0] - START_XY[0]) * frac
    y = START_XY[1] + (GOAL_XY[1] - START_XY[1]) * frac
    return x, y


def make_episode_dir(
    root: Path, method_id: str, n_decisions: int = 12, success: bool = True
) -> Path:
    """Create an H7-layout episode at root/runs/<method_id>/ep_fix1/r0/."""
    profile = parse_method_id(method_id)
    is_imggoal = profile.planner == "imggoal"
    has_map = profile.map_variant is not None
    hold_s = IMGGOAL_HOLD_S if is_imggoal else PTGOAL_HOLD_S
    duration_s = n_decisions * hold_s
    trace_span_s = duration_s + hold_s + 1.0  # Extend beyond the final freeze-frame time.

    run_dir = Path(root) / "runs" / method_id / "ep_fix1" / "r0"
    ego_dir = run_dir / "frames" / "ego"
    depth_dir = run_dir / "depth"
    goal_dir = run_dir / "goal"
    method_dir = run_dir / "method"
    for d in (ego_dir, depth_dir, goal_dir):
        d.mkdir(parents=True, exist_ok=True)

    # First-person frames at 10 Hz.
    n_frames = int(round(trace_span_s * 10)) + 1
    for k in range(n_frames):
        t_ms = k * 100
        cv2.imwrite(
            str(ego_dir / f"{t_ms:08d}.jpg"), _make_ego_frame(t_ms), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

    # trace.csv: a 10 Hz straight path from START_XY to GOAL_XY.
    with open(run_dir / "trace.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TRACE_COLUMNS))
        writer.writeheader()
        speed = (GOAL_XY[0] - START_XY[0]) / duration_s if duration_s > 0 else 0.0
        n = int(round(trace_span_s * 10)) + 1
        for k in range(n):
            t = round(k / 10.0, 3)
            x, y = _lerp_xy(t / duration_s if duration_s > 0 else 1.0)
            writer.writerow(
                {
                    "t_sim": t,
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "yaw": 0.0,
                    "up_z": 1.0,
                    "vx_body": speed if t < duration_s else 0.0,
                    "wz_body": 0.0,
                }
            )

    goal_jpeg = _jpeg(np.full((EGO_SIZE[1], EGO_SIZE[0], 3), (200, 160, 60), dtype=np.uint8))
    (goal_dir / "goal.jpg").write_bytes(goal_jpeg)

    subgoal_jpegs = (
        [_solid_jpeg(FRAME_SIZE, (30 + 40 * k, 160, 200 - 30 * k)) for k in range(N_SUBGOALS)]
        if (is_imggoal and has_map)
        else []
    )
    images_dir = method_dir / "vlm" / "images"
    memory = VlmMemory() if is_imggoal else None
    # The final two steps enter FINAL_APPROACH after subgoals are exhausted.
    nav_end = max(0, n_decisions - 2) if has_map else 0

    framework_rows: List[Dict[str, Any]] = []
    vlm_step_rows: List[Dict[str, Any]] = []
    vlm_call_rows: List[Dict[str, Any]] = []
    endpoint_name = ENDPOINT_NAME.get(profile.vlm or "", "")

    for i in range(n_decisions):
        t_sim = round(i * hold_s, 3)
        x_i, y_i = _lerp_xy(t_sim / duration_s if duration_s > 0 else 1.0)
        vx = (GOAL_XY[0] - START_XY[0]) / duration_s if duration_s > 0 else 0.0
        done_flag = bool(success and i == n_decisions - 1)
        t_ms_i = int(round(t_sim * 1000))
        cv2.imwrite(str(depth_dir / f"{t_ms_i:08d}.png"), _make_depth_frame(*EGO_SIZE))

        if is_imggoal:
            in_nav = i < nav_end
            phase = "NAVIGATE" if in_nav else "FINAL_APPROACH"
            subgoal_idx = (i * N_SUBGOALS // max(nav_end, 1)) if in_nav else None
            subgoal_idx = min(subgoal_idx, N_SUBGOALS - 1) if subgoal_idx is not None else None
            subgoal_view = (
                SubgoalView(subgoal_jpegs[subgoal_idx], subgoal_idx + 1, N_SUBGOALS)
                if subgoal_idx is not None
                else None
            )
            if done_flag:
                phase = "DONE"
            current_img = _make_ego_frame(t_ms_i)
            current_jpeg = _jpeg(current_img)
            assert memory is not None
            inputs = StepInputs(
                decision=i,
                goal_jpeg=goal_jpeg,
                current_jpeg=current_jpeg,
                subgoal=subgoal_view,
                history=memory.history(i),
                log=memory.log(),
                memory_note=memory.note,
                feedback=(),
            )
            req = build_request(inputs)
            manifest = _manifest_and_save(req, images_dir)
            usage = UsageRecord(
                provider="dashscope" if profile.vlm == "qwen" else "openai",
                model="fixture-model",
                input_tokens=3000,
                cached_tokens=1800,
                output_tokens=450,
                reasoning_tokens=0,
                list_usd=0.02,
                billed_usd=0.01,
            )
            vlm_call_rows.append(
                {
                    "decision": i,
                    "attempt": 0,
                    "status": "ok",
                    "error": "",
                    "endpoint": endpoint_name,
                    "fingerprint": prompt_fingerprint(),
                    "parts": manifest,
                    "usage": asdict(usage),
                }
            )
            observation = f"corridor frame {i}"
            memory_note = f"heading toward goal, step {i}"
            reply = {
                "observation": observation,
                "target_visible": True,
                "subgoal_reached": False,
                "done": done_flag,
                "memory": memory_note,
                "vx": vx,
                "vy": 0.0,
                "wz": 0.0,
            }
            vlm_step_rows.append(
                {
                    "decision": i,
                    "phase": phase,
                    "subgoal": None if subgoal_idx is None else [subgoal_idx + 1, N_SUBGOALS],
                    "status": "ok",
                    "error": "",
                    "calls": 1,
                    "reply": reply,
                    "cmd": [vx, 0.0, 0.0],
                    "hold_s": hold_s,
                    "done": done_flag,
                    "clamped": False,
                    "latency": {"vlm_api_s": 1.2, "server_ms": 900.0, "retries": 0.0},
                }
            )
            memory.record(i, vx, 0.0, 0.0, observation, status="ok")
            memory.set_note(memory_note)
            memory.add_frame(i, cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB))
            debug = {
                "latency": {"vlm_api_s": 1.2},
                "phase": phase,
                "subgoal": None if subgoal_idx is None else [subgoal_idx + 1, N_SUBGOALS],
                "status": "ok",
            }
            note = ""
        else:
            phase = "DONE" if done_flag else "TRACK"
            debug = {"latency": {"loc_s": 0.0}, "phase": phase, "kf": f"kf{i % 4}"}
            note = phase

        framework_rows.append(
            {
                "i": i,
                "t_sim": t_sim,
                "budget_left_s": max(0.0, 180.0 - t_sim),
                "act_wall_s": 0.05,
                "action": {"vx": vx, "vy": 0.0, "wz": 0.0, "hold_s": hold_s, "done": done_flag},
                "applied": {"vx": vx, "vy": 0.0, "wz": 0.0, "hold_s": hold_s},
                "clamp": {"vx": False, "vy": False, "wz": False, "nonfinite": False, "hold": False},
                "note": note,
                "debug": debug,
                "odom": {"x": x_i, "y": y_i, "yaw": 0.0},
                "proximity": False,
            }
        )

    with open(run_dir / "steps.jsonl", "w", encoding="utf-8") as fh:
        for row in framework_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if is_imggoal:
        (method_dir / "vlm").mkdir(parents=True, exist_ok=True)
        with open(method_dir / "vlm" / "steps.jsonl", "w", encoding="utf-8") as fh:
            for row in vlm_step_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(method_dir / "vlm" / "calls.jsonl", "w", encoding="utf-8") as fh:
            for row in vlm_call_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = {
        "termination": "done" if success else "timeout",
        "success": success,
        "success_loose": success,
        "final_dist_m": 0.1 if success else 2.5,
        "final_yaw_err_deg": 5.0 if success else 40.0,
        "ever_within_pose": success,
        "premature_done": False,
        "falls": 0,
        "fallen_at_s": None,
        "stuck_events": 0,
        "proximity_events": 0,
        "path_length_m": SHORTEST_M * (1.0 if success else 1.4),
        "shortest_path_m": SHORTEST_M,
        "spl": 1.0 if success else 0.0,
        "plr": 1.0 if success else None,
        "sim_time_s": duration_s,
        "decisions": n_decisions,
        "act_wall_s": {"mean": 0.05, "median": 0.05, "p95": 0.06, "max": 0.08},
        "compute_per_sim_s": 0.1,
    }
    # Rendering consumes call logs, not policy summary aggregates. Keep run.json schema-valid
    # by deriving its usage totals from those same calls instead of synthesizing a second log.
    usage_total = {
        "calls": len(vlm_call_rows),
        **{
            key: sum(row["usage"][key] for row in vlm_call_rows)
            for key in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
        },
        **{
            key: round(sum(row["usage"][key] for row in vlm_call_rows), 4)
            for key in ("list_usd", "billed_usd")
        },
    }
    run = {
        "schema": "vnav.run/v1",
        "method_id": method_id,
        "episode_set": "fixture_set",
        "episode_token": "ep_fix1",
        "repeat_index": 0,
        "status": "scored",
        "attempt": 1,
        "git": {"sha": "fixture"},
        "server": {"host": "fixture"},
        "settings": {
            "budget_s": 180.0,
            "limits": {"vx_min": -0.8, "vx_max": 0.8, "vy_max": 0.3, "wz_max": 0.7},
            "hold": {"quantum_s": 0.1, "min_s": 0.1, "max_s": 2.0},
            "success": {"dist_m": 0.5, "yaw_deg": 45.0, "loose_dist_m": 1.0},
            "seed": 0,
            "method_config_sha256": "fixture",
            "map": (
                None
                if not has_map
                else {
                    "map_id": "fixture-map",
                    "variant": profile.map_variant,
                    "sha256": "fixture",
                    "edge_type": "topo",
                }
            ),
            "localizer": {"kind": "fixture", "sigma_xy_m": 0.05, "sigma_yaw_rad": 0.02},
        },
        "episode": {
            "scene": SCENE,
            "start": {"x": START_XY[0], "y": START_XY[1], "yaw": 0.0},
            "goal": {"x": GOAL_XY[0], "y": GOAL_XY[1], "yaw": 0.0},
            "shortest_m": SHORTEST_M,
            "straight_m": SHORTEST_M,
            "detour": 1.0,
            "tier": "near",
            "same_room": False,
            "start_room": "lobby",
            "goal_room": "lounge",
            "waypoints": [[START_XY[0], START_XY[1]], [GOAL_XY[0], GOAL_XY[1]]],
        },
        "timing": {
            "wall_start": "2026-10-20T00:00:00",
            "wall_end": "2026-10-20T00:00:30",
            "sim_s": duration_s,
            "decisions": n_decisions,
            "act_wall_s": out["act_wall_s"],
            "compute_per_sim_s": 0.1,
            "method_latency": {},
        },
        "outcome": out,
        "metrics": {
            "success": success,
            "success_loose": success,
            "spl": out["spl"],
            "plr": out["plr"],
            "path_length_m": out["path_length_m"],
            "shortest_path_m": SHORTEST_M,
        },
        "method_stats": {
            "loc": {"used": 0, "ok": 0, "denied": 0},
            "clamped": {"vx": 0, "vy": 0, "wz": 0, "nonfinite": 0, "hold": 0},
            "server_clamped": 0,
        },
        "usage": usage_total,
        "infra": {"retries": 0, "attempts": 1, "errors": []},
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir
