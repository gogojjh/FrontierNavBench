# Upstream: tools/viz/video.py
# Original SHA256: b250b1d34e0a03412540f5971af890d6ad1ccb413a8f52e05779ee613dfb8904.
"""Bounded video rendering, safe OpenCV forking, encoding and command-line output."""

import argparse
import json
import math
import multiprocessing
import os
import resource
import shutil
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from . import drawing as draw
from . import tracks as bagdata
from .tracks import index_at

Rule = Tuple[str, Callable[[float], Optional[Any]]]


@dataclass(frozen=True)
class Timeline:
    """Timeline."""

    t0: float
    t_end: float
    speed: float
    fps: int
    hold_s: float

    @property
    def n_run(self) -> int:
        span = self.t_end - self.t0
        if span <= 0:
            return 1
        return max(1, int(math.ceil(span / self.speed * self.fps)))

    @property
    def n_total(self) -> int:
        return self.n_run + int(round(self.hold_s * self.fps))

    def time_at(self, k: int) -> Tuple[float, bool]:
        """Time at."""
        if k < self.n_run:
            t = self.t0 + k / self.fps * self.speed
            return min(t, self.t_end), False
        return self.t_end, True


class StateTimeline:
    """Statetimeline."""

    def __init__(
        self, t0: float, t_end: float, step_s: float, rules: Sequence[Rule], min_dwell_s: float
    ) -> None:
        priority = {name: i for i, (name, _fn) in enumerate(rules)}
        n_steps = int(math.ceil((t_end - t0) / step_s)) + 1 if t_end > t0 else 1
        timeline = np.minimum(t0 + np.arange(n_steps, dtype=np.float64) * step_s, t_end)
        names: List[str] = []
        ctxs: List[Any] = []
        current_name: Optional[str] = None
        current_ctx: Any = None
        dwell_start = float(timeline[0])
        for t in timeline.tolist():
            candidate_name, candidate_ctx = current_name, current_ctx
            for name, fn in rules:
                found = fn(t)
                if found is not None:
                    candidate_name, candidate_ctx = name, found
                    break
            if current_name is None or candidate_name != current_name:
                first_step = current_name is None
                higher = (not first_step) and priority[candidate_name] < priority[current_name]
                if first_step or higher or (t - dwell_start) >= min_dwell_s:
                    current_name, current_ctx, dwell_start = candidate_name, candidate_ctx, t
            names.append(current_name)
            ctxs.append(current_ctx)
        self._timeline = timeline
        self._names = names
        self._ctxs = ctxs

    def at(self, t: float) -> Tuple[str, Any]:
        idx = min(max(index_at(self._timeline, t), 0), len(self._timeline) - 1)
        return self._names[idx], self._ctxs[idx]


class SyncInputs(NamedTuple):
    """Syncinputs."""

    turn_t: Optional[float]
    t0: float
    odom: bagdata.PoseTrack
    ego: Optional[bagdata.FrameTrack]
    third: Optional[bagdata.FrameTrack]


@dataclass
class VideoJob:
    """Videojob."""

    timeline: Timeline
    frame: Callable[[int], np.ndarray]
    size: Tuple[int, int]
    info: Dict[str, Any]
    sync_inputs: SyncInputs


class VideoError(Exception):
    """Videoerror."""


@dataclass(frozen=True)
class VideoResult:
    """Videoresult."""

    mp4: Optional[Path]
    json_path: Path
    contact: Optional[Path]
    stills: Tuple[Path, ...]

    frames: int

    encoder: str
    seconds: float


ENCODER_GSTREAMER, ENCODER_OPENCV_AVC1, ENCODER_OPENCV_MP4V = (
    "gstreamer-x264-i420",
    "opencv-avc1",
    "opencv-mp4v",
)
GST_BITRATE_KBPS = 10000
GST_KEY_INT_SECONDS = 2.0


def _open_encoder(
    path: Path, size: Tuple[int, int], fps: float, log: Callable[[str], None]
) -> Tuple[cv2.VideoWriter, str]:
    """Open encoder."""
    key_int_max = max(1, round(GST_KEY_INT_SECONDS * fps))
    gst_pipeline = (
        "appsrc block=true ! videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc bitrate={GST_BITRATE_KBPS} speed-preset=veryfast key-int-max={key_int_max} ! "
        f'h264parse ! mp4mux ! filesink location="{path}"'
    )
    writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, fps, size, True)
    if writer.isOpened():
        return writer, ENCODER_GSTREAMER
    writer.release()
    log("GStreamer(I420) 编码器打不开，退回 OpenCV avc1")

    writer = cv2.VideoWriter(
        str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"avc1"), fps, size, True
    )
    if writer.isOpened():
        return writer, ENCODER_OPENCV_AVC1
    writer.release()
    log("OpenCV avc1 编码器打不开，退回 OpenCV mp4v")

    writer = cv2.VideoWriter(
        str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"mp4v"), fps, size, True
    )
    if writer.isOpened():
        return writer, ENCODER_OPENCV_MP4V
    writer.release()
    raise VideoError("gstreamer-x264-i420 / opencv-avc1 / opencv-mp4v 三个编码器都打不开")


def _verify_output(mp4_path: Path, expected_frames: int, size: Tuple[int, int]) -> Dict[str, Any]:
    """Verify output."""
    cap = cv2.VideoCapture(str(mp4_path))
    try:
        opened = cap.isOpened()
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    finally:
        cap.release()
    frame_count_ok = opened and abs(frames - expected_frames) <= 1
    size_ok = opened and (width, height) == tuple(size)
    return {
        "opened": opened,
        "frames": frames,
        "width": width,
        "height": height,
        "frame_count_ok": frame_count_ok,
        "size_ok": size_ok,
        "ok": bool(frame_count_ok and size_ok),
    }


CONTACT_SHEET_COLS, CONTACT_SHEET_ROWS = 3, 3
CONTACT_SHEET_COUNT = CONTACT_SHEET_COLS * CONTACT_SHEET_ROWS
CONTACT_JPEG_QUALITY, CONTACT_LABEL_SIZE, CONTACT_LABEL_MARGIN_PX, CONTACT_LABEL_OFFSET_PX = (
    90,
    20,
    6,
    4,
)
CONTACT_LABEL_CHIP_BGR = (20, 20, 20)
STILLS_RUN_COUNT, STILLS_JPEG_QUALITY = 6, 95


def _pick_indices(n: int, count: int) -> List[int]:
    """Pick indices."""
    if n <= 0:
        return [0] * count
    if count <= 1:
        return [n - 1]
    return [int(round(i * (n - 1) / (count - 1))) for i in range(count)]


def _stills_indices(n_run: int, n_total: int) -> List[int]:
    """Stills indices."""
    return _pick_indices(n_run, STILLS_RUN_COUNT) + [n_total - 1]


def _build_contact_sheet(
    timeline: Timeline, size: Tuple[int, int], thumbs: Dict[int, np.ndarray], indices: List[int]
) -> np.ndarray:
    """Build contact sheet."""

    def _label_chip(canvas: np.ndarray, x0: int, y0: int, label: str) -> None:
        text_w, text_h = draw.text_size(label, CONTACT_LABEL_SIZE, True)
        cx0, cy0 = x0 + CONTACT_LABEL_OFFSET_PX, y0 + CONTACT_LABEL_OFFSET_PX
        cx1, cy1 = (
            cx0 + text_w + 2 * CONTACT_LABEL_MARGIN_PX,
            cy0 + text_h + 2 * CONTACT_LABEL_MARGIN_PX,
        )
        cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), CONTACT_LABEL_CHIP_BGR, -1)
        draw.blit_text(
            canvas,
            cx0 + CONTACT_LABEL_MARGIN_PX,
            cy0 + CONTACT_LABEL_MARGIN_PX,
            label,
            CONTACT_LABEL_SIZE,
            True,
            (255, 255, 255),
        )

    w, h = size
    cell_w, cell_h = w // CONTACT_SHEET_COLS, h // CONTACT_SHEET_ROWS
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for pos, k in enumerate(indices):
        row, col = divmod(pos, CONTACT_SHEET_COLS)
        y0, x0 = row * cell_h, col * cell_w
        canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = thumbs[k]
        t, _final = timeline.time_at(k)
        _label_chip(canvas, x0, y0, draw.format_mmss(t - timeline.t0))
    return canvas


SYNC_WINDOW_BEFORE_S, SYNC_WINDOW_AFTER_S = 3.0, 6.0
SYNC_YAW_THRESHOLD_DEG = 2.0
SYNC_FRAME_DIFF_MIN_ABS = 2.0
SYNC_FRAME_DIFF_MEDIAN_MULT, SYNC_OK_MAX_DIFF_S = 3.0, 0.2
SYNC_THIRD_PERSON_ROI: Tuple[float, float, float, float] = (0.35, 0.45, 0.65, 0.85)

SYNC_THIRD_FRAME_DIFF_MIN_ABS = 0.8
SYNC_GRAY_SIZE: Tuple[int, int] = (160, 120)


def _odom_onset(odom: bagdata.PoseTrack, window_start: float, window_end: float) -> Optional[float]:
    """Odom onset."""
    if len(odom.stamps) == 0:
        return None
    _x, _y, yaw_ref = odom.at(window_start)
    mask = (odom.stamps > window_start) & (odom.stamps <= window_end)
    for i in np.nonzero(mask)[0].tolist():
        if abs(math.degrees(float(odom.yaw[i]) - yaw_ref)) > SYNC_YAW_THRESHOLD_DEG:
            return float(odom.stamps[i])
    return None


def _read_gray_small(
    path: str, roi: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """Read gray small."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((SYNC_GRAY_SIZE[1], SYNC_GRAY_SIZE[0]), dtype=np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, SYNC_GRAY_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    if roi is None:
        return small
    h, w = small.shape
    return small[int(roi[1] * h) : int(roi[3] * h), int(roi[0] * w) : int(roi[2] * w)]


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _frame_onset(
    track: Optional[bagdata.FrameTrack],
    tc: float,
    window_start: float,
    window_end: float,
    roi: Optional[Tuple[float, float, float, float]] = None,
    min_abs: float = SYNC_FRAME_DIFF_MIN_ABS,
) -> Optional[float]:
    """Frame onset."""
    if track is None or len(track.stamps) < 2:
        return None
    stamps = track.stamps
    cache: Dict[int, np.ndarray] = {}

    def gray_at(i: int) -> np.ndarray:
        if i not in cache:
            cache[i] = _read_gray_small(track.paths[i], roi)
        return cache[i]

    before_idxs = np.nonzero(stamps < tc)[0].tolist()
    before_diffs = [
        _mean_abs_diff(gray_at(a), gray_at(b))
        for a, b in zip(before_idxs, before_idxs[1:])
        if b == a + 1
    ]
    threshold = max(
        min_abs,
        SYNC_FRAME_DIFF_MEDIAN_MULT * (float(np.median(before_diffs)) if before_diffs else 0.0),
    )

    win_idxs = np.nonzero((stamps >= window_start) & (stamps <= window_end))[0].tolist()
    for i in win_idxs:
        if i != 0 and _mean_abs_diff(gray_at(i - 1), gray_at(i)) > threshold:
            return float(stamps[i])
    return None


def sync_check(
    turn_t: Optional[float],
    t0: float,
    odom: bagdata.PoseTrack,
    ego: Optional[bagdata.FrameTrack],
    third: Optional[bagdata.FrameTrack],
) -> Dict[str, Any]:
    """Sync check."""
    if turn_t is None:
        return {"ok": None, "reason": "没有给转身时刻，找不到对齐时刻"}

    def rel(v: Optional[float]) -> Optional[float]:
        return None if v is None else v - t0

    window_start, window_end = turn_t - SYNC_WINDOW_BEFORE_S, turn_t + SYNC_WINDOW_AFTER_S
    odom_onset = _odom_onset(odom, window_start, window_end)
    ego_onset = _frame_onset(ego, turn_t, window_start, window_end)
    third_onset = (
        _frame_onset(
            third,
            turn_t,
            window_start,
            window_end,
            SYNC_THIRD_PERSON_ROI,
            SYNC_THIRD_FRAME_DIFF_MIN_ABS,
        )
        if third is not None
        else None
    )
    onsets = [v for v in (odom_onset, ego_onset, third_onset) if v is not None]
    if len(onsets) >= 2:
        max_diff_s: Optional[float] = max(onsets) - min(onsets)
        ok: Optional[bool] = max_diff_s <= SYNC_OK_MAX_DIFF_S
    else:
        max_diff_s, ok = None, False
    return {
        "turn_cmd_t": turn_t - t0,
        "odom_onset": rel(odom_onset),
        "ego_onset": rel(ego_onset),
        "third_onset": rel(third_onset),
        "max_diff_s": max_diff_s,
        "ok": ok,
    }


CHUNK_SIZE, MAX_WORKERS = 4, 6
MAX_IN_FLIGHT = 48
SEMAPHORE_POLL_TIMEOUT_S, PROGRESS_LOG_STEP = 0.2, 0.05

_frame_fn: Optional[Callable[[int], np.ndarray]] = None


def _init_worker() -> None:
    """Init worker."""
    cv2.setNumThreads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


def _render_one(k: int) -> np.ndarray:
    """Render one."""
    assert _frame_fn is not None
    return _frame_fn(k)


def _consume_bounded(
    pool: "multiprocessing.pool.Pool",
    indices: Sequence[int],
    consume: Callable[[int, np.ndarray], None],
    semaphore: Optional[threading.Semaphore] = None,
) -> None:
    """Consume bounded."""
    sem = semaphore if semaphore is not None else threading.Semaphore(MAX_IN_FLIGHT)
    abort = threading.Event()

    def index_gen() -> Iterator[int]:
        for k in indices:
            while not sem.acquire(timeout=SEMAPHORE_POLL_TIMEOUT_S):
                if abort.is_set():
                    return
            if abort.is_set():
                return
            yield k

    try:
        results = pool.imap(_render_one, index_gen(), chunksize=CHUNK_SIZE)
        for k, frame in zip(indices, results):
            consume(k, frame)
            sem.release()
    except BaseException:
        abort.set()
        raise


# VNAV LOCAL PATCH BEGIN: parent OpenCV threads before fork.
def render_frames(
    frame: Callable[[int], np.ndarray],
    indices: Sequence[int],
    consume: Callable[[int, np.ndarray], None],
    workers: Optional[int] = None,
) -> None:
    """Render ordered frames without inheriting active OpenCV worker threads."""
    original_threads = cv2.getNumThreads()
    try:
        # Worker initialization is too late to release locks inherited during fork.
        cv2.setNumThreads(1)
        _render_frames_fork(frame, indices, consume, workers)
    finally:
        cv2.setNumThreads(original_threads)


# VNAV LOCAL PATCH END: parent OpenCV threads before fork.


def _render_frames_fork(
    frame: Callable[[int], np.ndarray],
    indices: Sequence[int],
    consume: Callable[[int, np.ndarray], None],
    workers: Optional[int] = None,
) -> None:
    """Render frames fork."""
    global _frame_fn
    n_workers = (
        workers if workers is not None else max(1, min(MAX_WORKERS, (os.cpu_count() or 1) // 2))
    )
    _frame_fn = frame
    ctx = multiprocessing.get_context("fork")
    pool = ctx.Pool(processes=n_workers, initializer=_init_worker)
    try:
        _consume_bounded(pool, indices, consume)
    except BaseException:
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass
        raise
    else:
        pool.close()
        pool.join()
    finally:
        _frame_fn = None


PrepareFn = Callable[[Path, Path, Callable[[str], None]], VideoJob]


def make_video(
    source: Path,
    prepare: PrepareFn,
    out: Optional[Path] = None,
    stills: bool = False,
    log: Callable[[str], None] = print,
) -> VideoResult:
    """Make video."""
    t_start_all = time.monotonic()
    source = Path(source)
    if out is not None:
        out_dir, name = Path(out).parent, Path(out).stem
    else:
        run_dir = source if source.is_dir() else source.parent
        out_dir, name = run_dir / "video", run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(out_dir / "make_video.log", "a", encoding="utf-8")

    def _log(msg: str) -> None:
        log(msg)
        log_fh.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")
        log_fh.flush()

    def _cleanup(writer: Optional[cv2.VideoWriter], partial_path: Optional[Path]) -> None:
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if partial_path is not None:
            try:
                if partial_path.exists():
                    partial_path.unlink()
            except Exception:
                pass

    writer: Optional[cv2.VideoWriter] = None
    partial_path: Optional[Path] = None
    frames_dir: Optional[Path] = None
    try:
        try:
            frames_dir = Path(tempfile.mkdtemp(prefix=".frames-", dir=out_dir))
            t_prepare_start = time.monotonic()
            job = prepare(source, frames_dir, _log)
            prepare_seconds = time.monotonic() - t_prepare_start

            n_total, n_run = job.timeline.n_total, job.timeline.n_run
            contact_thumbs: Dict[int, np.ndarray] = {}
            stills_paths: List[Path] = []
            mp4_final: Optional[Path] = None
            encoder_name = ""
            verify: Optional[Dict[str, Any]] = None
            written = 0
            next_threshold = PROGRESS_LOG_STEP

            def _progress(n_written: int, total: int) -> None:
                nonlocal next_threshold
                frac = n_written / total if total > 0 else 0.0
                if total <= 0 or frac < next_threshold:
                    return
                elapsed = time.monotonic() - t_render_start
                eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0.0
                _log(
                    f"render {int(round(frac * 100))}% ({n_written}/{total}), "
                    f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
                )
                while next_threshold <= frac:
                    next_threshold += PROGRESS_LOG_STEP

            if stills:
                target_indices = _stills_indices(n_run, n_total)
                stills_dir = out_dir / "stills"
                stills_dir.mkdir(parents=True, exist_ok=True)
            else:
                target_indices = list(range(n_total))
                contact_indices = set(_pick_indices(n_total, CONTACT_SHEET_COUNT))
                cell_w, cell_h = (
                    job.size[0] // CONTACT_SHEET_COLS,
                    job.size[1] // CONTACT_SHEET_ROWS,
                )
                mp4_tmp = out_dir / f"{name}.partial.mp4"
                mp4_final = out_dir / f"{name}.mp4"
                partial_path = mp4_tmp
                writer, encoder_name = _open_encoder(mp4_tmp, job.size, job.timeline.fps, _log)

            def _consume(k: int, frame: np.ndarray) -> None:
                nonlocal written
                if stills:
                    path = stills_dir / f"{name}_{len(stills_paths):05d}.jpg"
                    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, STILLS_JPEG_QUALITY])
                    stills_paths.append(path)
                else:
                    writer.write(frame)
                    if k in contact_indices:
                        contact_thumbs[k] = cv2.resize(
                            frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA
                        )
                written += 1
                _progress(written, len(target_indices))

            n_workers = max(1, min(MAX_WORKERS, (os.cpu_count() or 1) // 2))
            _log(f"rendering {len(target_indices)} frames with {n_workers} worker process(es)")
            t_render_start = time.monotonic()
            render_frames(job.frame, target_indices, _consume, workers=n_workers)
            render_seconds = time.monotonic() - t_render_start

            if not stills:
                writer.release()
                writer = None
                if not mp4_tmp.exists() or mp4_tmp.stat().st_size == 0:
                    raise VideoError(f"编码器没有产出有效文件：{mp4_tmp}")
                mp4_tmp.rename(mp4_final)
                partial_path = None
                _log(f"encoded {written} frames via {encoder_name}; verifying output")
                verify = _verify_output(mp4_final, n_total, job.size)

            _log("running sync check")
            si = job.sync_inputs
            sync_result = sync_check(si.turn_t, si.t0, si.odom, si.ego, si.third)

            peak_rss_mb = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            ) / 1024.0

            contact_path: Optional[Path] = None
            if not stills:
                sheet = _build_contact_sheet(
                    job.timeline,
                    job.size,
                    contact_thumbs,
                    _pick_indices(n_total, CONTACT_SHEET_COUNT),
                )
                contact_path = out_dir / f"{name}_contact.jpg"
                cv2.imwrite(
                    str(contact_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, CONTACT_JPEG_QUALITY]
                )

            json_path = out_dir / f"{name}.json"
            payload = {
                "source": str(source),
                "name": name,
                "created": datetime.now().isoformat(timespec="seconds"),
                **job.info,
                "frames": n_total,
                "fps": job.timeline.fps,
                "playback_speed": job.timeline.speed,
                "end_hold_s": job.timeline.hold_s,
                "size": list(job.size),
                "encoder": encoder_name,
                "verify": verify,
                "timing": {
                    "prepare_s": round(prepare_seconds, 3),
                    "render_encode_s": round(render_seconds, 3),
                    "total_s": round(time.monotonic() - t_start_all, 3),
                    "workers": n_workers,
                },
                "peak_rss_mb": peak_rss_mb,
                "sync": sync_result,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            if verify is not None and not verify["ok"]:
                raise VideoError(f"出片校验不过：{verify}")

            _log(f"done: {mp4_final if mp4_final is not None else '(stills only)'}")
            result = VideoResult(
                mp4=mp4_final,
                json_path=json_path,
                contact=contact_path,
                stills=tuple(stills_paths),
                frames=n_total,
                encoder=encoder_name,
                seconds=time.monotonic() - t_start_all,
            )
        except KeyboardInterrupt:
            _log("interrupted by user; cleaning up")
            _cleanup(writer, partial_path)
            _log(f"resume offline: python3 {sys.argv[0]} {source}")
            raise
        except Exception as exc:
            _log(f"failed: {exc}")
            _cleanup(writer, partial_path)
            _log(f"resume offline: python3 {sys.argv[0]} {source}")
            raise VideoError(str(exc)) from exc
        else:
            return result
    finally:
        if frames_dir is not None:
            shutil.rmtree(frames_dir, ignore_errors=True)
        log_fh.close()


def mp4_to_gif(
    src: str, dst: str, start_s: float, dur_s: float, fps: float = 5.0, width: int = 640
) -> int:
    """Mp4 to gif."""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"打不开视频：{src}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start_frame = int(round(start_s * src_fps))
        end_frame = int(round((start_s + dur_s) * src_fps))
        step = max(1, int(round(src_fps / fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames: List[Image.Image] = []
        frame_idx = start_frame
        while frame_idx < end_frame:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if w != width:
                rgb = cv2.resize(
                    rgb, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA
                )
            frames.append(Image.fromarray(rgb))
            frame_idx += 1
            for _ in range(step - 1):
                if frame_idx >= end_frame or not cap.grab():
                    break
                frame_idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"[{start_s}, {start_s + dur_s}) 这段没读到帧：{src}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        dst,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=round(1000.0 / fps),
        loop=0,
        optimize=True,
    )
    return Path(dst).stat().st_size


EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2
EXIT_SIGINT, EXIT_SIGTERM, EXIT_SIGHUP = 130, 143, 129

_SIGNAL_EXIT_CODE = {signal.SIGTERM: EXIT_SIGTERM, signal.SIGHUP: EXIT_SIGHUP}


class _Terminated(Exception):
    """Terminated."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"收到信号 {signum}")


PrepareFor = Callable[[argparse.Namespace], PrepareFn]


def run_cli(
    parser: argparse.ArgumentParser, prepare_for: PrepareFor, argv: Optional[Sequence[str]] = None
) -> int:
    """Run cli."""

    def _raise_terminated(signum: int, frame: object) -> None:
        raise _Terminated(signum)

    def _exit_code_for(signum: int) -> int:
        return _SIGNAL_EXIT_CODE.get(signum, EXIT_SIGTERM)

    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _raise_terminated)
    signal.signal(signal.SIGHUP, _raise_terminated)

    try:
        result = make_video(args.source, prepare_for(args), out=args.out, stills=args.stills)
    except KeyboardInterrupt:
        print("已中断(Ctrl-C)", file=sys.stderr)
        return EXIT_SIGINT
    except _Terminated as exc:
        print(f"已终止(信号 {exc.signum})", file=sys.stderr)
        return _exit_code_for(exc.signum)
    except VideoError as exc:
        if isinstance(exc.__cause__, _Terminated):
            print(f"已终止(信号 {exc.__cause__.signum})", file=sys.stderr)
            return _exit_code_for(exc.__cause__.signum)
        print(f"出片失败：{exc}", file=sys.stderr)
        print(f"离线补做：python3 {sys.argv[0]} {args.source}", file=sys.stderr)
        return EXIT_FAILURE

    print(f"json: {result.json_path}")
    if args.stills:
        for path in result.stills:
            print(f"still: {path}")
    else:
        print(f"mp4: {result.mp4}")
        print(f"contact sheet: {result.contact}")
    return EXIT_OK
