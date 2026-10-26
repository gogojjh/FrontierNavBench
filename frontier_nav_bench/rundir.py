"""Episode artifact directories (evaluation layer, frozen after T-I0; plan section 6.7).

``<campaign>/runs/<method_id>/<episode_token>/r<k>/`` contains:
- ``run.json`` (``vnav.run/v1``), atomically renamed as the final step; its
  presence marks completion and causes resume to skip that run.
- ``trace.csv`` (world truth at 10 Hz) and ``steps.jsonl`` (one row per decision).
- ``frames/ego/<t_ms>.jpg`` (10 Hz first-person images), ``frames/third/`` for
  selected videos, and decision-time ``depth/<t_ms>.png``.
- ``goal/goal.jpg|.depth.png`` copies of frozen goal images, method-owned
  ``method/`` output and ``logs/``.
- ``attempts/<n>/`` archives infrastructure failures separately from final results.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import jsonschema

RUN_SCHEMA_ID = "vnav.run/v1"
STATUSES = ("scored", "unscored_infra", "unscored_guard", "excluded_debug")
"""unscored_guard records an act call over 300 wall seconds or an episode cost above $5."""
TRACE_COLUMNS = ("t_sim", "x", "y", "yaw", "up_z", "vx_body", "wz_body", "z")
RUN_JSON = "run.json"
_ATTEMPT_ITEMS = ("trace.csv", "steps.jsonl", "frames", "depth", "goal", "method", "logs")

RUN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "schema", "method_id", "episode_set", "episode_token", "repeat_index", "status",
        "attempt", "git", "server", "settings", "episode", "timing", "outcome", "metrics",
        "method_stats", "usage", "infra",
    ],
    "properties": {
        "schema": {"const": RUN_SCHEMA_ID},
        "status": {"enum": list(STATUSES)},
        "repeat_index": {"type": "integer", "minimum": 0},
        "attempt": {"type": "integer", "minimum": 1},
        "outcome": {"type": ["object", "null"]},
        "metrics": {"type": ["object", "null"]},
        "settings": {"type": "object", "required": ["budget_s", "limits", "hold", "success"]},
        "episode": {"type": "object", "required": ["scene", "start", "goal", "shortest_m", "tier"]},
        "usage": {"type": "object", "required": ["billed_usd", "list_usd", "calls"]},
    },
}  # fmt: skip


def json_safe(value: Any) -> Any:
    """Convert numpy scalars to Python values for JSON; stringify other unsupported objects."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def validate_run(doc: Dict[str, Any]) -> None:
    jsonschema.validate(doc, RUN_SCHEMA)


class RunDir:
    def __init__(self, campaign_root: Path, method_id: str, token: str, repeat: int) -> None:
        self.root = Path(campaign_root) / "runs" / method_id / token / f"r{repeat}"
        self._steps_handle: Optional[Any] = None
        self._trace_handle: Optional[Any] = None
        self._trace_writer: Optional[Any] = None

    # -- State --
    @property
    def run_json(self) -> Path:
        return self.root / RUN_JSON

    def is_complete(self) -> bool:
        return self.run_json.is_file()

    def load(self) -> Dict[str, Any]:
        return json.loads(self.run_json.read_text(encoding="utf-8"))

    def attempts_done(self) -> int:
        adir = self.root / "attempts"
        return len([p for p in adir.iterdir() if p.is_dir()]) if adir.is_dir() else 0

    def begin_attempt(self) -> int:
        """Begin an attempt, archiving interrupted or failed partial output under attempts/<n>/."""
        self.close_streams()
        self.root.mkdir(parents=True, exist_ok=True)
        leftovers = [self.root / n for n in _ATTEMPT_ITEMS if (self.root / n).exists()]
        number = self.attempts_done() + 1
        if leftovers:
            dest = self.root / "attempts" / str(number)
            dest.mkdir(parents=True, exist_ok=True)
            for item in leftovers:
                shutil.move(str(item), str(dest / item.name))
            number += 1
        for sub in ("frames/ego", "depth", "goal", "method", "logs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return number

    def park_attempt(self, error: str) -> None:
        """Archive an infrastructure-failed attempt and record its reason."""
        self.close_streams()
        number = self.attempts_done() + 1
        dest = self.root / "attempts" / str(number)
        dest.mkdir(parents=True, exist_ok=True)
        for name in _ATTEMPT_ITEMS:
            if (self.root / name).exists():
                shutil.move(str(self.root / name), str(dest / name))
        (dest / "error.txt").write_text(error + "\n", encoding="utf-8")

    # -- Writing --
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def write_bytes(self, rel: str, data: bytes) -> None:
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def write_frames(self, sub: str, frames: Iterable[Any]) -> int:
        n = 0
        for t_sim, jpeg in frames:
            self.write_bytes(f"frames/{sub}/{int(round(float(t_sim) * 1000)):08d}.jpg", jpeg)
            n += 1
        return n

    def write_depth(self, t_sim: float, png: bytes) -> None:
        self.write_bytes(f"depth/{int(round(t_sim * 1000)):08d}.png", png)

    def append_step(self, row: Dict[str, Any]) -> None:
        if self._steps_handle is None:
            self._steps_handle = open(self.root / "steps.jsonl", "a", encoding="utf-8")
        self._steps_handle.write(
            json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._steps_handle.flush()

    def append_trace(self, rows: Iterable[Dict[str, float]]) -> None:
        if self._trace_handle is None:
            path = self.root / "trace.csv"
            fresh = not path.exists()
            self._trace_handle = open(path, "a", encoding="utf-8", newline="")
            self._trace_writer = csv.DictWriter(self._trace_handle, fieldnames=TRACE_COLUMNS)
            if fresh:
                self._trace_writer.writeheader()
        for row in rows:
            self._trace_writer.writerow({k: f"{float(row[k]):.6f}" for k in TRACE_COLUMNS})  # type:
            # ignore[union-attr]
        self._trace_handle.flush()

    def close_streams(self) -> None:
        for handle in (self._steps_handle, self._trace_handle):
            if handle is not None:
                handle.close()
        self._steps_handle = self._trace_handle = self._trace_writer = None

    def finish(self, doc: Dict[str, Any]) -> None:
        """Validate and atomically write run.json as the final episode-completion step."""
        self.close_streams()
        validate_run(json_safe(doc))
        write_json_atomic(self.run_json, doc)


def read_trace(path: Path) -> List[Dict[str, float]]:
    with open(path, encoding="utf-8") as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def read_steps(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
