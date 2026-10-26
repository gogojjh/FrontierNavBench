"""MessagePack control envelopes and the benchmark RGB-D wire constants."""

# Adapted from nav_sim/nav_sim/protocol.py.
# Source SHA256: fdb767f71284f28cc54a13857582f9d778f2123743810d2ef6d4c86292fbc877.
# Copyright (c) 2026 Jianhao Jiao. MIT license; see LICENSE.
from __future__ import annotations

from typing import Any, Dict

import msgpack

SCHEMA_VERSION = 1
TOPIC_CTRL = "ctrl"
ALL_TOPICS = (TOPIC_CTRL,)
DEPTH_ENC_ZSTD = "u16mm+zstd"
DEPTH_ENC_PNG = "u16mm+png"
DEPTH_ENCODINGS = (DEPTH_ENC_ZSTD, DEPTH_ENC_PNG)
DEPTH_SCALE_M_PER_UNIT = 0.001
DEPTH_MAX_RANGE_M = 8.0
DEPTH_INVALID = 0
_REQUIRED_ENVELOPE_KEYS = ("v", "topic", "seq", "t_sim", "t_send_wall", "frame_id", "payload")


class ProtocolError(Exception):
    pass


class ProtocolVersionError(ProtocolError):
    pass


class ProtocolFieldError(ProtocolError):
    pass


def encode(
    topic: str, seq: int, t_sim: float, t_send_wall: float, frame_id: str, payload: Dict[str, Any]
) -> bytes:
    """Encode the versioned control envelope using MessagePack binary values."""
    if topic not in ALL_TOPICS:
        raise ProtocolFieldError(f"不认识的消息种类 {topic!r};只能是:{', '.join(ALL_TOPICS)}")
    if seq < 0:
        raise ProtocolFieldError(f"序号不能是负数,收到 {seq}")
    envelope = {
        "v": SCHEMA_VERSION,
        "topic": topic,
        "seq": int(seq),
        "t_sim": float(t_sim),
        "t_send_wall": float(t_send_wall),
        "frame_id": str(frame_id),
        "payload": payload,
    }
    return msgpack.packb(envelope, use_bin_type=True)


def decode(raw: bytes) -> Dict[str, Any]:
    """Decode an envelope and reject missing fields or incompatible versions."""
    try:
        message = msgpack.unpackb(raw, raw=False)
    except Exception as exc:
        raise ProtocolFieldError(f"这段字节解不成消息:{exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolFieldError("消息的最外层必须是字典")
    version = message.get("v")
    if version != SCHEMA_VERSION:
        raise ProtocolVersionError(
            f"对面用的消息格式是 v{version},我们只认 v{SCHEMA_VERSION};两边要一起升级"
        )
    missing = [key for key in _REQUIRED_ENVELOPE_KEYS if key not in message]
    if missing:
        raise ProtocolFieldError(f"消息缺字段:{', '.join(missing)}")
    return message
