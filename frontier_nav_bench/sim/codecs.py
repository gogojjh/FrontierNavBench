"""JPEG and lossless millimetre depth encoding used by the simulator and evaluator."""

# Adapted from nav_sim/nav_sim/codecs.py.
# Source SHA256: b441691131ed6239997128bb8306bcc69a9260e384ae2c1f47937d8200b0323b.
# Copyright (c) 2026 Jianhao Jiao. MIT license; see LICENSE.
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from frontier_nav_bench.sim import protocol

JPEG_QUALITY = 85
PNG_COMPRESSION = 1
_zstd_module = None
_zstd_checked = False


class CodecError(Exception):
    pass


def _zstd():
    global _zstd_module, _zstd_checked
    if not _zstd_checked:
        _zstd_checked = True
        try:
            import zstandard

            _zstd_module = zstandard
        except ImportError:
            _zstd_module = None
    return _zstd_module


def preferred_depth_encoding() -> str:
    return protocol.DEPTH_ENC_ZSTD if _zstd() is not None else protocol.DEPTH_ENC_PNG


def encode_rgb(image_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode BGR pixels as JPEG using the frozen image quality."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise CodecError(f"彩色图要是 高×宽×3 的形状,收到 {image_bgr.shape}")
    (ok, buffer) = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise CodecError("OpenCV 压不出这张 JPEG")
    return buffer.tobytes()


def decode_rgb(data: bytes) -> np.ndarray:
    """Decode JPEG bytes into BGR pixels."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise CodecError("这段字节解不成彩色图")
    return image


def meters_to_millimeters(
    depth_m: np.ndarray, max_range_m: float = protocol.DEPTH_MAX_RANGE_M
) -> np.ndarray:
    """Quantize valid 0–8 m depths to uint16 millimetres; invalid pixels become zero."""
    if max_range_m <= 0:
        raise CodecError(f"量程要大于 0,收到 {max_range_m}")
    values = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = (values > 0.0) & (values <= max_range_m)
    millimeters = np.zeros(values.shape, dtype=np.uint16)
    millimeters[valid] = np.clip(
        np.round(values[valid] / protocol.DEPTH_SCALE_M_PER_UNIT), 1, 65535
    ).astype(np.uint16)
    return millimeters


def millimeters_to_meters(depth_mm: np.ndarray) -> np.ndarray:
    """Convert millimetre depth to float32 metres while preserving invalid zeros."""
    return np.asarray(depth_mm, dtype=np.float32) * np.float32(protocol.DEPTH_SCALE_M_PER_UNIT)


def encode_depth(
    depth_m: np.ndarray,
    encoding: Optional[str] = None,
    max_range_m: float = protocol.DEPTH_MAX_RANGE_M,
) -> tuple:
    """Encode uint16 millimetre depths losslessly as PNG or optional Zstandard."""
    chosen = encoding or preferred_depth_encoding()
    millimeters = meters_to_millimeters(depth_m, max_range_m)
    if chosen == protocol.DEPTH_ENC_ZSTD:
        module = _zstd()
        if module is None:
            raise CodecError("这台机器上没装 zstd,用 u16mm+png 或先装 zstandard")
        return (module.ZstdCompressor(level=1).compress(millimeters.tobytes()), chosen)
    if chosen == protocol.DEPTH_ENC_PNG:
        (ok, buffer) = cv2.imencode(
            ".png", millimeters, [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION]
        )
        if not ok:
            raise CodecError("OpenCV 压不出这张 16 位 PNG")
        return (buffer.tobytes(), chosen)
    raise CodecError(f"不认识的深度压法 {chosen!r}")


def decode_depth(data: bytes, width: int, height: int, encoding: str) -> np.ndarray:
    """Decode wire depth and verify its declared image dimensions."""
    if encoding == protocol.DEPTH_ENC_ZSTD:
        module = _zstd()
        if module is None:
            raise CodecError("收到 zstd 压的深度图,但这台机器上没装 zstandard")
        raw = module.ZstdDecompressor().decompress(data)
        image = np.frombuffer(raw, dtype=np.uint16)
        if image.size != width * height:
            raise CodecError(f"解出来 {image.size} 个像素,和说好的 {width}x{height} 对不上")
        return image.reshape(height, width)
    if encoding == protocol.DEPTH_ENC_PNG:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise CodecError("这段字节解不成深度图")
        if image.shape != (height, width):
            raise CodecError(f"解出来 {image.shape},和说好的 ({height}, {width}) 对不上")
        return image.astype(np.uint16)
    raise CodecError(f"不认识的深度压法 {encoding!r}")
