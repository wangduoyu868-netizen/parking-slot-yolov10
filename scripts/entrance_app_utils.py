"""Small geometry helpers for the EntranceDetect Gradio demo."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


TYPE_NAMES = {
    0: "rect",
    1: "rect-long",
    2: "acute/obtuse",
}


def normalize_vec(vec: Iterable[float], fallback: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    arr = np.array(list(vec), dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        arr = np.array(fallback, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
    return arr / max(norm, 1e-8)


def build_slot_quad(
    p1: np.ndarray,
    p2: np.ndarray,
    body_dir: np.ndarray,
    *,
    depth_ratio: float,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    entrance_len = float(np.linalg.norm(p2 - p1))
    depth = float(np.clip(entrance_len * depth_ratio, depth_min, depth_max))
    direction = normalize_vec(body_dir)
    offset = direction * depth
    return np.array([p1, p2, p2 + offset, p1 + offset], dtype=np.float32)


def direction_angle_deg(body_dir: Iterable[float]) -> float:
    dx, dy = normalize_vec(body_dir)
    return math.degrees(math.atan2(float(dy), float(dx)))
