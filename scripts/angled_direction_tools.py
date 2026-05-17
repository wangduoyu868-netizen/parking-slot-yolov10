"""Shared helpers for acute/obtuse PS2.0 direction annotation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

Point = Tuple[float, float]
IMG_W = 600
IMG_H = 600


def unit(vec: Point, fallback: Point = (1.0, 0.0)) -> Point:
    x, y = vec
    n = math.hypot(x, y)
    if n <= 1e-8:
        return fallback
    return x / n, y / n


def rotate(vec: Point, degrees: float) -> Point:
    rad = math.radians(degrees)
    c = math.cos(rad)
    s = math.sin(rad)
    x, y = vec
    return unit((x * c - y * s, x * s + y * c))


def angle_candidates(entrance_dir: Point, angle_degrees: float) -> tuple[Point, Point]:
    e = unit(entrance_dir)
    return rotate(e, angle_degrees), rotate(e, -angle_degrees)


def format_eline(values: Iterable[float]) -> str:
    vals = list(values)
    return " ".join([*[f"{float(v):.6f}" for v in vals[:6]], str(int(float(vals[6])))]).rstrip()


def image_path_for(data_root: Path, split: str, stem: str) -> Path:
    return data_root / "images" / split / f"{stem}.jpg"


def label_path_for(data_root: Path, split: str, stem: str, suffix: str) -> Path:
    return data_root / "labels" / split / f"{stem}{suffix}"
