"""Convert PS2.0 marks + slots JSON labels to PolygonPlus labels.

Output label format per slot:
    .txt:  class cx cy w h
    .poly: x1 y1 x2 y2 x3 y3 x4 y4 body_dx body_dy slot_type

Coordinates are normalized to [0, 1]. The first two polygon points are the
entrance endpoints, and the last two are the rear endpoints.

This is intentionally different from ordinary YOLO box labels: it preserves
ordered entrance semantics and body direction, so a YOLO-style polygon head can
learn long entrance lines and parallelogram slots directly.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

IMG_W = 600
IMG_H = 600
RANDOM_SEED = 42

Point = Tuple[float, float]


def normalize_items(items):
    """Normalize PS2.0 single-item and multi-item JSON fields."""
    if not items:
        return []
    if isinstance(items[0], (int, float)):
        return [items]
    return items


def unit(vec: Point, fallback: Point = (0.0, 1.0)) -> Point:
    x, y = vec
    n = math.hypot(x, y)
    if n <= 1e-8:
        return fallback
    return x / n, y / n


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def mark_direction(mark: Sequence[float]) -> Point:
    """Return unit mark direction from mark point to direction endpoint."""
    if len(mark) < 4:
        return 0.0, 1.0
    x, y = float(mark[0]), float(mark[1])
    return unit((float(mark[2]) - x, float(mark[3]) - y))


def average_body_direction(mark_a: Sequence[float], mark_b: Sequence[float]) -> Point:
    """Convert PS2.0 mark direction to body direction.

    PS2.0 mark direction points toward the vehicle-body direction in this
    training setup. If the two endpoints disagree, use their normalized average
    and fall back to the entrance normal.
    """
    da = mark_direction(mark_a)
    db = mark_direction(mark_b)
    body = unit((da[0] + db[0], da[1] + db[1]), fallback=(0.0, 0.0))
    return body


def estimate_depth(entrance_len: float, body_dir: Point, entrance_dir: Point) -> float:
    """Estimate rear depth from entrance length and slot geometry.

    PS2.0 JSON gives entrance point pairs, not complete rear corners. This
    pseudo-polygon depth is a training target for full-edge drawing. It is
    deliberately type-aware and conservative, so the model learns a stable
    geometric prior before optional edge refinement.
    """
    # Parallel slots have long entrance lines and shorter depth. Perpendicular
    # slots have shorter entrance lines and deeper bodies.
    if entrance_len >= 250:
        return clamp(entrance_len * 0.48, 120, 190)

    # If the body direction is nearly parallel to the entrance, this is an
    # angled/parallelogram case, and a little extra depth helps cover the slot.
    parallel = abs(dot(body_dir, entrance_dir))
    if parallel > 0.35:
        return clamp(entrance_len * 1.15, 115, 230)

    return clamp(entrance_len * 1.35, 100, 245)


def classify_slot_type(entrance_len: float, body_dir: Point, entrance_dir: Point) -> int:
    if entrance_len >= 250:
        return 1
    if abs(dot(body_dir, entrance_dir)) > 0.35:
        return 2
    return 0


def order_entrance(p1: Point, p2: Point, body_dir: Point) -> Tuple[Point, Point]:
    """Return entrance_left, entrance_right for a viewer looking into the slot."""
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    cross = vx * body_dir[1] - vy * body_dir[0]
    # If p1->p2 is left-to-right with respect to body direction, keep it.
    if cross > 0:
        return p1, p2
    return p2, p1


def polygon_from_slot(marks: Sequence[Sequence[float]], slot: Sequence[int]):
    if len(slot) < 2:
        return None
    i, j = int(slot[0]) - 1, int(slot[1]) - 1
    if i < 0 or j < 0 or i >= len(marks) or j >= len(marks):
        return None

    ma, mb = marks[i], marks[j]
    if len(ma) < 2 or len(mb) < 2:
        return None

    p1 = (float(ma[0]), float(ma[1]))
    p2 = (float(mb[0]), float(mb[1]))
    entrance_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if entrance_len < 5:
        return None

    entrance_dir = unit((p2[0] - p1[0], p2[1] - p1[1]))
    body_dir = average_body_direction(ma, mb)
    if math.hypot(body_dir[0], body_dir[1]) <= 1e-8:
        normal = (-entrance_dir[1], entrance_dir[0])
        center = (IMG_W / 2.0, IMG_H / 2.0)
        mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        outward = (mid[0] - center[0], mid[1] - center[1])
        body_dir = normal if dot(normal, outward) >= 0 else (-normal[0], -normal[1])

    left, right = order_entrance(p1, p2, body_dir)
    entrance_dir = unit((right[0] - left[0], right[1] - left[1]))
    slot_type = classify_slot_type(entrance_len, body_dir, entrance_dir)
    depth = estimate_depth(entrance_len, body_dir, entrance_dir)

    rear_right_raw = (right[0] + body_dir[0] * depth, right[1] + body_dir[1] * depth)
    rear_left_raw = (left[0] + body_dir[0] * depth, left[1] + body_dir[1] * depth)
    raw_quad = [left, right, rear_right_raw, rear_left_raw]

    quad = [
        (clamp(x, 0, IMG_W), clamp(y, 0, IMG_H))
        for x, y in raw_quad
    ]

    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x_min, x_max = clamp(min(xs), 0, IMG_W), clamp(max(xs), 0, IMG_W)
    y_min, y_max = clamp(min(ys), 0, IMG_H), clamp(max(ys), 0, IMG_H)
    if x_max - x_min < 2 or y_max - y_min < 2:
        return None

    cx = ((x_min + x_max) / 2.0) / IMG_W
    cy = ((y_min + y_max) / 2.0) / IMG_H
    bw = (x_max - x_min) / IMG_W
    bh = (y_max - y_min) / IMG_H

    norm_quad = []
    for x, y in quad:
        norm_quad.extend([clamp(x / IMG_W, 0.0, 1.0), clamp(y / IMG_H, 0.0, 1.0)])

    return [0, cx, cy, bw, bh, *norm_quad, body_dir[0], body_dir[1], slot_type]


def format_box_label(values: Sequence[float]) -> str:
    cls = int(values[0])
    floats = [f"{float(v):.6f}" for v in values[1:5]]
    return " ".join([str(cls), *floats])


def format_poly_label(values: Sequence[float]) -> str:
    floats = [f"{float(v):.6f}" for v in values[5:-1]]
    return " ".join([*floats, str(int(values[-1]))])


def collect_names(json_dir: Path, img_dir: Path) -> List[str]:
    names = []
    for p in json_dir.glob("*.json"):
        if (img_dir / f"{p.stem}.jpg").exists():
            names.append(p.stem)
    return sorted(names)


def convert_sample(
    name: str,
    img_dir: Path,
    json_dir: Path,
    out_img_dir: Path,
    out_label_dir: Path,
) -> int:
    img_src = img_dir / f"{name}.jpg"
    json_src = json_dir / f"{name}.json"
    out_img = out_img_dir / f"{name}.jpg"
    out_txt = out_label_dir / f"{name}.txt"
    out_poly = out_label_dir / f"{name}.poly"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, out_img)

    with json_src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    marks = normalize_items(data.get("marks", []))
    slots = normalize_items(data.get("slots", []))
    box_labels = []
    poly_labels = []
    for slot in slots:
        label = polygon_from_slot(marks, slot)
        if label is not None:
            box_labels.append(format_box_label(label))
            poly_labels.append(format_poly_label(label))

    out_txt.write_text("\n".join(box_labels), encoding="utf-8")
    out_poly.write_text("\n".join(poly_labels), encoding="utf-8")
    return len(box_labels)


def convert_split(
    names: Iterable[str],
    split: str,
    img_dir: Path,
    json_dir: Path,
    out_root: Path,
) -> int:
    total = 0
    for name in names:
        total += convert_sample(
            name,
            img_dir,
            json_dir,
            out_root / "images" / split,
            out_root / "labels" / split,
        )
    return total


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-train", default=r"E:\Programs\download\ps2.0\training")
    ap.add_argument("--json-train", default=str(Path("E:/") / "谷歌下载" / "ps_json_label" / "ps_json_label" / "training"))
    ap.add_argument("--img-test", default=r"E:\Programs\download\ps2.0\testing\all")
    ap.add_argument("--json-test", default=str(Path("E:/") / "谷歌下载" / "ps_json_label" / "ps_json_label" / "testing" / "all"))
    ap.add_argument("--out-root", default=r"E:\parking_yolov10_polygon")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    return ap.parse_args()


def main():
    args = parse_args()
    img_train = Path(args.img_train)
    json_train = Path(args.json_train)
    img_test = Path(args.img_test)
    json_test = Path(args.json_test)
    out_root = Path(args.out_root)

    names = collect_names(json_train, img_train)
    random.seed(args.seed)
    random.shuffle(names)
    val_count = int(len(names) * args.val_ratio)
    val_names = names[:val_count]
    train_names = names[val_count:]

    train_slots = convert_split(train_names, "train", img_train, json_train, out_root)
    val_slots = convert_split(val_names, "val", img_train, json_train, out_root)

    test_names = collect_names(json_test, img_test)
    test_slots = convert_split(test_names, "test", img_test, json_test, out_root)

    print(f"Train images: {len(train_names)}  slots: {train_slots}")
    print(f"Val images:   {len(val_names)}  slots: {val_slots}")
    print(f"Test images:  {len(test_names)}  slots: {test_slots}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
