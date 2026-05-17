"""Convert PS2.0 JSON to YOLO entrance-line labels.

This avoids pseudo full-polygon labels. Each object is a true PS2.0 slot
entrance line derived directly from `slots` point pairs.

Output:
    .txt: class cx cy w h
    .eline: x1 y1 x2 y2 body_dx body_dy slot_type
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Sequence, Tuple

from scipy.io import loadmat

IMG_W = 600
IMG_H = 600
RANDOM_SEED = 42

Point = Tuple[float, float]


def normalize_items(items):
    if not items:
        return []
    if isinstance(items[0], (int, float)):
        return [items]
    return items


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def unit(vec: Point, fallback: Point = (0.0, 1.0)) -> Point:
    x, y = vec
    n = math.hypot(x, y)
    if n <= 1e-8:
        return fallback
    return x / n, y / n


def mark_direction(mark: Sequence[float]) -> Point:
    if len(mark) < 4:
        return 0.0, 1.0
    x, y = float(mark[0]), float(mark[1])
    return unit((float(mark[2]) - x, float(mark[3]) - y))


def has_mark_direction(mark: Sequence[float]) -> bool:
    return len(mark) >= 4


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross_abs(a: Point, b: Point) -> float:
    return abs(a[0] * b[1] - a[1] * b[0])


def entrance_normal_toward_slot(p1: Point, p2: Point, entrance_dir: Point) -> Point:
    """Estimate the slot-side direction when mark directions follow entrance.

    In PS2.0, some marking-point directions describe the entrance line itself
    instead of the parking-body side. In AVM images, the visible entrance is
    usually closer to the ego vehicle near image center, so the slot body tends
    to extend away from the image center.
    """
    mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    away = (mid[0] - IMG_W * 0.5, mid[1] - IMG_H * 0.5)
    n1 = unit((-entrance_dir[1], entrance_dir[0]))
    n2 = (-n1[0], -n1[1])
    return n1 if dot(n1, away) >= dot(n2, away) else n2


def choose_body_direction(mark_a, mark_b, entrance_dir: Point, p1: Point, p2: Point) -> Point:
    """Choose a stable body direction from the two PS2.0 mark directions.

    Some PS2.0 slots have one endpoint direction along the entrance line and
    the other along the parking-body side. Averaging those two creates a fake
    diagonal direction. When endpoint directions disagree, choose the one less
    parallel to the entrance line. If both endpoint directions still follow the
    entrance line, fall back to an entrance normal.
    """
    if not has_mark_direction(mark_a) or not has_mark_direction(mark_b):
        return entrance_normal_toward_slot(p1, p2, entrance_dir)
    d1 = mark_direction(mark_a)
    d2 = mark_direction(mark_b)
    if dot(d1, d2) > 0.85:
        body_dir = unit((d1[0] + d2[0], d1[1] + d2[1]))
    else:
        body_dir = d1 if cross_abs(d1, entrance_dir) >= cross_abs(d2, entrance_dir) else d2
    if cross_abs(body_dir, entrance_dir) < 0.35:
        return entrance_normal_toward_slot(p1, p2, entrance_dir)
    return body_dir


def classify_slot_type(entrance_len: float, body_dir: Point, entrance_dir: Point) -> int:
    if entrance_len >= 250:
        return 1
    if abs(dot(body_dir, entrance_dir)) > 0.35:
        return 2
    return 0


def classify_slot_type_from_slot(slot, entrance_len: float, body_dir: Point, entrance_dir: Point) -> int:
    if len(slot) >= 3 and int(slot[2]) in (2, 3):
        return 2
    if len(slot) >= 4 and abs(float(slot[3]) - 90.0) > 10.0:
        return 2
    return classify_slot_type(entrance_len, body_dir, entrance_dir)


def entrance_from_slot(marks, slot, line_pad=18.0):
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
    body_dir = choose_body_direction(ma, mb, entrance_dir, p1, p2)
    slot_type = classify_slot_type_from_slot(slot, entrance_len, body_dir, entrance_dir)

    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    x_min = clamp(min(xs) - line_pad, 0, IMG_W)
    x_max = clamp(max(xs) + line_pad, 0, IMG_W)
    y_min = clamp(min(ys) - line_pad, 0, IMG_H)
    y_max = clamp(max(ys) + line_pad, 0, IMG_H)

    box = [
        0,
        ((x_min + x_max) / 2) / IMG_W,
        ((y_min + y_max) / 2) / IMG_H,
        (x_max - x_min) / IMG_W,
        (y_max - y_min) / IMG_H,
    ]
    extra = [
        clamp(p1[0] / IMG_W, 0, 1),
        clamp(p1[1] / IMG_H, 0, 1),
        clamp(p2[0] / IMG_W, 0, 1),
        clamp(p2[1] / IMG_H, 0, 1),
        body_dir[0],
        body_dir[1],
        slot_type,
    ]
    return box, extra


def load_annotation(name, img_dir, json_dir):
    json_path = json_dir / f"{name}.json" if json_dir else None
    if json_path and json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return normalize_items(data.get("marks", [])), normalize_items(data.get("slots", []))

    mat_path = img_dir / f"{name}.mat"
    if not mat_path.exists():
        return [], []
    data = loadmat(str(mat_path))
    marks = data.get("marks", [])
    slots = data.get("slots", [])
    return normalize_items(marks.tolist()), normalize_items(slots.tolist())


def convert_sample(name, img_dir, json_dir, out_img_dir, out_label_dir):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_dir / f"{name}.jpg", out_img_dir / f"{name}.jpg")

    marks, slots = load_annotation(name, img_dir, json_dir)

    boxes = []
    extras = []
    for slot in slots:
        item = entrance_from_slot(marks, slot)
        if item is None:
            continue
        box, extra = item
        boxes.append(" ".join([str(int(box[0])), *[f"{x:.6f}" for x in box[1:]]]))
        extras.append(" ".join([*[f"{x:.6f}" for x in extra[:-1]], str(int(extra[-1]))]))

    (out_label_dir / f"{name}.txt").write_text("\n".join(boxes), encoding="utf-8")
    (out_label_dir / f"{name}.eline").write_text("\n".join(extras), encoding="utf-8")
    return len(boxes)


def collect_names(json_dir: Path | None, img_dir: Path):
    names = set()
    if json_dir and json_dir.exists():
        names.update(p.stem for p in json_dir.glob("*.json") if (img_dir / f"{p.stem}.jpg").exists())
    names.update(p.stem for p in img_dir.glob("*.mat") if (img_dir / f"{p.stem}.jpg").exists())
    return sorted(names)


def convert_split(names, split, img_dir, json_dir, out_root):
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
    ap.add_argument("--out-root", default=r"E:\parking_yolov10_entrance")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--val-from-test", action="store_true")
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
    test_names = collect_names(json_test, img_test)
    if args.val_from_test:
        val_names = test_names
        train_names = names

    train_slots = convert_split(train_names, "train", img_train, json_train, out_root)
    if args.val_from_test:
        val_slots = convert_split(val_names, "val", img_test, json_test, out_root)
    else:
        val_slots = convert_split(val_names, "val", img_train, json_train, out_root)
    test_slots = convert_split(test_names, "test", img_test, json_test, out_root)

    print(f"Train images: {len(train_names)}  entrances: {train_slots}")
    print(f"Val images:   {len(val_names)}  entrances: {val_slots}")
    print(f"Test images:  {len(test_names)}  entrances: {test_slots}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
