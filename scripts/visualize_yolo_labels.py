"""
在 YOLO 格式数据上可视化标签。

- original（默认）：若提供与 split 对应的 JSON 目录，则按 PS2.0 标注绘制**车位入口线**（原几何尺度）；
  YOLO txt 仅画**中心点**（训练目标是 marking point，小框不代表车位范围）。
- patch：保留旧行为，绘制 YOLO 归一化框（16×16 一类的小框），便于核对转换是否正确。
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2

# 与 convert_ps20_to_yolo.py 默认一致，可用命令行覆盖
_DEFAULT_DATA_ROOT = r"E:\parking_yolov10_data"
_DEFAULT_OUT_DIR = r"E:\parking_yolov10_preview"
_DEFAULT_JSON_TRAIN = r"E:\谷歌下载\ps_json_label\ps_json_label\training"
_DEFAULT_JSON_TEST = r"E:\谷歌下载\ps_json_label\ps_json_label\testing\all"


def _normalize_marks_slots(data: Dict[str, Any]) -> Tuple[List, List]:
    marks = data.get("marks", [])
    slots = data.get("slots", [])
    if not marks:
        marks = []
    elif isinstance(marks[0], (int, float)):
        marks = [marks]
    if not slots:
        slots = []
    elif isinstance(slots[0], (int, float)):
        slots = [slots]
    return marks, slots


def draw_gt_slot_lines_on_image(img, data: Dict[str, Any]) -> None:
    """按 JSON 中 marks + slots 画车位入口线（原图尺度）。"""
    marks, slots = _normalize_marks_slots(data)

    for slot in slots:
        if not isinstance(slot, (list, tuple)) or len(slot) < 2:
            continue
        idx1 = int(slot[0]) - 1
        idx2 = int(slot[1]) - 1
        if idx1 < 0 or idx1 >= len(marks) or idx2 < 0 or idx2 >= len(marks):
            continue
        if len(marks[idx1]) < 2 or len(marks[idx2]) < 2:
            continue
        x1, y1 = int(marks[idx1][0]), int(marks[idx1][1])
        x2, y2 = int(marks[idx2][0]), int(marks[idx2][1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(
            img,
            "S",
            (mx, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


def draw_gt_marks_on_image(img, data: Dict[str, Any], radius: int = 4) -> None:
    """可选：画出标注中的 marking 点。"""
    marks, _ = _normalize_marks_slots(data)
    for idx, mark in enumerate(marks):
        if not isinstance(mark, (list, tuple)) or len(mark) < 2:
            continue
        x, y = int(mark[0]), int(mark[1])
        cv2.circle(img, (x, y), radius, (0, 255, 0), -1)
        cv2.putText(
            img,
            str(idx + 1),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 0),
            1,
        )


def draw_yolo_overlay(
    img,
    lines,
    *,
    mode: str,
    mark_radius: int = 5,
):
    h, w = img.shape[:2]
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h

        if mode == "patch":
            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (int(x_center), int(y_center)), 2, (0, 0, 255), -1)
            cv2.putText(
                img,
                str(cls_id),
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        else:
            # original / center：只画训练目标中心（ marking point ）
            cv2.circle(img, (int(x_center), int(y_center)), mark_radius, (255, 128, 0), -1)
            cv2.putText(
                img,
                str(cls_id),
                (int(x_center) + 6, int(y_center) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 128, 0),
                1,
            )


def draw_labels_on_image(
    img_path: str,
    txt_path: str,
    save_path: str,
    *,
    json_path: Optional[str],
    style: str,
    draw_json_marks: bool,
):
    img = cv2.imread(img_path)
    if img is None:
        print(f"读取图片失败: {img_path}")
        return

    if json_path and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        draw_gt_slot_lines_on_image(img, data)
        if draw_json_marks:
            draw_gt_marks_on_image(img, data)
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    overlay_mode = "patch" if style == "patch" else "center"
    draw_yolo_overlay(img, lines, mode=overlay_mode)

    cv2.imwrite(save_path, img)


def json_dir_for_split(split: str, json_train: str, json_test: str) -> str:
    if split in ("train", "val"):
        return json_train
    return json_test


def process_split(split: str, args) -> None:
    img_dir = os.path.join(args.data_root, "images", split)
    label_dir = os.path.join(args.data_root, "labels", split)
    save_dir = os.path.join(args.out_dir, split)
    if args.no_json:
        jdir = None
    else:
        jdir = json_dir_for_split(split, args.json_train, args.json_test)
        if not jdir or not os.path.isdir(jdir):
            jdir = None

    os.makedirs(save_dir, exist_ok=True)

    names = []
    for file in os.listdir(img_dir):
        if file.lower().endswith(".jpg"):
            name = os.path.splitext(file)[0]
            txt_path = os.path.join(label_dir, name + ".txt")
            if os.path.exists(txt_path):
                names.append(name)

    if len(names) == 0:
        print(f"{split} 没有可视化样本")
        return

    if jdir is None and not args.no_json and args.style == "original":
        print(f"{split}: 无有效 JSON 目录，仅绘制 YOLO 中心点（非车位框）")

    rng = random.Random(args.seed)
    sample_names = rng.sample(names, min(args.samples_per_split, len(names)))

    for name in sample_names:
        img_path = os.path.join(img_dir, name + ".jpg")
        txt_path = os.path.join(label_dir, name + ".txt")
        save_path = os.path.join(save_dir, name + ".jpg")
        json_path = os.path.join(jdir, name + ".json") if jdir is not None else None
        draw_labels_on_image(
            img_path,
            txt_path,
            save_path,
            json_path=json_path,
            style=args.style,
            draw_json_marks=args.draw_json_marks,
        )

    print(f"{split} 可视化完成，保存到: {save_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO 标签可视化：原车位线（JSON）+ 标记点（YOLO 中心）")
    ap.add_argument("--data_root", type=str, default=_DEFAULT_DATA_ROOT)
    ap.add_argument("--out_dir", type=str, default=_DEFAULT_OUT_DIR)
    ap.add_argument(
        "--style",
        choices=("original", "patch"),
        default="original",
        help="original=JSON 车位线+YOLO 中心点；patch=旧版小框",
    )
    ap.add_argument(
        "--json_train",
        type=str,
        default=_DEFAULT_JSON_TRAIN,
        help="train/val 的 PS2.0 json 目录",
    )
    ap.add_argument(
        "--json_test",
        type=str,
        default=_DEFAULT_JSON_TEST,
        help="test 的 PS2.0 json 目录",
    )
    ap.add_argument("--no_json", action="store_true", help="不读 JSON，只画 YOLO（中心或 patch）")
    ap.add_argument(
        "--draw_json_marks",
        action="store_true",
        help="在 original 模式下同时画出 JSON 中的 marking 点（绿）",
    )
    ap.add_argument("--samples_per_split", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        process_split(split, args)

    print("全部可视化完成！")


if __name__ == "__main__":
    main()
