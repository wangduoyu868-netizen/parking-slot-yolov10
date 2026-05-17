import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ===== 默认路径（与旧版脚本一致，可用命令行覆盖）=====
_DEFAULT_IMG_DIR = r"E:\Programs\download\ps2.0\testing\all"
_DEFAULT_JSON_DIR = r"E:\谷歌下载\ps_json_label\ps_json_label\testing\all"
_DEFAULT_TXT_DIR = r"E:\parking_yolov10_runs\test_pred_points\labels"
_DEFAULT_OUT_VIS_DIR = r"E:\parking_slot_eval_vis"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_METRICS_JSON = os.path.join(_REPO_ROOT, "runs", "eval", "slot_line_metrics.json")


def save_slot_line_metrics(path: str, r: Dict[str, Any], config: Dict[str, Any]) -> None:
    """将评测指标与完整参数写入 UTF-8 JSON，便于留档与复现。"""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "metrics": {
            "tp": int(r["tp"]),
            "fp": int(r["fp"]),
            "fn": int(r["fn"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1"]),
            "num_images": int(r["num_images"]),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def point_line_distance(px, py, x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    denom = math.sqrt(A * A + B * B)
    if denom < 1e-6:
        return 1e9
    return abs(A * px + B * py + C) / denom


def auto_compute_length_thresholds(distances, margin=0.15):
    """智能长短边阈值计算，返回 (short_min, short_max, long_min, long_max)。失败返回 None。"""
    distances = np.array(distances, dtype=float)
    if len(distances) < 3:
        return None

    median_d = np.median(distances)
    if median_d < 1e-6:
        return None

    # 过滤过小距离（内侧点间距等噪声），小于中位数 50% 的视为非车位线
    valid = distances[distances > median_d * 0.55]
    if len(valid) < 2:
        valid = distances

    # 判断能否分为有意义的两簇：最大距离 > 中位数 × 1.3 才认为有长短之分
    can_split = len(valid) >= 4 and (np.max(valid) / median_d) > 1.3

    if can_split:
        med_v = np.median(valid)
        short_d = valid[valid <= med_v]
        long_d = valid[valid > med_v]
        if len(short_d) >= 2 and len(long_d) >= 2:
            short_min = int(np.percentile(short_d, 5) * (1 - margin))
            short_max = int(np.percentile(short_d, 95) * (1 + margin))
            long_min = int(np.percentile(long_d, 5) * (1 - margin))
            long_max = int(np.percentile(long_d, 95) * (1 + margin))
            if short_max >= long_min:
                mid = (short_max + long_min) // 2
                short_max = mid
                long_min = mid + 1
            return (short_min, short_max, long_min, long_max)

    # 无法分簇：用全部有效距离的范围作为单一区间
    lo = int(np.percentile(valid, 5) * (1 - margin))
    hi = int(np.percentile(valid, 95) * (1 + margin))
    return (lo, hi, hi + 1, hi + 1)


def load_pred_points(txt_path, img_w, img_h, min_conf: float):
    points = []
    if not os.path.exists(txt_path):
        return points

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h

        conf = 1.0
        if len(parts) >= 6:
            conf = float(parts[5])

        if conf >= min_conf:
            points.append((xc, yc, conf))

    return points


def find_best_line_group(points, remaining_indices, dist_thresh):
    best_group = []

    if len(remaining_indices) < 2:
        return best_group

    for i in range(len(remaining_indices)):
        for j in range(i + 1, len(remaining_indices)):
            idx1 = remaining_indices[i]
            idx2 = remaining_indices[j]

            x1, y1, _ = points[idx1]
            x2, y2, _ = points[idx2]

            if math.hypot(x2 - x1, y2 - y1) < 5:
                continue

            group = []
            for idx in remaining_indices:
                px, py, _ = points[idx]
                d = point_line_distance(px, py, x1, y1, x2, y2)
                if d < dist_thresh:
                    group.append(idx)

            if len(group) > len(best_group):
                best_group = group

    return best_group


def sort_points_along_line(points, group_indices):
    coords = np.array([[points[idx][0], points[idx][1]] for idx in group_indices], dtype=np.float32)

    center = coords.mean(axis=0)
    coords_centered = coords - center

    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]

    projections = coords_centered @ main_dir
    sorted_pairs = sorted(zip(group_indices, projections), key=lambda x: x[1])

    return [idx for idx, _ in sorted_pairs]


def build_pred_slot_lines(
    img_path,
    txt_path,
    *,
    line_dist_thresh: int,
    min_conf: float,
    max_lines: int,
    short_min: int,
    short_max: int,
    long_min: int,
    long_max: int,
    auto_length: bool = False,
):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    points = load_pred_points(txt_path, w, h, min_conf)
    remaining = list(range(len(points)))
    line_groups = []

    for _ in range(max_lines):
        group = find_best_line_group(points, remaining, line_dist_thresh)
        if len(group) < 2:
            break
        line_groups.append(group)
        remaining = [idx for idx in remaining if idx not in group]

    # 收集所有候选线段
    candidates = []
    for group in line_groups:
        sorted_group = sort_points_along_line(points, group)
        for k in range(len(sorted_group) - 1):
            idx1 = sorted_group[k]
            idx2 = sorted_group[k + 1]
            x1, y1, _ = points[idx1]
            x2, y2, _ = points[idx2]
            dist = math.hypot(x2 - x1, y2 - y1)
            candidates.append(((x1, y1), (x2, y2), dist))

    # 自动计算阈值或使用手动值
    if auto_length:
        all_dists = [c[2] for c in candidates]
        auto_thresh = auto_compute_length_thresholds(all_dists)
        if auto_thresh:
            s_min, s_max, l_min, l_max = auto_thresh
        else:
            s_min, s_max = short_min, short_max
            l_min, l_max = long_min, long_max
    else:
        s_min, s_max = short_min, short_max
        l_min, l_max = long_min, long_max

    # 过滤
    pred_lines = []
    for (x1, y1), (x2, y2), dist in candidates:
        if (s_min <= dist <= s_max) or (l_min <= dist <= l_max):
            pred_lines.append(((x1, y1), (x2, y2)))

    return pred_lines, points


def build_gt_slot_lines(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    gt_lines = []
    for slot in slots:
        if not isinstance(slot, (list, tuple)) or len(slot) < 2:
            continue

        idx1 = int(slot[0]) - 1
        idx2 = int(slot[1]) - 1

        if idx1 < 0 or idx1 >= len(marks) or idx2 < 0 or idx2 >= len(marks):
            continue

        x1, y1 = float(marks[idx1][0]), float(marks[idx1][1])
        x2, y2 = float(marks[idx2][0]), float(marks[idx2][1])

        gt_lines.append(((x1, y1), (x2, y2)))

    return gt_lines, marks


def line_match_score(pred_line, gt_line):
    (px1, py1), (px2, py2) = pred_line
    (gx1, gy1), (gx2, gy2) = gt_line

    d11 = math.hypot(px1 - gx1, py1 - gy1)
    d22 = math.hypot(px2 - gx2, py2 - gy2)
    cost_a = max(d11, d22)

    d12 = math.hypot(px1 - gx2, py1 - gy2)
    d21 = math.hypot(px2 - gx1, py2 - gy1)
    cost_b = max(d12, d21)

    return min(cost_a, cost_b)


def greedy_match(pred_lines, gt_lines, thresh):
    matched_pred = set()
    matched_gt = set()

    pairs = []
    for i, p in enumerate(pred_lines):
        for j, g in enumerate(gt_lines):
            score = line_match_score(p, g)
            if score <= thresh:
                pairs.append((score, i, j))

    pairs.sort(key=lambda x: x[0])

    matches = []
    for score, i, j in pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        matches.append((i, j, score))

    tp = len(matches)
    fp = len(pred_lines) - tp
    fn = len(gt_lines) - tp

    return tp, fp, fn, matches


def draw_vis(img_path, save_path, gt_lines, pred_lines, matches):
    img = cv2.imread(img_path)

    for line in gt_lines:
        (x1, y1), (x2, y2) = line
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    for line in pred_lines:
        (x1, y1), (x2, y2) = line
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

    cv2.imwrite(save_path, img)


def run_slot_line_eval(
    img_dir: str,
    json_dir: str,
    txt_dir: str,
    *,
    out_vis_dir: Optional[str] = None,
    vis_num: int = 20,
    line_dist_thresh: int = 15,
    min_conf: float = 0.52,
    max_lines: int = 2,
    short_min: int = 140,
    short_max: int = 220,
    long_min: int = 320,
    long_max: int = 390,
    endpoint_thresh: int = 20,
    auto_length: bool = False,
) -> Dict[str, Any]:
    """
    在测试集上统计车位线 TP/FP/FN 及 P/R/F1。
    txt_dir 下为与 json 同名的 YOLO 预测 txt（含置信度列时优于 min_conf 的点会参与）。
    """
    if out_vis_dir:
        os.makedirs(out_vis_dir, exist_ok=True)

    names = []
    for file in os.listdir(json_dir):
        if file.lower().endswith(".json"):
            names.append(os.path.splitext(file)[0])
    names = sorted(names)

    all_tp = all_fp = all_fn = 0

    for idx, name in enumerate(names):
        img_path = os.path.join(img_dir, name + ".jpg")
        json_path = os.path.join(json_dir, name + ".json")
        txt_path = os.path.join(txt_dir, name + ".txt")

        if not os.path.exists(img_path):
            continue

        pred_lines, _ = build_pred_slot_lines(
            img_path,
            txt_path,
            line_dist_thresh=line_dist_thresh,
            min_conf=min_conf,
            max_lines=max_lines,
            short_min=short_min,
            short_max=short_max,
            long_min=long_min,
            long_max=long_max,
            auto_length=auto_length,
        )
        gt_lines, _ = build_gt_slot_lines(json_path)

        tp, fp, fn, matches = greedy_match(pred_lines, gt_lines, endpoint_thresh)

        all_tp += tp
        all_fp += fp
        all_fn += fn

        if out_vis_dir and idx < vis_num:
            save_path = os.path.join(out_vis_dir, name + "_eval.jpg")
            draw_vis(img_path, save_path, gt_lines, pred_lines, matches)

    precision = all_tp / (all_tp + all_fp + 1e-9)
    recall = all_tp / (all_tp + all_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_images": len(names),
    }


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="车位线几何恢复 + 与 GT 线段匹配评估")
    p.add_argument("--img_dir", type=str, default=_DEFAULT_IMG_DIR)
    p.add_argument("--json_dir", type=str, default=_DEFAULT_JSON_DIR)
    p.add_argument("--txt_dir", type=str, default=_DEFAULT_TXT_DIR)
    p.add_argument("--out_vis_dir", type=str, default=_DEFAULT_OUT_VIS_DIR)
    p.add_argument("--vis_num", type=int, default=20)
    p.add_argument("--line_dist_thresh", type=int, default=15)
    p.add_argument("--min_conf", type=float, default=0.52)
    p.add_argument("--max_lines", type=int, default=2)
    p.add_argument("--short_min", type=int, default=140)
    p.add_argument("--short_max", type=int, default=220)
    p.add_argument("--long_min", type=int, default=320)
    p.add_argument("--long_max", type=int, default=390)
    p.add_argument("--endpoint_thresh", type=int, default=20)
    p.add_argument("--auto_length", action="store_true", help="自动计算长短边阈值")
    p.add_argument("--no_vis", action="store_true", help="不写可视化图")
    p.add_argument(
        "--out_metrics",
        type=str,
        default=_DEFAULT_METRICS_JSON,
        help="将 TP/FP/FN/P/R/F1 及参数保存到此 JSON（默认 runs/eval/slot_line_metrics.json）",
    )
    p.add_argument("--no_save_metrics", action="store_true", help="不写入指标 JSON")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    out_vis = None if args.no_vis else args.out_vis_dir
    r = run_slot_line_eval(
        args.img_dir,
        args.json_dir,
        args.txt_dir,
        out_vis_dir=out_vis,
        vis_num=args.vis_num,
        line_dist_thresh=args.line_dist_thresh,
        min_conf=args.min_conf,
        max_lines=args.max_lines,
        short_min=args.short_min,
        short_max=args.short_max,
        long_min=args.long_min,
        long_max=args.long_max,
        endpoint_thresh=args.endpoint_thresh,
        auto_length=args.auto_length,
    )
    print("===== Slot Line Evaluation =====")
    print("TP =", r["tp"])
    print("FP =", r["fp"])
    print("FN =", r["fn"])
    print("Precision =", r["precision"])
    print("Recall =", r["recall"])
    print("F1 =", r["f1"])
    if out_vis:
        print(f"可视化结果已保存到: {out_vis}")

    if not args.no_save_metrics:
        cfg = {
            "img_dir": args.img_dir,
            "json_dir": args.json_dir,
            "txt_dir": args.txt_dir,
            "out_vis_dir": args.out_vis_dir,
            "vis_num": args.vis_num,
            "no_vis": args.no_vis,
            "line_dist_thresh": args.line_dist_thresh,
            "min_conf": args.min_conf,
            "max_lines": args.max_lines,
            "short_min": args.short_min,
            "short_max": args.short_max,
            "long_min": args.long_min,
            "long_max": args.long_max,
            "endpoint_thresh": args.endpoint_thresh,
        }
        save_slot_line_metrics(args.out_metrics, r, cfg)
        print(f"指标已保存到: {args.out_metrics}")


if __name__ == "__main__":
    main()
