import os
import json
import math
import cv2
import numpy as np

# ===== 路径 =====
IMG_DIR = r"E:\Programs\download\ps2.0\testing\all"
JSON_DIR = r"E:\谷歌下载\ps_json_label\ps_json_label\testing\all"
TXT_DIR = r"E:\parking_yolov10_runs\test_pred_points\labels"
OUT_VIS_DIR = r"E:\parking_slot_eval_vis"

os.makedirs(OUT_VIS_DIR, exist_ok=True)

# ===== 参数 =====
LINE_DIST_THRESH = 15
MIN_CONF = 0.52
MAX_LINES = 2

SHORT_MIN = 140
SHORT_MAX = 220
LONG_MIN = 320
LONG_MAX = 390

ENDPOINT_THRESH = 20   # 端点匹配阈值（像素）

# 可视化前几张
VIS_NUM = 20


def is_valid_slot_length(dist):
    return (SHORT_MIN <= dist <= SHORT_MAX) or (LONG_MIN <= dist <= LONG_MAX)


def point_line_distance(px, py, x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    denom = math.sqrt(A * A + B * B)
    if denom < 1e-6:
        return 1e9
    return abs(A * px + B * py + C) / denom


def load_pred_points(txt_path, img_w, img_h):
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

        if conf >= MIN_CONF:
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


def build_pred_slot_lines(img_path, txt_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    points = load_pred_points(txt_path, w, h)
    remaining = list(range(len(points)))
    line_groups = []

    for _ in range(MAX_LINES):
        group = find_best_line_group(points, remaining, LINE_DIST_THRESH)
        if len(group) < 2:
            break
        line_groups.append(group)
        remaining = [idx for idx in remaining if idx not in group]

    pred_lines = []

    for group in line_groups:
        sorted_group = sort_points_along_line(points, group)
        for k in range(len(sorted_group) - 1):
            idx1 = sorted_group[k]
            idx2 = sorted_group[k + 1]

            x1, y1, _ = points[idx1]
            x2, y2, _ = points[idx2]

            dist = math.hypot(x2 - x1, y2 - y1)
            if not is_valid_slot_length(dist):
                continue

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

    # 先画GT：绿色
    for line in gt_lines:
        (x1, y1), (x2, y2) = line
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 再画Pred：红色
    for line in pred_lines:
        (x1, y1), (x2, y2) = line
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

    cv2.imwrite(save_path, img)


all_tp = 0
all_fp = 0
all_fn = 0

names = []
for file in os.listdir(JSON_DIR):
    if file.lower().endswith(".json"):
        names.append(os.path.splitext(file)[0])

names = sorted(names)

for idx, name in enumerate(names):
    img_path = os.path.join(IMG_DIR, name + ".jpg")
    json_path = os.path.join(JSON_DIR, name + ".json")
    txt_path = os.path.join(TXT_DIR, name + ".txt")

    if not os.path.exists(img_path):
        continue

    pred_lines, pred_points = build_pred_slot_lines(img_path, txt_path)
    gt_lines, marks = build_gt_slot_lines(json_path)

    tp, fp, fn, matches = greedy_match(pred_lines, gt_lines, ENDPOINT_THRESH)

    all_tp += tp
    all_fp += fp
    all_fn += fn

    if idx < VIS_NUM:
        save_path = os.path.join(OUT_VIS_DIR, name + "_eval.jpg")
        draw_vis(img_path, save_path, gt_lines, pred_lines, matches)

precision = all_tp / (all_tp + all_fp + 1e-9)
recall = all_tp / (all_tp + all_fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("===== Slot Line Evaluation =====")
print("TP =", all_tp)
print("FP =", all_fp)
print("FN =", all_fn)
print("Precision =", precision)
print("Recall =", recall)
print("F1 =", f1)
print(f"可视化结果已保存到: {OUT_VIS_DIR}")