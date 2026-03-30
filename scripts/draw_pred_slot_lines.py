import os
import cv2
import math
import numpy as np

IMG_DIR = r"E:\Programs\download\ps2.0\testing\all"
TXT_DIR = r"E:\parking_yolov10_runs\test_pred_points\labels"
OUT_DIR = r"E:\parking_slot_pred_vis_v2"

os.makedirs(OUT_DIR, exist_ok=True)

# 先只跑几张图
sample_names = ["0001", "0002", "0003", "0004", "0005", "0006"]

# 参数
LINE_DIST_THRESH = 15
MIN_CONF = 0.52
MAX_LINES = 2

# ===== GT统计后得到的双长度约束 =====
SHORT_MIN = 140
SHORT_MAX = 220
LONG_MIN = 320
LONG_MAX = 390

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

for name in sample_names:
    img_path = os.path.join(IMG_DIR, name + ".jpg")
    txt_path = os.path.join(TXT_DIR, name + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"读取图片失败: {img_path}")
        continue

    h, w = img.shape[:2]
    points = load_pred_points(txt_path, w, h)

    # 画点
    for i, (x, y, conf) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.putText(
            img,
            f"P{i+1}",
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    remaining = list(range(len(points)))
    line_groups = []

    for _ in range(MAX_LINES):
        group = find_best_line_group(points, remaining, LINE_DIST_THRESH)
        if len(group) < 2:
            break
        line_groups.append(group)
        remaining = [idx for idx in remaining if idx not in group]

    colors = [(0, 0, 255), (0, 165, 255)]  # 红、橙
    slot_count = 0

    for line_idx, group in enumerate(line_groups):
        sorted_group = sort_points_along_line(points, group)

        for k in range(len(sorted_group) - 1):
            idx1 = sorted_group[k]
            idx2 = sorted_group[k + 1]

            x1, y1, _ = points[idx1]
            x2, y2, _ = points[idx2]

            dist = math.hypot(x2 - x1, y2 - y1)

            # ===== 长度过滤核心 =====
            if not is_valid_slot_length(dist):
                continue

            cv2.line(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                colors[line_idx % len(colors)],
                2,
            )

            mx = int((x1 + x2) / 2)
            my = int((y1 + y2) / 2)
            slot_count += 1
            cv2.putText(
                img,
                f"S{slot_count}",
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colors[line_idx % len(colors)],
                2,
            )

    save_path = os.path.join(OUT_DIR, name + "_pred_slots_v2.jpg")
    cv2.imwrite(save_path, img)
    print(f"已保存: {save_path}")

print("长度过滤版预测 slot line 可视化完成！")