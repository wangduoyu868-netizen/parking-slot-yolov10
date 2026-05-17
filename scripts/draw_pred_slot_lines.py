import os
import cv2
import math
import numpy as np

IMG_DIR = r"E:\Programs\download\ps2.0\testing\all"
TXT_DIR = r"E:\parking_yolov10_runs\test_pred_points\labels"
OUT_DIR = r"E:\parking_slot_pred_vis_v2"

os.makedirs(OUT_DIR, exist_ok=True)

sample_names = ["0001", "0002", "0003", "0004", "0005", "0006"]

# 参数
LINE_DIST_THRESH = 15
MIN_CONF = 0.52
MAX_LINES = 2

# 默认手动阈值（回退用）
SHORT_MIN_DEFAULT = 140
SHORT_MAX_DEFAULT = 220
LONG_MIN_DEFAULT = 320
LONG_MAX_DEFAULT = 390

AUTO_LENGTH = True  # 是否启用自动阈值计算


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


def is_valid_slot_length(dist, s_min, s_max, l_min, l_max):
    return (s_min <= dist <= s_max) or (l_min <= dist <= l_max)


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

    # 收集所有候选线段距离
    all_candidates = []
    for group in line_groups:
        sorted_group = sort_points_along_line(points, group)
        for k in range(len(sorted_group) - 1):
            idx1 = sorted_group[k]
            idx2 = sorted_group[k + 1]
            x1, y1, _ = points[idx1]
            x2, y2, _ = points[idx2]
            dist = math.hypot(x2 - x1, y2 - y1)
            all_candidates.append((x1, y1, x2, y2, dist))

    # 自动计算阈值
    if AUTO_LENGTH and len(all_candidates) >= 4:
        all_dists = [c[4] for c in all_candidates]
        auto_thresh = auto_compute_length_thresholds(all_dists)
        if auto_thresh:
            s_min, s_max, l_min, l_max = auto_thresh
            print(f"  {name}: 自动阈值 short=[{s_min},{s_max}] long=[{l_min},{l_max}]")
        else:
            s_min, s_max = SHORT_MIN_DEFAULT, SHORT_MAX_DEFAULT
            l_min, l_max = LONG_MIN_DEFAULT, LONG_MAX_DEFAULT
    else:
        s_min, s_max = SHORT_MIN_DEFAULT, SHORT_MAX_DEFAULT
        l_min, l_max = LONG_MIN_DEFAULT, LONG_MAX_DEFAULT

    colors = [(0, 0, 255), (0, 165, 255)]  # 红、橙
    slot_count = 0

    for x1, y1, x2, y2, dist in all_candidates:
        if not is_valid_slot_length(dist, s_min, s_max, l_min, l_max):
            continue

        # 找到所属 line_group 的颜色
        line_idx = 0  # 简化：用红色
        for li, group in enumerate(line_groups):
            sorted_group = sort_points_along_line(points, group)
            for k in range(len(sorted_group) - 1):
                gi1 = sorted_group[k]
                gi2 = sorted_group[k + 1]
                gx1, gy1, _ = points[gi1]
                gx2, gy2, _ = points[gi2]
                if abs(gx1 - x1) < 1 and abs(gy1 - y1) < 1 and abs(gx2 - x2) < 1 and abs(gy2 - y2) < 1:
                    line_idx = li
                    break

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
