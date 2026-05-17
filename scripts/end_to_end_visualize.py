import os
import sys
import math
import cv2
import numpy as np
import torch
import torch.nn as nn

# 注册 C2fCBAM + DirectionDetect 以支持权重加载
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
import ultralytics.nn.tasks as ult_tasks
from custom_modules.c2f_cbam import C2fCBAM
from custom_modules.direction_detect import DirectionDetect
ult_tasks.__dict__["C2fCBAM"] = C2fCBAM
ult_tasks.__dict__["DirectionDetect"] = DirectionDetect

_original_parse_model = ult_tasks.parse_model
def _patched_parse_model(d, ch, verbose=True):
    all_layers = d["backbone"] + d["head"]
    cbam_positions = []
    for i, layer in enumerate(all_layers):
        if layer[2] == "C2fCBAM":
            cbam_positions.append(i)
            layer[2] = "C2f"
    dir_detect_info = None
    nc_yaml = d.get("nc", 1)
    for i, layer in enumerate(all_layers):
        if layer[2] == "DirectionDetect":
            args = layer[3]
            nc = nc_yaml if (not args or args[0] == "nc") else int(args[0])
            ne = int(args[1]) if len(args) > 1 and args[1] != "ne" else 2
            dir_detect_info = (i, nc, ne)
            layer[2] = "v10Detect"
            layer[3] = [nc]
    model, save = _original_parse_model(d, ch, verbose)
    for idx in cbam_positions:
        old_c2f = model[idx]
        hidden_c = old_c2f.cv2.conv.out_channels
        n_bn = len(old_c2f.m)
        shortcut = old_c2f.m[0].add
        in_ch = old_c2f.cv1.conv.in_channels
        new = C2fCBAM(in_ch, hidden_c, n_bn, shortcut)
        new.load_state_dict(old_c2f.state_dict(), strict=False)
        new.i, new.f, new.type, new.np = old_c2f.i, old_c2f.f, old_c2f.type, old_c2f.np
        model[idx] = new
    if dir_detect_info is not None:
        idx, nc, ne = dir_detect_info
        old_head = model[idx]
        ch_in = [m[0].conv.in_channels for m in old_head.cv2] if hasattr(old_head, "cv2") else None
        new_head = DirectionDetect(nc=nc, ne=ne, ch=tuple(ch_in) if ch_in else (256, 512, 1024))
        try:
            new_head.load_state_dict(old_head.state_dict(), strict=False)
        except Exception:
            pass
        new_head.i, new_head.f, new_head.type, new_head.np = (
            old_head.i, old_head.f, old_head.type, old_head.np
        )
        model[idx] = new_head
    return model, save
ult_tasks.parse_model = _patched_parse_model

# 自定义 Predictor 以保留方向向量
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

class DirectionDetectionPredictor(DetectionPredictor):
    def construct_result(self, pred, img, orig_img, img_path):
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
        if pred.shape[1] > 6:
            result.direction = pred[:, 6:].cpu().numpy()
        else:
            result.direction = np.zeros((0, 2), dtype=np.float32)
        return result

# Monkey-patch DetectionPredictor to use our construct_result
_orig_construct_result = DetectionPredictor.construct_result
DetectionPredictor.construct_result = DirectionDetectionPredictor.construct_result

from ultralytics import YOLO
from torchvision import transforms, models

# =========================
# 路径
# =========================
DET_MODEL_PATH = r"E:\Programs\download\ps2.0\parking-slot-yolov10\runs\detect\ps20_direction_yolov10s\weights\best.pt"
CLS_MODEL_PATH = r"E:\parking_slot_cls_runs\best_mobilenetv3_small.pth"
SOURCE_DIR = r"E:\Programs\download\ps2.0\testing\all"
OUT_DIR = r"E:\parking_end2end_vis"

os.makedirs(OUT_DIR, exist_ok=True)

# 先只可视化几张，后面再批量
sample_names = ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg", "0005.jpg", "0006.jpg", "0010.jpg", "0020.jpg", "0050.jpg"]

# =========================
# marking point -> slot line 的参数
# =========================
DET_CONF = 0.52
LINE_DIST_THRESH = 15
MAX_LINES = 10
MAX_PAIR_DIST = 250
MIN_FACING_PROJ = 0.3

SHORT_MIN = 30
SHORT_MAX = 400
LONG_MIN = 30
LONG_MAX = 400

AUTO_LENGTH = True  # 是否启用自动阈值计算

# =========================
# ROI 裁剪参数（和你现在较好的版本保持一致）
# =========================
PATCH_W = 256
PATCH_H = 256
DEPTH_RATIO = 1.35
DEPTH_MIN = 120
DEPTH_MAX = 280

# =========================
# 分类模型类别顺序
# 你之前打印的是 ['occupied', 'vacant']
# 必须保持一致
# =========================
CLASS_NAMES = ["occupied", "vacant"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 加载模型
# =========================
det_model = YOLO(DET_MODEL_PATH)

weights = models.MobileNet_V3_Small_Weights.DEFAULT
cls_model = models.mobilenet_v3_small(weights=weights)
in_features = cls_model.classifier[3].in_features
cls_model.classifier[3] = nn.Linear(in_features, 2)
cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=device))
cls_model = cls_model.to(device)
cls_model.eval()

cls_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# 工具函数
# =========================
def is_valid_slot_length(dist, s_min, s_max, l_min, l_max):
    return (s_min <= dist <= s_max) or (l_min <= dist <= l_max)


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

def point_line_distance(px, py, x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    denom = math.sqrt(A * A + B * B)
    if denom < 1e-6:
        return 1e9
    return abs(A * px + B * py + C) / denom

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

def _edge_alignment_score(quad, edges):
    """计算四边形四条边与边缘图的对齐分数。"""
    h, w = edges.shape
    total = 0.0
    for i in range(4):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % 4]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            continue
        n_samples = max(int(length / 3), 2)
        for k in range(n_samples):
            t = k / (n_samples - 1)
            sx = int(x1 + t * (x2 - x1))
            sy = int(y1 + t * (y2 - y1))
            if 0 <= sx < w and 0 <= sy < h:
                total += edges[sy, sx] / 255.0
    return total


def _find_body_depth(p1, p2, edges, entrance_len, n):
    """沿垂直方向搜索边缘峰值，确定车身墙深度。"""
    h, w = edges.shape
    search_start = entrance_len * 0.4
    search_end = min(entrance_len * 2.5, min(w, h) * 0.6)
    step = 2.0
    band = 6

    depths = []
    for ep in [p1, p2]:
        profile = []
        positions = []
        for d in np.arange(search_start, search_end, step):
            cx = int(ep[0] + n[0] * d)
            cy = int(ep[1] + n[1] * d)
            if cx < band or cx >= w - band or cy < band or cy >= h - band:
                continue
            roi = edges[max(0, cy - band):cy + band + 1,
                        max(0, cx - band):cx + band + 1]
            profile.append(float(np.sum(roi) / 255))
            positions.append(d)
        if len(profile) < 5:
            continue
        arr = np.array(profile, dtype=np.float32)
        ks = max(3, int(entrance_len * 0.05 / step))
        if ks % 2 == 0:
            ks += 1
        if ks >= len(arr):
            ks = len(arr) - (1 - len(arr) % 2)
        if ks >= 3:
            kernel = np.ones(ks, dtype=np.float32) / ks
            arr = np.convolve(arr, kernel, mode='same')
        peak_idx = int(np.argmax(arr))
        if arr[peak_idx] > 0:
            depths.append(positions[peak_idx])
    if not depths:
        return None
    return float(np.mean(depths))


def build_slot_quad(p1, p2, img_w, img_h, all_points=None, edges=None, pred_dir=None):
    """构建车位四边形：支持矩形(90°)和平行四边形。

    方向判定：优先使用预测方向向量，回退到"远离中心"启发式。
    角度搜索：VPS-Net 旋转矩阵法，用边缘对齐评分选最佳角度。
    """
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    vec = p2 - p1
    line_len = np.linalg.norm(vec)
    if line_len < 1e-6:
        return None

    t = vec / line_len
    n_perp = np.array([-t[1], t[0]], dtype=np.float32)

    # ── 1. 车身方向 ──
    body_dir_set = False
    if pred_dir is not None:
        body_dir = np.array([-pred_dir[0], -pred_dir[1]], dtype=np.float32)
        norm = np.linalg.norm(body_dir)
        if norm > 1e-6:
            body_dir = body_dir / norm
            body_dir_set = True

    if not body_dir_set:
        center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)
        mid = (p1 + p2) / 2.0
        outward = mid - center
        if outward[0] * n_perp[0] + outward[1] * n_perp[1] >= 0:
            sign = 1.0
        else:
            sign = -1.0
        body_dir = n_perp * sign

    # ── 2. 深度：边缘峰值搜索 + 固定比例兜底 ──
    depth = np.clip(line_len * DEPTH_RATIO, DEPTH_MIN, DEPTH_MAX)
    if edges is not None:
        d = _find_body_depth(p1, p2, edges, line_len, body_dir)
        if d is not None:
            depth = np.clip(d, DEPTH_MIN, DEPTH_MAX)

    # ── 3. 角度搜索：以 body_dir 为基础旋转 ──
    candidate_angles = [50, 60, 67, 75, 82, 90, 98, 105, 115, 122, 129, 135]

    if edges is not None:
        best_quad = None
        best_score = -1

        for angle_deg in candidate_angles:
            rad = math.radians(angle_deg)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            ox = (cos_a * body_dir[0] + sin_a * t[0]) * depth
            oy = (cos_a * body_dir[1] + sin_a * t[1]) * depth
            offset = np.array([ox, oy], dtype=np.float32)

            c1 = p1 + offset
            c2 = p2 + offset
            quad = np.array([p1, p2, c2, c1], dtype=np.float32)

            a1 = abs((quad[1][0] - quad[0][0]) * (quad[2][1] - quad[0][1]) -
                      (quad[2][0] - quad[0][0]) * (quad[1][1] - quad[0][1])) * 0.5
            a2 = abs((quad[2][0] - quad[0][0]) * (quad[3][1] - quad[0][1]) -
                      (quad[3][0] - quad[0][0]) * (quad[2][1] - quad[0][1])) * 0.5
            if (a1 + a2) < line_len * line_len * 0.3:
                continue

            score = _edge_alignment_score(quad, edges)
            if abs(angle_deg - 90) <= 2:
                score *= 1.05

            if score > best_score:
                best_score = score
                best_quad = quad

        if best_quad is not None:
            return best_quad

    # 回退: 纯矩形
    offset = body_dir * depth
    return np.array([p1, p2, p2 + offset, p1 + offset], dtype=np.float32)

def warp_patch(img, quad):
    dst = np.array([
        [0, 0],
        [PATCH_W - 1, 0],
        [PATCH_W - 1, PATCH_H - 1],
        [0, PATCH_H - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    patch = cv2.warpPerspective(img, M, (PATCH_W, PATCH_H))
    return patch

def draw_quad(img, quad, color, thickness=2):
    quad_int = quad.astype(int)
    for i in range(4):
        p1 = tuple(quad_int[i])
        p2 = tuple(quad_int[(i + 1) % 4])
        cv2.line(img, p1, p2, color, thickness)


def draw_text_with_bg(img, text, pos, font_scale, fg, bg=(0, 0, 0), thickness=2, pad=4):
    """绘制带背景的文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def fill_quad(img, quad, color, alpha=0.2):
    """半透明填充四边形"""
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, quad.astype(np.int32), color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def detect_marking_points(img):
    results = det_model(img, conf=DET_CONF, verbose=False)[0]

    points = []
    if results.boxes is None or len(results.boxes) == 0:
        return points

    xywh = results.boxes.xywh.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    directions = getattr(results, "direction", None)

    for i, (box, conf) in enumerate(zip(xywh, confs)):
        xc, yc, w, h = box
        if directions is not None and i < len(directions):
            dx, dy = float(directions[i][0]), float(directions[i][1])
        else:
            dx, dy = 0.0, 0.0
        points.append((float(xc), float(yc), dx, dy, float(conf)))

    return points

def build_pred_slot_lines(points, img_w, img_h):
    """统一的入口线检测（perp + facing 两种几何关系）。

    1. 垂直型：mark 方向 ⊥ 连线（cross > 0.8），同向（dot > 0.5）
    2. 朝向型：两个 mark 互相朝向对方（proj > 0.3）
    body_dir 指向画面中心（车位内部方向）。
    """
    if len(points) < 2:
        return [], (SHORT_MIN, SHORT_MAX, LONG_MIN, LONG_MAX)

    center = np.array([img_w / 2.0, img_h / 2.0])
    candidates = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            xi, yi, dxi, dyi, ci = points[i]
            xj, yj, dxj, dyj, cj = points[j]
            dist = math.hypot(xj - xi, yj - yi)
            if dist < 5 or dist > MAX_PAIR_DIST:
                continue

            vx, vy = xj - xi, yj - yi
            vn = math.hypot(vx, vy)
            if vn < 1e-6:
                continue
            vnx, vny = vx / vn, vy / vn

            cross_i = abs(dxi * vny - dyi * vnx)
            cross_j = abs(dxj * vny - dyj * vnx)
            proj_i = dxi * vnx + dyi * vny
            proj_j = dxj * (-vnx) + dyj * (-vny)
            dot_dir = dxi * dxj + dyi * dyj

            is_perp = cross_i > 0.8 and cross_j > 0.8 and dot_dir > 0.5
            is_facing = proj_i > MIN_FACING_PROJ and proj_j > MIN_FACING_PROJ

            if not is_perp and not is_facing:
                continue

            if is_perp:
                score = (cross_i + cross_j) / 2 + 1.0
            else:
                score = proj_i + proj_j

            mid = np.array([(xi + xj) / 2, (yi + yj) / 2])
            to_center = center - mid
            tc_norm = np.linalg.norm(to_center)
            if tc_norm > 1e-6:
                to_center = to_center / tc_norm
            body_dir = (float(to_center[0]), float(to_center[1]))

            candidates.append({
                'p1': (xi, yi), 'p2': (xj, yj),
                'dist': dist, 'score': score,
                'body_dir': body_dir,
            })

    if not candidates:
        return [], (SHORT_MIN, SHORT_MAX, LONG_MIN, LONG_MAX)

    candidates.sort(key=lambda c: c['score'], reverse=True)
    selected = [(c['p1'], c['p2'], c['dist'], c['body_dir'])
                for c in candidates[:MAX_LINES]]

    if AUTO_LENGTH and len(selected) >= 2:
        all_dists = [c[2] for c in selected]
        auto_thresh = auto_compute_length_thresholds(all_dists)
        if auto_thresh:
            s_min, s_max, l_min, l_max = auto_thresh
        else:
            s_min, s_max, l_min, l_max = SHORT_MIN, SHORT_MAX, LONG_MIN, LONG_MAX
    else:
        s_min, s_max, l_min, l_max = SHORT_MIN, SHORT_MAX, LONG_MIN, LONG_MAX

    pred_lines = []
    for (x1, y1), (x2, y2), dist, avg_dir in selected:
        if is_valid_slot_length(dist, s_min, s_max, l_min, l_max):
            pred_lines.append(((x1, y1), (x2, y2), avg_dir))

    return pred_lines, (s_min, s_max, l_min, l_max)

def classify_patch(patch):
    x = cls_transform(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cls_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])
    return pred_name, pred_conf

# =========================
# 主程序
# =========================
COLOR_OCCUPIED = (40, 40, 220)    # 深红
COLOR_VACANT = (40, 200, 40)      # 亮绿
COLOR_POINT = (255, 180, 0)       # 蓝色标记点
COLOR_REJECTED = (180, 180, 180)  # 灰色（被过滤的线段）
s_min, s_max, l_min, l_max = SHORT_MIN, SHORT_MAX, LONG_MIN, LONG_MAX

for name in sample_names:
    img_path = os.path.join(SOURCE_DIR, name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"读取失败: {img_path}")
        continue

    vis = img.copy()
    h, w = img.shape[:2]

    # 1. marking point 检测
    points = detect_marking_points(img)
    print(f"{name}: 检测到 {len(points)} 个标记点")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    # 2. 构建 slot lines（方向向量配对）
    pred_lines, (s_min, s_max, l_min, l_max) = build_pred_slot_lines(points, w, h)

    # 3. 对每个有效 slot line 裁剪ROI并分类
    occupied = 0
    vacant = 0
    slot_data = []

    for (x1, y1), (x2, y2), avg_dir in pred_lines:
        dist = math.hypot(x2 - x1, y2 - y1)
        quad = build_slot_quad((x1, y1), (x2, y2), w, h, all_points=points, edges=edges,
                               pred_dir=avg_dir)
        if quad is None:
            continue
        patch = warp_patch(img, quad)
        state, conf = classify_patch(patch)

        color = COLOR_OCCUPIED if state == "occupied" else COLOR_VACANT
        if state == "occupied":
            occupied += 1
        else:
            vacant += 1
        slot_data.append(((x1, y1), (x2, y2), quad, state, conf, dist, color))

    # 画车位（半透明填充 + 粗线）
    for i, (p1, p2, quad, state, conf, dist, color) in enumerate(slot_data):
        fill_quad(vis, quad, color, alpha=0.15)
        draw_quad(vis, quad, color, 2)

        # 入口线加粗
        cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 4, cv2.LINE_AA)

        # 入口线两端画圆点
        cv2.circle(vis, (int(p1[0]), int(p1[1])), 6, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(p1[0]), int(p1[1])), 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(vis, (int(p2[0]), int(p2[1])), 6, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(p2[0]), int(p2[1])), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # 文字标签：S1 / occupied 0.92
        mx = int((p1[0] + p2[0]) / 2)
        my = int((p1[1] + p2[1]) / 2)
        label_cn = "occupied" if state == "occupied" else "vacant"
        label = f"S{i+1} {label_cn} {conf:.0%}"
        draw_text_with_bg(vis, label, (mx - 35, my - 12), 0.6, color, bg=(30, 30, 30), thickness=2)

        # 距离标注在线段旁边
        offset_y = -25 if i % 2 == 0 else 20
        dist_label = f"{dist:.0f}px"
        draw_text_with_bg(vis, dist_label, (mx - 15, my + offset_y), 0.4,
                        (255, 255, 255), bg=(30, 30, 30), thickness=1)

    # 画未被 slot 使用的标记点（灰色小点）
    used_points = set()
    for (p1, p2) in [(d[0], d[1]) for d in slot_data]:
        for px, py, dx, dy, conf in points:
            if math.hypot(px - p1[0], py - p1[1]) < 3 or math.hypot(px - p2[0], py - p2[1]) < 3:
                used_points.add((px, py))

    for i, (px, py, dx, dy, conf) in enumerate(points):
        if (px, py) in used_points:
            continue
        cv2.circle(vis, (int(px), int(py)), 5, COLOR_REJECTED, -1, cv2.LINE_AA)
        cv2.putText(vis, f"P{i+1}", (int(px) + 6, int(py) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_REJECTED, 1, cv2.LINE_AA)

    # 信息面板（左上角）
    total = occupied + vacant
    panel_lines = [
        f"Points: {len(points)}  |  Slots: {total}",
        f"Vacant: {vacant}  |  Occupied: {occupied}",
        f"Auto threshold: short[{s_min},{s_max}] long[{l_min},{l_max}]",
    ]
    panel_h = 10 + len(panel_lines) * 26
    cv2.rectangle(vis, (8, 8), (420, panel_h), (30, 30, 30), -1)
    cv2.rectangle(vis, (8, 8), (420, panel_h), (80, 80, 80), 1)
    for i, line in enumerate(panel_lines):
        cv2.putText(vis, line, (16, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    save_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + "_end2end.jpg")
    cv2.imwrite(save_path, vis)
    print(f"已保存: {save_path}")

print("端到端可视化完成！")