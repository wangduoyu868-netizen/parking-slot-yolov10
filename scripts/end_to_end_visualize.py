import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms, models

# =========================
# 路径
# =========================
DET_MODEL_PATH = r"E:\parking_yolov10_runs\yolov10s_baseline\weights\best.pt"
CLS_MODEL_PATH = r"E:\parking_slot_cls_runs\best_mobilenetv3_small.pth"
SOURCE_DIR = r"E:\Programs\download\ps2.0\testing\all"
OUT_DIR = r"E:\parking_end2end_vis"

os.makedirs(OUT_DIR, exist_ok=True)

# 先只可视化几张，后面再批量
sample_names = ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg", "0005.jpg", "0006.jpg"]

# =========================
# marking point -> slot line 的参数
# =========================
DET_CONF = 0.52
LINE_DIST_THRESH = 15
MAX_LINES = 2

SHORT_MIN = 140
SHORT_MAX = 220
LONG_MIN = 320
LONG_MAX = 390

# =========================
# ROI 裁剪参数（和你现在较好的版本保持一致）
# =========================
PATCH_W = 256
PATCH_H = 256
MARGIN_RATIO = 0.18
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

def build_slot_quad(p1, p2, img_w, img_h):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    vec = p2 - p1
    line_len = np.linalg.norm(vec)
    if line_len < 1e-6:
        return None

    t = vec / line_len
    n1 = np.array([-t[1], t[0]], dtype=np.float32)
    n2 = -n1

    mid = (p1 + p2) / 2.0
    center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)

    depth = np.clip(line_len * DEPTH_RATIO, DEPTH_MIN, DEPTH_MAX)

    cand1 = mid + n1 * depth
    cand2 = mid + n2 * depth

    d1 = np.linalg.norm(cand1 - center)
    d2 = np.linalg.norm(cand2 - center)

    n = n1 if d1 > d2 else n2

    margin = line_len * MARGIN_RATIO

    a = p1 - t * margin
    b = p2 + t * margin
    c = b + n * depth
    d = a + n * depth

    def clip_point(pt):
        x, y = pt
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        return np.array([x, y], dtype=np.float32)

    a = clip_point(a)
    b = clip_point(b)
    c = clip_point(c)
    d = clip_point(d)

    quad = np.array([a, b, c, d], dtype=np.float32)
    return quad

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

def detect_marking_points(img):
    results = det_model(img, conf=DET_CONF, verbose=False)[0]

    points = []
    if results.boxes is None or len(results.boxes) == 0:
        return points

    xywh = results.boxes.xywh.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    for box, conf in zip(xywh, confs):
        xc, yc, w, h = box
        points.append((float(xc), float(yc), float(conf)))

    return points

def build_pred_slot_lines(points):
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

    return pred_lines

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

    # 2. 构建 slot lines
    pred_lines = build_pred_slot_lines(points)

    # 3. 对每个 slot line 裁剪ROI并分类
    for slot_id, line in enumerate(pred_lines):
        (x1, y1), (x2, y2) = line
        quad = build_slot_quad((x1, y1), (x2, y2), w, h)
        if quad is None:
            continue

        patch = warp_patch(img, quad)
        state, conf = classify_patch(patch)

        # occupied -> 红色；vacant -> 绿色
        color = (0, 0, 255) if state == "occupied" else (0, 255, 0)

        # 画slot入口线
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

        # 画ROI四边形
        draw_quad(vis, quad, color, 2)

        # 写文字
        mx = int((x1 + x2) / 2)
        my = int((y1 + y2) / 2)
        text = f"{state}:{conf:.2f}"
        cv2.putText(vis, text, (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    save_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + "_end2end.jpg")
    cv2.imwrite(save_path, vis)
    print(f"已保存: {save_path}")

print("端到端可视化完成！")