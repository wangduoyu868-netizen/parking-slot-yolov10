import os
import json
import math
import csv
import cv2
import numpy as np

# =========================
# 路径
# =========================
IMG_DIR = r"E:\Programs\download\ps2.0\testing\all"
JSON_DIR = r"E:\谷歌下载\ps_json_label\ps_json_label\testing\all"

OUT_ROOT = r"E:\parking_slot_roi_dataset"
PATCH_DIR = os.path.join(OUT_ROOT, "patches")
VIS_DIR = os.path.join(OUT_ROOT, "vis")

os.makedirs(PATCH_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# =========================
# 参数
# =========================
PATCH_W = 256
PATCH_H = 256

DEPTH_RATIO = 1.35
DEPTH_MIN = 120
DEPTH_MAX = 280

VIS_NUM = 20             # 可视化前20张

# =========================
# 工具函数
# =========================
def normalize_marks_slots(data):
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

def clip_point(pt, w, h):
    x, y = pt
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return np.array([x, y], dtype=np.float32)

def build_slot_quad(p1, p2, img_w, img_h):
    """根据 slot line 两个端点生成车位 ROI 四边形，用标记点分布确定方向。"""
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    vec = p2 - p1
    line_len = np.linalg.norm(vec)
    if line_len < 1e-6:
        return None

    t = vec / line_len
    n_perp = np.array([-t[1], t[0]], dtype=np.float32)

    depth = np.clip(line_len * DEPTH_RATIO, DEPTH_MIN, DEPTH_MAX)

    mid = (p1 + p2) / 2.0
    center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)

    cand1 = mid + n_perp * depth
    cand2 = mid - n_perp * depth

    d1 = np.linalg.norm(cand1 - center)
    d2 = np.linalg.norm(cand2 - center)

    n = n_perp if d1 < d2 else -n_perp

    a = p1.copy()
    b = p2.copy()
    c = b + n * depth
    d_pt = a + n * depth

    def clip_point(pt):
        x, y = pt
        return np.array([max(0, min(img_w - 1, x)), max(0, min(img_h - 1, y))], dtype=np.float32)

    return np.array([clip_point(a), clip_point(b), clip_point(c), clip_point(d_pt)], dtype=np.float32)

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

def draw_quad(img, quad, color=(0, 0, 255), thickness=2):
    quad_int = quad.astype(int)
    for i in range(4):
        p1 = tuple(quad_int[i])
        p2 = tuple(quad_int[(i + 1) % 4])
        cv2.line(img, p1, p2, color, thickness)

# =========================
# 主程序
# =========================
meta_rows = []
all_names = sorted([os.path.splitext(f)[0] for f in os.listdir(JSON_DIR) if f.lower().endswith(".json")])

for img_idx, name in enumerate(all_names):
    img_path = os.path.join(IMG_DIR, name + ".jpg")
    json_path = os.path.join(JSON_DIR, name + ".json")

    img = cv2.imread(img_path)
    if img is None:
        print(f"读取图片失败: {img_path}")
        continue

    h, w = img.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    marks, slots = normalize_marks_slots(data)

    vis_img = img.copy()
    slot_counter = 0

    for slot in slots:
        if not isinstance(slot, (list, tuple)) or len(slot) < 2:
            continue

        idx1 = int(slot[0]) - 1
        idx2 = int(slot[1]) - 1

        if idx1 < 0 or idx1 >= len(marks) or idx2 < 0 or idx2 >= len(marks):
            continue

        p1 = (float(marks[idx1][0]), float(marks[idx1][1]))
        p2 = (float(marks[idx2][0]), float(marks[idx2][1]))

        quad = build_slot_quad(p1, p2, w, h)
        if quad is None:
            continue

        patch = warp_patch(img, quad)

        patch_name = f"{name}_slot_{slot_counter:02d}.jpg"
        patch_path = os.path.join(PATCH_DIR, patch_name)
        cv2.imwrite(patch_path, patch)

        # 可视化画框
        draw_quad(vis_img, quad, color=(0, 0, 255), thickness=2)

        # 标一下slot编号
        mid = ((quad[0] + quad[1] + quad[2] + quad[3]) / 4.0).astype(int)
        cv2.putText(
            vis_img,
            f"S{slot_counter}",
            (int(mid[0]), int(mid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        meta_rows.append([
            patch_name,
            name,
            slot_counter,
            patch_path,
            ""   # 这里先空着，后面你手工填 vacant / occupied
        ])

        slot_counter += 1

    if img_idx < VIS_NUM:
        vis_path = os.path.join(VIS_DIR, name + "_roi_vis.jpg")
        cv2.imwrite(vis_path, vis_img)

# 保存CSV清单
csv_path = os.path.join(OUT_ROOT, "slot_roi_metadata.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["patch_name", "image_name", "slot_id", "patch_path", "label"])
    writer.writerows(meta_rows)

print("ROI裁剪完成！")
print(f"patch保存目录: {PATCH_DIR}")
print(f"可视化目录: {VIS_DIR}")
print(f"元数据CSV: {csv_path}")
print(f"总patch数量: {len(meta_rows)}")