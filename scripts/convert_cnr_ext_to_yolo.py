"""
将 CNR-EXT FULL_IMAGE_1000x750 全景图 + 根目录 camera*.csv 转为 YOLO 检测标签。

说明：
- 图片实际为 1000x750；CSV 中 X,Y,W,H 处于各摄像头原始标注分辨率，需按该机位在 CSV 中的
  最大外接范围缩放到 1000x750（与 convert_ps20_to_yolo 思路不同，但适配本数据集）。
- 标注语义是「车位矩形」，不是 PS2.0 的 marking point。若要与现有 slot-line 几何脚本直接
  对接，需要改推理后处理；或把 LABEL_MODE 设为 mark_center 用车位中心小框做实验性对齐。
"""

import csv
import os
import random
import re
import shutil

# =========================
# 1. 改成你的路径
# =========================
CNR_ROOT = r"E:\Programs\download\CNR-EXT_FULL_IMAGE_1000x750"
# 天气子目录名（与数据集一致）
WEATHER_DIRS = ("SUNNY", "RAINY", "OVERCAST")

OUT_ROOT = r"E:\parking_yolov10_data_cnr"

# 输出图像尺寸（与文件名 1000x750 一致；若你改了分辨率，请同步修改）
IMG_W = 1000
IMG_H = 750

VAL_RATIO = 0.1
RANDOM_SEED = 42

# slot_bbox: 车位框（推荐，语义正确）
# mark_center: 以车位中心为 YOLO 小框（BOX_SIZE），仅用于与「点检测」管线做实验对比
LABEL_MODE = "slot_bbox"
BOX_SIZE = 16  # 仅 mark_center 时使用（像素）

# 类别：与 PS2.0 单类检测可共用 id=0，但语义不同；建议单独数据集 yaml 训练 CNR 模型
CLASS_ID = 0


def make_dirs():
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)


def clamp(v, low, high):
    return max(low, min(v, high))


def load_camera_boxes(cnr_root):
    """camera_id -> list of (x, y, w, h) in annotation pixel space."""
    boxes_by_cam = {}
    ann_extent = {}
    pattern = re.compile(r"camera(\d+)\.csv$", re.I)
    for name in os.listdir(cnr_root):
        m = pattern.match(name)
        if not m:
            continue
        cam_id = int(m.group(1))
        path = os.path.join(cnr_root, name)
        rows = []
        max_x2 = 0
        max_y2 = 0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = int(row["X"])
                y = int(row["Y"])
                w = int(row["W"])
                h = int(row["H"])
                rows.append((x, y, w, h))
                max_x2 = max(max_x2, x + w)
                max_y2 = max(max_y2, y + h)
        boxes_by_cam[cam_id] = rows
        ann_extent[cam_id] = (max(max_x2, 1), max(max_y2, 1))
    return boxes_by_cam, ann_extent


def ann_to_image_xywh(x, y, w, h, ann_w, ann_h):
    """标注坐标 -> 1000x750 图像上的像素框（左上 + 宽高）。"""
    sx = IMG_W / float(ann_w)
    sy = IMG_H / float(ann_h)
    x1 = x * sx
    y1 = y * sy
    w1 = w * sx
    h1 = h * sy
    return x1, y1, w1, h1


def to_yolo_lines(boxes_ann, ann_w, ann_h):
    lines = []
    for x, y, w, h in boxes_ann:
        x1, y1, w1, h1 = ann_to_image_xywh(x, y, w, h, ann_w, ann_h)
        if LABEL_MODE == "mark_center":
            cx = x1 + 0.5 * w1
            cy = y1 + 0.5 * h1
            half = BOX_SIZE / 2.0
            x1 = cx - half
            y1 = cy - half
            w1 = float(BOX_SIZE)
            h1 = float(BOX_SIZE)
        cx = x1 + 0.5 * w1
        cy = y1 + 0.5 * h1
        cx_n = clamp(cx / IMG_W, 0.0, 1.0)
        cy_n = clamp(cy / IMG_H, 0.0, 1.0)
        w_n = clamp(w1 / IMG_W, 1e-6, 1.0)
        h_n = clamp(h1 / IMG_H, 1e-6, 1.0)
        lines.append(f"{CLASS_ID} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    return lines


def collect_samples(cnr_root, img_root_name="FULL_IMAGE_1000x750"):
    """
    返回 list of dict:
    cam_id, src_img, out_stem (唯一文件名 stem)
    """
    img_root = os.path.join(cnr_root, img_root_name)
    if not os.path.isdir(img_root):
        raise FileNotFoundError(img_root)

    samples = []
    cam_dir_pat = re.compile(r"camera(\d+)$", re.I)

    for weather in WEATHER_DIRS:
        wdir = os.path.join(img_root, weather)
        if not os.path.isdir(wdir):
            continue
        for date_name in sorted(os.listdir(wdir)):
            dpath = os.path.join(wdir, date_name)
            if not os.path.isdir(dpath):
                continue
            for entry in sorted(os.listdir(dpath)):
                m = cam_dir_pat.match(entry)
                if not m:
                    continue
                cam_id = int(m.group(1))
                cdir = os.path.join(dpath, entry)
                for fn in sorted(os.listdir(cdir)):
                    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    stem = f"{weather}__{date_name}__{entry}__{os.path.splitext(fn)[0]}"
                    samples.append(
                        {
                            "cam_id": cam_id,
                            "src_img": os.path.join(cdir, fn),
                            "out_stem": stem.replace(" ", "_"),
                        }
                    )
    return samples


def main():
    make_dirs()
    boxes_by_cam, ann_extent = load_camera_boxes(CNR_ROOT)
    if not boxes_by_cam:
        raise SystemExit(f"未在 {CNR_ROOT} 找到 camera*.csv")

    samples = collect_samples(CNR_ROOT)
    missing_cam = sorted({s["cam_id"] for s in samples if s["cam_id"] not in boxes_by_cam})
    if missing_cam:
        print("警告：以下摄像头在图片中出现但无对应 CSV，将跳过：", missing_cam)
    samples = [s for s in samples if s["cam_id"] in boxes_by_cam]

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    val_n = int(len(samples) * VAL_RATIO)
    val_set = set(range(val_n))
    print(f"总样本: {len(samples)}, 验证: {val_n}, 训练: {len(samples) - val_n}")
    print(f"LABEL_MODE={LABEL_MODE}, IMG_W={IMG_W}, IMG_H={IMG_H}")

    for i, s in enumerate(samples):
        split = "val" if i in val_set else "train"
        cam_id = s["cam_id"]
        ann_w, ann_h = ann_extent[cam_id]
        lines = to_yolo_lines(boxes_by_cam[cam_id], ann_w, ann_h)

        out_img = os.path.join(OUT_ROOT, "images", split, s["out_stem"] + ".jpg")
        out_txt = os.path.join(OUT_ROOT, "labels", split, s["out_stem"] + ".txt")
        shutil.copy2(s["src_img"], out_img)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print("转换完成:", OUT_ROOT)


if __name__ == "__main__":
    main()
