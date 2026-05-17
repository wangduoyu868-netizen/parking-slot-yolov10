import os
import json
import random
import shutil

# =========================
# 1. 这里改成你自己的真实路径
# =========================
IMG_TRAIN_SRC = r"E:\Programs\download\ps2.0\training"
JSON_TRAIN_SRC = r"E:\谷歌下载\ps_json_label\ps_json_label\training"

IMG_TEST_SRC = r"E:\Programs\download\ps2.0\testing\all"
JSON_TEST_SRC = r"E:\谷歌下载\ps_json_label\ps_json_label\testing\all"

OUT_ROOT = r"E:\parking_yolov10_data"

# =========================
# 2. 参数
# =========================
IMG_W = 600
IMG_H = 600
BOX_SIZE = 16          # 每个mark扩成16x16的小框，后面可改成20试试
VAL_RATIO = 0.1        # 训练集里拿10%做验证集
RANDOM_SEED = 42

# 是否用marks最后一位作为类别
# False = 全部都当成一个类别 marking_point
# True  = 用 mark[4] 当类别（比如 0 / 1）
USE_SHAPE_AS_CLASS = False

def make_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

def clamp(v, low, high):
    return max(low, min(v, high))

def mark_to_yolo_line(mark):
    # mark格式：[x, y, dir_x, dir_y, shape]
    # 输出7值格式：class_id x y w h dx dy
    x = float(mark[0])
    y = float(mark[1])

    if USE_SHAPE_AS_CLASS:
        cls_id = int(mark[4])
    else:
        cls_id = 0

    w = BOX_SIZE
    h = BOX_SIZE

    # 防止超边界
    x = clamp(x, 0, IMG_W)
    y = clamp(y, 0, IMG_H)

    x_center = x / IMG_W
    y_center = y / IMG_H
    width = w / IMG_W
    height = h / IMG_H

    # 方向向量：从mark指向dir终点，归一化到单位向量
    if len(mark) >= 4:
        dir_x = float(mark[2])
        dir_y = float(mark[3])
        dx = dir_x - x
        dy = dir_y - y
        norm = (dx * dx + dy * dy) ** 0.5
        if norm > 1e-6:
            dx /= norm
            dy /= norm
        else:
            dx, dy = 0.0, 0.0
    else:
        dx, dy = 0.0, 0.0

    return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {dx:.6f} {dy:.6f}"

def convert_one_sample(img_src_path, json_src_path, out_img_path, out_txt_path):
    # 复制图片
    shutil.copy2(img_src_path, out_img_path)

    # 读json
    with open(json_src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    marks = data.get("marks", [])

    # -------- 关键修复：统一marks格式 --------
    # 情况1：没有mark
    if not marks:
        marks = []

    # 情况2：只有一个mark，格式可能是 [x, y, dx, dy, cls]
    # 这时 marks[0] 是 float/int，不是 list
    elif isinstance(marks[0], (int, float)):
        marks = [marks]

    # 情况3：正常多个mark，格式是 [[...], [...], ...]
    # 不需要处理
    # --------------------------------------

    # 写YOLO标签
    lines = []
    for mark in marks:
        if isinstance(mark, (list, tuple)) and len(mark) >= 2:
            line = mark_to_yolo_line(mark)
            lines.append(line)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
def collect_train_names():
    names = []
    for file in os.listdir(JSON_TRAIN_SRC):
        if file.lower().endswith(".json"):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(IMG_TRAIN_SRC, name + ".jpg")
            json_path = os.path.join(JSON_TRAIN_SRC, file)
            if os.path.exists(img_path) and os.path.exists(json_path):
                names.append(name)
    return names

def convert_train_val():
    names = collect_train_names()
    random.seed(RANDOM_SEED)
    random.shuffle(names)

    val_count = int(len(names) * VAL_RATIO)
    val_names = set(names[:val_count])
    train_names = set(names[val_count:])

    print(f"训练样本数: {len(train_names)}")
    print(f"验证样本数: {len(val_names)}")

    for name in train_names:
        img_src = os.path.join(IMG_TRAIN_SRC, name + ".jpg")
        json_src = os.path.join(JSON_TRAIN_SRC, name + ".json")

        out_img = os.path.join(OUT_ROOT, "images", "train", name + ".jpg")
        out_txt = os.path.join(OUT_ROOT, "labels", "train", name + ".txt")

        convert_one_sample(img_src, json_src, out_img, out_txt)

    for name in val_names:
        img_src = os.path.join(IMG_TRAIN_SRC, name + ".jpg")
        json_src = os.path.join(JSON_TRAIN_SRC, name + ".json")

        out_img = os.path.join(OUT_ROOT, "images", "val", name + ".jpg")
        out_txt = os.path.join(OUT_ROOT, "labels", "val", name + ".txt")

        convert_one_sample(img_src, json_src, out_img, out_txt)

def convert_test():
    count = 0
    for file in os.listdir(JSON_TEST_SRC):
        if file.lower().endswith(".json"):
            name = os.path.splitext(file)[0]
            img_src = os.path.join(IMG_TEST_SRC, name + ".jpg")
            json_src = os.path.join(JSON_TEST_SRC, file)

            if os.path.exists(img_src):
                out_img = os.path.join(OUT_ROOT, "images", "test", name + ".jpg")
                out_txt = os.path.join(OUT_ROOT, "labels", "test", name + ".txt")
                convert_one_sample(img_src, json_src, out_img, out_txt)
                count += 1

    print(f"测试样本数: {count}")

if __name__ == "__main__":
    make_dirs()
    convert_train_val()
    convert_test()
    print("转换完成！")