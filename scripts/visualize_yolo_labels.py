import os
import random
import cv2

DATA_ROOT = r"E:\parking_yolov10_data"
OUT_DIR = r"E:\parking_yolov10_preview"

SAMPLES_PER_SPLIT = 10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def draw_labels_on_image(img_path, txt_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"读取图片失败: {img_path}")
        return

    h, w = img.shape[:2]

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 画中心点
        cv2.circle(img, (int(x_center), int(y_center)), 2, (0, 0, 255), -1)

        # 类别文字
        cv2.putText(img, str(cls_id), (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imwrite(save_path, img)

def process_split(split):
    img_dir = os.path.join(DATA_ROOT, "images", split)
    label_dir = os.path.join(DATA_ROOT, "labels", split)
    save_dir = os.path.join(OUT_DIR, split)

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

    sample_names = random.sample(names, min(SAMPLES_PER_SPLIT, len(names)))

    for name in sample_names:
        img_path = os.path.join(img_dir, name + ".jpg")
        txt_path = os.path.join(label_dir, name + ".txt")
        save_path = os.path.join(save_dir, name + ".jpg")
        draw_labels_on_image(img_path, txt_path, save_path)

    print(f"{split} 可视化完成，保存到: {save_dir}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        process_split(split)

    print("全部可视化完成！")