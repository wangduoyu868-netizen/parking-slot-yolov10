import os
import json
import cv2

IMG_DIR = r"E:\Programs\download\ps2.0\testing\all"
JSON_DIR = r"E:\谷歌下载\ps_json_label\ps_json_label\testing\all"
OUT_DIR = r"E:\parking_slot_gt_vis"

os.makedirs(OUT_DIR, exist_ok=True)

# 先只测一张，跑通后再改成多张
sample_names = ["0001"]

for name in sample_names:
    img_path = os.path.join(IMG_DIR, name + ".jpg")
    json_path = os.path.join(JSON_DIR, name + ".json")

    img = cv2.imread(img_path)
    if img is None:
        print(f"读取图片失败: {img_path}")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    marks = data.get("marks", [])
    slots = data.get("slots", [])

    # -------- 统一 marks 格式 --------
    if not marks:
        marks = []
    elif isinstance(marks[0], (int, float)):
        marks = [marks]

    # -------- 统一 slots 格式 --------
    if not slots:
        slots = []
    elif isinstance(slots[0], (int, float)):
        slots = [slots]

    # 先画所有 mark 点
    for idx, mark in enumerate(marks):
        if not isinstance(mark, (list, tuple)) or len(mark) < 2:
            continue

        x = int(mark[0])
        y = int(mark[1])

        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            img,
            str(idx + 1),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # 再画 slot 入口线
    for slot in slots:
        if not isinstance(slot, (list, tuple)) or len(slot) < 2:
            continue

        idx1 = int(slot[0]) - 1
        idx2 = int(slot[1]) - 1

        if idx1 < 0 or idx1 >= len(marks) or idx2 < 0 or idx2 >= len(marks):
            continue

        if len(marks[idx1]) < 2 or len(marks[idx2]) < 2:
            continue

        x1, y1 = int(marks[idx1][0]), int(marks[idx1][1])
        x2, y2 = int(marks[idx2][0]), int(marks[idx2][1])

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        cv2.putText(
            img,
            "S",
            (mx, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    save_path = os.path.join(OUT_DIR, name + "_gt_slots.jpg")
    cv2.imwrite(save_path, img)
    print(f"已保存: {save_path}")

print("GT slot line 可视化完成！")