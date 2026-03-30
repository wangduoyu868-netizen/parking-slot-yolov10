import os
import json
import math
import numpy as np

JSON_DIR = r"E:\谷歌下载\ps_json_label\ps_json_label\training"

lengths = []

for file in os.listdir(JSON_DIR):
    if not file.lower().endswith(".json"):
        continue

    json_path = os.path.join(JSON_DIR, file)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    marks = data.get("marks", [])
    slots = data.get("slots", [])

    # 统一marks格式
    if not marks:
        marks = []
    elif isinstance(marks[0], (int, float)):
        marks = [marks]

    # 统一slots格式
    if not slots:
        slots = []
    elif isinstance(slots[0], (int, float)):
        slots = [slots]

    for slot in slots:
        if not isinstance(slot, (list, tuple)) or len(slot) < 2:
            continue

        idx1 = int(slot[0]) - 1
        idx2 = int(slot[1]) - 1

        if idx1 < 0 or idx1 >= len(marks) or idx2 < 0 or idx2 >= len(marks):
            continue

        if len(marks[idx1]) < 2 or len(marks[idx2]) < 2:
            continue

        x1, y1 = float(marks[idx1][0]), float(marks[idx1][1])
        x2, y2 = float(marks[idx2][0]), float(marks[idx2][1])

        dist = math.hypot(x2 - x1, y2 - y1)
        lengths.append(dist)

lengths = np.array(lengths)

print("GT slot line 数量:", len(lengths))
print("最小值:", np.min(lengths))
print("最大值:", np.max(lengths))
print("均值:", np.mean(lengths))
print("中位数:", np.median(lengths))
print("5%分位数:", np.percentile(lengths, 5))
print("10%分位数:", np.percentile(lengths, 10))
print("25%分位数:", np.percentile(lengths, 25))
print("75%分位数:", np.percentile(lengths, 75))
print("90%分位数:", np.percentile(lengths, 90))
print("95%分位数:", np.percentile(lengths, 95))