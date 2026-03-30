import csv
import os
import cv2

CSV_PATH = r"E:\parking_slot_roi_dataset\slot_roi_metadata.csv"
WINDOW_NAME = "Slot Patch Labeling"

# 标签定义
# 0 -> vacant
# 1 -> occupied
# s -> skip
# q -> quit and save

def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def save_rows(csv_path, rows):
    if len(rows) == 0:
        return

    fieldnames = rows[0].keys()
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

rows = load_rows(CSV_PATH)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 800)

total = len(rows)
idx = 0

while idx < total:
    row = rows[idx]
    patch_path = row["patch_path"]
    current_label = row.get("label", "").strip()

    # 已经标过的跳过
    if current_label in ["vacant", "occupied", "skip"]:
        idx += 1
        continue

    if not os.path.exists(patch_path):
        print(f"文件不存在，自动跳过: {patch_path}")
        row["label"] = "skip"
        idx += 1
        continue

    img = cv2.imread(patch_path)
    if img is None:
        print(f"读取失败，自动跳过: {patch_path}")
        row["label"] = "skip"
        idx += 1
        continue

    show = img.copy()

    # 放大显示
    show = cv2.resize(show, (512, 512), interpolation=cv2.INTER_CUBIC)

    info1 = f"{idx+1}/{total}"
    info2 = os.path.basename(patch_path)
    info3 = "0=vacant  1=occupied  s=skip  q=quit"

    cv2.putText(show, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(show, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
    cv2.putText(show, info3, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow(WINDOW_NAME, show)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('0'):
        row["label"] = "vacant"
        idx += 1
    elif key == ord('1'):
        row["label"] = "occupied"
        idx += 1
    elif key == ord('s'):
        row["label"] = "skip"
        idx += 1
    elif key == ord('q'):
        print("退出并保存进度...")
        save_rows(CSV_PATH, rows)
        break
    else:
        # 其他键不处理，继续等
        continue

    # 每标一张就保存一次，防止中途退出丢失
    save_rows(CSV_PATH, rows)

cv2.destroyAllWindows()
print("标注结束，结果已保存到 CSV。")