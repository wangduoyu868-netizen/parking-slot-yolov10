import csv
import os
import random
import shutil

CSV_PATH = r"E:\parking_slot_roi_dataset\slot_roi_metadata.csv"
OUT_ROOT = r"E:\parking_slot_cls_dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

random.seed(RANDOM_SEED)

def make_dirs():
    for split in ["train", "val", "test"]:
        for cls_name in ["vacant", "occupied"]:
            os.makedirs(os.path.join(OUT_ROOT, split, cls_name), exist_ok=True)

def load_labeled_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip().lower()
            if label in ["vacant", "occupied"]:
                rows.append(row)
    return rows

def split_class_rows(rows):
    by_class = {"vacant": [], "occupied": []}
    for row in rows:
        by_class[row["label"].strip().lower()].append(row)

    for cls_name in by_class:
        random.shuffle(by_class[cls_name])

    split_data = {"train": [], "val": [], "test": []}

    for cls_name, cls_rows in by_class.items():
        n = len(cls_rows)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        train_rows = cls_rows[:n_train]
        val_rows = cls_rows[n_train:n_train + n_val]
        test_rows = cls_rows[n_train + n_val:]

        split_data["train"].extend(train_rows)
        split_data["val"].extend(val_rows)
        split_data["test"].extend(test_rows)

        print(f"{cls_name}: 总数={n}, train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    return split_data

def copy_files(split_data):
    for split, rows in split_data.items():
        for row in rows:
            label = row["label"].strip().lower()
            src_path = row["patch_path"]

            if not os.path.exists(src_path):
                print(f"缺失文件，跳过: {src_path}")
                continue

            filename = os.path.basename(src_path)
            dst_path = os.path.join(OUT_ROOT, split, label, filename)
            shutil.copy2(src_path, dst_path)

def main():
    make_dirs()
    rows = load_labeled_rows(CSV_PATH)
    print(f"有效标注样本数: {len(rows)}")

    split_data = split_class_rows(rows)
    copy_files(split_data)

    print("分类数据集构建完成！")
    print(f"输出目录: {OUT_ROOT}")

if __name__ == "__main__":
    main()