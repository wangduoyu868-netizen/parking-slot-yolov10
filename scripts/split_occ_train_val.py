"""
将仅含 train/occupied、train/vacant 的数据集划出 val（按类分层随机）。

目录结构（输入）：
  DATA_ROOT/train/occupied/*.jpg
  DATA_ROOT/train/vacant/*.jpg

输出：
  DATA_ROOT/val/occupied/
  DATA_ROOT/val/vacant/

用法:
  python scripts/split_occ_train_val.py --data_root E:\\cnr_occ_dataset --val_ratio 0.15
"""

import argparse
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    train_root = os.path.join(args.data_root, "train")
    for cls in ("occupied", "vacant"):
        src = os.path.join(train_root, cls)
        if not os.path.isdir(src):
            print("跳过（不存在）:", src)
            continue
        files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val_ratio)) if len(files) > 1 else 0
        val_files = set(files[:n_val])
        dst_val = os.path.join(args.data_root, "val", cls)
        os.makedirs(dst_val, exist_ok=True)
        for fn in val_files:
            shutil.move(os.path.join(src, fn), os.path.join(dst_val, fn))
        print(f"{cls}: 总 {len(files)}, 移到 val {len(val_files)}, 剩余 train {len(files) - len(val_files)}")


if __name__ == "__main__":
    main()
