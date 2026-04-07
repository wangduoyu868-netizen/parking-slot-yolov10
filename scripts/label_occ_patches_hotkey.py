"""
用键盘一张张把 raw_crops 里的车位图标成 occupied / vacant。

依赖: opencv-python（项目已有）

用法（在 parking-slot-yolov10 目录下）:
  python scripts/label_occ_patches_hotkey.py --src E:\\cnr_occ_dataset\\raw_crops --train_root E:\\cnr_occ_dataset\\train

默认「移动」：标完后文件从 raw_crops 挪到 train/occupied 或 train/vacant。
若需保留 raw 里原文件，加 --copy（会写入 src 下的 .labeled_done.txt 跳过已标过的）。

快捷键（窗口需处于前台）:
  O 或 1  → 已占用 → train/occupied/
  V 或 2  → 空车位 → train/vacant/
  S       → 跳过（本张不处理，下次还会出现）
  U       → 撤销上一张（仅本会话内）
  Q       → 退出

小图会自动放大到至少 --min_display 的短边（仅显示用，保存/移动仍是原文件）。
窗口为可拉伸模式（WINDOW_NORMAL），可手动拉大窗口。

说明: OpenCV 窗口标题/图上提示为英文；终端里可打印中文。
"""

import argparse
import os
import random
import shutil
import sys

import cv2

WIN = "occ_label [O]=occupied [V]=vacant [S]=skip [U]=undo [Q]=quit"


def ensure_dirs(train_root):
    for sub in ("occupied", "vacant"):
        os.makedirs(os.path.join(train_root, sub), exist_ok=True)


def list_images(folder):
    ex = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    names = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(ex) and not fn.startswith("."):
            names.append(fn)
    return names


def load_done(done_file):
    if not os.path.isfile(done_file):
        return set()
    with open(done_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_done(done_file, name):
    with open(done_file, "a", encoding="utf-8") as f:
        f.write(name + "\n")


def fit_show(bgr, max_h: int, min_short: int):
    """小图放大便于辨认；过大则缩小到 max_h。仅用于预览。"""
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return bgr
    out = bgr
    short = min(h, w)
    if min_short > 0 and short < min_short:
        s = min_short / float(short)
        out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        h, w = out.shape[:2]
    if max_h > 0 and h > max_h:
        s = max_h / float(h)
        out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="raw_crops 目录")
    ap.add_argument("--train_root", type=str, required=True, help="含 occupied/ vacant 子目录的上级，如 .../train")
    ap.add_argument("--copy", action="store_true", help="复制到 train 并记录 .labeled_done.txt，不删 raw")
    ap.add_argument("--shuffle", action="store_true", help="随机顺序")
    ap.add_argument(
        "--min_display",
        type=int,
        default=520,
        help="显示时短边至少放大到该像素（0 表示不放大）",
    )
    ap.add_argument(
        "--max_display",
        type=int,
        default=1200,
        help="显示时高度超过该值则缩小（0 不限制）",
    )
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    train_root = os.path.abspath(args.train_root)
    if not os.path.isdir(src):
        print("错误: 源目录不存在:", src)
        sys.exit(1)
    ensure_dirs(train_root)
    done_file = os.path.join(src, ".labeled_done.txt")
    done_set = load_done(done_file) if args.copy else set()

    names = list_images(src)
    if args.copy:
        names = [n for n in names if n not in done_set]
    if args.shuffle:
        random.seed(42)
        random.shuffle(names)

    occ_dir = os.path.join(train_root, "occupied")
    vac_dir = os.path.join(train_root, "vacant")
    undo_stack = []
    idx = 0
    total = len(names)

    print(f"共 {total} 张待标注（{'复制' if args.copy else '移动'} 模式）")
    print("快捷键: O/1=occupied  V/2=vacant  S=跳过  U=撤销  Q=退出")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while idx < total:
        fn = names[idx]
        path = os.path.join(src, fn)
        if not os.path.isfile(path):
            idx += 1
            continue

        img = cv2.imread(path)
        if img is None:
            print("无法读取，跳过:", path)
            idx += 1
            continue

        vis = fit_show(img.copy(), max_h=args.max_display, min_short=args.min_display)
        bar = f"{idx+1}/{total}  {fn}"
        cv2.putText(vis, bar[:80], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(
            vis,
            "O occupied  V vacant  S skip  U undo  Q quit",
            (8, vis.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.imshow(WIN, vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("已退出，进度:", idx, "/", total)
            break
        if key in (ord("s"), ord("S")):
            print("跳过:", fn)
            idx += 1
            continue
        if key in (ord("u"), ord("U")):
            if not undo_stack:
                print("没有可撤销的操作")
                continue
            op, p_from, p_to = undo_stack.pop()
            try:
                if op == "move":
                    shutil.move(p_to, p_from)
                else:
                    os.remove(p_to)
                    if os.path.isfile(done_file):
                        lines = [ln for ln in open(done_file, encoding="utf-8").read().splitlines() if ln.strip() != os.path.basename(p_from)]
                        with open(done_file, "w", encoding="utf-8") as f:
                            f.write("\n".join(lines) + ("\n" if lines else ""))
                print("已撤销:", os.path.basename(p_to))
            except OSError as e:
                print("撤销失败:", e)
            idx = max(0, idx - 1)
            continue

        dst_dir = None
        if key in (ord("o"), ord("O"), ord("1")):
            dst_dir = occ_dir
        elif key in (ord("v"), ord("V"), ord("2")):
            dst_dir = vac_dir
        else:
            continue

        dst = os.path.join(dst_dir, fn)
        if os.path.exists(dst):
            root, ext = os.path.splitext(fn)
            k = 1
            while os.path.exists(dst):
                dst = os.path.join(dst_dir, f"{root}_{k}{ext}")
                k += 1

        try:
            if args.copy:
                shutil.copy2(path, dst)
                append_done(done_file, fn)
                undo_stack.append(("copy", path, dst))
            else:
                shutil.move(path, dst)
                undo_stack.append(("move", path, dst))
            label = "occupied" if dst_dir == occ_dir else "vacant"
            print(f"[{idx+1}/{total}] -> {label}: {fn}")
        except OSError as e:
            print("保存失败:", e)
            continue

        idx += 1

    cv2.destroyAllWindows()
    if idx >= total:
        print("全部处理完。")


if __name__ == "__main__":
    main()
