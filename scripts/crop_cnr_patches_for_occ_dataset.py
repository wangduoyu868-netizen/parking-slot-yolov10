"""
从 CNR-EXT 全景图裁剪车位小图，供占用分类（occupied / vacant）域微调标注。

两种方式（--box_source）：
  gt_csv  : 用根目录 camera*.csv 的车位框（与 convert_cnr_ext_to_yolo 同一套缩放），车位对齐稳定，推荐。
  detector: 用你训练好的 YOLO 检测框裁剪。

输出：
  OUT_DIR/raw_crops/*.jpg  以及 manifest.csv（源图、框、文件名）。

裁剪以「单个车位」为主：默认只做很小外扩，并限制裁块最长边（避免把相邻多辆车圈进同一张 patch）。
看不清时在标注脚本里用 --min_display 放大预览，或略增 --pad_ratio（勿过大）。

下一步（人工）：
  将 raw_crops 中图片按真实空/占分类，复制到例如：
    DATASET/train/occupied/  DATASET/train/vacant/
  再运行 scripts/split_occ_train_val.py 划出 val，最后：
    python scripts/finetune_occ_classifier_cnr.py --data_root DATASET ...

可选 --pseudo_cls_pt：用旧分类器自动分到 train/occupied|vacant（噪声大，仅作冷启动，务必人工抽查纠错）。

数量：默认只导出约 --max_crops 块（默认 1000），源图顺序会先随机打乱再裁，减轻标注量。
全量请设 --max_crops 0。
"""

import argparse
import csv
import math
import os
import random
import sys

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from convert_cnr_ext_to_yolo import (  # noqa: E402
    IMG_H,
    IMG_W,
    collect_samples,
    load_camera_boxes,
)


def ann_to_xyxy(x, y, w, h, ann_w, ann_h):
    sx = IMG_W / float(ann_w)
    sy = IMG_H / float(ann_h)
    x1 = x * sx
    y1 = y * sy
    x2 = x1 + w * sx
    y2 = y1 + h * sy
    return x1, y1, x2, y2


def _clamp_roi_max_side(xi1, yi1, xi2, yi2, w_img, h_img, max_long_side: int):
    """裁块过长边超过 max_long_side 时，以当前矩形中心为基准同比缩小（只缩不放大）。"""
    if max_long_side <= 0:
        return xi1, yi1, xi2, yi2
    cw = xi2 - xi1
    ch = yi2 - yi1
    ml = max(cw, ch)
    if ml <= max_long_side:
        return xi1, yi1, xi2, yi2
    s = max_long_side / float(ml)
    cx = (xi1 + xi2) * 0.5
    cy = (yi1 + yi2) * 0.5
    nw = cw * s
    nh = ch * s
    xi1n = int(max(0, round(cx - nw * 0.5)))
    yi1n = int(max(0, round(cy - nh * 0.5)))
    xi2n = int(min(w_img - 1, round(cx + nw * 0.5)))
    yi2n = int(min(h_img - 1, round(cy + nh * 0.5)))
    if xi2n <= xi1n:
        xi2n = min(w_img - 1, xi1n + 1)
    if yi2n <= yi1n:
        yi2n = min(h_img - 1, yi1n + 1)
    return xi1n, yi1n, xi2n, yi2n


def crop_pad(
    img_bgr,
    x1,
    y1,
    x2,
    y2,
    w_img,
    h_img,
    *,
    pad_ratio: float,
    pad_px: float,
    box_scale: float,
    min_crop_side: int,
    max_long_side: int,
):
    """
    以框中心为基准缩放 box_scale，再外扩 pad_ratio*max边长 + pad_px；
    可选 min_crop_side 把过小的块略放大（易把邻车带进来，默认关）；
    最后 max_long_side 限制最长边，防止单 patch 覆盖半幅图。
    """
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max((x2 - x1) * box_scale, 1.0)
    bh = max((y2 - y1) * box_scale, 1.0)
    x1 = cx - bw * 0.5
    x2 = cx + bw * 0.5
    y1 = cy - bh * 0.5
    y2 = cy + bh * 0.5

    pad = pad_ratio * max(bw, bh) + float(pad_px)
    xi1 = int(max(0, math.floor(x1 - pad)))
    yi1 = int(max(0, math.floor(y1 - pad)))
    xi2 = int(min(w_img - 1, math.ceil(x2 + pad)))
    yi2 = int(min(h_img - 1, math.ceil(y2 + pad)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None

    if min_crop_side > 0:
        cw = xi2 - xi1
        ch = yi2 - yi1
        if cw < min_crop_side:
            need = min_crop_side - cw
            d0, d1 = need // 2, need - need // 2
            xi1 = max(0, xi1 - d0)
            xi2 = min(w_img - 1, xi2 + d1)
        if ch < min_crop_side:
            need = min_crop_side - ch
            d0, d1 = need // 2, need - need // 2
            yi1 = max(0, yi1 - d0)
            yi2 = min(h_img - 1, yi2 + d1)
        if xi2 <= xi1 or yi2 <= yi1:
            return None

    xi1, yi1, xi2, yi2 = _clamp_roi_max_side(xi1, yi1, xi2, yi2, w_img, h_img, max_long_side)

    crop = img_bgr[yi1:yi2, xi1:xi2]
    return crop if crop.size > 0 else None


def run_pseudo_sort(crops_manifest, cls_pt, legacy_bgr, out_train, device="cuda"):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    net = models.mobilenet_v3_small(weights=weights)
    inf = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(inf, 2)
    net.load_state_dict(torch.load(cls_pt, map_location=dev))
    net = net.to(dev)
    net.eval()
    layout = ("occupied", "vacant")

    for sub in ("occupied", "vacant"):
        os.makedirs(os.path.join(out_train, sub), exist_ok=True)

    for row in crops_manifest:
        path = row["crop_path"]
        im = cv2.imread(path)
        if im is None:
            continue
        arr = im if legacy_bgr else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = tfm(arr).unsqueeze(0).to(dev)
        with torch.no_grad():
            pred = int(torch.argmax(net(x), dim=1).item())
        cls_name = layout[pred]
        dst = os.path.join(out_train, cls_name, os.path.basename(path))
        cv2.imwrite(dst, im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnr_root", type=str, default=r"E:\Programs\download\CNR-EXT_FULL_IMAGE_1000x750")
    ap.add_argument("--out_dir", type=str, required=True, help="输出根目录（将创建 raw_crops/）")
    ap.add_argument("--box_source", choices=("gt_csv", "detector"), default="gt_csv")
    ap.add_argument("--det_model", type=str, default="", help="box_source=detector 时 YOLO .pt 路径")
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument(
        "--pad_ratio",
        type=float,
        default=0.06,
        help="在框外再扩：比例 × 框最大边长（大了容易圈进多辆车）",
    )
    ap.add_argument(
        "--pad_px",
        type=float,
        default=6.0,
        help="在上述比例外再增加的像素边距",
    )
    ap.add_argument(
        "--box_scale",
        type=float,
        default=1.0,
        help="以框中心缩放车位框后再外扩；1.0=与标注框一致",
    )
    ap.add_argument(
        "--min_crop_side",
        type=int,
        default=0,
        help="裁块过小时向四周扩到最短边≥该值（>0 易带入邻车，默认关闭）",
    )
    ap.add_argument(
        "--max_long_side",
        type=int,
        default=200,
        help="裁块最长边上限（像素），防止单 patch 过大；0 表示不限制",
    )
    ap.add_argument("--max_images", type=int, default=0, help="最多处理多少张源图；0 不限制（在达到 max_crops 前持续扫）")
    ap.add_argument(
        "--max_crops",
        type=int,
        default=1000,
        help="最多保存多少块裁切；0 表示不限制（全量可能十多万块）",
    )
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="限制数量时打乱源图顺序的种子，便于复现",
    )
    ap.add_argument("--pseudo_cls_pt", type=str, default="", help="若填路径，则在 out_dir/train 下按旧模型伪标签分类（慎用）")
    ap.add_argument("--pseudo_legacy_bgr", action="store_true", help="伪标签推理时与 app_gradio legacy 一致")
    args = ap.parse_args()

    raw_dir = os.path.join(args.out_dir, "raw_crops")
    os.makedirs(raw_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")

    rows_out = []
    n_img = 0
    crop_count = 0
    cap = args.max_crops if args.max_crops > 0 else None

    if args.box_source == "detector":
        if not args.det_model:
            raise SystemExit("detector 模式需要 --det_model")
        from ultralytics import YOLO

        model = YOLO(args.det_model)
        samples = collect_samples(args.cnr_root)
        random.seed(args.shuffle_seed)
        random.shuffle(samples)
        if args.max_images > 0:
            samples = samples[: args.max_images]
        stop_all = False
        for s in samples:
            if stop_all:
                break
            img = cv2.imread(s["src_img"])
            if img is None:
                continue
            h, w = img.shape[:2]
            res = model(img, conf=args.det_conf, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            for j in range(len(xyxy)):
                if cap is not None and crop_count >= cap:
                    stop_all = True
                    break
                x1, y1, x2, y2 = [float(v) for v in xyxy[j]]
                crop = crop_pad(
                    img,
                    x1,
                    y1,
                    x2,
                    y2,
                    w,
                    h,
                    pad_ratio=args.pad_ratio,
                    pad_px=args.pad_px,
                    box_scale=args.box_scale,
                    min_crop_side=args.min_crop_side,
                    max_long_side=args.max_long_side,
                )
                if crop is None or crop.size == 0:
                    continue
                name = f"{s['out_stem']}__det{j}.jpg"
                path = os.path.join(raw_dir, name)
                cv2.imwrite(path, crop)
                rows_out.append(
                    {
                        "crop_path": path,
                        "src_image": s["src_img"],
                        "box": f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}",
                        "source": "detector",
                    }
                )
                crop_count += 1
            n_img += 1
    else:
        boxes_by_cam, ann_extent = load_camera_boxes(args.cnr_root)
        samples = collect_samples(args.cnr_root)
        samples = [s for s in samples if s["cam_id"] in boxes_by_cam]
        random.seed(args.shuffle_seed)
        random.shuffle(samples)
        if args.max_images > 0:
            samples = samples[: args.max_images]
        stop_all = False
        for s in samples:
            if stop_all:
                break
            img = cv2.imread(s["src_img"])
            if img is None:
                continue
            h, w = img.shape[:2]
            ann_w, ann_h = ann_extent[s["cam_id"]]
            for j, (gx, gy, gw, gh) in enumerate(boxes_by_cam[s["cam_id"]]):
                if cap is not None and crop_count >= cap:
                    stop_all = True
                    break
                x1, y1, x2, y2 = ann_to_xyxy(gx, gy, gw, gh, ann_w, ann_h)
                crop = crop_pad(
                    img,
                    x1,
                    y1,
                    x2,
                    y2,
                    w,
                    h,
                    pad_ratio=args.pad_ratio,
                    pad_px=args.pad_px,
                    box_scale=args.box_scale,
                    min_crop_side=args.min_crop_side,
                    max_long_side=args.max_long_side,
                )
                if crop is None or crop.size == 0:
                    continue
                name = f"{s['out_stem']}__gt{j}.jpg"
                path = os.path.join(raw_dir, name)
                cv2.imwrite(path, crop)
                rows_out.append(
                    {
                        "crop_path": path,
                        "src_image": s["src_img"],
                        "box": f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}",
                        "source": "gt_csv",
                    }
                )
                crop_count += 1
            n_img += 1

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        mw = csv.DictWriter(f, fieldnames=["crop_path", "src_image", "box", "source"])
        mw.writeheader()
        mw.writerows(rows_out)

    print(f"已扫源图约 {n_img} 张，保存裁切 {len(rows_out)} 块 → {raw_dir}")
    if cap is not None and len(rows_out) >= cap:
        print(f"（已达 --max_crops={cap} 上限后停止；全量请用 --max_crops 0）")
    print(f"清单: {manifest_path}")

    if args.pseudo_cls_pt:
        train_root = os.path.join(args.out_dir, "train")
        print("伪标签分类到", train_root, "（请务必人工纠错后再微调）")
        run_pseudo_sort(rows_out, args.pseudo_cls_pt, args.pseudo_legacy_bgr, train_root)


if __name__ == "__main__":
    main()
