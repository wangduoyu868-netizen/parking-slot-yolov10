"""Comprehensive evaluation for GNN Slot Detector.

Metrics covering DMPR-PS, GCN Parking Slot, and VPSNet papers:
  - Point detection: P/R/F1 @ multiple dist thresholds, position error, direction error
  - Slot detection: P/R/F1 @ multiple dist thresholds, mAP @ IoU thresholds
  - Per-type: T-type vs L-type breakdown
  - Speed: FPS, inference time

Usage:
    python scripts/eval_gnn_slot.py
    python scripts/eval_gnn_slot.py --weights runs/gnn_slot/best_f1.pt --conf 0.1
"""
import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from custom_modules.gnn_slot_detector import GNNSlotDetector
from custom_modules.gnn_slot_dataset import GNNSlotDataset, build_entrance_lines


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------
def point_match(pred_xy, gt_xy, thresh=0.04):
    """Match predicted points to GT points (normalized coords).

    Returns: pred_matched[], gt_matched[], matches[] as (pred_idx, gt_idx)
    """
    pred_matched = [False] * len(pred_xy)
    gt_matched = [False] * len(gt_xy)
    matches = []

    # Greedy match by distance
    dists = []
    for i, p in enumerate(pred_xy):
        for j, g in enumerate(gt_xy):
            d = math.hypot(p[0] - g[0], p[1] - g[1])
            dists.append((d, i, j))
    dists.sort()

    for d, i, j in dists:
        if d >= thresh:
            break
        if not pred_matched[i] and not gt_matched[j]:
            pred_matched[i] = True
            gt_matched[j] = True
            matches.append((i, j))

    return pred_matched, gt_matched, matches


def slot_match_endpoints(p1, p2, g1, g2, thresh=0.04):
    """Check if predicted slot matches GT slot by endpoint distance."""
    d1a = (p1[0] - g1[0]) ** 2 + (p1[1] - g1[1]) ** 2
    d1b = (p1[0] - g2[0]) ** 2 + (p1[1] - g2[1]) ** 2
    d2a = (p2[0] - g1[0]) ** 2 + (p2[1] - g1[1]) ** 2
    d2b = (p2[0] - g2[0]) ** 2 + (p2[1] - g2[1]) ** 2
    return (d1a < thresh and d2b < thresh) or (d1b < thresh and d2a < thresh)


def compute_iou_slot(p1, p2, g1, g2):
    """Approximate IoU between two slot line segments.

    Uses midpoint + length as proxy, returns IoU in [0, 1].
    """
    pm = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    gm = ((g1[0] + g2[0]) / 2, (g1[1] + g2[1]) / 2)
    pl = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    gl = math.hypot(g2[0] - g1[0], g2[1] - g1[1])
    mid_dist = math.hypot(pm[0] - gm[0], pm[1] - gm[1])
    len_diff = abs(pl - gl)
    # Proxy IoU: penalize distance + length difference
    union = max(pl, gl) + mid_dist + len_diff * 0.5
    inter = max(0, min(pl, gl) - mid_dist)
    return inter / max(union, 1e-8)


def classify_slot_type(dx1, dy1, dx2, dy2):
    """Classify slot as 'T' (perpendicular) or 'L' (angled/parallel)."""
    # Check if directions are roughly perpendicular to the connecting line
    vx, vy = dx2 - dx1, dy2 - dy1
    vn = math.hypot(vx, vy)
    if vn < 1e-6:
        return "unknown"
    vnx, vny = vx / vn, vy / vn
    cross_i = abs(dx1 * vny - dy1 * vnx)
    cross_j = abs(dx2 * vny - dy2 * vnx)
    if cross_i > 0.7 and cross_j > 0.7:
        return "T"
    else:
        return "L"


# ---------------------------------------------------------------------------
# Point detection evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_point_detection(model, dataset, device, dist_thresholds):
    """Point-level P/R/F1 at multiple distance thresholds."""
    model.eval()

    results = {t: {"tp": 0, "fp": 0, "fn": 0} for t in dist_thresholds}
    all_pos_errors = []
    all_dir_errors = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        img_path = sample["im_file"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        # Inference
        img_r = cv2.resize(img, (512, 512))
        t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0).to(device)
        result = model(t)

        detected = result["detected_points"][0]
        detected_dirs = result["detected_dirs"][0]
        npred = result["npoints"][0].item()
        pred_pts = [detected[i].cpu().numpy() for i in range(npred)]

        # GT points
        stem = Path(img_path).stem
        label_path = Path(dataset.root) / "labels" / dataset.split / f"{stem}.txt"
        dir_path = Path(dataset.root) / "labels" / dataset.split / f"{stem}.dir"
        if not label_path.exists() or not dir_path.exists():
            continue

        gt_pts = []
        gt_dirs = []
        with open(label_path) as fl, open(dir_path) as fd:
            for ll, dl in zip(fl, fd):
                parts = ll.strip().split()
                dp = dl.strip().split()
                gt_pts.append((float(parts[1]), float(parts[2])))
                gt_dirs.append((float(dp[0]), float(dp[1])))

        # Also get direction predictions
        for thresh in dist_thresholds:
            _, _, matches = point_match(pred_pts, gt_pts, thresh=thresh)
            tp = len(matches)
            fp = len(pred_pts) - tp
            fn = len(gt_pts) - tp
            results[thresh]["tp"] += tp
            results[thresh]["fp"] += fp
            results[thresh]["fn"] += fn

        # Position & direction error (use most lenient threshold)
        max_thresh = max(dist_thresholds)
        _, _, matches = point_match(pred_pts, gt_pts, thresh=max_thresh)
        for pi, gi in matches:
            px, py = pred_pts[pi]
            gx, gy = gt_pts[gi]
            pos_err = math.hypot((px - gx) * W, (py - gy) * H)
            all_pos_errors.append(pos_err)

            # Direction error
            gdx, gdy = gt_dirs[gi]
            g_angle = math.atan2(gdy, gdx)
            # Get predicted direction from detected_dirs
            cos_v, sin_v = detected_dirs[pi].cpu().numpy()
            p_angle = math.atan2(float(sin_v), float(cos_v))
            # Smallest angle difference
            diff = abs(p_angle - g_angle)
            diff = min(diff, 2 * math.pi - diff)
            all_dir_errors.append(math.degrees(diff))

    # Compute metrics
    out = {}
    for thresh in dist_thresholds:
        r = results[thresh]
        p = r["tp"] / max(r["tp"] + r["fp"], 1)
        rec = r["tp"] / max(r["tp"] + r["fn"], 1)
        f1 = 2 * p * rec / max(p + rec, 1e-8)
        out[thresh] = {"precision": p, "recall": rec, "f1": f1,
                       "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]}
    out["pos_error"] = float(np.mean(all_pos_errors)) if all_pos_errors else 0.0
    out["pos_error_median"] = float(np.median(all_pos_errors)) if all_pos_errors else 0.0
    out["dir_error"] = float(np.mean(all_dir_errors)) if all_dir_errors else 0.0
    out["dir_error_median"] = float(np.median(all_dir_errors)) if all_dir_errors else 0.0
    return out


# ---------------------------------------------------------------------------
# Slot detection evaluation (multi-threshold + mAP)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_slot_detection(model, dataset, device, dist_thresholds, iou_thresholds,
                        edge_thresh=0.5):
    """Slot-level P/R/F1 @ multiple dist thresholds + mAP @ IoU thresholds."""
    model.eval()

    # Per distance threshold stats
    slot_results = {t: {"tp": 0, "fp": 0, "fn": 0} for t in dist_thresholds}
    # Per type stats (at default threshold 0.04)
    type_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # For mAP: collect all predictions with scores and GT per image
    all_pred_slots = []  # list of (img_idx, p1, p2, score)
    all_gt_slots = []    # list of (img_idx, g1, g2, slot_type)
    img_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        img_path = sample["im_file"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        # Inference
        img_r = cv2.resize(img, (512, 512))
        t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0).to(device)
        result = model(t)

        detected = result["detected_points"][0]
        edge_pred = result["edge_pred"][0, 0]
        npred = result["npoints"][0].item()
        N = detected.shape[0]

        # Predicted slots
        pred_slots = []
        for i in range(npred):
            for j in range(i + 1, npred):
                score = edge_pred[i * N + j].item()
                if score > edge_thresh:
                    p1 = detected[i].cpu().numpy()
                    p2 = detected[j].cpu().numpy()
                    pred_slots.append((p1, p2, score))
                    all_pred_slots.append((img_count, p1, p2, score))

        # GT slots
        stem = Path(img_path).stem
        label_path = Path(dataset.root) / "labels" / dataset.split / f"{stem}.txt"
        dir_path = Path(dataset.root) / "labels" / dataset.split / f"{stem}.dir"
        if not label_path.exists() or not dir_path.exists():
            img_count += 1
            continue

        points_raw = []
        with open(label_path) as fl, open(dir_path) as fd:
            for ll, dl in zip(fl, fd):
                parts = ll.strip().split()
                dp = dl.strip().split()
                points_raw.append((float(parts[1]) * W, float(parts[2]) * H,
                                   float(dp[0]), float(dp[1])))

        gt_pairs = build_entrance_lines(points_raw, W, H)
        gt_slots = []
        for i, j in gt_pairs:
            g1 = np.array([points_raw[i][0] / W, points_raw[i][1] / H])
            g2 = np.array([points_raw[j][0] / W, points_raw[j][1] / H])
            slot_type = classify_slot_type(points_raw[i][2], points_raw[i][3],
                                           points_raw[j][2], points_raw[j][3])
            gt_slots.append((g1, g2, slot_type))
            all_gt_slots.append((img_count, g1, g2, slot_type))

        # Match at each distance threshold
        for thresh in dist_thresholds:
            gt_matched = [False] * len(gt_slots)
            for pp1, pp2, score in pred_slots:
                found = False
                for gi, (gp1, gp2, _) in enumerate(gt_slots):
                    if not gt_matched[gi] and slot_match_endpoints(pp1, pp2, gp1, gp2, thresh):
                        gt_matched[gi] = True
                        found = True
                        break
                if found:
                    slot_results[thresh]["tp"] += 1
                else:
                    slot_results[thresh]["fp"] += 1
            slot_results[thresh]["fn"] += sum(1 for m in gt_matched if not m)

        # Per-type at threshold 0.04
        gt_matched_type = [False] * len(gt_slots)
        for pp1, pp2, score in pred_slots:
            found = False
            for gi, (gp1, gp2, stype) in enumerate(gt_slots):
                if not gt_matched_type[gi] and slot_match_endpoints(pp1, pp2, gp1, gp2, 0.016):
                    gt_matched_type[gi] = True
                    found = True
                    type_results[stype]["tp"] += 1
                    break
            if not found:
                # FP attributed to the GT type present, or "unknown"
                pass
        for gi, (_, _, stype) in enumerate(gt_slots):
            if not gt_matched_type[gi]:
                type_results[stype]["fn"] += 1

        img_count += 1

    # Compute slot P/R/F1
    out = {}
    for thresh in dist_thresholds:
        r = slot_results[thresh]
        p = r["tp"] / max(r["tp"] + r["fp"], 1)
        rec = r["tp"] / max(r["tp"] + r["fn"], 1)
        f1 = 2 * p * rec / max(p + rec, 1e-8)
        out[thresh] = {"precision": p, "recall": rec, "f1": f1,
                       "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]}

    # Per-type
    type_out = {}
    for stype, r in type_results.items():
        p = r["tp"] / max(r["tp"] + r["fp"], 1)
        rec = r["tp"] / max(r["tp"] + r["fn"], 1)
        f1 = 2 * p * rec / max(p + rec, 1e-8)
        type_out[stype] = {"precision": p, "recall": rec, "f1": f1,
                           "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]}
    out["per_type"] = type_out

    # mAP at IoU thresholds
    out["mAP"] = compute_map(all_pred_slots, all_gt_slots, iou_thresholds)

    return out


def compute_map(pred_slots, gt_slots, iou_thresholds):
    """Compute mAP at multiple IoU thresholds.

    pred_slots: [(img_idx, p1, p2, score), ...]
    gt_slots: [(img_idx, g1, g2, type), ...]
    """
    # Group GT by image
    gt_by_img = defaultdict(list)
    for img_idx, g1, g2, stype in gt_slots:
        gt_by_img[img_idx].append((g1, g2))

    # Sort predictions by score descending
    sorted_preds = sorted(pred_slots, key=lambda x: x[3], reverse=True)

    map_results = {}
    for iou_thresh in iou_thresholds:
        gt_matched = defaultdict(lambda: [False] * len(gt_by_img.get(0, [])))
        # Reset for each IoU threshold
        gt_matched_map = {}
        for img_idx in gt_by_img:
            gt_matched_map[img_idx] = [False] * len(gt_by_img[img_idx])

        tp_acc = 0
        fp_acc = 0
        precisions = []
        recalls = []
        total_gt = len(gt_slots)

        for rank, (img_idx, p1, p2, score) in enumerate(sorted_preds):
            gts = gt_by_img.get(img_idx, [])
            matched = gt_matched_map.get(img_idx, [])

            best_iou = 0
            best_gi = -1
            for gi, (g1, g2) in enumerate(gts):
                if not matched[gi]:
                    iou = compute_iou_slot(p1, p2, g1, g2)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi

            if best_iou >= iou_thresh and best_gi >= 0:
                matched[best_gi] = True
                tp_acc += 1
            else:
                fp_acc += 1

            prec = tp_acc / max(tp_acc + fp_acc, 1)
            rec = tp_acc / max(total_gt, 1)
            precisions.append(prec)
            recalls.append(rec)

        # AP (area under PR curve, 11-point interpolation)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p_at_t = max([p for p, r in zip(precisions, recalls) if r >= t], default=0)
            ap += p_at_t
        ap /= 11.0
        map_results[iou_thresh] = ap

    # mAP (mean over IoU thresholds)
    map_results["mAP"] = float(np.mean(list(map_results.values())))
    return map_results


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_speed(model, device, n_runs=100, warmup=10):
    """Benchmark inference speed."""
    model.eval()
    dummy = torch.randn(1, 3, 512, 512).to(device)

    # Warmup
    for _ in range(warmup):
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    fps = n_runs / dt
    ms_per_img = dt / n_runs * 1000
    return {"fps": fps, "ms": ms_per_img, "n_runs": n_runs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(
        _REPO / "runs" / "gnn_slot" / "best_f1.pt"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--edge", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0,
                    help="Limit eval to N images (0=all)")
    args = ap.parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model = GNNSlotDetector()
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded: {args.weights}")

    ds = GNNSlotDataset("E:/parking_yolov10_data", split="val", imgsz=512)
    if args.max_images > 0:
        ds.samples = ds.samples[:args.max_images]
    print(f"Val set: {len(ds)} images")

    dist_thresholds = [0.016, 0.02, 0.04, 0.06]
    iou_thresholds = [0.25, 0.5, 0.75]

    # ---- Point Detection ----
    print("\n" + "=" * 60)
    print("Point Detection Evaluation")
    print("=" * 60)
    pt_results = eval_point_detection(model, ds, device, dist_thresholds)

    print(f"\n{'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 56)
    for t in dist_thresholds:
        r = pt_results[t]
        print(f"{t:>8.3f} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")

    print(f"\nPosition Error: {pt_results['pos_error']:.2f} px (mean), "
          f"{pt_results['pos_error_median']:.2f} px (median)")
    print(f"Direction Error: {pt_results['dir_error']:.2f} deg (mean), "
          f"{pt_results['dir_error_median']:.2f} deg (median)")

    # ---- Slot Detection ----
    print("\n" + "=" * 60)
    print("Slot Detection Evaluation")
    print("=" * 60)
    slot_results = eval_slot_detection(model, ds, device, dist_thresholds,
                                        iou_thresholds, edge_thresh=args.edge)

    print(f"\n{'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 56)
    for t in dist_thresholds:
        r = slot_results[t]
        print(f"{t:>8.3f} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")

    # ---- mAP ----
    print(f"\nmAP (IoU thresholds):")
    for it in iou_thresholds:
        print(f"  mAP@{it}: {slot_results['mAP'][it]:.4f}")
    print(f"  mAP (mean): {slot_results['mAP']['mAP']:.4f}")

    # ---- Per-type ----
    if slot_results["per_type"]:
        print(f"\nPer-type Slot Metrics (dist=0.016):")
        print(f"{'Type':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-" * 56)
        for stype, r in sorted(slot_results["per_type"].items()):
            print(f"{stype:>8} {r['precision']:>8.4f} {r['recall']:>8.4f} "
                  f"{r['f1']:>8.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")

    # ---- Speed ----
    print("\n" + "=" * 60)
    print("Speed Benchmark")
    print("=" * 60)
    speed = eval_speed(model, device)
    print(f"  FPS: {speed['fps']:.1f}")
    print(f"  Latency: {speed['ms']:.1f} ms/image")
    print(f"  (averaged over {speed['n_runs']} runs)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY (for paper)")
    print("=" * 60)
    t04 = slot_results[0.04]
    print(f"Point Detection @ d=0.04: P={pt_results[0.04]['precision']:.4f} "
          f"R={pt_results[0.04]['recall']:.4f} F1={pt_results[0.04]['f1']:.4f}")
    print(f"Position Error: {pt_results['pos_error']:.2f} px")
    print(f"Direction Error: {pt_results['dir_error']:.2f} deg")
    print(f"Slot Detection @ d=0.04: P={t04['precision']:.4f} "
          f"R={t04['recall']:.4f} F1={t04['f1']:.4f}")
    print(f"mAP@0.25={slot_results['mAP'][0.25]:.4f} "
          f"mAP@0.5={slot_results['mAP'][0.5]:.4f} "
          f"mAP@0.75={slot_results['mAP'][0.75]:.4f}")
    print(f"mAP={slot_results['mAP']['mAP']:.4f}")
    print(f"Speed: {speed['fps']:.1f} FPS ({speed['ms']:.1f} ms)")


if __name__ == "__main__":
    main()
