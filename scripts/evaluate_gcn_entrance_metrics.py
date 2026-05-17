"""Evaluate GCN parking-slot predictions with entrance-line metrics.

This script keeps the original GCN checkpoint/model untouched. It runs image
inference, converts predicted slots into entrance-line predictions, and applies
the same endpoint/direction matching protocol used for EntranceDetect.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

_REPO = Path(__file__).resolve().parent.parent


def unit(vec):
    x, y = float(vec[0]), float(vec[1])
    n = math.hypot(x, y)
    if n <= 1e-8:
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([x / n, y / n], dtype=np.float32)


def direction_error_deg(pred, gt):
    p = unit(pred)
    g = unit(gt)
    dot = float(np.clip(np.dot(p, g), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def endpoint_errors_px(pred, gt, img_size=600.0):
    p1 = np.array(pred[:2], dtype=np.float32) * img_size
    p2 = np.array(pred[2:4], dtype=np.float32) * img_size
    g1 = np.array(gt[:2], dtype=np.float32) * img_size
    g2 = np.array(gt[2:4], dtype=np.float32) * img_size
    direct = (np.linalg.norm(p1 - g1), np.linalg.norm(p2 - g2))
    swapped = (np.linalg.norm(p1 - g2), np.linalg.norm(p2 - g1))
    return direct if max(direct) <= max(swapped) else swapped


def line_length_px(line, img_size=600.0):
    p1 = np.array(line[:2], dtype=np.float32) * img_size
    p2 = np.array(line[2:4], dtype=np.float32) * img_size
    return float(np.linalg.norm(p2 - p1))


def load_gt(eline_path: Path):
    rows = []
    if not eline_path.exists():
        return rows
    for line in eline_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        vals = [float(x) for x in parts[:6]] + [int(float(parts[6]))]
        rows.append(
            {
                "line": np.array(vals[:4], dtype=np.float32),
                "dir": np.array(vals[4:6], dtype=np.float32),
                "type": int(vals[6]),
            }
        )
    return rows


def gcn_slot_to_pred(slot):
    score, line = slot
    if hasattr(score, "detach"):
        score = float(score.detach().cpu().item())
    else:
        score = float(score)
    line = np.asarray(line, dtype=np.float32)
    dx = float(line[2] - line[0])
    dy = float(line[3] - line[1])
    # GCN demo expands the parking slot body with (dy, -dx), so use the same
    # directed entrance-line convention for strict direction matching.
    body_dir = unit([dy, -dx])
    return {
        "conf": score,
        "line": line[:4],
        "dir": body_dir,
        "type": 0,
    }


def match_predictions(preds, gts, point_thresh: float, dir_thresh: float, point_only: bool = False):
    matched_gt = set()
    matches = []
    false_pos = 0

    for pred in preds:
        best = None
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            e1, e2 = endpoint_errors_px(pred["line"], gt["line"])
            dir_err = direction_error_deg(pred["dir"], gt["dir"])
            score = max(e1, e2) if point_only else max(e1, e2) + dir_err
            ok = max(e1, e2) <= point_thresh and (point_only or dir_err <= dir_thresh)
            if best is None or score < best["score"]:
                best = {
                    "gt_idx": gi,
                    "ok": ok,
                    "score": score,
                    "endpoint_mean": (e1 + e2) * 0.5,
                    "endpoint_max": max(e1, e2),
                    "dir_err": dir_err,
                    "length_err": abs(line_length_px(pred["line"]) - line_length_px(gt["line"])),
                    "type_correct": pred["type"] == gt["type"],
                    "gt_type": gt["type"],
                }
        if best and best["ok"]:
            matched_gt.add(best["gt_idx"])
            matches.append(best)
        else:
            false_pos += 1

    false_neg = len(gts) - len(matched_gt)
    return matches, false_pos, false_neg


def empty_stats():
    return {
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "endpoint_mean_errors": [],
        "endpoint_max_errors": [],
        "direction_errors": [],
        "length_errors": [],
        "type_correct": 0,
    }


def add_match(stats, match):
    stats["tp"] += 1
    stats["endpoint_mean_errors"].append(match["endpoint_mean"])
    stats["endpoint_max_errors"].append(match["endpoint_max"])
    stats["direction_errors"].append(match["dir_err"])
    stats["length_errors"].append(match["length_err"])
    stats["type_correct"] += int(match["type_correct"])


def finalize_stats(stats):
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) else 0.0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def mean_std(values):
        arr = np.array(values, dtype=np.float32)
        if arr.size == 0:
            return 0.0, 0.0
        return float(arr.mean()), float(arr.std())

    ep_mean, ep_std = mean_std(stats["endpoint_mean_errors"])
    ep_max_mean, ep_max_std = mean_std(stats["endpoint_max_errors"])
    dir_mean, dir_std = mean_std(stats["direction_errors"])
    len_mean, len_std = mean_std(stats["length_errors"])
    type_acc = stats["type_correct"] / stats["tp"] if stats["tp"] else 0.0
    return {
        "gt": stats["gt"],
        "tp": stats["tp"],
        "fp": stats["fp"],
        "fn": stats["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "endpoint_mean_px": ep_mean,
        "endpoint_mean_std_px": ep_std,
        "endpoint_max_px": ep_max_mean,
        "endpoint_max_std_px": ep_max_std,
        "direction_error_deg": dir_mean,
        "direction_error_std_deg": dir_std,
        "length_error_px": len_mean,
        "length_error_std_px": len_std,
        "type_accuracy": type_acc,
    }


def write_outputs(out_dir: Path, metrics, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "paper_metrics_per_image.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt", "pred", "tp", "fp", "fn", "infer_ms"])
        writer.writeheader()
        writer.writerows(rows)

    def pct(v):
        return f"{v * 100:.2f}"

    lines = [
        "| Group | GT | TP | FP | FN | Precision | Recall | F1 | Endpoint Mean px | Direction deg | Length px | Type Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("all",):
        m = metrics[group]
        lines.append(
            f"| {group} | {m['gt']} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | "
            f"{m['endpoint_mean_px']:.2f} | {m['direction_error_deg']:.2f} | "
            f"{m['length_error_px']:.2f} | {pct(m['type_accuracy'])} |"
        )
    lines.append("")
    speed = metrics["speed"]
    lines.append(f"Speed: {speed['infer_ms_per_image']:.3f} ms/image, FPS {speed['fps']:.2f}")
    (out_dir / "paper_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcn-root", required=True, help="Path to gcn-parking-slot repository")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="Entrance-line dataset root with images/labels split folders")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--point-thresh", type=float, default=12.0)
    ap.add_argument("--dir-thresh", type=float, default=10.0)
    ap.add_argument("--point-only", action="store_true")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_gcn(args, device):
    gcn_root = Path(args.gcn_root).resolve()
    sys.path.insert(0, str(gcn_root))
    from psdet.models.builder import build_model
    from psdet.utils.config import cfg_from_file

    cfg = cfg_from_file(str(Path(args.cfg)))
    model = build_model(cfg.model)
    model.load_params_from_file(args.weights, logger=None, to_cpu=False)
    model.to(device).eval()
    return model


def preprocess(img_path: Path, device):
    image = Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
    tensor = T.ToTensor()(image).unsqueeze(0).to(device)
    return tensor


def predict_one(model, img_path: Path, device):
    data_dict = {"image": preprocess(img_path, device)}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_dicts, _ = model(data_dict)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0
    preds = [gcn_slot_to_pred(slot) for slot in pred_dicts["slots_pred"][0]]
    preds.sort(key=lambda row: row["conf"], reverse=True)
    return preds, infer_ms


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_gcn(args, device)

    root = Path(args.data)
    img_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split
    images = sorted(img_dir.glob("*.jpg"))

    stats_all = empty_stats()
    per_image = []
    infer_times = []

    for idx, img_path in enumerate(images, 1):
        gts = load_gt(label_dir / f"{img_path.stem}.eline")
        preds, infer_ms = predict_one(model, img_path, device)
        infer_times.append(infer_ms)

        matches, fp, fn = match_predictions(
            preds,
            gts,
            args.point_thresh,
            args.dir_thresh,
            point_only=args.point_only,
        )
        stats_all["gt"] += len(gts)
        stats_all["fp"] += fp
        stats_all["fn"] += fn
        for match in matches:
            add_match(stats_all, match)

        per_image.append(
            {
                "image": img_path.name,
                "gt": len(gts),
                "pred": len(preds),
                "tp": len(matches),
                "fp": fp,
                "fn": fn,
                "infer_ms": f"{infer_ms:.4f}",
            }
        )
        if idx % 200 == 0:
            print(f"Evaluated {idx}/{len(images)} images")

    speed_ms = float(np.mean(infer_times)) if infer_times else 0.0
    metrics = {
        "all": finalize_stats(stats_all),
        "speed": {
            "images": len(images),
            "infer_ms_per_image": speed_ms,
            "fps": 1000.0 / speed_ms if speed_ms > 0 else 0.0,
            "device": str(device),
        },
        "protocol": {
            "point_threshold_px": args.point_thresh,
            "direction_threshold_deg": args.dir_thresh,
            "point_only": args.point_only,
            "split": args.split,
            "weights": args.weights,
            "gcn_root": args.gcn_root,
        },
    }
    write_outputs(Path(args.out), metrics, per_image)
    print(json.dumps(metrics, indent=2))
    print(f"Saved paper metrics to {args.out}")


if __name__ == "__main__":
    main()
