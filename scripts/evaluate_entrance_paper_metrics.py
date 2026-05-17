"""Evaluate EntranceDetect with parking-slot paper metrics.

Metrics follow the common PS2.0 parking-slot protocol:
  - TP: both entrance endpoints within --point-thresh pixels and body direction
    error within --dir-thresh degrees.
  - Report Precision, Recall, F1, endpoint error, direction error, type accuracy,
    and speed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import scripts.train_entrance_yolo  # noqa: F401,E402 - registers custom modules
from ultralytics import YOLO  # noqa: E402


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


def preprocess(img_path: Path, imgsz: int, device):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def predict_one(model, img_path: Path, imgsz: int, conf: float, device):
    x = preprocess(img_path, imgsz, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model.model(x)[0][0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    arr = pred.detach().cpu().numpy()
    keep = arr[:, 4] >= conf
    arr = arr[keep]
    order = np.argsort(-arr[:, 4])
    arr = arr[order]

    preds = []
    for row in arr:
        type_probs = row[12:15]
        preds.append(
            {
                "conf": float(row[4]),
                "line": row[6:10].astype(np.float32),
                "dir": unit(row[10:12]),
                "type": int(np.argmax(type_probs)),
            }
        )
    return preds, infer_ms


SUBTEST_DISPLAY_NAMES = {
    "indoor-parking lot": "Indoor",
    "outdoor-normal daylight": "Outdoor normal",
    "outdoor-street light": "Street light",
    "outdoor-shadow": "Outdoor shadow",
    "outdoor-rainy": "Outdoor rainy",
    "outdoor-slanted": "Slanted",
}


def file_md5(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_subtest_lookup(subtest_root: Path, image_paths):
    if not subtest_root.exists():
        return {}

    hash_to_subset = {}
    for subdir in sorted(p for p in subtest_root.iterdir() if p.is_dir() and p.name != "all"):
        subset_name = SUBTEST_DISPLAY_NAMES.get(subdir.name, subdir.name)
        for image_path in subdir.glob("*.jpg"):
            hash_to_subset[file_md5(image_path)] = subset_name

    lookup = {}
    for image_path in image_paths:
        subset_name = hash_to_subset.get(file_md5(image_path))
        if subset_name:
            lookup[image_path.name] = subset_name
    return lookup


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
    with (out_dir / "paper_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["group", *[k for k in metrics["all"].keys() if k != "speed"]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group in ("all", "right_angle", "acute_obtuse"):
            row = {"group": group}
            row.update(metrics[group])
            writer.writerow(row)
    with (out_dir / "paper_metrics_per_image.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt", "pred", "tp", "fp", "fn", "infer_ms"])
        writer.writeheader()
        writer.writerows(rows)
    if metrics.get("subtests"):
        with (out_dir / "paper_metrics_subtests.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["subtest", "gt", "tp", "fp", "fn", "precision", "recall", "f1", "endpoint_mean_px"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for subtest, values in metrics["subtests"].items():
                row = {"subtest": subtest}
                row.update({k: values[k] for k in fieldnames if k != "subtest"})
                writer.writerow(row)

    def pct(v):
        return f"{v * 100:.2f}"

    lines = [
        "| Group | GT | TP | FP | FN | Precision | Recall | F1 | Endpoint Mean px | Direction deg | Length px | Type Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("all", "right_angle", "acute_obtuse"):
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

    if metrics.get("subtests"):
        subtest_lines = [
            "| Sub-Test Set | GT | TP | FP | FN | Precision | Recall | F1 | Endpoint Mean px |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for subtest, m in metrics["subtests"].items():
            subtest_lines.append(
                f"| {subtest} | {m['gt']} | {m['tp']} | {m['fp']} | {m['fn']} | "
                f"{pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | "
                f"{m['endpoint_mean_px']:.2f} |"
            )
        (out_dir / "paper_metrics_subtests.md").write_text("\n".join(subtest_lines), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(_REPO / "runs" / "detect" / "ps20_entrance_angled_os5_80e" / "weights" / "best.pt"))
    ap.add_argument("--data", default=r"E:\parking_yolov10_entrance_angled_os5")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=608)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--point-thresh", type=float, default=12.0)
    ap.add_argument("--dir-thresh", type=float, default=10.0)
    ap.add_argument("--point-only", action="store_true", help="Only require endpoint distance for TP matching; ignore body direction error.")
    ap.add_argument("--subtest-root", default=r"E:\Programs\download\ps2.0\testing", help="Original PS2.0 testing root with sub-test folders.")
    ap.add_argument("--out", default=str(_REPO / "runs" / "detect" / "ps20_entrance_angled_os5_80e" / "paper_metrics"))
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    model.model.to(device).eval()

    root = Path(args.data)
    img_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split
    images = sorted(img_dir.glob("*.jpg"))
    subtest_lookup = build_subtest_lookup(Path(args.subtest_root), images)

    stats_all = empty_stats()
    stats_right = empty_stats()
    stats_angled = empty_stats()
    stats_subtests = {name: empty_stats() for name in SUBTEST_DISPLAY_NAMES.values()}
    per_image = []
    infer_times = []

    for idx, img_path in enumerate(images, 1):
        gts = load_gt(label_dir / f"{img_path.stem}.eline")
        preds, infer_ms = predict_one(model, img_path, args.imgsz, args.conf, device)
        infer_times.append(infer_ms)

        matches, fp, fn = match_predictions(preds, gts, args.point_thresh, args.dir_thresh, point_only=args.point_only)
        stats_all["gt"] += len(gts)
        stats_all["fp"] += fp
        stats_all["fn"] += fn
        for match in matches:
            add_match(stats_all, match)

        right_gts = [gt for gt in gts if gt["type"] != 2]
        angled_gts = [gt for gt in gts if gt["type"] == 2]
        right_preds = [pred for pred in preds if pred["type"] != 2]
        angled_preds = [pred for pred in preds if pred["type"] == 2]
        right_matches, right_fp, right_fn = match_predictions(right_preds, right_gts, args.point_thresh, args.dir_thresh, point_only=args.point_only)
        angled_matches, angled_fp, angled_fn = match_predictions(angled_preds, angled_gts, args.point_thresh, args.dir_thresh, point_only=args.point_only)

        stats_right["gt"] += len(right_gts)
        stats_right["fp"] += right_fp
        stats_right["fn"] += right_fn
        for match in right_matches:
            add_match(stats_right, match)

        stats_angled["gt"] += len(angled_gts)
        stats_angled["fp"] += angled_fp
        stats_angled["fn"] += angled_fn
        for match in angled_matches:
            add_match(stats_angled, match)

        subtest_name = subtest_lookup.get(img_path.name)
        if subtest_name in stats_subtests:
            stats_subtests[subtest_name]["gt"] += len(gts)
            stats_subtests[subtest_name]["fp"] += fp
            stats_subtests[subtest_name]["fn"] += fn
            for match in matches:
                add_match(stats_subtests[subtest_name], match)

        per_image.append({"image": img_path.name, "gt": len(gts), "pred": len(preds), "tp": len(matches), "fp": fp, "fn": fn, "infer_ms": f"{infer_ms:.4f}"})
        if idx % 200 == 0:
            print(f"Evaluated {idx}/{len(images)} images")

    speed_ms = float(np.mean(infer_times)) if infer_times else 0.0
    metrics = {
        "all": finalize_stats(stats_all),
        "right_angle": finalize_stats(stats_right),
        "acute_obtuse": finalize_stats(stats_angled),
        "speed": {
            "images": len(images),
            "infer_ms_per_image": speed_ms,
            "fps": 1000.0 / speed_ms if speed_ms > 0 else 0.0,
            "device": str(device),
        },
        "protocol": {
            "point_threshold_px": args.point_thresh,
            "direction_threshold_deg": args.dir_thresh,
            "confidence_threshold": args.conf,
            "point_only": args.point_only,
            "split": args.split,
            "weights": args.weights,
        },
    }
    if subtest_lookup:
        metrics["subtests"] = {
            name: finalize_stats(stats_subtests[name])
            for name in SUBTEST_DISPLAY_NAMES.values()
            if stats_subtests[name]["gt"] > 0
        }
    write_outputs(Path(args.out), metrics, per_image)
    print(json.dumps(metrics, indent=2))
    print(f"Saved paper metrics to {args.out}")


if __name__ == "__main__":
    main()
