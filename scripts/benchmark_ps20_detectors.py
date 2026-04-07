"""
在 PS2.0 转换后的同一 YOLO data.yaml 上，依次训练（或仅验证）多个检测模型并汇总 CSV，
便于对比 YOLOv5s / YOLOv8s / YOLOv10s 与本课题设置。

默认 data 指向仓库内 configs/ps20_marks.yaml（其中 path 需指向你本机的 YOLO 数据根目录）。

示例（仓库根目录执行）:
  python scripts/benchmark_ps20_detectors.py --epochs 100 --batch 16
  python scripts/benchmark_ps20_detectors.py --skip_train --weights runs/detect/ps20_bench_yolov8s/weights/best.pt ...
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def _sanitize_run_name(model_id: str) -> str:
    s = model_id.replace(".pt", "").replace(".yaml", "")
    s = re.sub(r"[^\w\-]+", "_", s)
    return s[:80] or "model"


def _extract_box_metrics(metrics) -> dict[str, float]:
    """兼容不同 ultralytics 小版本的指标字段名。"""
    box = metrics.box
    map50 = float(getattr(box, "map50", 0.0) or 0.0)
    map5095 = float(getattr(box, "map", 0.0) or 0.0)
    prec = getattr(box, "mp", None)
    rec = getattr(box, "mr", None)
    if prec is None:
        prec = getattr(box, "p", None)
    if rec is None:
        rec = getattr(box, "r", None)
    if prec is not None:
        try:
            prec = float(prec)
        except (TypeError, ValueError):
            prec = float(getattr(prec, "mean", lambda: 0.0)())
    else:
        prec = 0.0
    if rec is not None:
        try:
            rec = float(rec)
        except (TypeError, ValueError):
            rec = float(getattr(rec, "mean", lambda: 0.0)())
    else:
        rec = 0.0
    return {
        "mAP50": map50,
        "mAP50-95": map5095,
        "precision": prec,
        "recall": rec,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="PS2.0 标记点检测：多模型训练/验证并导出 CSV")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_REPO_ROOT / "configs" / "ps20_marks.yaml"),
        help="YOLO data yaml（与 convert_ps20_to_yolo 输出一致）",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["yolov5su.pt", "yolov8s.pt", "yolov10s.pt"],
        help="预训练权重或模型名；Ultralytics 中 YOLOv5 small 常用 yolov5su.pt",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="", help="空则自动；可填 0 或 cpu")
    ap.add_argument("--project", type=str, default=str(_REPO_ROOT / "runs" / "detect"))
    ap.add_argument("--name_prefix", type=str, default="ps20_bench")
    ap.add_argument(
        "--skip_train",
        action="store_true",
        help="不训练；需配合 --weights，且与 --models 数量一致（models 仅作表格中的名称标签）",
    )
    ap.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help="与 models 一一对应的 .pt；与 --skip_train 联用",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=str(_REPO_ROOT / "runs" / "detect" / "ps20_detector_benchmark.csv"),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    data_yaml = Path(args.data).resolve()
    if not data_yaml.is_file():
        print(f"找不到 data yaml: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    if args.skip_train:
        if not args.weights or len(args.weights) != len(args.models):
            print("使用 --skip_train 时，必须提供与 --models 数量相同的 --weights", file=sys.stderr)
            sys.exit(1)

    project = Path(args.project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    device_kw = {} if not args.device else {"device": args.device}

    for i, model_id in enumerate(args.models):
        tag = _sanitize_run_name(model_id)
        run_name = f"{args.name_prefix}_{tag}"
        t0 = time.perf_counter()

        if args.skip_train:
            weight_path = Path(args.weights[i]).resolve()
            if not weight_path.is_file():
                print(f"跳过：权重不存在 {weight_path}", file=sys.stderr)
                continue
            model = YOLO(str(weight_path))
            train_sec = 0.0
        else:
            model = YOLO(model_id)
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                project=str(project),
                name=run_name,
                seed=args.seed,
                workers=0,
                exist_ok=True,
                verbose=True,
                **device_kw,
            )
            train_sec = time.perf_counter() - t0
            best_pt = project / run_name / "weights" / "best.pt"
            if not best_pt.is_file():
                print(f"训练未产生 best.pt: {best_pt}", file=sys.stderr)
                continue
            model = YOLO(str(best_pt))

        val_t0 = time.perf_counter()
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            split="val",
            plots=False,
            verbose=False,
            workers=0,
            **device_kw,
        )
        val_sec = time.perf_counter() - val_t0
        m = _extract_box_metrics(metrics)

        row = {
            "model": model_id if not args.skip_train else f"{model_id} (val_only)",
            "weights": str(args.weights[i]) if args.skip_train else str(project / run_name / "weights" / "best.pt"),
            "mAP50": round(m["mAP50"], 6),
            "mAP50-95": round(m["mAP50-95"], 6),
            "precision": round(m["precision"], 6),
            "recall": round(m["recall"], 6),
            "train_sec": round(train_sec, 2),
            "val_sec": round(val_sec, 2),
            "run_name": run_name,
        }
        rows.append(row)
        print(row)

    if not rows:
        print("没有可写入的结果。", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "model",
        "weights",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "train_sec",
        "val_sec",
        "run_name",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"已写入: {out_csv}")


if __name__ == "__main__":
    main()
