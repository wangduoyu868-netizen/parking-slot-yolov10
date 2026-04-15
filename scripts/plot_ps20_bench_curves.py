"""
读取 ps20_bench_yolov5su / yolov8s / yolov10s 的 results.csv，绘制同一坐标下的训练曲线对比图。

依赖: matplotlib（通常随 ultralytics 已安装）
用法（仓库根目录）:
  python scripts/plot_ps20_bench_curves.py
  python scripts/plot_ps20_bench_curves.py --out docs/my_curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_results(csv_path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")

    epochs: List[int] = []
    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "train/box_loss",
        "val/box_loss",
    ]
    series: Dict[str, List[float]] = {k: [] for k in keys}

    for row in rows:
        try:
            epochs.append(int(float(row["epoch"])))
        except (KeyError, ValueError) as e:
            raise ValueError(f"bad epoch in {csv_path}") from e
        for k in keys:
            if k not in row:
                raise KeyError(f"missing column {k!r} in {csv_path}")
            series[k].append(float(row[k]))
    return epochs, series


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot PS2.0 YOLO bench curves (3 models)")
    ap.add_argument(
        "--out",
        type=str,
        default=str(_repo_root() / "docs" / "ps20_bench_curves.png"),
        help="Output PNG path",
    )
    ns = ap.parse_args()

    import matplotlib.pyplot as plt

    root = _repo_root()
    runs = [
        ("YOLOv5su", root / "runs" / "detect" / "ps20_bench_yolov5su" / "results.csv"),
        ("YOLOv8s", root / "runs" / "detect" / "ps20_bench_yolov8s" / "results.csv"),
        ("YOLOv10s", root / "runs" / "detect" / "ps20_bench_yolov10s" / "results.csv"),
    ]

    loaded: List[Tuple[str, List[int], Dict[str, List[float]]]] = []
    for label, path in runs:
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        ep, ser = load_results(path)
        loaded.append((label, ep, ser))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle("PS2.0 marking-point detection — training curves (same data, 100 epochs)", fontsize=14)

    plot_specs = [
        ("metrics/mAP50(B)", "mAP50", axes[0, 0]),
        ("metrics/mAP50-95(B)", "mAP50-95", axes[0, 1]),
        ("metrics/precision(B)", "Precision (B)", axes[0, 2]),
        ("metrics/recall(B)", "Recall (B)", axes[1, 0]),
        ("train/box_loss", "Train box loss", axes[1, 1]),
        ("val/box_loss", "Val box loss", axes[1, 2]),
    ]

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for key, title, ax in plot_specs:
        for (label, ep, ser), c in zip(loaded, colors):
            ax.plot(ep, ser[key], label=label, color=c, linewidth=1.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        if ax in axes[1, :]:
            ax.set_xlabel("epoch")

    plt.tight_layout()
    out_path = Path(ns.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
