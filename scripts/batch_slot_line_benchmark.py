"""
对多个已训练检测权重：在 PS2.0 测试图上导出 YOLO txt，再调用 evaluate_slot_lines 算车位线 P/R/F1。

典型流程（仓库根目录、使用 parking_env）:
  python scripts/batch_slot_line_benchmark.py ^
    --benchmark_csv runs/detect/ps20_detector_benchmark.csv

或手动指定权重与标签名:
  python scripts/batch_slot_line_benchmark.py ^
    --weights runs/detect/ps20_bench_yolov5su/weights/best.pt runs/detect/ps20_bench_yolov8s/weights/best.pt ^
    --tags yolov5su yolov8s

依赖: ultralytics, opencv-python（与 evaluate_slot_lines 相同）
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate_slot_lines import (  # noqa: E402
    _DEFAULT_IMG_DIR,
    _DEFAULT_JSON_DIR,
    run_slot_line_eval,
)


def _sanitize_tag(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s)
    return s[:64] or "model"


def _read_benchmark_weights(csv_path: Path) -> list[tuple[str, str, str]]:
    """(predict 子目录名, 表格展示用模型名, 权重路径)"""
    rows: list[tuple[str, str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("weights") or "").strip()
            if not w:
                continue
            folder = (row.get("run_name") or "").strip()
            display = (row.get("model") or "").strip()
            if not folder:
                folder = display or Path(w).parent.parent.name
            if not display:
                display = Path(w).stem
            rows.append((folder, display, w))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="批量预测 + 车位线指标汇总")
    ap.add_argument(
        "--benchmark_csv",
        type=str,
        default=str(_REPO_ROOT / "runs" / "detect" / "ps20_detector_benchmark.csv"),
        help="含 weights 列的 CSV；与 --weights 二选一",
    )
    ap.add_argument("--weights", nargs="+", default=None)
    ap.add_argument("--tags", nargs="+", default=None, help="与 --weights 一一对应；缺省用文件名 stem")
    ap.add_argument("--img_dir", type=str, default=_DEFAULT_IMG_DIR)
    ap.add_argument("--json_dir", type=str, default=_DEFAULT_JSON_DIR)
    ap.add_argument(
        "--predict_project",
        type=str,
        default=str(_REPO_ROOT / "runs" / "slot_line_preds"),
        help="ultralytics predict 的 project 根目录，其下每个 tag 一个子目录",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--predict_conf", type=float, default=0.25, help="predict 置信度阈值（仍会写入 txt，评估里用 min_conf 过滤点）")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--skip_predict", action="store_true", help="仅评估，不跑 predict（labels 已存在）")
    ap.add_argument("--vis_num", type=int, default=20)
    ap.add_argument("--no_vis", action="store_true")
    ap.add_argument(
        "--out_csv",
        type=str,
        default=str(_REPO_ROOT / "runs" / "detect" / "ps20_slot_line_benchmark.csv"),
    )
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装 ultralytics", file=sys.stderr)
        sys.exit(1)

    if args.weights:
        triples: list[tuple[str, str, str]] = []
        tags = args.tags or []
        for i, w in enumerate(args.weights):
            folder = tags[i] if i < len(tags) else Path(w).parent.parent.name
            display = Path(w).stem
            triples.append((folder, display, w))
    else:
        p = Path(args.benchmark_csv).resolve()
        if not p.is_file():
            print(f"找不到 {p}，请提供 --weights 或有效的 --benchmark_csv", file=sys.stderr)
            sys.exit(1)
        triples = _read_benchmark_weights(p)
        if not triples:
            print("benchmark CSV 中没有 weights 行", file=sys.stderr)
            sys.exit(1)

    project_root = Path(args.predict_project).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device_kw = {} if not args.device else {"device": args.device}
    summary_rows: list[dict[str, object]] = []

    for folder_name, display_model, weight_path in triples:
        tag = _sanitize_tag(folder_name)
        run_dir = project_root / tag
        labels_dir = run_dir / "labels"

        if not args.skip_predict:
            wp = Path(weight_path).resolve()
            if not wp.is_file():
                print(f"跳过：权重不存在 {wp}", file=sys.stderr)
                continue
            yolo = YOLO(str(wp))
            # stream=True：逐张处理，避免整目录预测时 Results 全部堆在内存里（Ultralytics 会 WARN）
            for _ in yolo.predict(
                source=args.img_dir,
                imgsz=args.imgsz,
                conf=args.predict_conf,
                save_txt=True,
                save_conf=True,
                save=False,
                project=str(project_root),
                name=tag,
                exist_ok=True,
                verbose=False,
                stream=True,
                **device_kw,
            ):
                pass

        if not labels_dir.is_dir():
            print(f"跳过：未找到标签目录 {labels_dir}", file=sys.stderr)
            continue

        vis_root = None if args.no_vis else str(project_root / f"{tag}_vis")
        r = run_slot_line_eval(
            args.img_dir,
            args.json_dir,
            str(labels_dir),
            out_vis_dir=vis_root,
            vis_num=args.vis_num,
        )
        row = {
            "tag": tag,
            "model": display_model,
            "weights": str(Path(weight_path).resolve()),
            "labels_dir": str(labels_dir),
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "precision": round(r["precision"], 6),
            "recall": round(r["recall"], 6),
            "f1": round(r["f1"], 6),
        }
        summary_rows.append(row)
        print(row)

    if not summary_rows:
        print("没有可写入的结果。", file=sys.stderr)
        sys.exit(1)

    fields = [
        "tag",
        "model",
        "weights",
        "labels_dir",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"已写入: {out_csv}")


if __name__ == "__main__":
    main()
