"""
Train YOLOv10s + CBAM + DirectionDetect on PS2.0 marking-point dataset (v2).

改进点：
  - 标签改为标准 5 值格式 (class x y w h)，兼容所有 YOLO pipeline
  - 方向向量从独立 .dir 文件加载（每行 dx dy）
  - DirectionYOLODataset 简化：标准 YOLODataset + 额外加载方向

Usage:
    python scripts/train_direction_v2.py
    python scripts/train_direction_v2.py --epochs 100 --batch 16
"""

from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 0. 注册自定义模块
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import ultralytics.nn.tasks as ult_tasks
from custom_modules.c2f_cbam import C2fCBAM
from custom_modules.direction_detect import DirectionDetect

ult_tasks.__dict__["C2fCBAM"] = C2fCBAM
ult_tasks.__dict__["DirectionDetect"] = DirectionDetect

# ---------------------------------------------------------------------------
# 1. Monkey-patch parse_model：处理 C2fCBAM + DirectionDetect
# ---------------------------------------------------------------------------
_original_parse_model = ult_tasks.parse_model


def _patched_parse_model(d, ch, verbose=True):
    all_layers = d["backbone"] + d["head"]
    cbam_positions = []

    for i, layer in enumerate(all_layers):
        if layer[2] == "C2fCBAM":
            cbam_positions.append(i)
            layer[2] = "C2f"

    dir_detect_info = None
    nc_yaml = d.get("nc", 1)
    for i, layer in enumerate(all_layers):
        if layer[2] == "DirectionDetect":
            args = layer[3]
            nc = nc_yaml if (not args or args[0] == "nc") else int(args[0])
            ne = int(args[1]) if len(args) > 1 and args[1] != "ne" else 2
            dir_detect_info = (i, nc, ne)
            layer[2] = "v10Detect"
            layer[3] = [nc]

    model, save = _original_parse_model(d, ch, verbose)

    # 恢复 C2fCBAM
    for idx in cbam_positions:
        old_c2f = model[idx]
        hidden_c = old_c2f.cv2.conv.out_channels
        n_bn = len(old_c2f.m)
        shortcut = old_c2f.m[0].add
        in_ch = old_c2f.cv1.conv.in_channels
        new = C2fCBAM(in_ch, hidden_c, n_bn, shortcut)
        new.load_state_dict(old_c2f.state_dict(), strict=False)
        new.i, new.f, new.type, new.np = old_c2f.i, old_c2f.f, old_c2f.type, old_c2f.np
        model[idx] = new

    # 恢复 DirectionDetect
    if dir_detect_info is not None:
        idx, nc, ne = dir_detect_info
        old_head = model[idx]
        ch_in = [m[0].conv.in_channels for m in old_head.cv2] if hasattr(old_head, "cv2") else None
        new_head = DirectionDetect(nc=nc, ne=ne, ch=tuple(ch_in) if ch_in else (256, 512, 1024))
        try:
            new_head.load_state_dict(old_head.state_dict(), strict=False)
        except Exception:
            pass
        new_head.i, new_head.f, new_head.type, new_head.np = (
            old_head.i, old_head.f, old_head.type, old_head.np
        )
        model[idx] = new_head

    return model, save


ult_tasks.parse_model = _patched_parse_model

# ---------------------------------------------------------------------------
# 2. DirectionYOLODataset：标准 YOLODataset + 从 .dir 文件加载方向
# ---------------------------------------------------------------------------
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.data.build import build_dataloader  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402


class DirectionYOLODataset(YOLODataset):
    """标准 YOLODataset + 从 .dir 文件加载方向向量。

    标签格式：标准 5 值 (class x y w h)
    方向格式：.dir 文件，每行 dx dy
    """

    def __init__(self, *args, **kwargs):
        self._directions = {}
        super().__init__(*args, **kwargs)

    def load_directions(self):
        """根据 label 文件路径加载对应的 .dir 方向文件。"""
        for lb_file in self.label_files:
            dir_file = Path(lb_file).with_suffix(".dir")
            if dir_file.exists():
                with open(dir_file, "r") as f:
                    dirs = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            dirs.append([float(parts[0]), float(parts[1])])
                        else:
                            dirs.append([0.0, 0.0])
                self._directions[lb_file] = np.array(dirs, dtype=np.float32)
            else:
                self._directions[lb_file] = np.zeros((0, 2), dtype=np.float32)

    def cache_labels(self, path=None):
        """调用标准 cache_labels 后加载方向。"""
        result = super().cache_labels(path)
        self.load_directions()
        return result

    def __getitem__(self, index):
        """在标准 item 基础上附加方向向量。"""
        item = super().__getitem__(index)
        im_file = item.get("im_file", "")
        # 从 im_file 推导 label_file
        lb_file = self.label_files[index] if index < len(self.label_files) else None
        if lb_file and lb_file in self._directions:
            item["direction"] = self._directions[lb_file]
        else:
            # 根据 bboxes 数量创建零方向
            n = item["bboxes"].shape[0] if hasattr(item["bboxes"], "shape") else 0
            item["direction"] = np.zeros((n, 2), dtype=np.float32)
        return item

    def collate_fn(self, batch):
        """自定义 collate，合并 direction 到 batch dict。"""
        batch_dict = super().collate_fn(batch)
        directions = []
        for item in batch:
            if "direction" in item:
                directions.append(item["direction"])
        if directions:
            all_dirs = []
            for i, d in enumerate(directions):
                if len(d) > 0:
                    d_tensor = torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d.float()
                    batch_col = torch.full((d_tensor.shape[0], 1), i, dtype=torch.float32)
                    all_dirs.append(torch.cat([batch_col, d_tensor], dim=1))
            if all_dirs:
                batch_dict["direction"] = torch.cat(all_dirs, 0)
            else:
                batch_dict["direction"] = torch.zeros((0, 3), dtype=torch.float32)
        return batch_dict


# ---------------------------------------------------------------------------
# 3. Monkey-patch 方向损失到 Trainer
# ---------------------------------------------------------------------------
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.utils.loss import E2ELoss  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from custom_modules.direction_loss import v8DirectionDetectionLoss  # noqa: E402

_original_init_criterion = DetectionModel.init_criterion


def _patched_init_criterion(self):
    if getattr(self, "end2end", False):
        return E2ELoss(self, loss_fn=v8DirectionDetectionLoss)
    return v8DirectionDetectionLoss(self)


DetectionModel.init_criterion = _patched_init_criterion

_original_build_dataset = DetectionTrainer.build_dataset
_original_get_dataloader = DetectionTrainer.get_dataloader


def _patched_build_dataset(self, img_path, mode="train", batch=None):
    """使用 DirectionYOLODataset 替代标准 YOLODataset。"""
    from ultralytics.utils import colorstr

    m = self.model.module if hasattr(self.model, "module") else self.model
    gs = max(int(m.stride.max() if m else 0), 32)
    cfg = self.args
    data = self.data

    dataset = DirectionYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=gs,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
    return dataset


def _patched_get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
    """使用自定义 dataset 构建 dataloader。"""
    from ultralytics.data import build_dataloader as bd
    from contextlib import contextmanager

    @contextmanager
    def cuda_context(enabled=True):
        yield

    assert mode in {"train", "val"}
    with cuda_context(rank == -1):
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        return bd(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=(mode == "train"),
            rank=rank,
        )


# ---------------------------------------------------------------------------
# 4. 运行训练
# ---------------------------------------------------------------------------
from ultralytics import YOLO  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv10s+CBAM+Direction on PS2.0 (v2)")
    ap.add_argument("--data", default=str(_REPO / "configs" / "ps20_marks.yaml"))
    ap.add_argument("--model", default=str(_REPO / "configs" / "yolov10s_cbam_direction.yaml"))
    ap.add_argument("--pretrained", default="yolov10s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--project", default=str(_REPO / "runs" / "detect"))
    ap.add_argument("--name", default="ps20_direction_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device_kw = {} if not args.device else {"device": args.device}
    data_path = str(Path(args.data).resolve())

    # 应用 monkey-patches
    DetectionTrainer.build_dataset = _patched_build_dataset
    DetectionTrainer.get_dataloader = _patched_get_dataloader

    if args.resume:
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_pt.is_file():
            raise SystemExit(f"No checkpoint found: {last_pt}")
        model = YOLO(str(last_pt))
        model.train(data=data_path, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, project=args.project, name=args.name,
                    resume=True, workers=0, exist_ok=True, verbose=True, **device_kw)
    else:
        model = YOLO(str(Path(args.model).resolve()))
        model.train(
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            pretrained=args.pretrained,
            seed=args.seed,
            workers=0,
            exist_ok=True,
            verbose=True,
            **device_kw,
        )

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.is_file():
        print(f"\nTraining complete! Best weights: {best_pt}")


if __name__ == "__main__":
    main()
