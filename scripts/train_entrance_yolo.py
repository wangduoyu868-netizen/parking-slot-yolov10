"""Train YOLOv10s + CBAM + EntranceDetect on true PS2.0 entrance-line labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import ultralytics.nn.tasks as ult_tasks
from custom_modules.c2f_cbam import C2fCBAM
from custom_modules.entrance_detect import EntranceDetect

ult_tasks.__dict__["C2fCBAM"] = C2fCBAM
ult_tasks.__dict__["EntranceDetect"] = EntranceDetect

_original_parse_model = ult_tasks.parse_model


def _patched_parse_model(d, ch, verbose=True):
    all_layers = d["backbone"] + d["head"]
    cbam_positions = []
    entrance_info = None

    for i, layer in enumerate(all_layers):
        if layer[2] == "C2fCBAM":
            cbam_positions.append(i)
            layer[2] = "C2f"

    nc_yaml = d.get("nc", 1)
    for i, layer in enumerate(all_layers):
        if layer[2] == "EntranceDetect":
            args = layer[3]
            nc = nc_yaml if (not args or args[0] == "nc") else int(args[0])
            ne = int(args[1]) if len(args) > 1 and args[1] != "ne" else 9
            entrance_info = (i, nc, ne)
            layer[2] = "v10Detect"
            layer[3] = [nc]

    model, save = _original_parse_model(d, ch, verbose)

    for idx in cbam_positions:
        old = model[idx]
        hidden_c = old.cv2.conv.out_channels
        n_bn = len(old.m)
        shortcut = old.m[0].add
        in_ch = old.cv1.conv.in_channels
        new = C2fCBAM(in_ch, hidden_c, n_bn, shortcut)
        new.load_state_dict(old.state_dict(), strict=False)
        new.i, new.f, new.type, new.np = old.i, old.f, old.type, old.np
        model[idx] = new

    if entrance_info is not None:
        idx, nc, ne = entrance_info
        old_head = model[idx]
        ch_in = [m[0].conv.in_channels for m in old_head.cv2] if hasattr(old_head, "cv2") else None
        new_head = EntranceDetect(nc=nc, ne=ne, ch=tuple(ch_in) if ch_in else (256, 512, 1024))
        try:
            new_head.load_state_dict(old_head.state_dict(), strict=False)
        except Exception:
            pass
        new_head.i, new_head.f, new_head.type, new_head.np = old_head.i, old_head.f, old_head.type, old_head.np
        model[idx] = new_head

    return model, save


ult_tasks.parse_model = _patched_parse_model

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.loss import E2ELoss  # noqa: E402
from custom_modules.entrance_loss import v8EntranceDetectionLoss  # noqa: E402


class EntranceYOLODataset(YOLODataset):
    """Standard YOLO entrance boxes plus aligned .eline labels."""

    def __init__(self, *args, **kwargs):
        self._entrances = {}
        super().__init__(*args, **kwargs)

    def load_entrances(self):
        for lb_file in self.label_files:
            eline_file = Path(lb_file).with_suffix(".eline")
            rows = []
            if eline_file.exists():
                with eline_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            rows.append([float(x) for x in parts[:7]])
            self._entrances[lb_file] = (
                np.array(rows, dtype=np.float32)
                if rows else np.zeros((0, 7), dtype=np.float32)
            )

    def cache_labels(self, path=None):
        result = super().cache_labels(path)
        self.load_entrances()
        return result

    def __getitem__(self, index):
        item = super().__getitem__(index)
        lb_file = self.label_files[index] if index < len(self.label_files) else None
        n = item["bboxes"].shape[0] if hasattr(item["bboxes"], "shape") else 0
        entrance = self._entrances.get(lb_file, np.zeros((0, 7), dtype=np.float32))
        if len(entrance) != n:
            fixed = np.zeros((n, 7), dtype=np.float32)
            fixed[: min(n, len(entrance))] = entrance[: min(n, len(entrance))]
            entrance = fixed
        item["entrance"] = entrance
        return item

    def collate_fn(self, batch):
        batch_dict = super().collate_fn(batch)
        rows = []
        for i, item in enumerate(batch):
            entrance = item.get("entrance")
            if entrance is None or len(entrance) == 0:
                continue
            ent_t = torch.from_numpy(entrance).float() if isinstance(entrance, np.ndarray) else entrance.float()
            batch_col = torch.full((ent_t.shape[0], 1), i, dtype=torch.float32)
            rows.append(torch.cat([batch_col, ent_t], dim=1))
        batch_dict["entrance"] = torch.cat(rows, 0) if rows else torch.zeros((0, 8), dtype=torch.float32)
        return batch_dict


def _patched_init_criterion(self):
    if getattr(self, "end2end", False):
        return E2ELoss(self, loss_fn=v8EntranceDetectionLoss)
    return v8EntranceDetectionLoss(self)


DetectionModel.init_criterion = _patched_init_criterion


def _patched_build_dataset(self, img_path, mode="train", batch=None):
    from ultralytics.utils import colorstr

    m = self.model.module if hasattr(self.model, "module") else self.model
    gs = max(int(m.stride.max() if m else 0), 32)
    cfg = self.args
    return EntranceYOLODataset(
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
        data=self.data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


DetectionTrainer.build_dataset = _patched_build_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(_REPO / "configs" / "yolov10s_cbam_entrance.yaml"))
    ap.add_argument("--data", default=str(_REPO / "configs" / "ps20_entrance.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--name", default="ps20_entrance_yolov10s")
    return ap.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(_REPO / "runs" / "detect"),
        name=args.name,
        task="detect",
        mosaic=0.0,
        close_mosaic=0,
        fliplr=0.0,
        flipud=0.0,
        translate=0.0,
        scale=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
    )


if __name__ == "__main__":
    main()
