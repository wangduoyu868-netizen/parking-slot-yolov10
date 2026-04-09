"""
使用 CNR-EXT 官方 150×150 patch 列表训练占用二分类（MobileNetV3-Small，与 app_gradio / finetune_occ_classifier_cnr 一致）。

数据布局（你本机）:
  e:\\Programs\\download\\CNR-EXT-Patches-150x150\\
    PATCHES\\   # SUNNY/OVERCAST/RAINY/.../camera*/ *.jpg
    LABELS\\    # train.txt, val.txt, test.txt

列表文件每行: 「相对 PATCHES 的路径」空格「0 或 1」
  默认约定: 0 = vacant（空）, 1 = occupied（占）
与 ImageFolder 训练脚本对齐的类别下标: 0 = occupied, 1 = vacant
  → 内部标签 target = 1 - csv_label（可用 --swap_labels 反转）

关于 CNRPark+EXT.csv:
  该 CSV 的 image_url 多为 CNRPark/A/free/... 全图路径，与 EXT 150 补丁命名（SUNNY/2015-...）不是同一套文件；
  本脚本以 PATCHES + LABELS/*.txt 为准。若需联合 CNRPark+，需另行对齐文件名或另写转换脚本。

示例（PowerShell 须一整条命令；勿单独执行以 -- 开头的行）:
  python scripts/train_occ_classifier_cnr_ext_patches.py --pretrained E:\\parking_slot_cls_runs\\best_mobilenetv3_small.pth

  或显式路径:
  & .\\venv\\Scripts\\python.exe scripts\\train_occ_classifier_cnr_ext_patches.py `
    --patches_root E:\\Programs\\download\\CNR-EXT-Patches-150x150\\PATCHES `
    --save_dir E:\\parking_slot_cls_runs_cnr_ext_p150
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def parse_list_line(line: str) -> Optional[Tuple[str, int]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    rel, lab_s = parts[0].strip(), parts[1].strip()
    try:
        lab = int(lab_s)
    except ValueError:
        return None
    if lab not in (0, 1):
        return None
    rel = rel.replace("/", os.sep)
    return rel, lab


def load_split_list(labels_dir: str, split: str) -> List[Tuple[str, int]]:
    path = os.path.join(labels_dir, f"{split}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    rows: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = parse_list_line(line)
            if p is None:
                continue
            rows.append(p)
    return rows


class CNRExtPatchDataset(Dataset):
    """从 train.txt / val.txt 读取相对路径与 0/1 标签。"""

    def __init__(
        self,
        patches_root: str,
        entries: List[Tuple[str, int]],
        transform,
        swap_labels: bool,
    ):
        self.patches_root = patches_root
        self.entries = entries
        self.transform = transform
        self.swap_labels = swap_labels

    def __len__(self):
        return len(self.entries)

    def _to_class_idx(self, raw: int) -> int:
        # raw 0 vacant, 1 occupied → ImageFolder 顺序 occupied=0, vacant=1
        if self.swap_labels:
            return raw
        return 1 - raw

    def __getitem__(self, idx):
        rel, raw = self.entries[idx]
        fp = os.path.join(self.patches_root, rel)
        img = Image.open(fp).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = self._to_class_idx(raw)
        return img, y


def evaluate(loader, model, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)

    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for gt, pd in zip(all_labels, all_preds):
        cm[gt][pd] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][k] for k in range(num_classes) if k != c)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return avg_loss, acc, sum(precisions) / num_classes, sum(recalls) / num_classes, sum(f1s) / num_classes, cm


def filter_existing(entries: List[Tuple[str, int]], patches_root: str) -> List[Tuple[str, int]]:
    ok = []
    missing = 0
    for rel, lab in entries:
        fp = os.path.join(patches_root, rel)
        if os.path.isfile(fp):
            ok.append((rel, lab))
        else:
            missing += 1
    if missing:
        print(f"警告: 有 {missing} 条在 PATCHES 下找不到文件，已跳过")
    return ok


def main():
    ap = argparse.ArgumentParser(description="CNR-EXT 150×150 patch 列表训练占用分类")
    ap.add_argument(
        "--patches_root",
        type=str,
        default=r"E:\Programs\download\CNR-EXT-Patches-150x150\PATCHES",
    )
    ap.add_argument(
        "--labels_dir",
        type=str,
        default=r"E:\Programs\download\CNR-EXT-Patches-150x150\LABELS",
    )
    ap.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="PS2.0 等预训练头权重 .pth；空则仅加载 ImageNet backbone + 随机头",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default=r"E:\parking_slot_cls_runs_cnr_ext_p150",
        help="权重保存目录（默认可改）",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224, help="输入分类网络边长（patch 150 会放大）")
    ap.add_argument(
        "--swap_labels",
        action="store_true",
        help="列表中 0/1 与默认 vacant/occupied 相反时使用",
    )
    ap.add_argument(
        "--eval_test",
        action="store_true",
        help="训练结束后用 LABELS/test.txt 再评一次（不参与选优）",
    )
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("logit 下标 0=occupied, 1=vacant（与 Gradio / ImageFolder 一致）")

    train_entries = filter_existing(load_split_list(args.labels_dir, "train"), args.patches_root)
    val_entries = filter_existing(load_split_list(args.labels_dir, "val"), args.patches_root)
    if len(train_entries) < 100 or len(val_entries) < 50:
        raise SystemExit(f"样本过少: train={len(train_entries)} val={len(val_entries)}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CNRExtPatchDataset(args.patches_root, train_entries, train_tf, args.swap_labels)
    val_ds = CNRExtPatchDataset(args.patches_root, val_entries, eval_tf, args.swap_labels)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin
    )

    num_classes = 2
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    inf = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(inf, num_classes)

    if args.pretrained and os.path.isfile(args.pretrained):
        state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state, strict=True)
        print("loaded pretrained:", args.pretrained)
    else:
        print("未提供 --pretrained，使用 ImageNet 权重 + 新分类头（随机初始化头需多训几轮）")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    out_name = "best_mobilenetv3_cnr_ext_p150.pth"

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        total = correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, vp, vr, vf1, vcm = evaluate(val_loader, model, criterion, device, num_classes)
        scheduler.step(vf1)
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} macroF1={vf1:.4f} "
            f"time={time.time()-t0:.1f}s"
        )
        if vf1 > best_f1:
            best_f1 = vf1
            best_wts = copy.deepcopy(model.state_dict())
            out_pt = os.path.join(args.save_dir, out_name)
            torch.save(model.state_dict(), out_pt)
            print("  >>> saved", out_pt)

    print("best val macro-F1:", best_f1)

    if args.eval_test:
        test_entries = filter_existing(load_split_list(args.labels_dir, "test"), args.patches_root)
        test_ds = CNRExtPatchDataset(args.patches_root, test_entries, eval_tf, args.swap_labels)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin
        )
        model.load_state_dict(best_wts)
        tl, ta, tp, tr, tf1, tcm = evaluate(test_loader, model, criterion, device, num_classes)
        print(f"TEST loss={tl:.4f} acc={ta:.4f} macroF1={tf1:.4f} cm={tcm}")


if __name__ == "__main__":
    main()
