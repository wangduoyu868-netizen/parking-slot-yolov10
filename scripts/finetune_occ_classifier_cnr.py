"""
在 CNR（或其它域）占用数据集上微调 MobileNetV3-Small 二分类。

数据目录（ImageFolder）：
  DATA_ROOT/train/occupied/
  DATA_ROOT/train/vacant/
  DATA_ROOT/val/occupied/
  DATA_ROOT/val/vacant/

先用 crop_cnr_patches_for_occ_dataset.py 得到 raw_crops，人工归类到 train 下各类，
再运行 split_occ_train_val.py，最后执行本脚本。

示例:
  python scripts/finetune_occ_classifier_cnr.py ^
    --data_root E:\\cnr_occ_cls ^
    --pretrained E:\\parking_slot_cls_runs\\best_mobilenetv3_small.pth ^
    --save_dir E:\\parking_slot_cls_runs_cnr
"""

import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


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

    avg_loss = running_loss / total
    acc = correct / total

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--pretrained", type=str, required=True, help="PS2.0 上训好的 .pth")
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("classes (ImageFolder order):", sorted(os.listdir(os.path.join(args.data_root, "train"))))

    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(args.img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
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

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    if not os.path.isdir(val_dir):
        raise SystemExit(f"缺少验证集目录: {val_dir}，请先运行 split_occ_train_val.py")
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    assert train_ds.classes == val_ds.classes, "train/val 类别名需一致"
    num_classes = len(train_ds.classes)
    assert num_classes == 2, "本脚本仅支持 occupied + vacant 二类"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    inf = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(inf, num_classes)
    model.load_state_dict(torch.load(args.pretrained, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

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

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc, vp, vr, vf1, vcm = evaluate(val_loader, model, criterion, device, num_classes)
        scheduler.step(vf1)
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} F1={vf1:.4f} "
            f"time={time.time()-t0:.1f}s"
        )
        if vf1 > best_f1:
            best_f1 = vf1
            best_wts = copy.deepcopy(model.state_dict())
            out_pt = os.path.join(args.save_dir, "best_mobilenetv3_small_cnr_ft.pth")
            torch.save(model.state_dict(), out_pt)
            print("  >>> saved", out_pt)

    print("best val macro-F1:", best_f1)
    print("ImageFolder class order (logit index):", train_ds.class_to_idx)


if __name__ == "__main__":
    main()
