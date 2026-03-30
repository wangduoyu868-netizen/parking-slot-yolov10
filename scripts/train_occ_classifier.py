import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_ROOT = r"E:\parking_slot_cls_dataset"
SAVE_DIR = r"E:\parking_slot_cls_runs"

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
IMG_SIZE = 224
NUM_WORKERS = 0   # Windows下先设成0，最稳

def evaluate(loader, model, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    all_preds = []
    all_labels = []

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

    # confusion matrix
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for gt, pd in zip(all_labels, all_preds):
        cm[gt][pd] += 1

    precisions = []
    recalls = []
    f1s = []

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

    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1s) / num_classes

    return avg_loss, acc, macro_precision, macro_recall, macro_f1, cm


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

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

        val_loss, val_acc, val_p, val_r, val_f1, val_cm = evaluate(
            val_loader, model, criterion, device, num_classes
        )
        scheduler.step(val_f1)

        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_P={val_p:.4f} val_R={val_r:.4f} val_F1={val_f1:.4f} "
              f"time={elapsed:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_mobilenetv3_small.pth"))
            print(">>> 保存最佳模型")

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, test_p, test_r, test_f1, test_cm = evaluate(
        test_loader, model, criterion, device, num_classes
    )

    print("\n===== Final Test Results =====")
    print(f"Test Loss = {test_loss:.4f}")
    print(f"Test Acc  = {test_acc:.4f}")
    print(f"Test P    = {test_p:.4f}")
    print(f"Test R    = {test_r:.4f}")
    print(f"Test F1   = {test_f1:.4f}")
    print("Test Confusion Matrix:")
    for row in test_cm:
        print(row)


if __name__ == "__main__":
    main()