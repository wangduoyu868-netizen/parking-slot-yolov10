# 对比实验训练教程

## 概述

本教程用于训练论文对比实验所需的基线模型，与本文方法（YOLOv10s+CBAM）在相同数据集上进行公平对比。

### 实验列表

| 编号 | 模型 | 配置文件 | 数据集 | 年份 | 预计耗时 |
|------|------|---------|--------|------|---------|
| 1 | YOLOv5s | `configs/yolov5s_entrance.yaml` | ps20_entrance | 2020 | ~11h |
| 2 | YOLOv8s | `configs/yolov8s_entrance.yaml` | ps20_entrance | 2023 | ~11h |
| 3 | YOLOv10s | `configs/yolov10s_entrance.yaml` | ps20_entrance | 2024 | ~11h |
| 4 | YOLOv11s | `configs/yolo11s_entrance.yaml` | ps20_entrance | 2024 | ~11h |
| 5 | YOLOv10s+CBAM（本文方法） | `configs/yolov10s_cbam_entrance.yaml` | ps20_entrance_angled_os5 | 2024 | 已完成 |

总计约 44 小时（依次执行）。

## 前置条件

1. 已安装 `parking_env` 环境（Miniconda）
2. 数据集 `E:\parking_yolov10_entrance` 存在
3. 磁盘空间充足（每个实验约 200MB 权重 + 日志）

确认环境：
```powershell
& "E:\miniconda3\envs\parking_env\python.exe" -c "import torch; print(torch.cuda.is_available())"
```
输出 `True` 表示 GPU 可用。

## 训练命令

**所有命令在项目根目录 `E:\Programs\download\ps2.0\parking-slot-yolov10` 下执行。**

### 实验 1：YOLOv5s（经典基线）

```powershell
& "E:\miniconda3\envs\parking_env\python.exe" scripts/train_entrance_yolo.py `
    --model configs/yolov5s_entrance.yaml `
    --data configs/ps20_entrance.yaml `
    --epochs 80 --batch 8 --imgsz 600 --device 0 `
    --name cmp_yolov5s_80e
```

### 实验 2：YOLOv8s（当前主流）

```powershell
& "E:\miniconda3\envs\parking_env\python.exe" scripts/train_entrance_yolo.py `
    --model configs/yolov8s_entrance.yaml `
    --data configs/ps20_entrance.yaml `
    --epochs 80 --batch 8 --imgsz 600 --device 0 `
    --name cmp_yolov8s_80e
```

### 实验 3：YOLOv10s（NMS-free）

```powershell
& "E:\miniconda3\envs\parking_env\python.exe" scripts/train_entrance_yolo.py `
    --model configs/yolov10s_entrance.yaml `
    --data configs/ps20_entrance.yaml `
    --epochs 80 --batch 8 --imgsz 600 --device 0 `
    --name cmp_yolov10s_80e
```

### 实验 4：YOLOv11s（最新版本）

```powershell
& "E:\miniconda3\envs\parking_env\python.exe" scripts/train_entrance_yolo.py `
    --model configs/yolo11s_entrance.yaml `
    --data configs/ps20_entrance.yaml `
    --epochs 80 --batch 8 --imgsz 600 --device 0 `
    --name cmp_yolo11s_80e
```

## 评估指标提取

训练脚本结束后会自动运行 `paper_metrics` 评估。如果没有自动运行，可手动执行：

```powershell
& "E:\miniconda3\envs\parking_env\python.exe" scripts/evaluate_entrance_paper_metrics.py `
    --weights runs/detect/cmp_yolov5s_80e/weights/best.pt `
    --data configs/ps20_entrance.yaml --split val

& "E:\miniconda3\envs\parking_env\python.exe" scripts/evaluate_entrance_paper_metrics.py `
    --weights runs/detect/cmp_yolov8s_80e/weights/best.pt `
    --data configs/ps20_entrance.yaml --split val

& "E:\miniconda3\envs\parking_env\python.exe" scripts/evaluate_entrance_paper_metrics.py `
    --weights runs/detect/cmp_yolov10s_80e/weights/best.pt `
    --data configs/ps20_entrance.yaml --split val

& "E:\miniconda3\envs\parking_env\python.exe" scripts/evaluate_entrance_paper_metrics.py `
    --weights runs/detect/cmp_yolo11s_80e/weights/best.pt `
    --data configs/ps20_entrance.yaml --split val
```

**注意**：本文方法（YOLOv10s+CBAM）使用的是 evalfix 版本的数据集评估，指标为：
- F1: 87.64%, P: 87.46%, R: 87.83%
- 端点误差: 4.82px, 方向误差: 0.84°, FPS: 36.53

## 训练输出

每个实验完成后，结果保存在：
```
runs/detect/cmp_<模型名>_80e/
├── weights/
│   ├── best.pt          # 最优权重（按 mAP50-95）
│   └── last.pt          # 最后一轮权重
├── results.csv          # 每轮训练指标
├── results.png          # 训练曲线图
├── paper_metrics/       # 论文级评估指标（自动生成）
│   ├── paper_metrics.md
│   ├── paper_metrics.csv
│   └── paper_metrics.json
├── BoxPR_curve.png      # PR 曲线
├── confusion_matrix.png # 混淆矩阵
└── args.yaml            # 训练参数记录
```

## 论文对比表模板

所有实验完成后，整理成论文中的对比表：

### 表 X 不同检测方法在 PS2.0 数据集上的性能对比

| 方法 | 骨干网络 | 年份 | P(%) | R(%) | F1(%) | 端点误差(px) | 方向误差(°) | 类型准确率(%) | FPS |
|------|---------|------|------|------|-------|-------------|------------|-------------|-----|
| DeepPS [1] | DCNN | 2018 | — | — | — | — | — | — | — |
| YOLOv5s | CSPDarknet | 2020 | | | | | | | |
| YOLOv8s | C2f | 2023 | | | | | | | |
| YOLOv10s | C2f+CIB | 2024 | | | | | | | |
| YOLOv11s | C3k2+C2PSA | 2024 | | | | | | | |
| **Ours (YOLOv10s+CBAM)** | **C2fCBAM** | **2024** | **87.46** | **87.83** | **87.64** | **4.82** | **0.84** | **100.00** | **36.53** |

> [1] Zhang L, Huang J, Li X, et al. Vision-Based Parking-Slot Detection: A DCNN-Based Approach and a Large-Scale Benchmark Dataset[J]. IEEE TIP, 2018.
> 注：DeepPS 数据引自原文，因数据集划分不同，仅作参考。

### 表 Y CBAM 注意力机制消融实验

| 配置 | P(%) | R(%) | F1(%) | FPS |
|------|------|------|-------|-----|
| YOLOv10s（基线） | | | | |
| YOLOv10s+CBAM | | | | |
| YOLOv10s+CBAM+Oversample | 87.46 | 87.83 | 87.64 | 36.53 |

## 常见问题

### Q: 训练中途报错 CUDA out of memory
将 `--batch 8` 改为 `--batch 4`，或降低 `--imgsz` 到 512。

### Q: 某个实验训练特别慢
检查是否有其他程序占用 GPU。可以用 `nvidia-smi` 查看 GPU 使用情况。

### Q: 训练完成后 paper_metrics 没有自动生成
手动运行上面的评估命令即可。

### Q: 多个实验可以同时跑吗
不建议。GPU 显存不足以同时训练多个模型，且会互相影响速度。建议依次执行。

### Q: 中途想暂停怎么办
训练脚本支持 `--resume` 参数。先按 `Ctrl+C` 停止，然后在原命令后加 `--resume` 即可从断点继续。

### Q: RT-DETR 可以加吗
RT-DETR 使用 `RTDETRDecoder` 检测头，与当前训练脚本的 `EntranceDetect` 不兼容，需要额外适配。如需添加，请联系指导老师或助教。
