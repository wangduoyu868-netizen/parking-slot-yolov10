# Parking Slot Detection and Occupancy Recognition based on YOLOv10

## 项目简介
本项目基于 YOLOv10 实现停车位标记点检测，并结合几何规则恢复停车位入口线，进一步完成车位占用状态识别。

## 功能
- PS2.0 标签转换为 YOLO 格式
- 标记点检测训练与可视化
- slot line 几何恢复
- slot line 定量评估
- ROI 自动裁剪
- vacant / occupied 二分类
- 端到端结果可视化

## 项目结构
- `scripts/`：核心脚本
- `configs/`：数据集配置文件
- `docs/`：结果示意图

## 主要结果
- Marking point detection:
  - Precision = 0.993
  - Recall = 0.993
  - mAP50 = 0.995
  - mAP50-95 = 0.896

- Slot-line detection:
  - Precision = 0.982
  - Recall = 0.923
  - F1 = 0.951

- Occupancy classification:
  - Accuracy = 0.9807
  - Precision = 0.9729
  - Recall = 0.9729
  - F1 = 0.9729

## 环境
- Python 3.10
- PyTorch
- Ultralytics
- OpenCV
- torchvision

## 说明
本仓库不包含 PS2.0 原始数据集和训练权重。