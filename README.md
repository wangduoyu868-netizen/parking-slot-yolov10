# Parking Slot YOLOv10

基于 Ultralytics YOLO 的停车位检测实验项目。当前仓库主要包含三类能力：

1. **入口线检测**：将 PS2.0 的停车位标记点和车位拓扑转换为入口线检测任务，联合预测入口线端点、车身方向和车位类型。
2. **车位框 / 多边形实验**：保留 CNR、polygon、direction 等早期实验脚本与配置，便于对照不同停车位表示方式。
3. **评估与演示工具**：提供入口线严格协议、point-only 协议、GCN baseline 转换评估、Gradio 可视化等脚本。

本仓库不包含数据集、训练权重、论文源码、训练输出和大型第三方 baseline 二进制文件。请自行准备 PS2.0、CNR、PKLot 等数据，并遵守对应数据集和模型许可证。

---

## 1. 仓库结构

```text
parking-slot-yolov10/
  configs/                  # 数据集与模型 YAML 配置
  custom_modules/           # 自定义 YOLO 检测头、损失函数与 GNN 模块
  scripts/                  # 数据转换、训练、评估和演示脚本
  tests/                    # 几何与方向相关的轻量测试
  docs/                     # 复现实验说明与代码文档
  requirements.txt
  README.md
```

以下本地文件和目录默认不进入 Git：

```text
runs/
outputs/
thesis/
thesis_local/
模板/
*.pt
*.pth
*.weights
*.caffemodel
*.onnx
```

---

## 2. 核心任务：停车位入口线检测

最终任务不是直接预测停车位四边形，而是将每个车位表示为：

```text
(x1, y1, x2, y2, body_dx, body_dy, type)
```

其中：

- `(x1, y1), (x2, y2)` 表示入口线两个端点。
- `(body_dx, body_dy)` 表示车位向内部延伸的车身方向向量。
- `type` 表示车位类型，包括直角车位、长入口车位和锐角/钝角车位。

模型基于 YOLOv10 风格检测器，并额外加入入口线预测分支，核心实现位于：

```text
custom_modules/entrance_detect.py
custom_modules/entrance_loss.py
```

主要训练与评估入口：

```text
scripts/convert_ps20_to_entrance_yolo.py
scripts/train_entrance_yolo.py
scripts/train_entrance_yolo_weighted_loss.py
scripts/evaluate_entrance_paper_metrics.py
```

---

## 3. 环境要求

推荐环境：

- Python 3.10
- PyTorch + CUDA，用于训练和 GPU 推理
- Ultralytics YOLO
- OpenCV
- Gradio，用于本地可视化演示

安装基础依赖：

```bash
pip install -r requirements.txt
```

本项目主要在 Windows 环境进行本地可视化，在 Linux/CUDA 服务器进行训练。部分脚本示例中会出现路径参数，换机器运行时需要替换为自己的数据集路径。

---

## 4. 数据准备

### 4.1 PS2.0 转入口线 YOLO 格式

准备 PS2.0 原始图像和 JSON 标注目录后运行：

```bash
python scripts/convert_ps20_to_entrance_yolo.py \
  --image-root /path/to/ps2.0 \
  --label-root /path/to/ps_json_label \
  --out-root /path/to/output/parking_yolov10_entrance
```

转换后会生成：

```text
images/train|val|test/
labels/train|val|test/*.txt      # YOLO 检测框标签
labels/train|val|test/*.eline    # 入口线标签
```

常用配置文件：

```text
configs/ps20_entrance.yaml
configs/ps20_entrance_angled.yaml
configs/ps20_entrance_angled_os5.yaml
configs/ps20_entrance_official.yaml
```

### 4.2 其他实验性转换脚本

仓库中还保留了早期或辅助任务的数据转换脚本：

```text
scripts/convert_ps20_to_yolo.py          # 标记点检测标签
scripts/convert_ps20_to_polygon.py       # 多边形检测标签
scripts/convert_cnr_ext_to_yolo.py       # CNR 车位框标签
```

---

## 5. 模型训练

### 5.1 EntranceDetect 基础训练

示例命令：

```bash
python scripts/train_entrance_yolo.py \
  --data configs/ps20_entrance_angled_os5.yaml \
  --model configs/yolov10s_cbam_entrance.yaml \
  --epochs 160 \
  --batch 16 \
  --imgsz 600 \
  --device 0 \
  --optimizer MuSGD \
  --name ps20_entrance_yolov10s_cbam
```

### 5.2 端点损失权重微调

入口线端点定位对最终停车位几何恢复影响较大，因此可以在基础训练后提高端点损失权重继续微调：

```bash
python scripts/train_entrance_yolo_weighted_loss.py \
  --data configs/ps20_entrance_angled_os5.yaml \
  --model configs/yolov10s_cbam_entrance.yaml \
  --weights /path/to/base/last.pt \
  --epochs 40 \
  --batch 16 \
  --imgsz 600 \
  --device 0 \
  --w-line 24 \
  --name ps20_entrance_wline24_ft40e
```

不同分支的命令行参数可能略有差异。长时间训练前建议先运行 `--help` 或检查脚本顶部参数定义。

### 5.3 相关模型配置

```text
configs/yolov10s_cbam_entrance.yaml
configs/yolov10s_entrance.yaml
configs/yolov8s_entrance.yaml
configs/yolov5s_entrance.yaml
configs/yolo11s_entrance.yaml
```

---

## 6. 模型评估

### 6.1 入口线指标

严格协议同时约束：

- 入口线端点距离
- 车身方向角度误差
- 车位类型判断
- Precision / Recall / F1
- 端点平均误差
- 方向平均误差

运行示例：

```bash
python scripts/evaluate_entrance_paper_metrics.py \
  --weights /path/to/weights.pt \
  --data /path/to/parking_yolov10_entrance_eval \
  --split val \
  --out outputs/eval_strict
```

point-only 协议只评估端点定位，不约束方向：

```bash
python scripts/evaluate_entrance_paper_metrics.py \
  --weights /path/to/weights.pt \
  --data /path/to/parking_yolov10_entrance_eval \
  --split val \
  --point-only \
  --out outputs/eval_point_only
```

### 6.2 Baseline 评估辅助脚本

```text
scripts/evaluate_gcn_entrance_metrics.py
scripts/eval_gnn_slot.py
```

这些脚本用于在条件允许时将图结构或车位连接预测转换到入口线评估协议下。官方预训练权重不包含在本仓库中。

---

## 7. Gradio 可视化演示

### 7.1 入口线检测演示

```bash
python scripts/app_gradio_entrance.py
```

该应用用于可视化：

- 入口线检测结果
- 恢复出的停车位四边形
- 车身方向向量
- 可选的占用状态分类结果

### 7.2 旧版 PS2.0 / CNR 演示

```bash
python scripts/app_gradio.py
python scripts/app_gradio_ps20_cnr.py
```

这两个脚本保留给早期标记点管线和 CNR 车位框管线使用。

---

## 8. 测试

语法检查：

```bash
python -m py_compile scripts/*.py custom_modules/*.py tests/*.py
```

如果安装了 `pytest`，可运行：

```bash
python -m pytest tests
```

当前轻量测试包括：

```text
tests/test_angled_direction_candidates.py
tests/test_entrance_app_geometry.py
tests/test_entrance_direction.py
```

---

## 9. 注意事项

- 本仓库只维护代码和配置，不存放大型权重和原始数据集。
- 不要提交 `runs/`、`outputs/`、`.pt`、`.pth`、`.weights`、`.caffemodel` 或原始数据集。
- PS2.0、CNR、PKLot、DMPR-PS、DeepPS、GCN/PSDet 等资源均有各自许可证和引用要求。
- 部分脚本来自研究实验过程，换机器运行前可能需要修改数据路径和权重路径。

---

## 10. 引用说明

如果使用本仓库，请同时引用原始数据集、YOLO/Ultralytics 相关组件，以及用于对比的 baseline 方法。该仓库本身是工程实现，不重新分发第三方数据集或预训练权重。
