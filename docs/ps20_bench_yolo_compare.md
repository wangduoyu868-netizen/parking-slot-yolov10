# PS2.0 标记点检测：YOLOv5su / YOLOv8s / YOLOv10s 对比摘要

数据来自本机一次完整训练与一次车位线评测流水线（**同 `configs/ps20_marks.yaml`**，训练 **100 epochs**，`results.csv` 取 **最后一行 epoch=100**；车位线指标来自 `runs/detect/ps20_slot_line_benchmark.csv`）。

---

## 1. YOLO 验证集检测指标（`results.csv` 第 100 epoch）

| Run | 模型 | Precision (B) | Recall (B) | mAP50 (B) | mAP50-95 (B) |
|-----|------|-----------------|------------|-----------|---------------|
| `ps20_bench_yolov5su` | YOLOv5su | 0.9974 | 0.9936 | **0.9949** | 0.8884 |
| `ps20_bench_yolov8s` | YOLOv8s | 0.9954 | **0.9967** | **0.9950** | 0.8924 |
| `ps20_bench_yolov10s` | YOLOv10s | 0.9935 | 0.9961 | 0.9948 | **0.8972** |

**简要**：三者 **mAP50** 几乎持平（≈0.9948–0.9950）；**mAP50-95** 以 **YOLOv10s 略高**；**Precision** 以 **v5su 略高**；**Recall** 以 **v8s 略高**。若写论文需强调 **`best.pt` 未必对应最后一 epoch**，可在各自 `results.csv` 里再按 `metrics/mAP50-95(B)` 找最优行对照。

---

## 2. 下游车位线评测（点 → 几何 → 与 GT 线段匹配）

与 `scripts/evaluate_slot_lines.py` 默认协议一致；详见 `ps20_slot_line_benchmark.csv` 中 `labels_dir` 与各 `weights`。

| tag | Precision | Recall | F1 | TP | FP | FN |
|-----|-----------|--------|-----|-----|-----|-----|
| ps20_bench_yolov5su | 0.9806 | 0.9252 | 0.9521 | 1916 | 38 | 155 |
| ps20_bench_yolov8s | 0.9806 | **0.9261** | **0.9526** | **1918** | 38 | **153** |
| ps20_bench_yolov10s | **0.9835** | 0.9218 | 0.9516 | 1909 | **32** | 162 |

**简要**：**v10s** 误检线最少（FP=32，P 最高）；**v8s** 漏检略少（FN=153，R/F1 略好）；三者 F1 差距约 **0.001**，需结合业务更重视 FP 还是 FN。

---

## 3. 训练曲线（不要只看最后一行）

### 3.1 单模型自带图（Ultralytics）

每个 run 目录下通常还有 **`results.png`**（多指标综合）、**`BoxPR_curve.png`** / **`BoxF1_curve.png`** 等，适合在**单模型**内看收敛与 PR/F1 形状。三份图并列打开，可对比「谁更早进入高 mAP 平台」「谁在中段抖动更大」。

### 3.2 三模型叠在同一张坐标里（推荐）

已提供脚本，从三份 **`results.csv`** 读全 epoch，输出一张 **6 子图** PNG：

- **mAP50、mAP50-95**：整体检测强度与定位（IoU 更严）谁更好；常见现象是后期三条线**贴得很近**，差异主要在 **mAP50-95** 或中段爬升速度。  
- **Precision / Recall**：是否某一模型长期偏高 P、另一模型长期偏高 R（与最后车位线里 FP/FN 倾向可对照）。  
- **Train / Val box loss**：谁下降更快、验证集是否更早平台（**过拟合**时 val loss 会抬头，你这次三份可肉眼看 `val/box_loss` 末段是否分叉）。

生成/更新合并图（仓库根目录）：

```bash
python scripts/plot_ps20_bench_curves.py
# 默认输出 docs/ps20_bench_curves.png
python scripts/plot_ps20_bench_curves.py --out runs/detect/ps20_bench_curves.png
```

**结合你本机数据可读出的现象（看曲线时核对）**：

- **YOLOv8s** 第 1 个 epoch 的 **`train/cls_loss`** 往往明显高于 v5su / v10s（分类头与数据分布初期不匹配），但随后快速下降——写报告时可写「初期 cls 损失高、收敛后与其他模型持平」。  
- **mAP50** 三者在约 **20～40 epoch** 后多已进入 **0.99+** 平台，**末段差异很小**，因此**车位线**阶段的 FP/FN 更依赖「几何 + 阈值」对**少量误点**的敏感性，而不仅是 mAP 小数点后第三位。

依赖：`matplotlib`（若报错可 `pip install matplotlib`）。

### 3.3 与 `best.pt` 的关系

曲线最高点未必在第 100 epoch。若要比「验证集最优」而非「最后一轮」，可在各 `results.csv` 里对 **`metrics/mAP50-95(B)`** 或 **`metrics/mAP50(B)`** 取 **argmax** 对应行，与 `weights/best.pt` 的保存逻辑对齐（Ultralytics 默认按 fitness 选 best）。

---

## 4. 路径索引

| 内容 | 路径 |
|------|------|
| YOLOv5su 训练输出 | `runs/detect/ps20_bench_yolov5su/` |
| YOLOv8s 训练输出 | `runs/detect/ps20_bench_yolov8s/` |
| YOLOv10s 训练输出 | `runs/detect/ps20_bench_yolov10s/` |
| 车位线汇总 CSV | `runs/detect/ps20_slot_line_benchmark.csv` |
| 三模型合并曲线图（脚本生成） | `docs/ps20_bench_curves.png` |
| 合并曲线脚本 | `scripts/plot_ps20_bench_curves.py` |

重新生成车位线表：在项目根执行 `python scripts/batch_slot_line_benchmark.py`（权重列表以 `runs/detect/ps20_detector_benchmark.csv` 为准，可按需修改）。

---

*本文件由仓库维护者根据当时 `results.csv` / CSV 整理，复制到论文时请再次核对原始文件。*
