# Parking Slot YOLOv10

基于 Ultralytics YOLO 的停车位检测实验项目。当前仓库主要包含三类能力：

1. **Entrance line detection**：将 PS2.0 的停车位标记点和车位拓扑转换为入口线检测任务，联合预测入口线端点、车身方向和车位类型。
2. **Parking slot box / polygon experiments**：保留 CNR、polygon、direction 等早期实验脚本与配置，便于对照不同表示方式。
3. **Evaluation and demo tools**：提供入口线严格协议、point-only 协议、GCN baseline 转换评估、Gradio 可视化等脚本。

本仓库不包含数据集、训练权重、论文源码、训练输出和大型第三方 baseline 二进制文件。请自行准备 PS2.0、CNR、PKLot 等数据，并遵守对应数据集和模型许可证。

---

## 1. Repository Structure

```text
parking-slot-yolov10/
  configs/                  # Dataset and model YAML configs
  custom_modules/           # Custom YOLO heads, losses, and GNN modules
  scripts/                  # Data conversion, training, evaluation, and demo scripts
  tests/                    # Lightweight geometry and direction tests
  docs/                     # Reproduction notes and code explanations
  requirements.txt
  README.md
```

Ignored local artifacts include:

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

## 2. Main Task: Entrance Line Detection

The final task formulation does not directly predict a parking slot quadrilateral. Instead, each parking slot is represented by:

```text
(x1, y1, x2, y2, body_dx, body_dy, type)
```

where:

- `(x1, y1), (x2, y2)` are the two entrance-line endpoints.
- `(body_dx, body_dy)` is the body/depth direction vector.
- `type` distinguishes right-angle, long-entrance, and acute/obtuse slots.

The model uses a YOLOv10-style detector with an additional entrance prediction branch implemented in:

```text
custom_modules/entrance_detect.py
custom_modules/entrance_loss.py
```

The key training entry points are:

```text
scripts/convert_ps20_to_entrance_yolo.py
scripts/train_entrance_yolo.py
scripts/train_entrance_yolo_weighted_loss.py
scripts/evaluate_entrance_paper_metrics.py
```

---

## 3. Environment

Recommended environment:

- Python 3.10
- PyTorch with CUDA for training
- Ultralytics YOLO
- OpenCV
- Gradio for local demos

Install minimal dependencies:

```bash
pip install -r requirements.txt
```

The project was developed mainly on Windows for local visualization and on Linux/CUDA servers for training. Some scripts use absolute paths in examples; replace them with your own dataset paths before running.

---

## 4. Data Preparation

### 4.1 PS2.0 to Entrance-Line YOLO Format

Prepare the PS2.0 raw image and JSON label directories, then run:

```bash
python scripts/convert_ps20_to_entrance_yolo.py \
  --image-root /path/to/ps2.0 \
  --label-root /path/to/ps_json_label \
  --out-root /path/to/output/parking_yolov10_entrance
```

The converter creates:

```text
images/train|val|test/
labels/train|val|test/*.txt      # YOLO detection boxes
labels/train|val|test/*.eline    # entrance-line labels
```

Typical configs:

```text
configs/ps20_entrance.yaml
configs/ps20_entrance_angled.yaml
configs/ps20_entrance_angled_os5.yaml
configs/ps20_entrance_official.yaml
```

### 4.2 Other Experimental Converters

Earlier or auxiliary representations are still available:

```text
scripts/convert_ps20_to_yolo.py          # marking-point style labels
scripts/convert_ps20_to_polygon.py       # polygon-style labels
scripts/convert_cnr_ext_to_yolo.py       # CNR slot-box labels
```

---

## 5. Training

### 5.1 EntranceDetect Training

Example:

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

Endpoint-loss fine-tuning can be run with:

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

The exact CLI options may differ by branch; run `--help` or inspect the script header before launching long jobs.

### 5.2 Related Configs

```text
configs/yolov10s_cbam_entrance.yaml
configs/yolov10s_entrance.yaml
configs/yolov8s_entrance.yaml
configs/yolov5s_entrance.yaml
configs/yolo11s_entrance.yaml
```

---

## 6. Evaluation

### 6.1 Entrance-Line Metrics

Strict protocol:

- Endpoint distance threshold
- Direction angle threshold
- Type accuracy
- Precision / Recall / F1
- Endpoint mean error
- Direction error

Run:

```bash
python scripts/evaluate_entrance_paper_metrics.py \
  --weights /path/to/weights.pt \
  --data /path/to/parking_yolov10_entrance_eval \
  --split val \
  --out outputs/eval_strict
```

Point-only protocol ignores the direction constraint and evaluates endpoint localization:

```bash
python scripts/evaluate_entrance_paper_metrics.py \
  --weights /path/to/weights.pt \
  --data /path/to/parking_yolov10_entrance_eval \
  --split val \
  --point-only \
  --out outputs/eval_point_only
```

### 6.2 Baseline Evaluation Helpers

```text
scripts/evaluate_gcn_entrance_metrics.py
scripts/eval_gnn_slot.py
```

These scripts are provided to convert graph/slot predictions to an entrance-line-style evaluation where possible. Official pretrained weights are not included.

---

## 7. Gradio Demo

### 7.1 Entrance-Line Demo

```bash
python scripts/app_gradio_entrance.py
```

This app visualizes:

- Detected entrance lines
- Recovered parking-slot quadrilaterals
- Body direction vectors
- Optional occupancy status when a classifier is provided

### 7.2 Legacy PS2.0 / CNR Demo

```bash
python scripts/app_gradio.py
python scripts/app_gradio_ps20_cnr.py
```

These two scripts are kept for the earlier marking-point and CNR box pipelines.

---

## 8. Tests

Run syntax checks:

```bash
python -m py_compile scripts/*.py custom_modules/*.py tests/*.py
```

Run tests if `pytest` is installed:

```bash
python -m pytest tests
```

Current lightweight tests cover:

```text
tests/test_angled_direction_candidates.py
tests/test_entrance_app_geometry.py
tests/test_entrance_direction.py
```

---

## 9. Important Notes

- This repository is code-only. Large model weights and datasets should be stored outside Git or released separately.
- Do not commit `runs/`, `outputs/`, `.pt`, `.pth`, `.weights`, `.caffemodel`, or raw datasets.
- The PS2.0, CNR, PKLot, DMPR-PS, DeepPS, and GCN/PSDet resources have their own licenses and citation requirements.
- Some scripts were used for research experiments and may require path edits before running on a new machine.

---

## 10. Citation

If you use this repository, cite the original datasets, YOLO/Ultralytics components, and any baseline method you compare against. This repository itself is an engineering implementation and does not redistribute third-party data or pretrained weights.
