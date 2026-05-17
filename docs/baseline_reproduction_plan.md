# 可复现基线实验准备记录

本文档记录用于论文对比实验的外部基线准备状态。优先级为 GCN Parking Slot，其次为 DMPR-PS。

## 1. GCN Parking Slot

代码目录：

```powershell
baselines\gcn-parking-slot
```

来源：

```text
https://github.com/Jiaolong/gcn-parking-slot
```

已完成：

- 已克隆公开代码。
- 已在 `datasets/parking_slot` 下建立数据目录联接：
  - `ps_json_label` -> `E:\谷歌下载\ps_json_label\ps_json_label`
  - `training` -> `E:\Programs\download\ps2.0\training`
  - `testing` -> `E:\Programs\download\ps2.0\testing`
- 已安装缺失依赖：`numba`、`tensorboardX`、`easydict`、`permissive_dict`。
- 已修复新版 `torchvision` 删除 `model_urls` 导致的导入错误。
- 已修复新版 `PyYAML` 需要显式 `Loader` 的兼容问题。
- 已新增 smoke 配置：`baselines\gcn-parking-slot\config\ps_gat_smoke.yaml`。
- 已完成一次数据加载和 GPU 前向检查：
  - 训练样本数：7844
  - batch image shape：`(2, 3, 512, 512)`
  - forward loss 正常输出

关键验证命令：

```powershell
cd E:\Programs\download\ps2.0\parking-slot-yolov10\baselines\gcn-parking-slot
$env:PYTHONPATH=(Resolve-Path .).Path
E:\miniconda3\envs\parking_env\python.exe tools\test.py -c config\ps_gat.yaml -m cache\missing.pth
```

预期结果：能加载验证集 2290 张图，最后因 `cache\missing.pth` 不存在而停止。这说明配置和数据路径已跑通。

下一步：

1. 优先尝试下载作者预训练权重 `Model1`。
2. 如果权重下载困难，则直接训练：

```powershell
cd E:\Programs\download\ps2.0\parking-slot-yolov10\baselines\gcn-parking-slot
$env:PYTHONPATH=(Resolve-Path .).Path
E:\miniconda3\envs\parking_env\python.exe tools\train.py -c config\ps_gat.yaml
```

训练完成后评估：

```powershell
E:\miniconda3\envs\parking_env\python.exe tools\test.py -c config\ps_gat.yaml -m cache\ps_gat\100\models\checkpoint_epoch_200.pth
```

论文用途：

- 作为近年可复现 GNN 基线。
- 与本文 YOLO entrance head 方法对比时，应强调二者评价协议可能不同，优先报告作者脚本输出的 slot detection AP/P/R，再补充本文统一入口线评价结果。

## 2. DMPR-PS

代码目录：

```powershell
baselines\DMPR-PS
```

来源：

```text
https://github.com/Teoge/DMPR-PS
```

已完成：

- 已克隆公开代码。
- `py_compile` 通过。
- 已修复 `visdom` 顶层强制导入问题，改为仅在 `--enable_visdom` 时懒加载。
- 已完成模块导入检查：`config`、`data`、`util`、`DirectionalPointDetector` 可正常导入。

下一步：

1. 准备测试集：

```powershell
cd E:\Programs\download\ps2.0\parking-slot-yolov10\baselines\DMPR-PS
E:\miniconda3\envs\parking_env\python.exe prepare_dataset.py `
  --dataset test `
  --label_directory E:\谷歌下载\ps_json_label\ps_json_label\testing\all `
  --image_directory E:\Programs\download\ps2.0\testing\all `
  --output_directory E:\parking_baselines\dmpr_ps
```

2. 下载作者预训练权重，或重新训练。

3. 点检测评估：

```powershell
E:\miniconda3\envs\parking_env\python.exe evaluate.py `
  --dataset_directory E:\parking_baselines\dmpr_ps\test `
  --detector_weights <DMPR_WEIGHTS>
```

4. 车位检测评估：

```powershell
E:\miniconda3\envs\parking_env\python.exe ps_evaluate.py `
  --label_directory E:\谷歌下载\ps_json_label\ps_json_label\testing\all `
  --image_directory E:\Programs\download\ps2.0\testing\all `
  --detector_weights <DMPR_WEIGHTS>
```

论文用途：

- 作为经典方向标记点回归基线。
- 年份较早，但代码和预训练权重公开，适合支撑“可复现对比”。

## 3. 建议实验顺序

1. 先跑 GCN 预训练模型评估。
2. 若 GCN 权重无法下载，则启动 GCN 本地训练。
3. 再跑 DMPR-PS 预训练模型评估。
4. 最后将外部基线结果与本文方法整理成两类表：
   - 外部基线复现表：GCN、DMPR-PS、本文方法。
   - 本文方法消融表：YOLOv10s、YOLOv10s+CBAM、方向修正、斜车位过采样。
