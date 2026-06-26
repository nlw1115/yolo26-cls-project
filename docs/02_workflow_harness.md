# 阶段化流程 Harness

## 总览

当前流程按四个阶段组织：

```mermaid
flowchart LR
  A["数据集阶段 00-02"] --> B["训练阶段 10-12"]
  B --> C["评估阶段 20-23"]
  C --> D["导出阶段 30-31"]
  D --> E["C# 部署与 demo"]
```

所有脚本都是 harness：它们负责参数、路径、默认值和产物组织；实际逻辑应由 `src/pollen_pipeline/` 模块实现。

## 数据集阶段

### `00_dataset_prepare_fused_labelme.ps1`

用途：整理融合图与 LabelMe JSON 到 `dataset/fused_labelme/`。

输入：

- `dataset/groups/` 下每个视场的原始多焦距图、融合图和可能存在的原始 JSON。

输出：

- `dataset/fused_labelme/<sample>.jpg`
- `dataset/fused_labelme/<sample>.json`

写入范围：`dataset/`。该脚本属于事实源整理入口，不能生成 YOLO txt。

### `01_dataset_prelabel_yolo_sam2.ps1`

用途：用多类别粗检测权重预测框中心点，再调用 SAM2 生成待复核 LabelMe 预标注。

输入：

- `dataset/fused_labelme/`
- 多类别检测权重 `WEIGHTS`

输出：

- `derived/prelabels/<run_id>/*.json`

规则：

- 预标注只进入 `derived/prelabels/`。
- 人工复核通过后，才手动合并进 `dataset/fused_labelme/`。
- 该阶段用于提高标注效率，不是生产检测模型的最终训练路线。

### `02_dataset_crop_classifier_from_detector.ps1`

用途：用检测权重对 train/val 融合图全量推理一次，再按预测框离线裁剪分类训练数据。

输入：

- `dataset/fused_labelme/`
- 检测权重 `WEIGHTS`
- 阈值、输入尺寸、匹配 IoU、crop scale 等参数

输出：

- `derived/detector_predictions/<run_id>/manifest.csv`
- `derived/classification_crops/<run_id>/train|val/<class_name>/*.jpg`
- `derived/classification_crops/<run_id>/crop_meta.json`

规则：

- 分类训练不在线调用检测模型。
- 只保存与 GT 匹配成功的预测框 crop。
- 未匹配预测框和漏检写入清单与统计，用于诊断检测权重和阈值。
- crop 数据不和原始数据集混放。

Linux 服务器可用同名 `.sh` 入口：

```bash
WEIGHTS=/content/yolo26-cls-projectv3/runs/detector/<run>/weights/best.pt \
RUN_ID=singlecls_prod_crop_conf035 \
DEVICE=0 \
CONF=0.35 \
IMGSZ=640 \
IOU=0.6 \
MATCH_IOU=0.5 \
CROP_SCALE=1.6 \
IMAGE_SIZE=224 \
bash scripts/02_dataset_crop_classifier_from_detector.sh
```

## 训练阶段

### `10_train_detector_single_prod.sh`

用途：生产化单类别检测训练，用于落地检测、计数和后续分类 crop。

关键规则：

- 从同一批 LabelMe JSON 读取。
- `class_mode=single` 在内存中合并所有类别为 `0=pollen`。
- 不生成单类别 YOLO 标签。

Linux 示例：

```bash
EPOCHS=100 \
BATCH=16 \
DEVICE=0 \
WORKERS=4 \
DATA_REPEAT=4 \
LR0=0.001 \
LRF=0.02 \
BOX=10.0 \
CLS=0.25 \
DFL=2.0 \
WEIGHT_DECAY=0.001 \
WARMUP_BIAS_LR=0.01 \
SEED=26052016 \
RUN_PREFIX=singlecls_prod_100e_seed_26052016 \
AUGMENT=true \
DEGREES=5 \
TRANSLATE=0.03 \
SCALE=0.10 \
FLIPUD=0.5 \
FLIPLR=0.5 \
HSV_S=0.03 \
HSV_V=0.05 \
bash scripts/10_train_detector_single_prod.sh
```

### `11_train_detector_multi_prelabel.sh`

用途：多类别粗检测训练，用于后续预标注。

关键规则：

- `class_mode=multi` 保留 `classes.yaml` 类别顺序。
- 目标是小规模数据快速收敛到可用粗模型，不一定追求最终生产精度。

### 历史 A/B 对比实验

A 多类别检测对比脚本和早期一键 B 方案对比脚本已退出当前主流程，归档到：

- `archive/2026-06-01-final-route-retired-experiments/40_exp_multiclass_detector_gpu.sh`
- `archive/2026-06-01-final-route-retired-experiments/41_exp_single_detector_classifier_gpu.sh`

实验结论、速度、精度、NMS 和量化结果见 `docs/05_final_route_and_experiment_archive.md`。当前主流程不再从 `scripts/` 暴露 A/B 一键对比入口。

### `12_train_classifier.sh`

用途：训练二级分类模型。

输入：

- `CROP_RUN=<crop_run_id>` 指向 `derived/classification_crops/<crop_run_id>/`
- 或显式设置 `CLASSIFIER_DATASET`

规则：

- 只读取离线 crop 数据。
- 不在分类训练阶段调用检测权重。
- 类别顺序来自 crop 目录和训练 metadata。

示例：

```bash
CROP_RUN=singlecls_prod_100e_crop_conf035 \
RUN_NAME=resnet34_from_singlecls_prod_100e_crop_conf035 \
MODEL=resnet34 \
EPOCHS=80 \
BATCH=64 \
LR=0.001 \
DEVICE=0 \
bash scripts/12_train_classifier.sh
```

## 评估阶段

### `20_eval_detector.sh`

用途：评估检测模型并输出人工复核 overlay。

输出：

- `metrics.csv`
- `per_image_metrics.csv`
- `predictions.csv`
- `detection_review.md`
- `overlays/<sample>.jpg`

overlay 规则：

- 绿色框：LabelMe GT。
- 蓝色框：成功匹配 GT 的检出框。
- 红色框：过检 FP。
- 漏检 FN：图上只有绿色 GT 框没有蓝色匹配框，并在报告中记录数量。

### `21_eval_classifier.sh`

用途：评估离线 crop 分类模型。

输入：

- 分类权重 `WEIGHTS`
- `CROP_RUN` 或 `CLASSIFIER_DATASET`

### `22_eval_pipeline.sh`

用途：评估检测 + 分类级联效果。

输入：

- `DETECTOR_WEIGHTS`
- `CLASSIFIER_WEIGHTS`

输出：

- `pipeline_predictions.csv`
- `pipeline_report.json`
- `metrics.csv`
- `per_class_metrics.csv`
- 混淆矩阵 CSV/PNG

指标口径：检测框先经分类模型得到最终类别，再按最终类别进行 class-aware 匹配，输出与多类别检测实验可对齐的 P、R、AP50、mAP50-95；同时在 `pipeline_report.json` 保留检测阶段 P/R 和匹配框分类准确率用于诊断。

### `23_visualize_pipeline.sh`

用途：生成检测 + 分类级联可视化图。

可视化规则：

- 检测框使用描边，不做实色填充。
- 分类名和分数直接画文字，不画实色标签底，避免遮挡其他目标。
- 分类名优先来自分类模型预测；没有加载分类模型时才显示检测类别。

## 导出阶段

### `30_export_openvino.ps1`

用途：将检测或分类 `.pt` 权重导出为 OpenVINO FP16。

规则：

- 检测：`pt -> onnx -> OpenVINO`。
- 分类：导出 ResNet/EfficientNet 到 OpenVINO。
- 类别、输入尺寸、任务类型、预处理参数尽量写入 XML metadata。
- JSON metadata 只作审计记录，不作为部署必需第三方文件。

### `31_package_csharp.ps1`

用途：调用 `deploy/csharp/publish.ps1` 打包 C# 算法核心和 demo。

输出：

- `deploy/csharp/dist/algorithm/PollenInference.dll`
- `deploy/csharp/dist/demo/PollenInferenceDemo.exe`

### `32_quantize_classifier_int8.ps1`

用途：将已经导出的分类器 OpenVINO 中间模型做 INT8 PTQ，生成最终落地分类器。

输入：

- `MODEL_XML` 指向分类器 OpenVINO `model.xml`。
- `dataset/` 中的 LabelMe 事实源。

输出：

- `runs/export_openvino_classifier_int8/model_int8.xml`
- `runs/export_openvino_classifier_int8/model_int8.bin`
- `runs/export_openvino_classifier_int8/quantize_meta.json`

规则：

- 只对分类器做 INT8 PTQ，不改变 YOLO26s 检测器的 FP32 路线。
- 校准样本从 train split 的 LabelMe crop 构建，遵循与分类训练一致的 crop 规则。
- 该脚本是当前主流程的一部分，不是临时实验脚本。

## 验证入口

### `90_smoke_cpu.ps1`

用途：本地 CPU 快速检查项目是否能导入、CLI help 是否可用、LabelMe 数据读取是否正常。

该脚本是最小 harness，不替代完整训练或评估。
