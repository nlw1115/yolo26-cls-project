# yolo26-cls-projectv3

这是花粉多焦距融合图的 YOLO26 检测 + 二级分类 v3 项目。当前生产路线为 **B 方案：单类别 YOLO26s 检出 + ResNet34 多分类级联**。核心规则是：数据集只维护 LabelMe JSON，训练时直接读取 LabelMe，不落盘 YOLO txt，也不维护 `annotations.jsonl` 派生标签。

当前部署精度策略：YOLO26s 检测器保持 OpenVINO FP32，检测后处理使用模型内部 NMS；ResNet34 分类器使用 OpenVINO INT8 PTQ；C# 侧只使用 OpenVINO。

## 目录

```text
dataset/
  groups/             每个视场的原始多焦距图、融合图、原始 LabelMe JSON
  fused_labelme/      训练用融合图与同名 LabelMe JSON
  classes.yaml        多类别名称顺序
  split.json          train/val 划分
derived/
  detector_predictions/     检测权重全量推理清单
  classification_crops/     离线裁剪后的分类训练数据
  prelabels/                YOLO+SAM2 待复核预标注
runs/                       训练、评估、导出输出
scripts/                    阶段化入口脚本
src/pollen_pipeline/        可复用算法与流程代码
```

## 常用流程

数据集阶段：

```powershell
.\scripts\00_dataset_prepare_fused_labelme.ps1

$env:WEIGHTS="runs\detector\multi_prelabel_fast\weights\best.pt"
.\scripts\01_dataset_prelabel_yolo_sam2.ps1

$env:WEIGHTS="runs\detector\singlecls_aug_mild_box10_seed_26052016\weights\best.pt"
.\scripts\02_dataset_crop_classifier_from_detector.ps1
```

训练阶段：

```bash
bash scripts/10_train_detector_single_prod.sh
CROP_RUN=<crop_run_id> bash scripts/12_train_classifier.sh
```

评估阶段：

```bash
WEIGHTS=runs/detector/<run>/weights/best.pt bash scripts/20_eval_detector.sh
CROP_RUN=<crop_run_id> WEIGHTS=runs/classifier/<run>/weights/best.pt bash scripts/21_eval_classifier.sh
DETECTOR_WEIGHTS=runs/detector/<run>/weights/best.pt CLASSIFIER_WEIGHTS=runs/classifier/<run>/weights/best.pt bash scripts/22_eval_pipeline.sh
```

多类别检测训练仅保留给预标注和历史归档，不再作为当前生产主线入口。

导出阶段：

```powershell
$env:TASK="detector"
$env:WEIGHTS="runs\detector\<run>\weights\best.pt"
$env:CLASS_MODE="single"
.\scripts\30_export_openvino.ps1

$env:TASK="classifier"
$env:WEIGHTS="runs\classifier\<run>\weights\best.pt"
$env:OUTPUT_DIR="runs\export_openvino\classifier_resnet34_fp"
.\scripts\30_export_openvino.ps1

$env:MODEL_XML="runs\export_openvino\classifier_resnet34_fp\model.xml"
$env:OUTPUT_DIR="runs\export_openvino\classifier_resnet34_int8"
.\scripts\32_quantize_classifier_int8.ps1
```

## 关键约定

- 单类别检测训练通过 `--class-mode single` 在内存中合并为 `pollen`。
- 多类别检测训练通过 `--class-mode multi` 保留 `classes.yaml` 类别，用于预标注和历史对比。
- 分类训练不在线调用检测模型；必须先用 `02_dataset_crop_classifier_from_detector.ps1` 离线裁剪。
- 检测 + 分类级联评估会把分类器预测类别写回检测框，再计算与多类别检测可对齐的 P、R、AP50、mAP50-95。
- 最终部署检测器不量化；分类器 INT8 PTQ 由 `scripts/32_quantize_classifier_int8.ps1` 生成。
- 检测评估 overlay 固定颜色：绿色 GT，蓝色 TP，红色 FP，漏检表现为只有绿色框没有蓝框。
## 文档入口

项目架构与后续维护规则见 `docs/`：

- `docs/00_harness_architecture.md`：harness 架构思想和分层边界。
- `docs/01_project_layout_and_boundaries.md`：目录职责、事实源和派生产物边界。
- `docs/02_workflow_harness.md`：数据集、训练、评估、导出阶段脚本说明。
- `docs/03_deployment_harness.md`：OpenVINO 与 C# 核心/demo 部署边界。
- `docs/04_documentation_sync.md`：代码、脚本、部署改动时必须同步更新文档的规则。
- `docs/05_final_route_and_experiment_archive.md`：最终技术路线、速度/精度/NMS/量化实验归档。
