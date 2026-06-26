# 目录职责与边界

## 顶层目录

```text
dataset/    人工维护的事实源
derived/    可重建的中间产物
runs/       训练、评估、导出和可视化结果
scripts/    用户常用阶段化 harness 入口
src/        Python 算法核心和流程模块
deploy/     C# 部署核心、demo harness 和打包脚本
configs/    稳定参数说明或后续 preset
weights/    可选的本地初始权重或外部权重落点
archive/    退出主流程的旧内容
docs/       架构、流程和同步规则
```

## `dataset/` 事实源契约

推荐布局：

```text
dataset/
  groups/
    Group-0001/
      focus_0.jpg
      focus_1.jpg
      focus_2.jpg
      fused.jpg
  fused_labelme/
    Group-0001.jpg
    Group-0001.json
  classes.yaml
  split.json
```

规则：

- `groups/` 保存每个视场的原始多焦距图和融合图，用于回溯融合质量。
- `fused_labelme/` 保存训练所需融合图和同名 LabelMe JSON。
- `classes.yaml` 是多类别类别顺序的唯一来源。
- `split.json` 是 train/val 划分的唯一来源。
- 不在 `dataset/` 写 YOLO txt、`annotations*.jsonl`、检测预测、分类 crop 或预标注临时结果。

## `derived/` 中间产物契约

`derived/` 存放可以由 `dataset/`、权重和参数重新生成的内容：

```text
derived/
  detector_predictions/<run_id>/manifest.csv
  classification_crops/<run_id>/train|val/<class_name>/*.jpg
  prelabels/<run_id>/*.json
```

规则：

- `detector_predictions/` 记录检测权重对融合图的一次性全量推理结果。
- `classification_crops/` 只保存检测框与 LabelMe GT 匹配成功的离线 crop。
- `prelabels/` 只保存待人工复核的预标注 LabelMe JSON，不自动覆盖事实源。
- 需要复现实验时优先用 `run_id`、权重路径、阈值和 `crop_meta.json` 回溯。

## `runs/` 结果契约

`runs/` 保存训练、评估、导出和可视化结果：

```text
runs/
  detector/<run_name>/
  classifier/<run_name>/
  detector_eval/<run_name>/
  classifier_eval/<run_name>/
  pipeline_eval/<run_name>/
  pipeline_visualize/<run_name>/
  export_openvino/<run_name>/
```

规则：

- 训练输出进入 `runs/detector/` 或 `runs/classifier/`。
- 评估输出进入对应 eval 目录。
- OpenVINO 导出输出进入 `runs/export_openvino/`。
- 可以删除后重建的实验产物不要混入 `dataset/`。

## `scripts/` harness 入口

脚本按阶段编号：

- `00-02`：数据集阶段。
- `10-12`：训练阶段。
- `20-23`：评估阶段。
- `30-31`：导出和打包阶段。
- `90`：快速验证入口。

脚本职责：

- 暴露用户最常改的环境变量。
- 设置稳定默认值。
- 调用 `python -m pollen_pipeline.<module>` 或 C# 打包脚本。
- 明确写入 `dataset/derived/runs/archive` 中的哪个目录。

脚本不应复制 Python 模块中的算法逻辑。

## `src/pollen_pipeline/` Python 核心模块

主要模块职责：

| 模块 | 职责 |
| --- | --- |
| `labelme_dataset.py` | 读取 LabelMe，支持 rectangle/polygon/mask 外接框，处理 single/multi 类别模式。 |
| `detector_train.py` | 将 LabelMe 直接适配到 YOLO26/Ultralytics 训练，不生成 YOLO 标签。 |
| `crop_classifier_from_detector.py` | 使用检测权重全量推理并离线裁剪分类训练数据。 |
| `classifier_train.py` | 训练离线 crop 分类模型。 |
| `detector_eval.py` | 检测评估、指标 CSV、review 报告和 overlay 图。 |
| `classifier_eval.py` | 分类模型评估。 |
| `pipeline_eval.py` | 检测 + 分类级联评估。 |
| `visualize_pipeline.py` | 检测 + 分类级联可视化。 |
| `export_openvino.py` | 检测模型导出为 OpenVINO 并写入 metadata。 |
| `export_classifier_openvino.py` | 分类模型导出为 OpenVINO 并写入 metadata。 |

## `deploy/` 部署边界

```text
deploy/csharp/
  PollenInference/      算法核心库
  PollenInferenceDemo/  WinForms demo harness
  publish.ps1           打包 harness
  dist/                 打包输出
```

规则：

- `PollenInference/` 是后续移交给其他部门调用的核心算法库。
- `PollenInferenceDemo/` 只用于人工查看、模型选择、图片浏览和结果可视化。
- demo 可以有 UI 状态和可视化选项，但核心推理逻辑应留在核心库或共享核心模块。
- 打包输出拆为 `dist/algorithm` 和 `dist/demo`，避免 demo 与算法核心混在一起交付。

## 禁止事项

- 不把 YOLO txt 或 `annotations*.jsonl` 重新引入主流程。
- 不为了单类别训练批量改写 LabelMe 标签。
- 不在分类训练 dataloader 中每次读取图片时调用检测模型。
- 不把预标注直接覆盖到 `dataset/fused_labelme/`。
- 不在 C# demo 中重新引入 NMS 参数；YOLO26 输出按 end-to-end 检测结果处理。
- 不把 UI/demo 逻辑写进可移交算法核心库。
