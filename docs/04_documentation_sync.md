# 文档同步规则

## 总规则

凡是改变用户入口、数据格式、产物位置、模型导出、评估指标、部署行为或可视化语义的修改，都必须同步更新文档。文档更新应和代码修改在同一次任务中完成，避免代码和说明分叉。

## 修改类型与同步位置

| 修改内容 | 必须同步 |
| --- | --- |
| 顶层架构、分层、核心与 harness 边界 | `docs/00_harness_architecture.md` |
| 目录结构、数据集格式、`dataset/derived/runs/archive` 写入规则 | `docs/01_project_layout_and_boundaries.md` |
| 脚本名称、环境变量、默认参数、输入输出 | `docs/02_workflow_harness.md` 和对应脚本顶部中文注释 |
| LabelMe 读取、single/multi 类别模式、禁止 YOLO txt 规则 | `docs/01_project_layout_and_boundaries.md` 和 `docs/02_workflow_harness.md` |
| 分类 crop 生成方式、manifest/crop_meta 字段、匹配规则 | `docs/02_workflow_harness.md` |
| 检测评估 overlay 颜色、TP/FP/FN 解释、报告文件 | `docs/02_workflow_harness.md` |
| OpenVINO 导出、metadata 字段、C# 加载规则 | `docs/03_deployment_harness.md` |
| C# 核心库接口、demo 行为、可视化文字/颜色规则 | `docs/03_deployment_harness.md` 和 `deploy/csharp/PollenInferenceDemo/README.md` |
| 最终路线、A/B/NMS/量化实验归档 | `docs/05_final_route_and_experiment_archive.md` |
| 用户最常用命令变化 | 根目录 `README.md` |
| 旧方案退出主流程 | 移入 `archive/`，并从推荐文档中移除或标注为归档 |

## 每次修改前的判断

修改前先判断：

1. 这次改动属于事实源、算法核心、harness、产物还是部署？
2. 是否改变用户要执行的命令？
3. 是否改变 `dataset/`、`derived/`、`runs/`、`deploy/csharp/dist/` 的写入位置？
4. 是否改变模型输入尺寸、类别、metadata、后处理或可视化含义？
5. 是否会影响后续其他部门调用 C# 核心库？

任意答案为“是”，都要同步更新对应文档。

## 完成前检查清单

提交或交付前检查：

- 脚本顶部中文注释是否仍准确描述用途、输入、输出和写入位置。
- 根 README 是否仍能让用户找到常用命令。
- `docs/` 是否说明了新增或修改后的架构边界。
- 文档中是否还引用已经归档或删除的主流程脚本。
- 部署文档是否和 C# demo 当前行为一致。
- 如果引入新产物，是否明确它属于 `dataset/`、`derived/`、`runs/`、`deploy/csharp/dist/` 还是 `archive/`。

## 当前同步状态

截至 2026-06-01，文档已覆盖以下主流程约定：

- LabelMe-only 数据集。
- single/multi 检测训练共享同一批 LabelMe 标注。
- 分类训练使用检测权重离线 crop，不在线调用检测模型。
- 预标注只进入 `derived/prelabels/`。
- 检测评估 overlay 使用 GT 绿框、TP 蓝框、FP 红框。
- C# 部署只保留 OpenVINO。
- 最终路线为 B 方案：YOLO26s 单类别检测 + ResNet34 多分类级联。
- YOLO26s 检测器保持 OpenVINO FP32，检测输出使用模型内部 NMS。
- ResNet34 分类器使用 OpenVINO INT8 PTQ。
- C# demo 支持检测 + 分类级联，demo 不提供 NMS 参数。
- demo 分类名和分数可视化无实色填充。
- A/B 对比、NMS、YOLO26s 量化、ResNet34 INT4 等实验结论归档在 `docs/05_final_route_and_experiment_archive.md`。
