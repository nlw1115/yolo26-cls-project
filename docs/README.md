# 项目文档入口

本目录用于维护 `yolo26-cls-projectv3` 的架构、流程、部署和变更同步说明。当前项目按 harness 架构思想组织：核心算法只负责稳定、可复用的能力，脚本、评估、导出、demo 和打包负责把核心能力接入具体场景。

## 文档索引

- [00_harness_architecture.md](00_harness_architecture.md)：项目采用 harness 架构的总体思想、分层和边界。
- [01_project_layout_and_boundaries.md](01_project_layout_and_boundaries.md)：目录职责、事实源、派生产物和禁止事项。
- [02_workflow_harness.md](02_workflow_harness.md)：数据集、训练、评估、导出四类 harness 的入口和产物。
- [03_deployment_harness.md](03_deployment_harness.md)：OpenVINO 导出、C# 核心库、WinForms demo 和打包边界。
- [04_documentation_sync.md](04_documentation_sync.md)：后续代码、脚本、模型流程变更时必须同步更新的文档规则。
- [05_final_route_and_experiment_archive.md](05_final_route_and_experiment_archive.md)：当前最终技术路线，以及 A/B、NMS、量化和速度精度实验归档。

## 阅读顺序

新成员或后续接手人员建议按以下顺序阅读：

1. 先读 `00_harness_architecture.md`，理解为什么项目要把算法核心和运行入口分离。
2. 再读 `01_project_layout_and_boundaries.md`，确认哪些目录能写、哪些目录不能写派生文件。
3. 需要训练或评估时读 `02_workflow_harness.md`。
4. 需要部署、移交算法核心或修改 demo 时读 `03_deployment_harness.md`。
5. 准备改项目结构、脚本参数或输出格式时读 `04_documentation_sync.md`。

## 文档维护原则

- 代码、脚本、部署行为发生变化时，同一提交或同一次修改中同步更新对应文档。
- 面向用户的入口必须在脚本顶部中文注释、根 README 或本目录文档中至少有一处能查到用途、输入和输出。
- 文档只描述当前主流程；旧实验、旧脚本和临时方案应归档到 `archive/`，不要继续写成推荐路线。
