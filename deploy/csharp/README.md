# C# 部署目录

- `PollenInference/`：算法核心库，只承载 OpenVINO 推理和后续 pipeline 逻辑，方便移交给其他部门调用。
- `PollenInferenceDemo/`：WinForms 演示界面，只调用算法核心库，不放业务算法。
- `publish.ps1`：打包 `dist/algorithm` 与 `dist/demo`。

当前 Python 导出入口是 `scripts/30_export_openvino.ps1` 和 `scripts/32_quantize_classifier_int8.ps1`。技术路线是 `pt -> onnx -> OpenVINO`，ONNX 只作为导出中间产物，C# 部署只使用 OpenVINO `model.xml/model.bin`。当前生产路线固定为 YOLO26s 检测器 OpenVINO FP32 + 内部 NMS，ResNet34 分类器 OpenVINO INT8 PTQ。
## 架构文档

C# 部署遵循项目 harness 架构：`PollenInference/` 是可移交算法核心，`PollenInferenceDemo/` 是人工演示和复核 harness。完整说明见 `..\..\docs\03_deployment_harness.md`。
