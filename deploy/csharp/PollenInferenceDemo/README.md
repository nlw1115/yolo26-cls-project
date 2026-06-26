# PollenInferenceDemo

WinForms 推理 demo，仅用于人工查看当前 OpenVINO 检测模型效果。算法核心在上级 `PollenInference/` 类库中，demo 只负责模型选择、图片浏览、结果显示和可视化。

## 当前路线

- 部署只加载 OpenVINO `model.xml/model.bin`。
- 当前生产组合是 YOLO26s 检测器 FP32 + 内部 NMS，ResNet34 分类器 INT8 PTQ。
- 支持检测模型与分类模型级联：先用 YOLO26 检出花粉框，再按训练时同样的黑边 letterbox crop 送入 ResNet/EfficientNet 分类模型。
- YOLO26 输出按模型内部 NMS 后的检测结果处理，demo 不再提供 NMS 参数。
- 类别、输入尺寸等优先从 OpenVINO XML metadata 读取；没有 metadata 时才使用 `assets/classes.txt` 兜底。
- 真值叠加读取同名 LabelMe JSON，不再读取 YOLO txt。
- 默认可视化颜色：检出框蓝色，LabelMe 真值框绿色。

## 本地运行

```powershell
dotnet run -c Release --project deploy/csharp/PollenInferenceDemo/PollenInferenceDemo.csproj
```

正式分发使用 `deploy/csharp/publish.ps1`，输出拆分为算法核心和 demo 两部分。

## 可视化约定

- 检测框和真值框只描边，不做实色填充。
- 加载分类模型后，目标标签优先显示 cls 预测分类名。
- 标签和分数直接绘制文字，不使用实色背景填充，避免遮挡相邻目标。
- 中文分类名使用 Windows 字体绘制，避免 OpenCV 默认字体无法显示中文。

更完整的部署和 demo harness 边界见 `..\..\..\docs\03_deployment_harness.md`。
