# PollenPipelineBenchmark

This console app benchmarks the final B-route deployment chain:

```text
YOLO26s detector OpenVINO FP32 with internal NMS
  -> bbox crop and letterbox
  -> ResNet34 classifier OpenVINO INT8 PTQ
```

It is intentionally separate from `PollenInferenceDemo` so throughput experiments do not change the WinForms demo or the algorithm library API.

## Scenarios

The benchmark writes `pipeline_benchmark.csv` and `pipeline_benchmark.json` with:

- `serial_chain`: one detector and one classifier, running images sequentially.
- `pipeline_parallel`: detector workers and classifier workers run as two independent stages connected by a bounded queue.
- `detector_batch`: tries OpenVINO model reshape to run detector input batch sizes.
- `classifier_crop_batch`: collects detector crops first, then measures classifier batch inference only.

`detector_workers=2` means two independently loaded detector model instances process different images in parallel. It is not the same as detector model batch size 2.

Each timed scenario also records `avg_cpu_percent`, `max_cpu_percent`, `peak_working_set_mb`, and `peak_private_memory_mb`. CPU percent is normalized by logical processor count, so `100%` means the process used all logical CPU cores during the sampling window.

The detector batch scenario may fail for the current YOLO26s export when internal layers are fixed to batch 1. The failure is recorded as `status=failed` instead of stopping the whole benchmark.

## Example

```powershell
dotnet run -c Release --project deploy/csharp/PollenPipelineBenchmark/PollenPipelineBenchmark.csproj -- `
  --detector "C:\Users\liwei_niu\Downloads\runs_analysis\nms_experiments\best_with_internal_nms_openvino.xml" `
  --classifier "C:\Users\liwei_niu\Downloads\runs_analysis\resnet34_int8_ptq\classifier_resnet34_int8.xml" `
  --images "D:\vibe_coding\yolo26-cls-projectv3\dataset\fused_labelme" `
  --image-limit 50 `
  --repeat 2 `
  --warmup 3 `
  --det-workers 1,2,3 `
  --cls-workers 1,2,4 `
  --batch-sizes 1,2,4,8 `
  --output "D:\vibe_coding\yolo26-cls-projectv3\runs\csharp_pipeline_benchmark"
```

For stable numbers, use a fixed validation image list, run Release builds, keep other CPU-heavy processes closed, and compare `ms_per_image`, `images_per_sec`, and `crops_per_sec` from the CSV.
