using OpenCvSharp;

namespace PollenInferenceDemo.Core;

public sealed record Detection(
    int ClassId,
    string ClassName,
    float Confidence,
    Rect BoundingBox,
    int? ClassifierClassId = null,
    string? ClassifierClassName = null,
    float? ClassifierConfidence = null)
{
    public string DisplayScore => $"{Confidence:F2}";
    public string DisplayLabel => ClassifierConfidence.HasValue
        ? $"{ClassifierClassName} {ClassifierConfidence.Value:F2}"
        : ClassName;
}

public sealed record ClassificationResult(
    int ClassId,
    string ClassName,
    float Confidence
);

public sealed record GroundTruthAnnotation(
    int ClassId,
    string ClassName,
    Rect BoundingBox
);

public sealed class InferenceResult
{
    public required Mat OriginalImage { get; init; }
    public required List<Detection> Detections { get; set; }
}

internal interface IDetector : IDisposable
{
    void Warmup();
    InferenceResult Predict(string imagePath, float confThreshold);
}
