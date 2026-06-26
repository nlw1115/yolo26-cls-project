using PollenInferenceDemo.Core;

namespace PollenInference;

public interface IPollenDetectorInitializer
{
    Task InitializeAsync(CancellationToken cancellationToken = default);
}

public interface IPollenDetectorInference
{
    Task<PollenDetectionResult> InferAsync(byte[] imageBytes, float confidenceThreshold, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<PollenDetectionResult>> InferBatchAsync(IReadOnlyList<byte[]> imageBatch, float confidenceThreshold, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<PollenDetectionResult>> InferParallelAsync(IReadOnlyList<byte[]> imageBatch, float confidenceThreshold, int degreeOfParallelism, CancellationToken cancellationToken = default);
}

public sealed class PollenDetectorOptions
{
    public InferenceEngine Engine { get; init; } = InferenceEngine.OpenVino;
    public int WarmupRuns { get; init; } = 10;
    public int ParallelWorkerCount { get; init; } = 1;
    public string? ExtractionRoot { get; init; }
}

public sealed record PollenBoundingBox(
    int Left,
    int Top,
    int Width,
    int Height);

public sealed record PollenTarget(
    int ClassId,
    string ClassName,
    float Confidence,
    PollenBoundingBox BoundingBox);

public sealed class PollenDetectionResult
{
    public required InferenceEngine Engine { get; init; }
    public required string AlgorithmVersion { get; init; }
    public required int ImageWidth { get; init; }
    public required int ImageHeight { get; init; }
    public required long ElapsedMs { get; init; }
    public required IReadOnlyList<PollenTarget> Targets { get; init; }
}

public static class PollenAlgorithmInfo
{
    public static string Version => Bundled.BundledAssetStore.GetAlgorithmVersion();
}
