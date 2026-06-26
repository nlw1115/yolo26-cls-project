using System.Collections.Concurrent;
using PollenInferenceDemo.Core;
using PollenInferenceDemo.Services;

namespace PollenInference;

public sealed class PollenDetector : IPollenDetectorInitializer, IPollenDetectorInference, IDisposable
{
    private readonly SemaphoreSlim _initLock = new(1, 1);
    private readonly PollenDetectorOptions _options;
    private readonly List<InferenceService> _workers = [];
    private bool _initialized;
    private bool _disposed;

    public PollenDetector(PollenDetectorOptions? options = null)
    {
        _options = options ?? new PollenDetectorOptions();
    }

    public string Version => PollenAlgorithmInfo.Version;
    public InferenceEngine Engine => _options.Engine;
    public bool IsReady => _initialized && _workers.Count > 0 && _workers.All(x => x.IsModelLoaded);

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (IsReady)
        {
            return;
        }

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (IsReady)
            {
                return;
            }

            Native.NativeDependencyBootstrapper.EnsureLoaded(_options.ExtractionRoot);
            var config = AppConfig.CreateBundled(_options.Engine, _options.WarmupRuns, _options.ExtractionRoot);
            DisposeWorkers();

            var workerCount = Math.Max(1, _options.ParallelWorkerCount);
            for (var i = 0; i < workerCount; i++)
            {
                var worker = new InferenceService();
                try
                {
                    await worker.LoadModelAsync(config, cancellationToken: cancellationToken);
                    _workers.Add(worker);
                }
                catch
                {
                    worker.Dispose();
                    DisposeWorkers();
                    throw;
                }
            }

            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    public Task<PollenDetectionResult> InferAsync(
        byte[] imageBytes,
        float confidenceThreshold,
        CancellationToken cancellationToken = default)
    {
        ValidateImageBytes(imageBytes);
        EnsureInitialized();
        return InferWithWorkerAsync(_workers[0], imageBytes, confidenceThreshold, cancellationToken);
    }

    public async Task<IReadOnlyList<PollenDetectionResult>> InferBatchAsync(
        IReadOnlyList<byte[]> imageBatch,
        float confidenceThreshold,
        CancellationToken cancellationToken = default)
    {
        ValidateBatch(imageBatch);
        EnsureInitialized();

        var results = new PollenDetectionResult[imageBatch.Count];
        for (var i = 0; i < imageBatch.Count; i++)
        {
            results[i] = await InferWithWorkerAsync(_workers[0], imageBatch[i], confidenceThreshold, cancellationToken);
        }

        return results;
    }

    public async Task<IReadOnlyList<PollenDetectionResult>> InferParallelAsync(
        IReadOnlyList<byte[]> imageBatch,
        float confidenceThreshold,
        int degreeOfParallelism,
        CancellationToken cancellationToken = default)
    {
        ValidateBatch(imageBatch);
        EnsureInitialized();

        var effectiveDegree = Math.Clamp(degreeOfParallelism, 1, _workers.Count);
        var queue = new ConcurrentQueue<int>(Enumerable.Range(0, imageBatch.Count));
        var results = new PollenDetectionResult[imageBatch.Count];
        var tasks = Enumerable.Range(0, effectiveDegree)
            .Select(workerIndex => Task.Run(async () =>
            {
                while (queue.TryDequeue(out var index))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    results[index] = await InferWithWorkerAsync(_workers[workerIndex], imageBatch[index], confidenceThreshold, cancellationToken);
                }
            }, cancellationToken))
            .ToArray();

        await Task.WhenAll(tasks);
        return results;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            DisposeWorkers();
        }
        finally
        {
            _disposed = true;
            _initLock.Dispose();
        }
    }

    private static void ValidateBatch(IReadOnlyList<byte[]> imageBatch)
    {
        ArgumentNullException.ThrowIfNull(imageBatch);
        for (var i = 0; i < imageBatch.Count; i++)
        {
            ValidateImageBytes(imageBatch[i]);
        }
    }

    private static void ValidateImageBytes(byte[] imageBytes)
    {
        ArgumentNullException.ThrowIfNull(imageBytes);
        if (imageBytes.Length == 0)
        {
            throw new ArgumentException("Image bytes cannot be empty.", nameof(imageBytes));
        }
    }

    private async Task<PollenDetectionResult> InferWithWorkerAsync(
        InferenceService worker,
        byte[] imageBytes,
        float confidenceThreshold,
        CancellationToken cancellationToken)
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"pollen_{Guid.NewGuid():N}.img");
        await File.WriteAllBytesAsync(tempPath, imageBytes, cancellationToken);

        try
        {
            var (result, elapsedMs) = await worker.PredictAsync(tempPath, confidenceThreshold, cancellationToken);
            try
            {
                return MapResult(result, elapsedMs);
            }
            finally
            {
                result.OriginalImage.Dispose();
            }
        }
        finally
        {
            TryDeleteTempFile(tempPath);
        }
    }

    private void EnsureInitialized()
    {
        ThrowIfDisposed();
        if (!IsReady)
        {
            throw new InvalidOperationException("Detector is not initialized. Call InitializeAsync first.");
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    private void DisposeWorkers()
    {
        foreach (var worker in _workers)
        {
            worker.Dispose();
        }

        _workers.Clear();
        _initialized = false;
    }

    private PollenDetectionResult MapResult(InferenceResult result, long elapsedMs)
    {
        return new PollenDetectionResult
        {
            Engine = Engine,
            AlgorithmVersion = Version,
            ImageWidth = result.OriginalImage.Width,
            ImageHeight = result.OriginalImage.Height,
            ElapsedMs = elapsedMs,
            Targets = result.Detections
                .Select(d => new PollenTarget(
                    d.ClassifierClassId ?? d.ClassId,
                    d.ClassifierClassName ?? d.ClassName,
                    d.ClassifierConfidence ?? d.Confidence,
                    new PollenBoundingBox(
                        d.BoundingBox.Left,
                        d.BoundingBox.Top,
                        d.BoundingBox.Width,
                        d.BoundingBox.Height)))
                .ToArray()
        };
    }

    private static void TryDeleteTempFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Ignore temp cleanup failures to keep inference results flowing.
        }
    }
}
