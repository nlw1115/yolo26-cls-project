using System.Diagnostics;
using PollenInferenceDemo.Core;
using PollenInferenceDemo.Engine;

namespace PollenInferenceDemo.Services;

public sealed class InferenceService : IDisposable
{
    private readonly SemaphoreSlim _lock = new(1, 1);
    private IDetector? _detector;
    private OpenVinoClassifier? _classifier;

    public bool IsModelLoaded => _detector is not null;
    public bool IsClassifierLoaded => _classifier is not null;
    public InferenceEngine LoadedEngine { get; private set; } = InferenceEngine.OpenVino;
    public string LoadedModelPath { get; private set; } = string.Empty;
    public string LoadedClassifierPath { get; private set; } = string.Empty;

    public async Task LoadModelAsync(AppConfig config, AppConfig? classifierConfig = null, CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            _detector?.Dispose();
            _detector = null;
            _classifier?.Dispose();
            _classifier = null;

            var detector = await Task.Run(() => DetectorFactory.Create(config), cancellationToken);
            _detector = detector;
            LoadedEngine = config.Engine;
            LoadedModelPath = config.ModelPath;
            LoadedClassifierPath = string.Empty;

            if (classifierConfig is not null)
            {
                var classifierInfo = ModelInfo.Load(classifierConfig);
                _classifier = await Task.Run(() => new OpenVinoClassifier(classifierConfig, classifierInfo), cancellationToken);
                LoadedClassifierPath = classifierConfig.ModelPath;
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task<(InferenceResult Result, long ElapsedMs)> PredictAsync(
        string imagePath,
        float confThreshold,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (_detector is null)
            {
                throw new InvalidOperationException("Model is not loaded.");
            }

            var detector = _detector;
            var classifier = _classifier;
            var sw = Stopwatch.StartNew();
            var result = await Task.Run(() => detector.Predict(imagePath, confThreshold), cancellationToken);
            if (classifier is not null && result.Detections.Count > 0)
            {
                result.Detections = result.Detections
                    .Select(d =>
                    {
                        var cls = classifier.Classify(result.OriginalImage, d);
                        return d with
                        {
                            ClassifierClassId = cls.ClassId,
                            ClassifierClassName = cls.ClassName,
                            ClassifierConfidence = cls.Confidence
                        };
                    })
                    .ToList();
            }
            sw.Stop();

            return (result, sw.ElapsedMilliseconds);
        }
        finally
        {
            _lock.Release();
        }
    }

    public void Dispose()
    {
        _detector?.Dispose();
        _classifier?.Dispose();
        _lock.Dispose();
    }
}
