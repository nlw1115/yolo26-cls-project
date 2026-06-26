using PollenInferenceDemo.Core;

namespace PollenInferenceDemo.Engine;

internal static class DetectorFactory
{
    public static IDetector Create(AppConfig config)
    {
        if (string.IsNullOrWhiteSpace(config.ModelPath) || !File.Exists(config.ModelPath))
        {
            throw new FileNotFoundException($"Model file not found: {config.ModelPath}");
        }

        var modelInfo = ModelInfo.Load(config);
        return config.Engine switch
        {
            InferenceEngine.OpenVino => new OpenVinoDetector(config, modelInfo),
            _ => throw new NotSupportedException($"Unsupported engine: {config.Engine}")
        };
    }
}
