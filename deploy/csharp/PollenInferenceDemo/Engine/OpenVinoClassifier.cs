using OpenCvSharp;
using OpenVinoSharp;
using PollenInferenceDemo.Core;

namespace PollenInferenceDemo.Engine;

internal sealed class OpenVinoClassifier : IDisposable
{
    private readonly AppConfig _config;
    private readonly ModelInfo _modelInfo;
    private readonly OpenVinoSharp.Core _core;
    private readonly CompiledModel _compiled;
    private readonly InferRequest _inferRequest;
    private readonly int _inputW;
    private readonly int _inputH;

    public OpenVinoClassifier(AppConfig config, ModelInfo modelInfo)
    {
        _config = config;
        _modelInfo = modelInfo;
        _inputW = modelInfo.InputWidth;
        _inputH = modelInfo.InputHeight;

        _core = new OpenVinoSharp.Core();
        var model = _core.read_model(config.ModelPath, "");
        _compiled = _core.compile_model(model, "CPU");
        _inferRequest = _compiled.create_infer_request();
        model.Dispose();
        Warmup();
    }

    public IReadOnlyList<string> ClassNames => _modelInfo.ClassNames;
    public double CropScale => _modelInfo.CropScale;

    public void Warmup()
    {
        var data = ImagePreprocessor.CreateWarmupInput(_inputW, _inputH);
        using var shape = new Shape([1, 3, _inputH, _inputW]);
        using var tensor = new Tensor(shape, data);
        _inferRequest.set_input_tensor(tensor);

        for (var i = 0; i < _config.WarmupRuns; i++)
        {
            _inferRequest.infer();
        }
    }

    public ClassificationResult Classify(Mat sourceImage, Detection detection)
    {
        using var crop = ImagePreprocessor.CropLetterboxFromBbox(sourceImage, detection.BoundingBox, CropScale, _inputW);
        var input = ImagePreprocessor.PrepareClassifier(crop, _inputW, _inputH);
        using var shape = new Shape([1, 3, _inputH, _inputW]);
        using var tensor = new Tensor(shape, input);
        _inferRequest.set_input_tensor(tensor);
        _inferRequest.infer();

        using var outTensor = _inferRequest.get_output_tensor();
        var outElems = (int)outTensor.get_size();
        var logits = outTensor.get_float_data(outElems);
        var probs = Softmax(logits);
        var bestIndex = 0;
        for (var i = 1; i < probs.Length; i++)
        {
            if (probs[i] > probs[bestIndex])
            {
                bestIndex = i;
            }
        }

        var className = bestIndex >= 0 && bestIndex < _modelInfo.ClassNames.Length
            ? _modelInfo.ClassNames[bestIndex]
            : $"class_{bestIndex}";
        return new ClassificationResult(bestIndex, className, probs[bestIndex]);
    }

    public void Dispose()
    {
        _inferRequest.Dispose();
        _compiled.Dispose();
        _core.Dispose();
    }

    private static float[] Softmax(float[] logits)
    {
        if (logits.Length == 0)
        {
            return [];
        }

        var max = logits.Max();
        var exp = new float[logits.Length];
        var sum = 0.0;
        for (var i = 0; i < logits.Length; i++)
        {
            exp[i] = MathF.Exp(logits[i] - max);
            sum += exp[i];
        }

        if (sum <= 0)
        {
            return exp;
        }

        for (var i = 0; i < exp.Length; i++)
        {
            exp[i] = (float)(exp[i] / sum);
        }

        return exp;
    }
}
