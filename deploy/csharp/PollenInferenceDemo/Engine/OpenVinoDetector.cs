using OpenCvSharp;
using OpenVinoSharp;
using PollenInferenceDemo.Core;

namespace PollenInferenceDemo.Engine;

internal sealed class OpenVinoDetector : IDetector
{
    private readonly AppConfig _config;
    private readonly ModelInfo _modelInfo;
    private readonly OpenVinoSharp.Core _core;
    private readonly CompiledModel _compiled;
    private readonly InferRequest _inferRequest;
    private readonly int _inputW;
    private readonly int _inputH;

    public OpenVinoDetector(AppConfig config, ModelInfo modelInfo)
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

    public InferenceResult Predict(string imagePath, float confThreshold)
    {
        using var src = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (src.Empty())
        {
            throw new FileNotFoundException($"Failed to read image: {imagePath}");
        }

        var prep = ImagePreprocessor.Prepare(src, _inputW, _inputH);
        using var shape = new Shape([1, 3, _inputH, _inputW]);
        using var tensor = new Tensor(shape, prep.InputTensor);
        _inferRequest.set_input_tensor(tensor);
        _inferRequest.infer();

        using var outTensor = _inferRequest.get_output_tensor();
        using var outShape = outTensor.get_shape();
        var outElems = (int)outTensor.get_size();
        var dims = ParseShape(outShape.to_string());
        if (dims.Length != 3)
        {
            throw new InvalidOperationException($"Unexpected output shape: {outShape.to_string()}");
        }
        var rows = dims[1];
        var cols = dims[2];
        var flat = outTensor.get_float_data(outElems);

        var detections = Postprocess.DecodeYoloOutput(
            flat,
            rows,
            cols,
            confThreshold,
            prep.ScaleX,
            prep.ScaleY,
            prep.SrcWidth,
            prep.SrcHeight,
            _modelInfo.ClassNames);

        return new InferenceResult
        {
            OriginalImage = src.Clone(),
            Detections = detections
        };
    }

    public void Dispose()
    {
        _inferRequest.Dispose();
        _compiled.Dispose();
        _core.Dispose();
    }

    private static int[] ParseShape(string shapeText)
    {
        // Format like "{1,300,6}"
        var chars = shapeText.Where(c => char.IsDigit(c) || c == ',' || c == '-').ToArray();
        var cleaned = new string(chars).Trim(',');
        if (string.IsNullOrWhiteSpace(cleaned))
        {
            return [];
        }
        return cleaned.Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(x => int.Parse(x.Trim()))
            .ToArray();
    }
}
