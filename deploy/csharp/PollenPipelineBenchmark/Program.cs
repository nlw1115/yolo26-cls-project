using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Xml.Linq;
using OpenCvSharp;
using OpenVinoSharp;
using PollenInferenceDemo.Core;

namespace PollenPipelineBenchmark;

internal static class Program
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        try
        {
            var options = BenchmarkOptions.Parse(args);
            Directory.CreateDirectory(options.OutputDir);

            var detectorInfo = ModelMetadata.Load(options.DetectorXml, fallbackInput: 960, fallbackClasses: ["pollen"]);
            var classifierInfo = ModelMetadata.Load(options.ClassifierXml, fallbackInput: 224, fallbackClasses: []);
            var images = ImageSet.Load(options.Images, options.Subset, options.ImageLimit);
            var workItems = Expand(images, options.Repeat).ToArray();

            if (workItems.Length == 0)
            {
                throw new InvalidOperationException("No images found. Use --images with a file or image directory.");
            }

            Console.WriteLine($"detector={options.DetectorXml}");
            Console.WriteLine($"classifier={options.ClassifierXml}");
            Console.WriteLine($"subset={options.Subset}, images={images.Count}, repeat={options.Repeat}, work_items={workItems.Length}");
            Console.WriteLine($"conf={options.ConfidenceThreshold}, nms_iou={options.NmsIouThreshold}");
            Console.WriteLine($"detector_input={detectorInfo.InputWidth}x{detectorInfo.InputHeight}, classifier_input={classifierInfo.InputWidth}x{classifierInfo.InputHeight}");

            var rows = new List<BenchmarkRow>();

            rows.Add(BenchmarkScenarios.RunSerial(options, detectorInfo, classifierInfo, workItems));

            foreach (var detWorkers in options.DetectorWorkers)
            {
                foreach (var clsWorkers in options.ClassifierWorkers)
                {
                    rows.Add(BenchmarkScenarios.RunPipeline(options, detectorInfo, classifierInfo, workItems, detWorkers, clsWorkers));
                }
            }

            foreach (var batchSize in options.BatchSizes)
            {
                rows.Add(BenchmarkScenarios.RunDetectorBatch(options, detectorInfo, workItems, batchSize));
            }

            foreach (var batchSize in options.BatchSizes)
            {
                rows.Add(BenchmarkScenarios.RunClassifierBatch(options, detectorInfo, classifierInfo, workItems, batchSize));
            }

            var csvPath = Path.Combine(options.OutputDir, "pipeline_benchmark.csv");
            BenchmarkRow.WriteCsv(csvPath, rows);
            var jsonPath = Path.Combine(options.OutputDir, "pipeline_benchmark.json");
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);

            Console.WriteLine();
            Console.WriteLine($"written: {csvPath}");
            Console.WriteLine($"written: {jsonPath}");
            Console.WriteLine();
            Console.WriteLine(BenchmarkRow.ToConsoleTable(rows));
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static IEnumerable<string> Expand(IReadOnlyList<string> images, int repeat)
    {
        for (var r = 0; r < repeat; r++)
        {
            foreach (var image in images)
            {
                yield return image;
            }
        }
    }
}

internal sealed record BenchmarkOptions
{
    public required string DetectorXml { get; init; }
    public required string ClassifierXml { get; init; }
    public required string Images { get; init; }
    public string Subset { get; init; } = "all";
    public string OutputDir { get; init; } = Path.Combine("runs", "csharp_pipeline_benchmark");
    public int Repeat { get; init; } = 1;
    public int ImageLimit { get; init; } = 0;
    public float ConfidenceThreshold { get; init; } = 0.25f;
    public float NmsIouThreshold { get; init; } = 0.6f;
    public int WarmupRuns { get; init; } = 3;
    public int[] DetectorWorkers { get; init; } = [1, 2];
    public int[] ClassifierWorkers { get; init; } = [1, 2, 4];
    public int[] BatchSizes { get; init; } = [1, 2, 4, 8];

    public static BenchmarkOptions Parse(string[] args)
    {
        if (args.Contains("--help", StringComparer.OrdinalIgnoreCase) || args.Contains("-h", StringComparer.OrdinalIgnoreCase))
        {
            PrintUsage();
            Environment.Exit(0);
        }

        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (!arg.StartsWith("--", StringComparison.Ordinal))
            {
                continue;
            }

            var key = arg[2..];
            if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                map[key] = "true";
                continue;
            }

            map[key] = args[++i];
        }

        var detector = Required(map, "detector");
        var classifier = Required(map, "classifier");
        var images = Required(map, "images");

        if (!File.Exists(detector))
        {
            throw new FileNotFoundException($"Detector XML not found: {detector}");
        }

        if (!File.Exists(classifier))
        {
            throw new FileNotFoundException($"Classifier XML not found: {classifier}");
        }

        if (!File.Exists(images) && !Directory.Exists(images))
        {
            throw new FileNotFoundException($"Image path not found: {images}");
        }

        return new BenchmarkOptions
        {
            DetectorXml = Path.GetFullPath(detector),
            ClassifierXml = Path.GetFullPath(classifier),
            Images = Path.GetFullPath(images),
            Subset = NormalizeSubset(Value(map, "subset", "all")),
            OutputDir = Path.GetFullPath(Value(map, "output", Path.Combine("runs", "csharp_pipeline_benchmark"))),
            Repeat = Math.Max(1, IntValue(map, "repeat", 1)),
            ImageLimit = Math.Max(0, IntValue(map, "image-limit", 0)),
            ConfidenceThreshold = FloatValue(map, "conf", 0.25f),
            NmsIouThreshold = FloatValue(map, "nms-iou", 0.6f),
            WarmupRuns = Math.Max(0, IntValue(map, "warmup", 3)),
            DetectorWorkers = IntList(map, "det-workers", [1, 2]),
            ClassifierWorkers = IntList(map, "cls-workers", [1, 2, 4]),
            BatchSizes = IntList(map, "batch-sizes", [1, 2, 4, 8])
        };
    }

    private static string Required(IReadOnlyDictionary<string, string> map, string key)
    {
        if (!map.TryGetValue(key, out var value) || string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"Missing --{key}");
        }

        return value;
    }

    private static string Value(IReadOnlyDictionary<string, string> map, string key, string fallback)
    {
        return map.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value) ? value : fallback;
    }

    private static string NormalizeSubset(string raw)
    {
        var value = raw.Trim().ToLowerInvariant();
        if (value is "train" or "val" or "all")
        {
            return value;
        }

        throw new ArgumentException("--subset must be train, val, or all.");
    }

    private static int IntValue(IReadOnlyDictionary<string, string> map, string key, int fallback)
    {
        return map.TryGetValue(key, out var value) && int.TryParse(value, out var parsed) ? parsed : fallback;
    }

    private static float FloatValue(IReadOnlyDictionary<string, string> map, string key, float fallback)
    {
        return map.TryGetValue(key, out var value) && float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : fallback;
    }

    private static int[] IntList(IReadOnlyDictionary<string, string> map, string key, int[] fallback)
    {
        if (!map.TryGetValue(key, out var value) || string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        var parsed = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(x => int.TryParse(x, out var v) ? v : 0)
            .Where(x => x > 0)
            .Distinct()
            .Order()
            .ToArray();
        return parsed.Length > 0 ? parsed : fallback;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
        Usage:
          dotnet run -c Release --project deploy/csharp/PollenPipelineBenchmark/PollenPipelineBenchmark.csproj -- \
            --detector <detector.xml> \
            --classifier <classifier.xml> \
            --images <image file or directory> \
            --subset train \
            --conf 0.25 \
            --nms-iou 0.6 \
            --output runs/csharp_pipeline_benchmark \
            --repeat 3 \
            --det-workers 1,2,3 \
            --cls-workers 1,2,4 \
            --batch-sizes 1,2,4,8

        Notes:
          The detector is expected to be the final YOLO26s OpenVINO FP32 model with internal NMS.
          The classifier is expected to be the final ResNet34 OpenVINO INT8 PTQ model.
        """);
    }
}

internal static class ImageSet
{
    private static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    };

    public static IReadOnlyList<string> Load(string path, string subset, int limit)
    {
        IEnumerable<string> images = File.Exists(path)
            ? [path]
            : Directory.EnumerateFiles(path, "*.*", SearchOption.AllDirectories)
                .Where(x => Extensions.Contains(Path.GetExtension(x)))
                .Order(StringComparer.OrdinalIgnoreCase);

        if (!File.Exists(path) && !string.Equals(subset, "all", StringComparison.OrdinalIgnoreCase))
        {
            var allowed = LoadSplit(Path.GetFullPath(path), subset);
            images = images.Where(x => allowed.Contains(Path.GetFileNameWithoutExtension(x)));
        }

        if (limit > 0)
        {
            images = images.Take(limit);
        }

        return images.Select(Path.GetFullPath).ToArray();
    }

    private static HashSet<string> LoadSplit(string imageRoot, string subset)
    {
        var dir = new DirectoryInfo(imageRoot);
        var candidates = new[]
        {
            Path.Combine(dir.FullName, "split.json"),
            Path.Combine(dir.Parent?.FullName ?? dir.FullName, "split.json")
        };
        var splitPath = candidates.FirstOrDefault(File.Exists)
            ?? throw new FileNotFoundException($"Cannot find split.json for subset '{subset}'.");

        using var doc = JsonDocument.Parse(File.ReadAllText(splitPath, Encoding.UTF8));
        if (!doc.RootElement.TryGetProperty(subset, out var node) || node.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidOperationException($"split.json does not contain array '{subset}': {splitPath}");
        }

        return node.EnumerateArray()
            .Where(x => x.ValueKind == JsonValueKind.String)
            .Select(x => x.GetString())
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Select(x => x!)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
    }
}

internal static class DetectionNms
{
    public static List<Detection> Apply(IReadOnlyList<Detection> detections, float iouThreshold)
    {
        if (detections.Count <= 1 || iouThreshold <= 0.0f || iouThreshold >= 1.0f)
        {
            return detections.OrderByDescending(x => x.Confidence).ToList();
        }

        var ordered = detections.OrderByDescending(x => x.Confidence).ToArray();
        var suppressed = new bool[ordered.Length];
        var keep = new List<Detection>(ordered.Length);
        for (var i = 0; i < ordered.Length; i++)
        {
            if (suppressed[i])
            {
                continue;
            }

            var current = ordered[i];
            keep.Add(current);
            for (var j = i + 1; j < ordered.Length; j++)
            {
                if (!suppressed[j] && IoU(current.BoundingBox, ordered[j].BoundingBox) >= iouThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return keep;
    }

    private static float IoU(Rect a, Rect b)
    {
        var x1 = Math.Max(a.Left, b.Left);
        var y1 = Math.Max(a.Top, b.Top);
        var x2 = Math.Min(a.Right, b.Right);
        var y2 = Math.Min(a.Bottom, b.Bottom);
        var interW = Math.Max(0, x2 - x1);
        var interH = Math.Max(0, y2 - y1);
        var inter = interW * interH;
        if (inter <= 0)
        {
            return 0.0f;
        }

        var union = a.Width * a.Height + b.Width * b.Height - inter;
        return union > 0 ? (float)inter / union : 0.0f;
    }
}

internal sealed record ModelMetadata(
    int InputWidth,
    int InputHeight,
    string[] ClassNames,
    double CropScale)
{
    public static ModelMetadata Load(string xmlPath, int fallbackInput, string[] fallbackClasses)
    {
        try
        {
            var doc = XDocument.Load(xmlPath);
            var node = doc.Descendants().FirstOrDefault(x =>
                string.Equals(x.Name.LocalName, "pollen_pipeline", StringComparison.OrdinalIgnoreCase));
            if (node is null)
            {
                return new ModelMetadata(fallbackInput, fallbackInput, fallbackClasses, 1.7);
            }

            var width = ReadInt(node, "input_width", fallbackInput);
            var height = ReadInt(node, "input_height", fallbackInput);
            var classes = ReadClasses(node, "class_names") ?? fallbackClasses;
            var cropScale = ReadCropScale(node, "preprocessing", 1.7);
            return new ModelMetadata(width, height, classes, cropScale);
        }
        catch
        {
            return new ModelMetadata(fallbackInput, fallbackInput, fallbackClasses, 1.7);
        }
    }

    private static string? ReadValue(XElement node, string key)
    {
        return node.Elements()
            .FirstOrDefault(x => string.Equals(x.Name.LocalName, key, StringComparison.OrdinalIgnoreCase))
            ?.Attribute("value")
            ?.Value;
    }

    private static int ReadInt(XElement node, string key, int fallback)
    {
        var raw = ReadValue(node, key);
        if (int.TryParse(raw, out var value))
        {
            return value;
        }

        try
        {
            using var doc = JsonDocument.Parse(raw ?? "");
            return doc.RootElement.ValueKind == JsonValueKind.Number && doc.RootElement.TryGetInt32(out value)
                ? value
                : fallback;
        }
        catch
        {
            return fallback;
        }
    }

    private static string[]? ReadClasses(XElement node, string key)
    {
        var raw = ReadValue(node, key);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        try
        {
            using var doc = JsonDocument.Parse(raw);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
            {
                return null;
            }

            var names = doc.RootElement.EnumerateArray()
                .Where(x => x.ValueKind == JsonValueKind.String)
                .Select(x => x.GetString())
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .Select(x => x!)
                .ToArray();
            return names.Length > 0 ? names : null;
        }
        catch
        {
            return [raw];
        }
    }

    private static double ReadCropScale(XElement node, string key, double fallback)
    {
        var raw = ReadValue(node, key);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return fallback;
        }

        try
        {
            using var doc = JsonDocument.Parse(raw);
            if (doc.RootElement.ValueKind == JsonValueKind.Object &&
                doc.RootElement.TryGetProperty("crop_scale", out var cropScale) &&
                cropScale.ValueKind == JsonValueKind.Number &&
                cropScale.TryGetDouble(out var value))
            {
                return value;
            }
        }
        catch
        {
            return fallback;
        }

        return fallback;
    }
}

internal sealed class DetectorRunner : IDisposable
{
    private readonly OpenVinoSharp.Core _core;
    private readonly CompiledModel _compiled;
    private readonly InferRequest _request;
    private readonly ModelMetadata _info;
    private readonly float _nmsIouThreshold;

    public DetectorRunner(string xmlPath, ModelMetadata info, int warmupRuns, float nmsIouThreshold)
    {
        _info = info;
        _nmsIouThreshold = nmsIouThreshold;
        _core = new OpenVinoSharp.Core();
        var model = _core.read_model(xmlPath, "");
        _compiled = _core.compile_model(
            model,
            "CPU",
            new Dictionary<string, string> { { "INFERENCE_PRECISION_HINT", "f32" } });
        _request = _compiled.create_infer_request();
        model.Dispose();
        Warmup(warmupRuns);
    }

    public InferenceResult Predict(string imagePath, float confThreshold)
    {
        using var src = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (src.Empty())
        {
            throw new FileNotFoundException($"Failed to read image: {imagePath}");
        }

        var prep = ImagePreprocessor.Prepare(src, _info.InputWidth, _info.InputHeight);
        using var shape = new Shape([1, 3, _info.InputHeight, _info.InputWidth]);
        using var tensor = new Tensor(shape, prep.InputTensor);
        _request.set_input_tensor(tensor);
        _request.infer();

        using var outTensor = _request.get_output_tensor();
        using var outShape = outTensor.get_shape();
        var dims = ShapeText.Parse(outShape.to_string());
        if (dims.Length != 3)
        {
            throw new InvalidOperationException($"Unexpected detector output shape: {outShape.to_string()}");
        }

        var flat = outTensor.get_float_data((int)outTensor.get_size());
        var detections = Postprocess.DecodeYoloOutput(
            flat,
            dims[1],
            dims[2],
            confThreshold,
            prep.ScaleX,
            prep.ScaleY,
            prep.SrcWidth,
            prep.SrcHeight,
            _info.ClassNames.Length > 0 ? _info.ClassNames : ["pollen"]);
        detections = DetectionNms.Apply(detections, _nmsIouThreshold);

        return new InferenceResult
        {
            OriginalImage = src.Clone(),
            Detections = detections
        };
    }

    private void Warmup(int warmupRuns)
    {
        if (warmupRuns <= 0)
        {
            return;
        }

        var data = ImagePreprocessor.CreateWarmupInput(_info.InputWidth, _info.InputHeight);
        using var shape = new Shape([1, 3, _info.InputHeight, _info.InputWidth]);
        using var tensor = new Tensor(shape, data);
        _request.set_input_tensor(tensor);
        for (var i = 0; i < warmupRuns; i++)
        {
            _request.infer();
        }
    }

    public void Dispose()
    {
        _request.Dispose();
        _compiled.Dispose();
        _core.Dispose();
    }
}

internal sealed class ClassifierRunner : IDisposable
{
    private readonly OpenVinoSharp.Core _core;
    private readonly CompiledModel _compiled;
    private readonly InferRequest _request;
    private readonly ModelMetadata _info;

    public ClassifierRunner(string xmlPath, ModelMetadata info, int warmupRuns)
    {
        _info = info;
        _core = new OpenVinoSharp.Core();
        var model = _core.read_model(xmlPath, "");
        _compiled = _core.compile_model(model, "CPU");
        _request = _compiled.create_infer_request();
        model.Dispose();
        Warmup(warmupRuns);
    }

    public ClassificationResult Classify(Mat sourceImage, Detection detection)
    {
        using var crop = ImagePreprocessor.CropLetterboxFromBbox(sourceImage, detection.BoundingBox, _info.CropScale, _info.InputWidth);
        var input = ImagePreprocessor.PrepareClassifier(crop, _info.InputWidth, _info.InputHeight);
        using var shape = new Shape([1, 3, _info.InputHeight, _info.InputWidth]);
        using var tensor = new Tensor(shape, input);
        _request.set_input_tensor(tensor);
        _request.infer();

        using var outTensor = _request.get_output_tensor();
        var logits = outTensor.get_float_data((int)outTensor.get_size());
        return ArgmaxSoftmax(logits, _info.ClassNames);
    }

    private void Warmup(int warmupRuns)
    {
        if (warmupRuns <= 0)
        {
            return;
        }

        var data = ImagePreprocessor.CreateWarmupInput(_info.InputWidth, _info.InputHeight);
        using var shape = new Shape([1, 3, _info.InputHeight, _info.InputWidth]);
        using var tensor = new Tensor(shape, data);
        _request.set_input_tensor(tensor);
        for (var i = 0; i < warmupRuns; i++)
        {
            _request.infer();
        }
    }

    public void Dispose()
    {
        _request.Dispose();
        _compiled.Dispose();
        _core.Dispose();
    }

    public static ClassificationResult ArgmaxSoftmax(float[] logits, string[] classNames)
    {
        if (logits.Length == 0)
        {
            return new ClassificationResult(0, "class_0", 0.0f);
        }

        var max = logits.Max();
        var expSum = 0.0;
        var bestIndex = 0;
        var bestExp = 0.0f;
        for (var i = 0; i < logits.Length; i++)
        {
            var exp = MathF.Exp(logits[i] - max);
            expSum += exp;
            if (exp > bestExp)
            {
                bestExp = exp;
                bestIndex = i;
            }
        }

        var score = expSum > 0 ? (float)(bestExp / expSum) : 0.0f;
        var name = bestIndex >= 0 && bestIndex < classNames.Length ? classNames[bestIndex] : $"class_{bestIndex}";
        return new ClassificationResult(bestIndex, name, score);
    }
}

internal static class BenchmarkScenarios
{
    public static BenchmarkRow RunSerial(
        BenchmarkOptions options,
        ModelMetadata detectorInfo,
        ModelMetadata classifierInfo,
        IReadOnlyList<string> workItems)
    {
        Console.WriteLine("running serial chain...");
        var totalDetections = 0;
        var totalCrops = 0;

        using var detector = new DetectorRunner(options.DetectorXml, detectorInfo, options.WarmupRuns, options.NmsIouThreshold);
        using var classifier = new ClassifierRunner(options.ClassifierXml, classifierInfo, options.WarmupRuns);

        using var resources = ResourceSampler.Start();
        var stopwatch = Stopwatch.StartNew();
        foreach (var image in workItems)
        {
            var result = detector.Predict(image, options.ConfidenceThreshold);
            totalDetections += result.Detections.Count;
            foreach (var detection in result.Detections)
            {
                _ = classifier.Classify(result.OriginalImage, detection);
                totalCrops++;
            }

            result.OriginalImage.Dispose();
        }

        stopwatch.Stop();
        var stats = resources.Stop();
        return BenchmarkRow.Success("serial_chain", workItems.Count, totalCrops, 1, 1, 1, stopwatch.Elapsed.TotalMilliseconds, totalDetections, stats);
    }

    public static BenchmarkRow RunPipeline(
        BenchmarkOptions options,
        ModelMetadata detectorInfo,
        ModelMetadata classifierInfo,
        IReadOnlyList<string> workItems,
        int detectorWorkers,
        int classifierWorkers)
    {
        Console.WriteLine($"running pipeline det_workers={detectorWorkers}, cls_workers={classifierWorkers}...");
        var queue = new BlockingCollection<DetectedImage>(boundedCapacity: Math.Max(8, classifierWorkers * 2));
        var imageIndex = 0;
        var totalDetections = 0;
        var totalCrops = 0;
        var errors = new ConcurrentQueue<Exception>();
        using var ready = new CountdownEvent(detectorWorkers + classifierWorkers);
        using var startGate = new ManualResetEventSlim(false);
        var stopwatch = new Stopwatch();

        var consumers = Enumerable.Range(0, classifierWorkers).Select(workerId => Task.Run(() =>
        {
            _ = workerId;
            var readySignaled = false;
            try
            {
                using var classifier = new ClassifierRunner(options.ClassifierXml, classifierInfo, options.WarmupRuns);
                ready.Signal();
                readySignaled = true;
                startGate.Wait();
                foreach (var item in queue.GetConsumingEnumerable())
                {
                    try
                    {
                        foreach (var detection in item.Result.Detections)
                        {
                            var ignored = classifier.Classify(item.Result.OriginalImage, detection);
                            _ = ignored;
                            Interlocked.Increment(ref totalCrops);
                        }
                    }
                    finally
                    {
                        item.Result.OriginalImage.Dispose();
                    }
                }
            }
            catch (Exception ex)
            {
                if (!readySignaled)
                {
                    ready.Signal();
                }

                errors.Enqueue(ex);
            }
        })).ToArray();

        var producers = Enumerable.Range(0, detectorWorkers).Select(workerId => Task.Run(() =>
        {
            _ = workerId;
            var readySignaled = false;
            try
            {
                using var detector = new DetectorRunner(options.DetectorXml, detectorInfo, options.WarmupRuns, options.NmsIouThreshold);
                ready.Signal();
                readySignaled = true;
                startGate.Wait();
                while (true)
                {
                    var current = Interlocked.Increment(ref imageIndex) - 1;
                    if (current >= workItems.Count)
                    {
                        break;
                    }

                    var result = detector.Predict(workItems[current], options.ConfidenceThreshold);
                    Interlocked.Add(ref totalDetections, result.Detections.Count);
                    queue.Add(new DetectedImage(current, result));
                }
            }
            catch (Exception ex)
            {
                if (!readySignaled)
                {
                    ready.Signal();
                }

                errors.Enqueue(ex);
            }
        })).ToArray();

        ready.Wait();
        if (!errors.IsEmpty)
        {
            queue.CompleteAdding();
            startGate.Set();
            Task.WaitAll(producers.Concat(consumers).ToArray());
            return BenchmarkRow.Failed(
                "pipeline_parallel",
                workItems.Count,
                totalCrops,
                detectorWorkers,
                classifierWorkers,
                1,
                0,
                totalDetections,
                string.Join(" | ", errors.Select(x => x.Message).Take(3)));
        }

        using var resources = ResourceSampler.Start();
        stopwatch.Start();
        startGate.Set();
        Task.WaitAll(producers);
        queue.CompleteAdding();
        Task.WaitAll(consumers);
        stopwatch.Stop();
        var stats = resources.Stop();

        if (!errors.IsEmpty)
        {
            return BenchmarkRow.Failed(
                "pipeline_parallel",
                workItems.Count,
                totalCrops,
                detectorWorkers,
                classifierWorkers,
                1,
                stopwatch.Elapsed.TotalMilliseconds,
                totalDetections,
                string.Join(" | ", errors.Select(x => x.Message).Take(3)),
                stats);
        }

        return BenchmarkRow.Success(
            "pipeline_parallel",
            workItems.Count,
            totalCrops,
            detectorWorkers,
            classifierWorkers,
            1,
            stopwatch.Elapsed.TotalMilliseconds,
            totalDetections,
            stats);
    }

    public static BenchmarkRow RunDetectorBatch(
        BenchmarkOptions options,
        ModelMetadata detectorInfo,
        IReadOnlyList<string> workItems,
        int batchSize)
    {
        Console.WriteLine($"running detector batch={batchSize}...");
        var totalDetections = 0;

        try
        {
            using var runner = new DetectorBatchRunner(options.DetectorXml, detectorInfo, batchSize, options.WarmupRuns, options.NmsIouThreshold);
            using var resources = ResourceSampler.Start();
            var stopwatch = Stopwatch.StartNew();
            foreach (var chunk in workItems.Chunk(batchSize))
            {
                totalDetections += runner.Predict(chunk, options.ConfidenceThreshold);
            }

            stopwatch.Stop();
            var stats = resources.Stop();
            return BenchmarkRow.Success("detector_batch", workItems.Count, 0, 1, 0, batchSize, stopwatch.Elapsed.TotalMilliseconds, totalDetections, stats);
        }
        catch (Exception ex)
        {
            return BenchmarkRow.Failed("detector_batch", workItems.Count, 0, 1, 0, batchSize, 0, totalDetections, ex.Message);
        }
    }

    public static BenchmarkRow RunClassifierBatch(
        BenchmarkOptions options,
        ModelMetadata detectorInfo,
        ModelMetadata classifierInfo,
        IReadOnlyList<string> workItems,
        int batchSize)
    {
        Console.WriteLine($"running classifier crop batch={batchSize}...");

        try
        {
            var crops = CollectClassifierInputs(options, detectorInfo, classifierInfo, workItems);
            using var runner = new ClassifierBatchRunner(options.ClassifierXml, classifierInfo, batchSize, options.WarmupRuns);
            using var resources = ResourceSampler.Start();
            var stopwatch = Stopwatch.StartNew();
            var classified = 0;
            foreach (var chunk in crops.Chunk(batchSize))
            {
                classified += runner.Classify(chunk);
            }

            stopwatch.Stop();
            var stats = resources.Stop();
            return BenchmarkRow.Success("classifier_crop_batch", workItems.Count, classified, 0, 1, batchSize, stopwatch.Elapsed.TotalMilliseconds, crops.Count, stats);
        }
        catch (Exception ex)
        {
            return BenchmarkRow.Failed("classifier_crop_batch", workItems.Count, 0, 0, 1, batchSize, 0, 0, ex.Message);
        }
    }

    private static List<float[]> CollectClassifierInputs(
        BenchmarkOptions options,
        ModelMetadata detectorInfo,
        ModelMetadata classifierInfo,
        IReadOnlyList<string> workItems)
    {
        var crops = new List<float[]>();
        using var detector = new DetectorRunner(options.DetectorXml, detectorInfo, options.WarmupRuns, options.NmsIouThreshold);

        foreach (var image in workItems)
        {
            var result = detector.Predict(image, options.ConfidenceThreshold);
            try
            {
                foreach (var detection in result.Detections)
                {
                    using var crop = ImagePreprocessor.CropLetterboxFromBbox(result.OriginalImage, detection.BoundingBox, classifierInfo.CropScale, classifierInfo.InputWidth);
                    crops.Add(ImagePreprocessor.PrepareClassifier(crop, classifierInfo.InputWidth, classifierInfo.InputHeight));
                }
            }
            finally
            {
                result.OriginalImage.Dispose();
            }
        }

        return crops;
    }
}

internal sealed class DetectorBatchRunner : IDisposable
{
    private readonly OpenVinoSharp.Core _core;
    private readonly CompiledModel _compiled;
    private readonly InferRequest _request;
    private readonly ModelMetadata _info;
    private readonly int _batchSize;
    private readonly float _nmsIouThreshold;

    public DetectorBatchRunner(string xmlPath, ModelMetadata info, int batchSize, int warmupRuns, float nmsIouThreshold)
    {
        _info = info;
        _batchSize = batchSize;
        _nmsIouThreshold = nmsIouThreshold;
        _core = new OpenVinoSharp.Core();
        var model = _core.read_model(xmlPath, "");
        using var shape = new Shape([batchSize, 3, info.InputHeight, info.InputWidth]);
        var partialShape = new PartialShape(shape);
        model.reshape(partialShape);
        _compiled = _core.compile_model(
            model,
            "CPU",
            new Dictionary<string, string> { { "INFERENCE_PRECISION_HINT", "f32" } });
        _request = _compiled.create_infer_request();
        model.Dispose();
        Warmup(warmupRuns);
    }

    public int Predict(IReadOnlyList<string> imagePaths, float confThreshold)
    {
        if (imagePaths.Count == 0)
        {
            return 0;
        }

        var realCount = imagePaths.Count;
        var paths = realCount == _batchSize
            ? imagePaths
            : imagePaths.Concat(Enumerable.Repeat(imagePaths[^1], _batchSize - realCount)).ToArray();

        var input = new float[_batchSize * 3 * _info.InputHeight * _info.InputWidth];
        var preps = new ImagePreprocessor.PrepResult[_batchSize];
        for (var i = 0; i < paths.Count; i++)
        {
            using var src = Cv2.ImRead(paths[i], ImreadModes.Color);
            if (src.Empty())
            {
                throw new FileNotFoundException($"Failed to read image: {paths[i]}");
            }

            var prep = ImagePreprocessor.Prepare(src, _info.InputWidth, _info.InputHeight);
            preps[i] = prep;
            Array.Copy(prep.InputTensor, 0, input, i * prep.InputTensor.Length, prep.InputTensor.Length);
        }

        using var shape = new Shape([_batchSize, 3, _info.InputHeight, _info.InputWidth]);
        using var tensor = new Tensor(shape, input);
        _request.set_input_tensor(tensor);
        _request.infer();

        using var outTensor = _request.get_output_tensor();
        using var outShape = outTensor.get_shape();
        var dims = ShapeText.Parse(outShape.to_string());
        var flat = outTensor.get_float_data((int)outTensor.get_size());
        return DecodeBatchedDetections(flat, dims, preps, realCount, confThreshold);
    }

    private int DecodeBatchedDetections(
        float[] flat,
        int[] dims,
        ImagePreprocessor.PrepResult[] preps,
        int imageCount,
        float confThreshold)
    {
        if (dims.Length == 3 && dims[0] == _batchSize)
        {
            var rows = dims[1];
            var cols = dims[2];
            var perImage = rows * cols;
            var total = 0;
            for (var i = 0; i < imageCount; i++)
            {
                var slice = new float[perImage];
                Array.Copy(flat, i * perImage, slice, 0, perImage);
                var detections = Postprocess.DecodeYoloOutput(
                    slice,
                    rows,
                    cols,
                    confThreshold,
                    preps[i].ScaleX,
                    preps[i].ScaleY,
                    preps[i].SrcWidth,
                    preps[i].SrcHeight,
                    _info.ClassNames.Length > 0 ? _info.ClassNames : ["pollen"]);
                total += DetectionNms.Apply(detections, _nmsIouThreshold).Count;
            }

            return total;
        }

        if (dims.Length == 3 && imageCount == 1)
        {
            var detections = Postprocess.DecodeYoloOutput(
                flat,
                dims[1],
                dims[2],
                confThreshold,
                preps[0].ScaleX,
                preps[0].ScaleY,
                preps[0].SrcWidth,
                preps[0].SrcHeight,
                _info.ClassNames.Length > 0 ? _info.ClassNames : ["pollen"]);
            return DetectionNms.Apply(detections, _nmsIouThreshold).Count;
        }

        throw new InvalidOperationException($"Unsupported batched detector output shape: {{{string.Join(",", dims)}}}");
    }

    private void Warmup(int warmupRuns)
    {
        if (warmupRuns <= 0)
        {
            return;
        }

        var data = new float[_batchSize * 3 * _info.InputWidth * _info.InputHeight];
        var one = ImagePreprocessor.CreateWarmupInput(_info.InputWidth, _info.InputHeight);
        for (var i = 0; i < _batchSize; i++)
        {
            Array.Copy(one, 0, data, i * one.Length, one.Length);
        }

        using var shape = new Shape([_batchSize, 3, _info.InputHeight, _info.InputWidth]);
        using var tensor = new Tensor(shape, data);
        _request.set_input_tensor(tensor);
        for (var i = 0; i < warmupRuns; i++)
        {
            _request.infer();
        }
    }

    public void Dispose()
    {
        _request.Dispose();
        _compiled.Dispose();
        _core.Dispose();
    }

}

internal sealed class ClassifierBatchRunner : IDisposable
{
    private readonly OpenVinoSharp.Core _core;
    private readonly CompiledModel _compiled;
    private readonly InferRequest _request;
    private readonly ModelMetadata _info;
    private readonly int _batchSize;

    public ClassifierBatchRunner(string xmlPath, ModelMetadata info, int batchSize, int warmupRuns)
    {
        _info = info;
        _batchSize = batchSize;
        _core = new OpenVinoSharp.Core();
        var model = _core.read_model(xmlPath, "");
        using var shape = new Shape([batchSize, 3, info.InputHeight, info.InputWidth]);
        var partialShape = new PartialShape(shape);
        model.reshape(partialShape);
        _compiled = _core.compile_model(model, "CPU");
        _request = _compiled.create_infer_request();
        model.Dispose();
        Warmup(warmupRuns);
    }

    public int Classify(IReadOnlyList<float[]> inputs)
    {
        if (inputs.Count == 0)
        {
            return 0;
        }

        var realCount = inputs.Count;
        var inputLen = 3 * _info.InputHeight * _info.InputWidth;
        var data = new float[_batchSize * inputLen];
        for (var i = 0; i < _batchSize; i++)
        {
            var source = inputs[Math.Min(i, inputs.Count - 1)];
            Array.Copy(source, 0, data, i * inputLen, inputLen);
        }

        using var shape = new Shape([_batchSize, 3, _info.InputHeight, _info.InputWidth]);
        using var tensor = new Tensor(shape, data);
        _request.set_input_tensor(tensor);
        _request.infer();

        using var outTensor = _request.get_output_tensor();
        using var outShape = outTensor.get_shape();
        var dims = ShapeText.Parse(outShape.to_string());
        var logits = outTensor.get_float_data((int)outTensor.get_size());

        var classes = InferClassCount(dims, logits.Length);
        if (classes <= 0)
        {
            throw new InvalidOperationException($"Unsupported classifier output shape: {{{string.Join(",", dims)}}}");
        }

        for (var i = 0; i < realCount; i++)
        {
            var slice = new float[classes];
            Array.Copy(logits, i * classes, slice, 0, classes);
            _ = ClassifierRunner.ArgmaxSoftmax(slice, _info.ClassNames);
        }

        return realCount;
    }

    private int InferClassCount(int[] dims, int length)
    {
        if (dims.Length == 2 && dims[0] == _batchSize)
        {
            return dims[1];
        }

        if (length % _batchSize == 0)
        {
            return length / _batchSize;
        }

        return -1;
    }

    private void Warmup(int warmupRuns)
    {
        if (warmupRuns <= 0)
        {
            return;
        }

        var one = ImagePreprocessor.CreateWarmupInput(_info.InputWidth, _info.InputHeight);
        var data = new float[_batchSize * one.Length];
        for (var i = 0; i < _batchSize; i++)
        {
            Array.Copy(one, 0, data, i * one.Length, one.Length);
        }

        using var shape = new Shape([_batchSize, 3, _info.InputHeight, _info.InputWidth]);
        using var tensor = new Tensor(shape, data);
        _request.set_input_tensor(tensor);
        for (var i = 0; i < warmupRuns; i++)
        {
            _request.infer();
        }
    }

    public void Dispose()
    {
        _request.Dispose();
        _compiled.Dispose();
        _core.Dispose();
    }
}

internal sealed record DetectedImage(int Index, InferenceResult Result);

internal sealed record BenchmarkRow(
    string Scenario,
    int Images,
    int Crops,
    int DetectorWorkers,
    int ClassifierWorkers,
    int BatchSize,
    double TotalMs,
    double MsPerImage,
    double ImagesPerSecond,
    double CropsPerSecond,
    int Detections,
    double AvgCpuPercent,
    double MaxCpuPercent,
    double PeakWorkingSetMb,
    double PeakPrivateMemoryMb,
    string Status,
    string Error)
{
    public static BenchmarkRow Success(
        string scenario,
        int images,
        int crops,
        int detectorWorkers,
        int classifierWorkers,
        int batchSize,
        double totalMs,
        int detections,
        ResourceStats? stats = null)
    {
        return Create(scenario, images, crops, detectorWorkers, classifierWorkers, batchSize, totalMs, detections, "ok", "", stats);
    }

    public static BenchmarkRow Failed(
        string scenario,
        int images,
        int crops,
        int detectorWorkers,
        int classifierWorkers,
        int batchSize,
        double totalMs,
        int detections,
        string error,
        ResourceStats? stats = null)
    {
        return Create(scenario, images, crops, detectorWorkers, classifierWorkers, batchSize, totalMs, detections, "failed", error, stats);
    }

    private static BenchmarkRow Create(
        string scenario,
        int images,
        int crops,
        int detectorWorkers,
        int classifierWorkers,
        int batchSize,
        double totalMs,
        int detections,
        string status,
        string error,
        ResourceStats? stats)
    {
        var seconds = totalMs > 0 ? totalMs / 1000.0 : 0.0;
        stats ??= ResourceStats.Empty;
        return new BenchmarkRow(
            scenario,
            images,
            crops,
            detectorWorkers,
            classifierWorkers,
            batchSize,
            totalMs,
            images > 0 ? totalMs / images : 0.0,
            seconds > 0 ? images / seconds : 0.0,
            seconds > 0 ? crops / seconds : 0.0,
            detections,
            stats.AvgCpuPercent,
            stats.MaxCpuPercent,
            stats.PeakWorkingSetMb,
            stats.PeakPrivateMemoryMb,
            status,
            error);
    }

    public static void WriteCsv(string path, IReadOnlyList<BenchmarkRow> rows)
    {
        using var writer = new StreamWriter(path, false, Encoding.UTF8);
        writer.WriteLine("scenario,images,crops,detector_workers,classifier_workers,batch_size,total_ms,ms_per_image,images_per_sec,crops_per_sec,detections,avg_cpu_percent,max_cpu_percent,peak_working_set_mb,peak_private_memory_mb,status,error");
        foreach (var row in rows)
        {
            writer.WriteLine(string.Join(",",
                Csv(row.Scenario),
                row.Images.ToString(CultureInfo.InvariantCulture),
                row.Crops.ToString(CultureInfo.InvariantCulture),
                row.DetectorWorkers.ToString(CultureInfo.InvariantCulture),
                row.ClassifierWorkers.ToString(CultureInfo.InvariantCulture),
                row.BatchSize.ToString(CultureInfo.InvariantCulture),
                row.TotalMs.ToString("F3", CultureInfo.InvariantCulture),
                row.MsPerImage.ToString("F3", CultureInfo.InvariantCulture),
                row.ImagesPerSecond.ToString("F3", CultureInfo.InvariantCulture),
                row.CropsPerSecond.ToString("F3", CultureInfo.InvariantCulture),
                row.Detections.ToString(CultureInfo.InvariantCulture),
                row.AvgCpuPercent.ToString("F2", CultureInfo.InvariantCulture),
                row.MaxCpuPercent.ToString("F2", CultureInfo.InvariantCulture),
                row.PeakWorkingSetMb.ToString("F1", CultureInfo.InvariantCulture),
                row.PeakPrivateMemoryMb.ToString("F1", CultureInfo.InvariantCulture),
                Csv(row.Status),
                Csv(row.Error)));
        }
    }

    public static string ToConsoleTable(IReadOnlyList<BenchmarkRow> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("scenario                  status   detW clsW batch images crops total_ms ms/img img/s crop/s avgCPU maxCPU peakWS");
        foreach (var row in rows)
        {
            sb.AppendLine(string.Format(
                CultureInfo.InvariantCulture,
                "{0,-25} {1,-8} {2,4} {3,4} {4,5} {5,6} {6,5} {7,8:F1} {8,6:F1} {9,5:F2} {10,6:F2} {11,6:F1} {12,6:F1} {13,6:F0}",
                row.Scenario.Length > 25 ? row.Scenario[..25] : row.Scenario,
                row.Status,
                row.DetectorWorkers,
                row.ClassifierWorkers,
                row.BatchSize,
                row.Images,
                row.Crops,
                row.TotalMs,
                row.MsPerImage,
                row.ImagesPerSecond,
                row.CropsPerSecond,
                row.AvgCpuPercent,
                row.MaxCpuPercent,
                row.PeakWorkingSetMb));
            if (!string.IsNullOrWhiteSpace(row.Error))
            {
                sb.AppendLine($"  error: {row.Error}");
            }
        }

        return sb.ToString();
    }

    private static string Csv(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
        {
            return value;
        }

        return $"\"{value.Replace("\"", "\"\"")}\"";
    }
}

internal static class ShapeText
{
    public static int[] Parse(string shapeText)
    {
        var chars = shapeText.Where(c => char.IsDigit(c) || c == ',' || c == '-').ToArray();
        var cleaned = new string(chars).Trim(',');
        if (string.IsNullOrWhiteSpace(cleaned))
        {
            return [];
        }

        return cleaned.Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(x => int.Parse(x.Trim(), CultureInfo.InvariantCulture))
            .ToArray();
    }
}

internal sealed record ResourceStats(
    double AvgCpuPercent,
    double MaxCpuPercent,
    double PeakWorkingSetMb,
    double PeakPrivateMemoryMb)
{
    public static ResourceStats Empty { get; } = new(0.0, 0.0, 0.0, 0.0);
}

internal sealed class ResourceSampler : IDisposable
{
    private readonly Process _process;
    private readonly CancellationTokenSource _cts = new();
    private readonly object _gate = new();
    private readonly int _processorCount;
    private readonly Task _task;
    private DateTime _lastWall;
    private TimeSpan _lastCpu;
    private double _cpuSum;
    private double _maxCpu;
    private double _peakWorkingSetMb;
    private double _peakPrivateMemoryMb;
    private int _samples;
    private bool _stopped;

    private ResourceSampler(TimeSpan interval)
    {
        _process = Process.GetCurrentProcess();
        _processorCount = Math.Max(1, Environment.ProcessorCount);
        _process.Refresh();
        _lastWall = DateTime.UtcNow;
        _lastCpu = _process.TotalProcessorTime;
        SampleMemoryOnly();
        _task = Task.Run(() => Loop(interval, _cts.Token));
    }

    public static ResourceSampler Start(int intervalMs = 100)
    {
        return new ResourceSampler(TimeSpan.FromMilliseconds(Math.Max(20, intervalMs)));
    }

    public ResourceStats Stop()
    {
        lock (_gate)
        {
            if (_stopped)
            {
                return BuildStats();
            }

            _stopped = true;
        }

        _cts.Cancel();
        try
        {
            _task.Wait();
        }
        catch (AggregateException ex) when (ex.InnerExceptions.All(x => x is TaskCanceledException))
        {
        }

        Sample();
        lock (_gate)
        {
            return BuildStats();
        }
    }

    public void Dispose()
    {
        if (!_stopped)
        {
            _ = Stop();
        }

        _cts.Dispose();
        _process.Dispose();
    }

    private async Task Loop(TimeSpan interval, CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            await Task.Delay(interval, token);
            Sample();
        }
    }

    private void Sample()
    {
        try
        {
            _process.Refresh();
            var nowWall = DateTime.UtcNow;
            var nowCpu = _process.TotalProcessorTime;
            var wallMs = (nowWall - _lastWall).TotalMilliseconds;
            var cpuMs = (nowCpu - _lastCpu).TotalMilliseconds;
            var cpuPercent = wallMs > 0 ? Math.Max(0.0, cpuMs / wallMs / _processorCount * 100.0) : 0.0;
            var workingSetMb = _process.WorkingSet64 / 1024.0 / 1024.0;
            var privateMb = _process.PrivateMemorySize64 / 1024.0 / 1024.0;

            lock (_gate)
            {
                _lastWall = nowWall;
                _lastCpu = nowCpu;
                _cpuSum += cpuPercent;
                _maxCpu = Math.Max(_maxCpu, cpuPercent);
                _peakWorkingSetMb = Math.Max(_peakWorkingSetMb, workingSetMb);
                _peakPrivateMemoryMb = Math.Max(_peakPrivateMemoryMb, privateMb);
                _samples++;
            }
        }
        catch
        {
        }
    }

    private void SampleMemoryOnly()
    {
        var workingSetMb = _process.WorkingSet64 / 1024.0 / 1024.0;
        var privateMb = _process.PrivateMemorySize64 / 1024.0 / 1024.0;
        lock (_gate)
        {
            _peakWorkingSetMb = Math.Max(_peakWorkingSetMb, workingSetMb);
            _peakPrivateMemoryMb = Math.Max(_peakPrivateMemoryMb, privateMb);
        }
    }

    private ResourceStats BuildStats()
    {
        return new ResourceStats(
            _samples > 0 ? _cpuSum / _samples : 0.0,
            _maxCpu,
            _peakWorkingSetMb,
            _peakPrivateMemoryMb);
    }
}
