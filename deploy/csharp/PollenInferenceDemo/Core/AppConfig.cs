using System.Text.Json;
using System.Xml.Linq;
using PollenInference.Bundled;

namespace PollenInferenceDemo.Core;

public enum InferenceEngine
{
    OpenVino
}

public sealed class AppConfig
{
    public InferenceEngine Engine { get; init; } = InferenceEngine.OpenVino;
    public string ModelPath { get; init; } = string.Empty;
    public string ClassNamesPath { get; init; } = GetDefaultClassNamesPath();
    public string? ModelInfoPath { get; init; } = GetDefaultModelInfoPath();
    public int WarmupRuns { get; init; } = 10;

    public static string ProjectRoot => FindProjectRoot();

    public static AppConfig CreateBundled(
        InferenceEngine engine = InferenceEngine.OpenVino,
        int warmupRuns = 10,
        string? extractionRoot = null)
    {
        return BundledAssetStore.CreateBundledConfig(engine, warmupRuns, extractionRoot);
    }

    public static string GetDefaultClassNamesPath()
    {
        return Path.Combine(ProjectRoot, "deploy", "csharp", "PollenInferenceDemo", "assets", "classes.txt");
    }

    public static string? GetDefaultModelInfoPath()
    {
        var outputsRoot = Path.Combine(ProjectRoot, "runs");
        var latestOutputModelInfo = FindLatestOutputFile(
            outputsRoot,
            "export_meta.json",
            path => IsOpenVinoExportPath(path));
        if (!string.IsNullOrWhiteSpace(latestOutputModelInfo))
        {
            return latestOutputModelInfo;
        }

        var candidate = Path.Combine(ProjectRoot, "runs", "export_openvino", "export_meta.json");
        return File.Exists(candidate) ? candidate : null;
    }

    public static string GetDefaultModelPath(InferenceEngine engine, string? modelInfoPath)
    {
        var pathFromMeta = TryGetModelPathFromMetadata(engine, modelInfoPath);
        if (!string.IsNullOrWhiteSpace(pathFromMeta))
        {
            return pathFromMeta;
        }

        var runsRoot = Path.Combine(ProjectRoot, "runs");
        var latestOutputPath = FindLatestOutputFile(runsRoot, "model.xml", IsOpenVinoExportPath);
        if (string.IsNullOrWhiteSpace(latestOutputPath))
        {
            latestOutputPath = FindLatestOutputFile(
                Path.Combine(ProjectRoot, "artifacts", "outputs"),
                "model.xml",
                IsOpenVinoExportPath);
        }

        if (!string.IsNullOrWhiteSpace(latestOutputPath))
        {
            return latestOutputPath;
        }

        return Path.Combine(ProjectRoot, "runs", "export_openvino", "model.xml");
    }

    private static string? TryGetModelPathFromMetadata(InferenceEngine engine, string? modelInfoPath)
    {
        if (string.IsNullOrWhiteSpace(modelInfoPath) || !File.Exists(modelInfoPath))
        {
            return null;
        }

        var json = File.ReadAllText(modelInfoPath);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var key = "openvino_xml";

        if (!root.TryGetProperty(key, out var node) || node.ValueKind != JsonValueKind.String)
        {
            return null;
        }

        var raw = node.GetString();
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        return Path.GetFullPath(raw);
    }

    private static string? FindLatestOutputFile(string root, string pattern, Func<string, bool> filter)
    {
        if (!Directory.Exists(root))
        {
            return null;
        }

        string? latestPath = null;
        var latestWriteTime = DateTime.MinValue;

        foreach (var path in Directory.EnumerateFiles(root, pattern, SearchOption.AllDirectories))
        {
            if (!filter(path))
            {
                continue;
            }

            var writeTime = File.GetLastWriteTimeUtc(path);
            if (writeTime <= latestWriteTime)
            {
                continue;
            }

            latestWriteTime = writeTime;
            latestPath = path;
        }

        return latestPath;
    }

    private static bool IsOpenVinoExportPath(string path)
    {
        var normalized = path.Replace(Path.AltDirectorySeparatorChar, Path.DirectorySeparatorChar);
        return normalized.Contains($"{Path.DirectorySeparatorChar}export_openvino{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) ||
               normalized.Contains($"{Path.DirectorySeparatorChar}openvino{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(Path.GetFileName(Path.GetDirectoryName(path)), "export_openvino", StringComparison.OrdinalIgnoreCase);
    }

    private static string FindProjectRoot()
    {
        var dir = AppContext.BaseDirectory;
        var d = new DirectoryInfo(dir);
        while (d is not null)
        {
            if (Directory.Exists(Path.Combine(d.FullName, "scripts")) &&
                Directory.Exists(Path.Combine(d.FullName, "dataset")))
            {
                return d.FullName;
            }

            d = d.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}

public sealed class ModelInfo
{
    public int InputWidth { get; init; } = 640;
    public int InputHeight { get; init; } = 640;
    public string InputName { get; init; } = "images";
    public string OutputName { get; init; } = "output0";
    public string[] ClassNames { get; init; } = ["pollen"];
    public double CropScale { get; init; } = 1.6;

    public static ModelInfo Load(AppConfig config)
    {
        var fromOpenVino = LoadFromOpenVinoMetadata(config);
        if (fromOpenVino is not null)
        {
            return fromOpenVino;
        }

        if (!string.IsNullOrWhiteSpace(config.ModelInfoPath) && File.Exists(config.ModelInfoPath))
        {
            var json = File.ReadAllText(config.ModelInfoPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            var width = TryInt(root, "input_width", 640);
            var height = TryInt(root, "input_height", 640);
            var inputName = TryString(root, "input_name", "images");
            var outputName = TryString(root, "output_name", "output0");
            var classes = TryArray(root, "class_names") ?? LoadClasses(config.ClassNamesPath);
            var cropScale = TryDoubleFromPreprocessing(root, "crop_scale", 1.6);

            return new ModelInfo
            {
                InputWidth = width,
                InputHeight = height,
                InputName = inputName,
                OutputName = outputName,
                ClassNames = classes,
                CropScale = cropScale
            };
        }

        return new ModelInfo
        {
            ClassNames = LoadClasses(config.ClassNamesPath)
        };
    }

    private static int TryInt(JsonElement root, string name, int fallback)
    {
        if (root.TryGetProperty(name, out var p) && p.ValueKind == JsonValueKind.Number && p.TryGetInt32(out var v))
        {
            return v;
        }

        return fallback;
    }

    private static string TryString(JsonElement root, string name, string fallback)
    {
        if (root.TryGetProperty(name, out var p) && p.ValueKind == JsonValueKind.String)
        {
            return p.GetString() ?? fallback;
        }

        return fallback;
    }

    private static string[]? TryArray(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var p) || p.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        var list = new List<string>();
        foreach (var item in p.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(item.GetString()))
            {
                list.Add(item.GetString()!);
            }
        }

        return list.Count > 0 ? list.ToArray() : null;
    }

    private static ModelInfo? LoadFromOpenVinoMetadata(AppConfig config)
    {
        if (string.IsNullOrWhiteSpace(config.ModelPath) ||
            !string.Equals(Path.GetExtension(config.ModelPath), ".xml", StringComparison.OrdinalIgnoreCase) ||
            !File.Exists(config.ModelPath))
        {
            return null;
        }

        try
        {
            var doc = XDocument.Load(config.ModelPath);
            var pipelineNode = doc.Descendants()
                .FirstOrDefault(x => string.Equals(x.Name.LocalName, "pollen_pipeline", StringComparison.OrdinalIgnoreCase));
            if (pipelineNode is null)
            {
                return null;
            }

            var width = TryIntValue(ReadMetadataValue(pipelineNode, "input_width"), 640);
            var height = TryIntValue(ReadMetadataValue(pipelineNode, "input_height"), 640);
            var inputName = ReadMetadataValue(pipelineNode, "input_name") ?? "images";
            var outputName = ReadMetadataValue(pipelineNode, "output_name") ?? "output0";
            var classNames = TryStringArrayValue(ReadMetadataValue(pipelineNode, "class_names")) ?? LoadClasses(config.ClassNamesPath);
            var cropScale = TryCropScale(ReadMetadataValue(pipelineNode, "preprocessing"), 1.6);

            return new ModelInfo
            {
                InputWidth = width,
                InputHeight = height,
                InputName = inputName,
                OutputName = outputName,
                ClassNames = classNames,
                CropScale = cropScale
            };
        }
        catch
        {
            return null;
        }
    }

    private static string? ReadMetadataValue(XElement pipelineNode, string key)
    {
        var node = pipelineNode.Elements()
            .FirstOrDefault(x => string.Equals(x.Name.LocalName, key, StringComparison.OrdinalIgnoreCase));
        return node?.Attribute("value")?.Value;
    }

    private static int TryIntValue(string? raw, int fallback)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return fallback;
        }

        if (int.TryParse(raw, out var parsed))
        {
            return parsed;
        }

        try
        {
            using var doc = JsonDocument.Parse(raw);
            return doc.RootElement.ValueKind == JsonValueKind.Number && doc.RootElement.TryGetInt32(out var jsonValue)
                ? jsonValue
                : fallback;
        }
        catch
        {
            return fallback;
        }
    }

    private static string[]? TryStringArrayValue(string? raw)
    {
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

            var values = doc.RootElement.EnumerateArray()
                .Where(x => x.ValueKind == JsonValueKind.String)
                .Select(x => x.GetString())
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .Select(x => x!)
                .ToArray();
            return values.Length > 0 ? values : null;
        }
        catch
        {
            return [raw];
        }
    }

    private static double TryDoubleFromPreprocessing(JsonElement root, string name, double fallback)
    {
        if (!root.TryGetProperty("preprocessing", out var preprocessing) || preprocessing.ValueKind != JsonValueKind.Object)
        {
            return fallback;
        }

        if (preprocessing.TryGetProperty(name, out var node) && node.ValueKind == JsonValueKind.Number && node.TryGetDouble(out var value))
        {
            return value;
        }

        return fallback;
    }

    private static double TryCropScale(string? raw, double fallback)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return fallback;
        }

        try
        {
            using var doc = JsonDocument.Parse(raw);
            if (doc.RootElement.ValueKind == JsonValueKind.Object &&
                doc.RootElement.TryGetProperty("crop_scale", out var node) &&
                node.ValueKind == JsonValueKind.Number &&
                node.TryGetDouble(out var value))
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

    private static string[] LoadClasses(string classPath)
    {
        if (!File.Exists(classPath))
        {
            return ["pollen"];
        }

        var lines = File.ReadAllLines(classPath)
            .Select(x => x.Trim())
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .ToArray();
        return lines.Length > 0 ? lines : ["pollen"];
    }
}
