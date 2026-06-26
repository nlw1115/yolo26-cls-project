using System.Reflection;
using PollenInferenceDemo.Core;

namespace PollenInference.Bundled;

internal static class BundledAssetStore
{
    private const string ClassesResourceName = "PollenInference.BundledAssets.classes.txt";
    private const string OpenVinoXmlResourceName = "PollenInference.BundledAssets.model.openvino.xml";
    private const string OpenVinoBinResourceName = "PollenInference.BundledAssets.model.openvino.bin";

    private static readonly Lock SyncRoot = new();
    private static BundledAssetLayout? _cachedLayout;

    public static AppConfig CreateBundledConfig(InferenceEngine engine, int warmupRuns, string? extractionRoot)
    {
        var layout = EnsureAssets(extractionRoot);
        return new AppConfig
        {
            Engine = engine,
            ModelPath = layout.OpenVinoXmlPath,
            ClassNamesPath = layout.ClassesPath,
            ModelInfoPath = null,
            WarmupRuns = warmupRuns
        };
    }

    public static string GetAlgorithmVersion()
    {
        var assembly = typeof(BundledAssetStore).Assembly;
        return assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? assembly.GetName().Version?.ToString()
            ?? "0.0.0";
    }

    private static BundledAssetLayout EnsureAssets(string? extractionRoot)
    {
        lock (SyncRoot)
        {
            var resolvedRoot = ResolveExtractionRoot(extractionRoot);
            if (_cachedLayout is not null && string.Equals(_cachedLayout.RootPath, resolvedRoot, StringComparison.OrdinalIgnoreCase))
            {
                return _cachedLayout;
            }

            Directory.CreateDirectory(resolvedRoot);
            var modelsRoot = Path.Combine(resolvedRoot, "models");
            var openVinoRoot = Path.Combine(modelsRoot, "openvino");
            Directory.CreateDirectory(modelsRoot);
            Directory.CreateDirectory(openVinoRoot);

            var classesPath = Path.Combine(resolvedRoot, "classes.txt");
            var openVinoXmlPath = Path.Combine(openVinoRoot, "model.xml");
            var openVinoBinPath = Path.Combine(openVinoRoot, "model.bin");

            WriteResourceIfNeeded(ClassesResourceName, classesPath);
            WriteResourceIfNeeded(OpenVinoXmlResourceName, openVinoXmlPath);
            WriteResourceIfNeeded(OpenVinoBinResourceName, openVinoBinPath);

            _cachedLayout = new BundledAssetLayout(
                resolvedRoot,
                classesPath,
                openVinoXmlPath,
                openVinoBinPath);
            return _cachedLayout;
        }
    }

    private static string ResolveExtractionRoot(string? extractionRoot)
    {
        var baseRoot = !string.IsNullOrWhiteSpace(extractionRoot)
            ? Path.GetFullPath(extractionRoot)
            : Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "PollenInference",
                "algorithm");

        var versionFolderName = SanitizePathSegment(GetAlgorithmVersion());
        return Path.Combine(baseRoot, versionFolderName);
    }

    private static string SanitizePathSegment(string raw)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var chars = raw.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray();
        return new string(chars);
    }

    private static void WriteResourceIfNeeded(string resourceName, string destinationPath)
    {
        var assembly = typeof(BundledAssetStore).Assembly;
        using var source = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Bundled resource not found: {resourceName}");

        if (File.Exists(destinationPath))
        {
            var existingLength = new FileInfo(destinationPath).Length;
            if (source.CanSeek && existingLength == source.Length)
            {
                return;
            }

            source.Position = 0;
        }

        using var target = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.Read);
        source.CopyTo(target);
    }

    private sealed record BundledAssetLayout(
        string RootPath,
        string ClassesPath,
        string OpenVinoXmlPath,
        string OpenVinoBinPath);
}
