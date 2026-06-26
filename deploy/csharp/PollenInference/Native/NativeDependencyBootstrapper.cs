using System.Reflection;
using System.Runtime.InteropServices;

namespace PollenInference.Native;

internal static class NativeDependencyBootstrapper
{
    private const string ResourcePrefix = "PollenInference.Native.";

    private static readonly Lock SyncRoot = new();
    private static string? _loadedRoot;

    public static string EnsureLoaded(string? extractionRoot)
    {
        lock (SyncRoot)
        {
            var resolvedRoot = ResolveExtractionRoot(extractionRoot);
            if (string.Equals(_loadedRoot, resolvedRoot, StringComparison.OrdinalIgnoreCase))
            {
                return resolvedRoot;
            }

            Directory.CreateDirectory(resolvedRoot);

            var assembly = typeof(NativeDependencyBootstrapper).Assembly;
            foreach (var resourceName in assembly.GetManifestResourceNames()
                         .Where(name => name.StartsWith(ResourcePrefix, StringComparison.OrdinalIgnoreCase)))
            {
                var fileName = resourceName[ResourcePrefix.Length..];
                ExtractResource(assembly, resourceName, Path.Combine(resolvedRoot, fileName));
            }

            PrependProcessPath(resolvedRoot);
            PreloadCoreLibraries(resolvedRoot);
            _loadedRoot = resolvedRoot;
            return resolvedRoot;
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

        var versionFolder = PollenAlgorithmInfo.Version.Replace('+', '_');
        return Path.Combine(baseRoot, versionFolder, "native");
    }

    private static void ExtractResource(Assembly assembly, string resourceName, string destinationPath)
    {
        using var source = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Native dependency resource not found: {resourceName}");

        if (File.Exists(destinationPath))
        {
            var fileInfo = new FileInfo(destinationPath);
            if (source.CanSeek && fileInfo.Length == source.Length)
            {
                return;
            }

            source.Position = 0;
        }

        using var target = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.Read);
        source.CopyTo(target);
    }

    private static void PrependProcessPath(string directory)
    {
        var currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        var segments = currentPath.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (segments.Contains(directory, StringComparer.OrdinalIgnoreCase))
        {
            return;
        }

        var updatedPath = string.IsNullOrWhiteSpace(currentPath)
            ? directory
            : $"{directory}{Path.PathSeparator}{currentPath}";
        Environment.SetEnvironmentVariable("PATH", updatedPath);
    }

    private static void PreloadCoreLibraries(string directory)
    {
        foreach (var fileName in new[]
                 {
                     "tbb12.dll",
                     "tbbbind_2_5.dll",
                     "tbbmalloc.dll",
                     "tbbmalloc_proxy.dll",
                     "openvino.dll",
                     "openvino_c.dll",
                     "OpenCvSharpExtern.dll"
                 })
        {
            var fullPath = Path.Combine(directory, fileName);
            if (!File.Exists(fullPath))
            {
                continue;
            }

            try
            {
                NativeLibrary.Load(fullPath);
            }
            catch
            {
                // Some libraries are loaded on demand; keep the bootstrap permissive.
            }
        }
    }
}
