using System.Text.Json;
using OpenCvSharp;

namespace PollenInferenceDemo.Core;

internal static class LabelMeAnnotationLoader
{
    public static List<GroundTruthAnnotation> LoadForImage(
        string imagePath,
        IReadOnlyList<string> classNames,
        int imageWidth,
        int imageHeight)
    {
        if (string.IsNullOrWhiteSpace(imagePath) || imageWidth <= 0 || imageHeight <= 0)
        {
            return [];
        }

        var labelPath = ResolveLabelPath(imagePath);
        if (string.IsNullOrWhiteSpace(labelPath) || !File.Exists(labelPath))
        {
            return [];
        }

        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(labelPath));
            if (!doc.RootElement.TryGetProperty("shapes", out var shapes) || shapes.ValueKind != JsonValueKind.Array)
            {
                return [];
            }

            var annotations = new List<GroundTruthAnnotation>();
            foreach (var shape in shapes.EnumerateArray())
            {
                var label = TryString(shape, "label") ?? "pollen";
                if (!TryReadPoints(shape, out var points) || points.Count == 0)
                {
                    continue;
                }

                var boundingBox = ToPixelRect(points, imageWidth, imageHeight);
                if (boundingBox.Width <= 0 || boundingBox.Height <= 0)
                {
                    continue;
                }

                var classId = FindClassId(classNames, label);
                annotations.Add(new GroundTruthAnnotation(classId, label, boundingBox));
            }

            return annotations;
        }
        catch
        {
            return [];
        }
    }

    private static string? ResolveLabelPath(string imagePath)
    {
        foreach (var candidate in EnumerateLabelCandidates(imagePath).Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private static IEnumerable<string> EnumerateLabelCandidates(string imagePath)
    {
        var fileStem = Path.GetFileNameWithoutExtension(imagePath);
        var sameDirectory = Path.ChangeExtension(imagePath, ".json");
        if (!string.IsNullOrWhiteSpace(sameDirectory))
        {
            yield return sameDirectory;
        }

        var directory = Path.GetDirectoryName(imagePath);
        if (string.IsNullOrWhiteSpace(directory))
        {
            yield break;
        }

        yield return Path.Combine(directory, "labelme.json");

        var dirInfo = new DirectoryInfo(directory);
        var fusedLabelmeRoot = Path.Combine(AppConfig.ProjectRoot, "dataset", "fused_labelme");
        if (Directory.Exists(fusedLabelmeRoot))
        {
            yield return Path.Combine(fusedLabelmeRoot, $"{fileStem}.json");
            yield return Path.Combine(fusedLabelmeRoot, $"{dirInfo.Name}.json");
        }
    }

    private static bool TryReadPoints(JsonElement shape, out List<Point2f> points)
    {
        points = [];
        if (!shape.TryGetProperty("points", out var pointsNode) || pointsNode.ValueKind != JsonValueKind.Array)
        {
            return false;
        }

        foreach (var pointNode in pointsNode.EnumerateArray())
        {
            if (pointNode.ValueKind != JsonValueKind.Array || pointNode.GetArrayLength() < 2)
            {
                continue;
            }

            var xNode = pointNode[0];
            var yNode = pointNode[1];
            if (xNode.ValueKind != JsonValueKind.Number || yNode.ValueKind != JsonValueKind.Number)
            {
                continue;
            }

            points.Add(new Point2f(xNode.GetSingle(), yNode.GetSingle()));
        }

        return points.Count > 0;
    }

    private static Rect ToPixelRect(IReadOnlyList<Point2f> points, int imageWidth, int imageHeight)
    {
        var minX = points.Min(p => p.X);
        var minY = points.Min(p => p.Y);
        var maxX = points.Max(p => p.X);
        var maxY = points.Max(p => p.Y);

        var left = ClampToRange(minX, 0, imageWidth - 1);
        var top = ClampToRange(minY, 0, imageHeight - 1);
        var right = ClampToRange(maxX, left + 1, imageWidth);
        var bottom = ClampToRange(maxY, top + 1, imageHeight);

        return new Rect(left, top, Math.Max(1, right - left), Math.Max(1, bottom - top));
    }

    private static int FindClassId(IReadOnlyList<string> classNames, string label)
    {
        for (var i = 0; i < classNames.Count; i++)
        {
            if (string.Equals(classNames[i], label, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        return -1;
    }

    private static string? TryString(JsonElement node, string propertyName)
    {
        return node.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;
    }

    private static int ClampToRange(float value, int minValue, int maxValue)
    {
        var rounded = (int)MathF.Round(value);
        return Math.Clamp(rounded, minValue, maxValue);
    }
}
