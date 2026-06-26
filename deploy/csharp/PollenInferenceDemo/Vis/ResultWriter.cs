using System.Text.Json;
using OpenCvSharp;
using PollenInferenceDemo.Core;

namespace PollenInferenceDemo.Vis;

internal static class ResultWriter
{
    public static void SaveImage(InferenceResult result, string outputPath)
    {
        using var canvas = ResultRenderer.DrawDetections(result);
        Cv2.ImWrite(outputPath, canvas);
    }

    public static void SaveJson(InferenceResult result, string outputPath)
    {
        var payload = new
        {
            detections = result.Detections.Select(d => new
            {
                class_id = d.ClassId,
                class_name = d.ClassName,
                confidence = d.Confidence,
                classifier_class_id = d.ClassifierClassId,
                classifier_class_name = d.ClassifierClassName,
                classifier_confidence = d.ClassifierConfidence,
                bbox = new { x = d.BoundingBox.Left, y = d.BoundingBox.Top, w = d.BoundingBox.Width, h = d.BoundingBox.Height }
            }).ToArray()
        };
        var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(outputPath, json);
    }
}
