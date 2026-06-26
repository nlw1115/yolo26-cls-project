using OpenCvSharp;

namespace PollenInferenceDemo.Core;

internal static class Postprocess
{
    public static List<Detection> DecodeYoloOutput(
        float[] output,
        int rows,
        int cols,
        float confThr,
        float scaleX,
        float scaleY,
        int srcW,
        int srcH,
        string[] classNames)
    {
        // Expected row format: [x1, y1, x2, y2, conf, cls]
        var detections = new List<Detection>();
        for (var i = 0; i < rows; i++)
        {
            var off = i * cols;
            var conf = output[off + 4];
            if (conf < confThr)
            {
                continue;
            }

            var clsId = (int)MathF.Round(output[off + 5]);
            var x1 = output[off];
            var y1 = output[off + 1];
            var x2 = output[off + 2];
            var y2 = output[off + 3];

            var left = ClampToInt(x1 * scaleX, 0, srcW - 1);
            var top = ClampToInt(y1 * scaleY, 0, srcH - 1);
            var right = ClampToInt(x2 * scaleX, 0, srcW - 1);
            var bottom = ClampToInt(y2 * scaleY, 0, srcH - 1);

            if (right <= left || bottom <= top)
            {
                continue;
            }

            var name = clsId >= 0 && clsId < classNames.Length ? classNames[clsId] : $"cls_{clsId}";
            detections.Add(new Detection(clsId, name, conf, new Rect(left, top, right - left, bottom - top)));
        }

        // YOLO26 end-to-end export already emits final candidate boxes; the demo only filters confidence.
        return detections.OrderByDescending(x => x.Confidence).ToList();
    }

    private static int ClampToInt(float value, int min, int max)
    {
        var v = (int)MathF.Round(value);
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

}
