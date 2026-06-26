using OpenCvSharp;

namespace PollenInferenceDemo.Core;

internal static class ImagePreprocessor
{
    private static readonly float[] ImagenetMean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] ImagenetStd = [0.229f, 0.224f, 0.225f];

    internal sealed class PrepResult
    {
        public required float[] InputTensor { get; init; }
        public required float ScaleX { get; init; }
        public required float ScaleY { get; init; }
        public required int SrcWidth { get; init; }
        public required int SrcHeight { get; init; }
    }

    public static PrepResult Prepare(Mat bgr, int inputW, int inputH)
    {
        var srcW = bgr.Width;
        var srcH = bgr.Height;
        var resized = new Mat();
        Cv2.Resize(bgr, resized, new OpenCvSharp.Size(inputW, inputH), 0, 0, InterpolationFlags.Linear);
        Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

        var chw = new float[3 * inputW * inputH];
        var idxR = 0;
        var idxG = inputW * inputH;
        var idxB = 2 * inputW * inputH;

        for (var y = 0; y < inputH; y++)
        {
            for (var x = 0; x < inputW; x++)
            {
                var px = resized.At<Vec3b>(y, x); // RGB after conversion
                chw[idxR++] = px.Item0 / 255.0f;
                chw[idxG++] = px.Item1 / 255.0f;
                chw[idxB++] = px.Item2 / 255.0f;
            }
        }

        resized.Dispose();

        return new PrepResult
        {
            InputTensor = chw,
            ScaleX = (float)srcW / inputW,
            ScaleY = (float)srcH / inputH,
            SrcWidth = srcW,
            SrcHeight = srcH
        };
    }

    public static float[] CreateWarmupInput(int inputW, int inputH)
    {
        var random = new Random(123);
        var data = new float[3 * inputW * inputH];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = (float)random.NextDouble();
        }
        return data;
    }

    public static Mat CropLetterboxFromBbox(Mat bgr, Rect bbox, double cropScale, int imageSize)
    {
        var boxW = Math.Max(1.0, bbox.Width);
        var boxH = Math.Max(1.0, bbox.Height);
        var cx = bbox.Left + boxW / 2.0;
        var cy = bbox.Top + boxH / 2.0;
        var cropW = boxW * cropScale;
        var cropH = boxH * cropScale;
        var left = (int)Math.Floor(cx - cropW / 2.0);
        var top = (int)Math.Floor(cy - cropH / 2.0);
        var right = (int)Math.Ceiling(cx + cropW / 2.0);
        var bottom = (int)Math.Ceiling(cy + cropH / 2.0);
        var width = Math.Max(1, right - left);
        var height = Math.Max(1, bottom - top);

        using var crop = new Mat(new OpenCvSharp.Size(width, height), bgr.Type(), Scalar.Black);
        var srcLeft = Math.Max(0, left);
        var srcTop = Math.Max(0, top);
        var srcRight = Math.Min(bgr.Width, right);
        var srcBottom = Math.Min(bgr.Height, bottom);
        if (srcRight > srcLeft && srcBottom > srcTop)
        {
            var srcRect = new Rect(srcLeft, srcTop, srcRight - srcLeft, srcBottom - srcTop);
            var dstRect = new Rect(srcLeft - left, srcTop - top, srcRect.Width, srcRect.Height);
            using var srcRoi = new Mat(bgr, srcRect);
            using var dstRoi = new Mat(crop, dstRect);
            srcRoi.CopyTo(dstRoi);
        }

        var scale = Math.Min((double)imageSize / crop.Width, (double)imageSize / crop.Height);
        var newW = Math.Max(1, Math.Min(imageSize, (int)Math.Round(crop.Width * scale)));
        var newH = Math.Max(1, Math.Min(imageSize, (int)Math.Round(crop.Height * scale)));
        using var resized = new Mat();
        Cv2.Resize(crop, resized, new OpenCvSharp.Size(newW, newH), 0, 0, InterpolationFlags.Linear);

        var canvas = new Mat(new OpenCvSharp.Size(imageSize, imageSize), bgr.Type(), Scalar.Black);
        var x0 = (imageSize - newW) / 2;
        var y0 = (imageSize - newH) / 2;
        using var canvasRoi = new Mat(canvas, new Rect(x0, y0, newW, newH));
        resized.CopyTo(canvasRoi);
        return canvas;
    }

    public static float[] PrepareClassifier(Mat bgr, int inputW, int inputH)
    {
        using var resized = new Mat();
        if (bgr.Width != inputW || bgr.Height != inputH)
        {
            Cv2.Resize(bgr, resized, new OpenCvSharp.Size(inputW, inputH), 0, 0, InterpolationFlags.Linear);
        }
        else
        {
            bgr.CopyTo(resized);
        }

        Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);
        var chw = new float[3 * inputW * inputH];
        var idxR = 0;
        var idxG = inputW * inputH;
        var idxB = 2 * inputW * inputH;

        for (var y = 0; y < inputH; y++)
        {
            for (var x = 0; x < inputW; x++)
            {
                var px = resized.At<Vec3b>(y, x);
                chw[idxR++] = ((px.Item0 / 255.0f) - ImagenetMean[0]) / ImagenetStd[0];
                chw[idxG++] = ((px.Item1 / 255.0f) - ImagenetMean[1]) / ImagenetStd[1];
                chw[idxB++] = ((px.Item2 / 255.0f) - ImagenetMean[2]) / ImagenetStd[2];
            }
        }

        return chw;
    }
}
