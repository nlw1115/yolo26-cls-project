using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.IO;
using OpenCvSharp;
using PollenInferenceDemo.Core;

namespace PollenInferenceDemo.Vis;

internal static class ResultRenderer
{
    public static Mat DrawDetections(
        InferenceResult result,
        DetectionRenderOptions? options = null,
        IReadOnlyCollection<GroundTruthAnnotation>? annotations = null)
    {
        return DrawDetections(result.OriginalImage, result.Detections, options, annotations);
    }

    public static Mat DrawDetections(
        Mat sourceImage,
        IReadOnlyCollection<Detection> detections,
        DetectionRenderOptions? options = null,
        IReadOnlyCollection<GroundTruthAnnotation>? annotations = null)
    {
        var renderOptions = options ?? DetectionRenderOptions.Default;
        var canvas = sourceImage.Clone();
        if (detections.Count == 0 && (!renderOptions.ShowAnnotations || annotations is null || annotations.Count == 0))
        {
            return canvas;
        }

        var metrics = CreateDrawingMetrics(canvas, renderOptions);
        var textCommands = new List<PositionedText>();
        foreach (var detection in detections)
        {
            if (renderOptions.ShowBoxes)
            {
                Cv2.Rectangle(canvas, detection.BoundingBox, ToScalar(renderOptions.BoxColor), metrics.BoxThickness, LineTypes.AntiAlias);
            }

            QueueDetectionAnnotationTexts(textCommands, canvas, detection, renderOptions, metrics);
        }

        if (renderOptions.ShowAnnotations && annotations is not null)
        {
            foreach (var annotation in annotations)
            {
                DrawGroundTruthAnnotation(canvas, annotation, renderOptions, metrics, textCommands);
            }
        }

        DrawTextCommands(canvas, textCommands);
        return canvas;
    }

    public static Mat DrawGroundTruths(
        Mat sourceImage,
        IReadOnlyCollection<GroundTruthAnnotation> annotations,
        DetectionRenderOptions? options = null)
    {
        var renderOptions = options ?? DetectionRenderOptions.Default;
        var canvas = sourceImage.Clone();
        if (!renderOptions.ShowAnnotations || annotations.Count == 0)
        {
            return canvas;
        }

        var metrics = CreateDrawingMetrics(canvas, renderOptions);
        var textCommands = new List<PositionedText>();
        foreach (var annotation in annotations)
        {
            DrawGroundTruthAnnotation(canvas, annotation, renderOptions, metrics, textCommands);
        }

        DrawTextCommands(canvas, textCommands);
        return canvas;
    }

    public static Bitmap ToBitmap(Mat mat)
    {
        var encoded = mat.ToBytes(".bmp");
        using var ms = new MemoryStream(encoded);
        using var temp = new Bitmap(ms);
        return new Bitmap(temp);
    }

    private static Mat BitmapToMat(Bitmap bitmap)
    {
        using var ms = new MemoryStream();
        bitmap.Save(ms, ImageFormat.Bmp);
        return Cv2.ImDecode(ms.ToArray(), ImreadModes.Color);
    }

    private static void QueueDetectionAnnotationTexts(
        List<PositionedText> textCommands,
        Mat canvas,
        Detection detection,
        DetectionRenderOptions options,
        DrawingMetrics metrics)
    {
        var scoreText = options.ShowScores ? detection.DisplayScore : null;
        var labelText = options.ShowLabels ? detection.DisplayLabel : null;

        if (string.IsNullOrWhiteSpace(scoreText) && string.IsNullOrWhiteSpace(labelText))
        {
            return;
        }

        var texts = new List<AnnotationText>(2);
        if (!string.IsNullOrWhiteSpace(scoreText))
        {
            texts.Add(CreateAnnotationText(scoreText, options.ScoreColor, metrics.ScoreFontScale, metrics.ScoreTextThickness));
        }
        if (!string.IsNullOrWhiteSpace(labelText))
        {
            texts.Add(CreateAnnotationText(labelText, options.LabelColor, metrics.LabelFontScale, metrics.LabelTextThickness));
        }

        QueueVerticalTextAnnotations(textCommands, canvas, detection.BoundingBox, texts, metrics.TextGap, metrics.AnnotationOffset);
    }

    private static void DrawGroundTruthAnnotation(
        Mat canvas,
        GroundTruthAnnotation annotation,
        DetectionRenderOptions options,
        DrawingMetrics metrics,
        List<PositionedText> textCommands)
    {
        Cv2.Rectangle(canvas, annotation.BoundingBox, ToScalar(options.AnnotationColor), metrics.AnnotationBoxThickness, LineTypes.AntiAlias);

        var texts = new List<AnnotationText>(1)
        {
            CreateAnnotationText(
                $"GT: {annotation.ClassName}",
                options.AnnotationColor,
                metrics.AnnotationFontScale,
                metrics.AnnotationTextThickness)
        };

        QueueInlineTextAnnotations(textCommands, canvas, annotation.BoundingBox, texts, metrics.TextGap, metrics.AnnotationOffset);
    }

    private static void QueueInlineTextAnnotations(
        List<PositionedText> textCommands,
        Mat canvas,
        Rect anchorBox,
        IReadOnlyList<AnnotationText> annotations,
        int textGap,
        int annotationOffset)
    {
        if (annotations.Count == 0)
        {
            return;
        }

        var totalWidth = annotations.Sum(x => x.Size.Width) + textGap * (annotations.Count - 1);
        var maxHeight = annotations.Max(x => x.Size.Height);

        var x = Math.Max(0, anchorBox.Left);
        if (x + totalWidth > canvas.Width)
        {
            x = Math.Max(0, canvas.Width - totalWidth - 1);
        }

        var y = anchorBox.Top - maxHeight - annotationOffset;
        if (y < 0)
        {
            y = Math.Min(
                Math.Max(0, anchorBox.Top + annotationOffset),
                Math.Max(0, canvas.Height - maxHeight - 1));
        }
        else
        {
            y = Math.Min(y, Math.Max(0, canvas.Height - maxHeight - 1));
        }

        foreach (var annotation in annotations)
        {
            textCommands.Add(new PositionedText(
                annotation.Text,
                annotation.Color,
                annotation.FontSizePx,
                annotation.TextThickness,
                new RectangleF(x, y, annotation.Size.Width, annotation.Size.Height)));

            x += annotation.Size.Width + textGap;
        }
    }

    private static void QueueVerticalTextAnnotations(
        List<PositionedText> textCommands,
        Mat canvas,
        Rect anchorBox,
        IReadOnlyList<AnnotationText> annotations,
        int textGap,
        int annotationOffset)
    {
        if (annotations.Count == 0)
        {
            return;
        }

        var x = Math.Max(0, anchorBox.Left);

        var y = anchorBox.Top - annotationOffset;
        foreach (var annotation in annotations)
        {
            y -= annotation.Size.Height;
            if (y < 0)
            {
                y = Math.Min(
                    Math.Max(0, anchorBox.Top + annotationOffset),
                    Math.Max(0, canvas.Height - annotation.Size.Height - 1));
            }

            textCommands.Add(new PositionedText(
                annotation.Text,
                annotation.Color,
                annotation.FontSizePx,
                annotation.TextThickness,
                new RectangleF(x, y, annotation.Size.Width, annotation.Size.Height)));

            y -= textGap;
        }
    }

    private static AnnotationText CreateAnnotationText(string text, Color color, double fontScale, int textThickness)
    {
        var fontSizePx = Math.Max(9, (int)Math.Round(fontScale * 28.0));
        var textSize = MeasureUnicodeText(text, fontSizePx, textThickness);
        return new AnnotationText(text, color, fontSizePx, textThickness, textSize);
    }

    private static System.Drawing.Size MeasureUnicodeText(string text, int fontSizePx, int textThickness)
    {
        using var bitmap = new Bitmap(1, 1);
        using var graphics = Graphics.FromImage(bitmap);
        using var font = CreateTextFont(fontSizePx, textThickness);
        graphics.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
        var measured = graphics.MeasureString(text, font, int.MaxValue, StringFormat.GenericTypographic);
        return new System.Drawing.Size(
            Math.Max(1, (int)Math.Ceiling(measured.Width) + 2),
            Math.Max(1, (int)Math.Ceiling(measured.Height) + 2));
    }

    private static Font CreateTextFont(int fontSizePx, int textThickness)
    {
        var style = textThickness > 1 ? FontStyle.Bold : FontStyle.Regular;
        return new Font("Microsoft YaHei UI", fontSizePx, style, GraphicsUnit.Pixel);
    }

    private static void DrawTextCommands(Mat canvas, IReadOnlyList<PositionedText> textCommands)
    {
        if (textCommands.Count == 0)
        {
            return;
        }

        using var bitmap = ToBitmap(canvas);
        using var graphics = Graphics.FromImage(bitmap);

        graphics.SmoothingMode = SmoothingMode.AntiAlias;
        graphics.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;

        foreach (var command in textCommands)
        {
            using var font = CreateTextFont(command.FontSizePx, command.TextThickness);
            using var brush = new SolidBrush(command.Color);
            graphics.DrawString(command.Text, font, brush, command.Bounds.Location, StringFormat.GenericTypographic);
        }

        using var updated = BitmapToMat(bitmap);
        updated.CopyTo(canvas);
    }

    private static DrawingMetrics CreateDrawingMetrics(Mat canvas, DetectionRenderOptions options)
    {
        var shortEdge = Math.Max(1, Math.Min(canvas.Width, canvas.Height));
        var baseFontScale = Math.Clamp(shortEdge / 1200.0, 0.55, 0.9);
        var baseBoxThickness = Math.Max(2, (int)Math.Round(shortEdge / 420.0));
        var baseTextThickness = Math.Max(1, baseBoxThickness - 1);
        var boxThickness = Math.Max(1, (int)Math.Round(baseBoxThickness * options.BoxThicknessScalePercent / 100.0));
        var labelFontScale = Math.Max(0.35, baseFontScale * options.LabelFontScalePercent / 100.0);
        var scoreFontScale = Math.Max(0.35, baseFontScale * options.ScoreFontScalePercent / 100.0);
        var labelTextThickness = Math.Max(1, (int)Math.Round(baseTextThickness * options.LabelFontScalePercent / 100.0));
        var scoreTextThickness = Math.Max(1, (int)Math.Round(baseTextThickness * options.ScoreFontScalePercent / 100.0));
        var annotationFontScale = Math.Max(0.35, baseFontScale * options.AnnotationFontScalePercent / 100.0);
        var annotationTextThickness = Math.Max(1, (int)Math.Round(baseTextThickness * options.AnnotationFontScalePercent / 100.0));
        var annotationBoxThickness = Math.Max(1, (int)Math.Round(baseBoxThickness * options.AnnotationBoxThicknessScalePercent / 100.0));

        return new DrawingMetrics(
            boxThickness,
            labelFontScale,
            labelTextThickness,
            scoreFontScale,
            scoreTextThickness,
            annotationFontScale,
            annotationTextThickness,
            annotationBoxThickness,
            TextGap: Math.Max(8, boxThickness * 3),
            AnnotationOffset: Math.Max(6, boxThickness * 2));
    }

    private static Scalar ToScalar(Color color)
    {
        return new Scalar(color.B, color.G, color.R);
    }

    private readonly record struct DrawingMetrics(
        int BoxThickness,
        double LabelFontScale,
        int LabelTextThickness,
        double ScoreFontScale,
        int ScoreTextThickness,
        double AnnotationFontScale,
        int AnnotationTextThickness,
        int AnnotationBoxThickness,
        int TextGap,
        int AnnotationOffset);

    private readonly record struct AnnotationText(
        string Text,
        Color Color,
        int FontSizePx,
        int TextThickness,
        System.Drawing.Size Size);

    private readonly record struct PositionedText(
        string Text,
        Color Color,
        int FontSizePx,
        int TextThickness,
        RectangleF Bounds);
}
