using System.Drawing;

namespace PollenInferenceDemo.Vis;

internal sealed record DetectionRenderOptions
{
    public bool ShowBoxes { get; init; } = true;
    public bool ShowLabels { get; init; } = true;
    public bool ShowScores { get; init; } = true;
    public bool ShowAnnotations { get; init; }
    public Color BoxColor { get; init; } = Color.FromArgb(59, 130, 246);
    public Color LabelColor { get; init; } = Color.FromArgb(255, 159, 67);
    public Color ScoreColor { get; init; } = Color.FromArgb(59, 130, 246);
    public Color AnnotationColor { get; init; } = Color.FromArgb(34, 197, 94);
    public int BoxThicknessScalePercent { get; init; } = 100;
    public int LabelFontScalePercent { get; init; } = 100;
    public int ScoreFontScalePercent { get; init; } = 100;
    public int AnnotationBoxThicknessScalePercent { get; init; } = 100;
    public int AnnotationFontScalePercent { get; init; } = 100;

    public static DetectionRenderOptions Default { get; } = new();
}
