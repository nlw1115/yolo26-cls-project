using System.Drawing;
using System.Text.Json;

namespace PollenInferenceDemo.UI;

internal sealed class MainFormState
{
    public int Version { get; set; } = 1;
    public string? Engine { get; set; }
    public string? ModelPath { get; set; }
    public string? ClassifierModelPath { get; set; }
    public string? ModelInfoPath { get; set; }
    public string? ImagePath { get; set; }
    public string? FolderPath { get; set; }
    public int WarmupRuns { get; set; } = 10;
    public decimal ConfThreshold { get; set; } = 0.25m;
    public bool AutoInferEnabled { get; set; }
    public int AutoIntervalMs { get; set; } = 1000;
    public bool ShowBoxes { get; set; } = true;
    public bool ShowLabels { get; set; } = true;
    public bool ShowScores { get; set; } = true;
    public bool ShowAnnotations { get; set; }
    public int BoxThicknessScalePercent { get; set; } = 100;
    public int LabelFontScalePercent { get; set; } = 100;
    public int ScoreFontScalePercent { get; set; } = 100;
    public int AnnotationBoxThicknessScalePercent { get; set; } = 100;
    public int AnnotationFontScalePercent { get; set; } = 100;
    public int BoxColorArgb { get; set; } = Color.FromArgb(59, 130, 246).ToArgb();
    public int LabelColorArgb { get; set; } = Color.FromArgb(255, 159, 67).ToArgb();
    public int ScoreColorArgb { get; set; } = Color.FromArgb(59, 130, 246).ToArgb();
    public int AnnotationColorArgb { get; set; } = Color.FromArgb(34, 197, 94).ToArgb();
    public string? SortColumn { get; set; }
    public bool SortAscending { get; set; } = true;
    public WindowBoundsState? WindowBounds { get; set; }
    public string? WindowState { get; set; }
    public int? MainSplitterDistance { get; set; }
    public int? RightSplitterDistance { get; set; }
    public Dictionary<string, int>? ImageGridColumnWidths { get; set; }
}

internal sealed class WindowBoundsState
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }

    public static WindowBoundsState FromRectangle(Rectangle rectangle)
    {
        return new WindowBoundsState
        {
            X = rectangle.X,
            Y = rectangle.Y,
            Width = rectangle.Width,
            Height = rectangle.Height
        };
    }

    public Rectangle ToRectangle()
    {
        return new Rectangle(X, Y, Width, Height);
    }
}

internal static class MainFormStateStore
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private static readonly string StateFilePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "PollenInferenceDemo",
        "ui_state.json");

    public static MainFormState? Load()
    {
        try
        {
            if (!File.Exists(StateFilePath))
            {
                return null;
            }

            var json = File.ReadAllText(StateFilePath);
            if (string.IsNullOrWhiteSpace(json))
            {
                return null;
            }

            return JsonSerializer.Deserialize<MainFormState>(json, JsonOptions);
        }
        catch
        {
            return null;
        }
    }

    public static void Save(MainFormState state)
    {
        try
        {
            var directory = Path.GetDirectoryName(StateFilePath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var json = JsonSerializer.Serialize(state, JsonOptions);
            File.WriteAllText(StateFilePath, json);
        }
        catch
        {
            // Keep the demo usable even if local settings fail to write.
        }
    }
}
