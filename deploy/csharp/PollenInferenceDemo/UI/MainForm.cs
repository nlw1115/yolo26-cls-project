using System.ComponentModel;
using OpenCvSharp;
using PollenInferenceDemo.Core;
using PollenInferenceDemo.Services;
using PollenInferenceDemo.Vis;

namespace PollenInferenceDemo.UI;

[DesignerCategory("Form")]
internal sealed partial class MainForm : Form
{
    private static readonly HashSet<string> SupportedImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };

    private static readonly Color DefaultBoxColor = Color.FromArgb(59, 130, 246);
    private static readonly Color DefaultLabelColor = Color.FromArgb(255, 159, 67);
    private static readonly Color DefaultScoreColor = Color.FromArgb(59, 130, 246);
    private static readonly Color DefaultAnnotationColor = Color.FromArgb(34, 197, 94);

    private const string ImageColName = "Name";
    private const string ImageColType = "Type";
    private const string ImageColSize = "Size";
    private const string ImageColModified = "Modified";

    private readonly InferenceService _inferenceService = new();
    private readonly List<ImageFileItem> _imageItems = [];
    private readonly ColorDialog _colorDialog = new() { FullOpen = true };
    private readonly List<Detection> _previewDetections = [];

    private int _currentIndex = -1;
    private bool _busy;
    private bool _suppressImageSelectionEvent;
    private string _currentSortColumn = ImageColName;
    private bool _sortAscending = true;
    private bool _autoInferRunning;
    private CancellationTokenSource? _autoInferCts;
    private Color _boxColor = DefaultBoxColor;
    private Color _labelColor = DefaultLabelColor;
    private Color _scoreColor = DefaultScoreColor;
    private Color _annotationColor = DefaultAnnotationColor;
    private bool _hasInferencePreview;
    private MainFormState? _restoredState;
    private bool _deferredLayoutRestorePending;

    public MainForm()
    {
        InitializeComponent();
        ConfigureDetectionGridColumns();
        ConfigureImageGridColumns();
        ConfigureDefaults();

        if (IsDesignerHosted())
        {
            return;
        }

        RestoreUiState();
        BindEvents();
        UpdateUiState();
    }

    private static bool IsDesignerHosted()
    {
        return LicenseManager.UsageMode == LicenseUsageMode.Designtime;
    }

    private void ConfigureDetectionGridColumns()
    {
        _gridDetections.Columns.Clear();
        _gridDetections.Columns.Add("DetectorClass", "检测类");
        _gridDetections.Columns.Add("DetectorConfidence", "检测分");
        _gridDetections.Columns.Add("ClassifierClass", "分类类");
        _gridDetections.Columns.Add("ClassifierConfidence", "分类分");
        _gridDetections.Columns.Add("X", "左");
        _gridDetections.Columns.Add("Y", "上");
        _gridDetections.Columns.Add("W", "宽");
        _gridDetections.Columns.Add("H", "高");
    }

    private void ConfigureImageGridColumns()
    {
        _gridImages.Columns.Clear();

        var nameCol = new DataGridViewTextBoxColumn
        {
            Name = ImageColName,
            HeaderText = "名称",
            Width = 200,
            SortMode = DataGridViewColumnSortMode.Programmatic
        };
        var typeCol = new DataGridViewTextBoxColumn
        {
            Name = ImageColType,
            HeaderText = "类型",
            Width = 90,
            SortMode = DataGridViewColumnSortMode.Programmatic
        };
        var sizeCol = new DataGridViewTextBoxColumn
        {
            Name = ImageColSize,
            HeaderText = "大小",
            Width = 100,
            ValueType = typeof(long),
            SortMode = DataGridViewColumnSortMode.Programmatic,
            DefaultCellStyle = { Alignment = DataGridViewContentAlignment.MiddleRight }
        };
        var modifiedCol = new DataGridViewTextBoxColumn
        {
            Name = ImageColModified,
            HeaderText = "修改时间",
            Width = 165,
            ValueType = typeof(DateTime),
            SortMode = DataGridViewColumnSortMode.Programmatic
        };

        _gridImages.Columns.AddRange(nameCol, typeCol, sizeCol, modifiedCol);
        UpdateImageGridSortGlyph();
    }

    private void ConfigureDefaults()
    {
        ConfigureUiTheme();

        if (IsDesignerHosted())
        {
            return;
        }

        _txtModelPath.Text = AppConfig.GetDefaultModelPath(InferenceEngine.OpenVino, null);
        _txtClassifierModelPath.Text = GetDefaultClassifierModelPath();
    }

    private void ConfigureUiTheme()
    {
        ConfigureSecondaryButton(_btnBrowseModel);
        ConfigureSecondaryButton(_btnBrowseClassifierModel);
        ConfigureSecondaryButton(_btnBrowseImage);
        ConfigureSecondaryButton(_btnBrowseFolder);
        ConfigureSecondaryButton(_btnResetPreview);
        ConfigureAccentButton(_btnLoadModel, Color.FromArgb(37, 99, 235));
        ConfigureAccentButton(_btnInferCurrent, Color.FromArgb(13, 148, 136));

        ConfigureColorButton(_btnBoxColor, "框颜色", _boxColor);
        ConfigureColorButton(_btnLabelColor, "标签色", _labelColor);
        ConfigureColorButton(_btnScoreColor, "分数色", _scoreColor);
        ConfigureColorButton(_btnAnnotationColor, "标注色", _annotationColor);
        UpdateStyleScaleLabel(_lblBoxThicknessValue, "粗细", _trkBoxThickness.Value);
        UpdateStyleScaleLabel(_lblLabelFontSizeValue, "字号", _trkLabelFontSize.Value);
        UpdateStyleScaleLabel(_lblScoreFontSizeValue, "字号", _trkScoreFontSize.Value);
        UpdateStyleScaleLabel(_lblAnnotationThicknessValue, "粗细", _trkAnnotationThickness.Value);
        UpdateStyleScaleLabel(_lblAnnotationFontSizeValue, "字号", _trkAnnotationFontSize.Value);

        _gridImages.BackgroundColor = Color.White;
        _gridDetections.BackgroundColor = Color.White;
        _gridImages.BorderStyle = BorderStyle.FixedSingle;
        _gridDetections.BorderStyle = BorderStyle.FixedSingle;
    }

    private void RestoreUiState()
    {
        _restoredState = MainFormStateStore.Load();
        if (_restoredState is null)
        {
            return;
        }

        ApplyRestoredFormBounds(_restoredState);
        ApplyRestoredControlState(_restoredState);
        RestoreImageContext(_restoredState);
        _deferredLayoutRestorePending = true;
    }

    private void BindEvents()
    {
        _btnBrowseModel.Click += (_, _) => BrowseModel();
        _btnBrowseClassifierModel.Click += (_, _) => BrowseClassifierModel();
        _btnLoadModel.Click += async (_, _) => await LoadModelAsync();

        _btnBrowseImage.Click += (_, _) => BrowseSingleImage();
        _btnBrowseFolder.Click += (_, _) => BrowseFolderImages();

        _btnInferCurrent.Click += async (_, _) => await HandleInferButtonClickAsync();
        _chkAutoInfer.CheckedChanged += (_, _) => HandleAutoInferCheckedChanged();
        _chkShowBoxes.CheckedChanged += (_, _) => RefreshInferencePreview();
        _chkShowLabels.CheckedChanged += (_, _) => RefreshInferencePreview();
        _chkShowScores.CheckedChanged += (_, _) => RefreshInferencePreview();
        _chkShowAnnotations.CheckedChanged += (_, _) => RefreshInferencePreview();
        _trkBoxThickness.ValueChanged += (_, _) => HandleStyleSliderChanged(_lblBoxThicknessValue, "粗细", _trkBoxThickness.Value);
        _trkLabelFontSize.ValueChanged += (_, _) => HandleStyleSliderChanged(_lblLabelFontSizeValue, "字号", _trkLabelFontSize.Value);
        _trkScoreFontSize.ValueChanged += (_, _) => HandleStyleSliderChanged(_lblScoreFontSizeValue, "字号", _trkScoreFontSize.Value);
        _trkAnnotationThickness.ValueChanged += (_, _) => HandleStyleSliderChanged(_lblAnnotationThicknessValue, "粗细", _trkAnnotationThickness.Value);
        _trkAnnotationFontSize.ValueChanged += (_, _) => HandleStyleSliderChanged(_lblAnnotationFontSizeValue, "字号", _trkAnnotationFontSize.Value);
        _btnBoxColor.Click += (_, _) => ChooseRenderColor(_btnBoxColor, "目标框颜色", color => _boxColor = color, _boxColor);
        _btnLabelColor.Click += (_, _) => ChooseRenderColor(_btnLabelColor, "标签颜色", color => _labelColor = color, _labelColor);
        _btnScoreColor.Click += (_, _) => ChooseRenderColor(_btnScoreColor, "分数颜色", color => _scoreColor = color, _scoreColor);
        _btnAnnotationColor.Click += (_, _) => ChooseRenderColor(_btnAnnotationColor, "标注颜色", color => _annotationColor = color, _annotationColor);
        _btnResetPreview.Click += (_, _) => _imageViewer.ResetView();

        _gridImages.ColumnHeaderMouseClick += (_, e) => HandleImageGridHeaderClick(e);
        _gridImages.SelectionChanged += (_, _) => HandleImageGridSelectionChanged();
        _gridImages.CellFormatting += (_, e) => HandleImageGridCellFormatting(e);
        Shown += HandleMainFormShown;
        FormClosing += HandleMainFormClosing;

        FormClosing += (_, _) =>
        {
            StopAutoInference("正在停止自动轮询...");
            _inferenceService.Dispose();
            ReplacePreviewImage(null);
        };
    }

    private void HandleMainFormShown(object? sender, EventArgs e)
    {
        Shown -= HandleMainFormShown;

        if (!_deferredLayoutRestorePending || _restoredState is null)
        {
            return;
        }

        ApplyRestoredWindowState(_restoredState);
        BeginInvoke(new Action(ApplyDeferredLayoutState));
    }

    private void HandleMainFormClosing(object? sender, FormClosingEventArgs e)
    {
        SaveUiState();
    }

    private void ApplyDeferredLayoutState()
    {
        if (!_deferredLayoutRestorePending || _restoredState is null)
        {
            return;
        }

        ApplySplitterDistance(_splitMain, _restoredState.MainSplitterDistance);
        ApplySplitterDistance(_splitRight, _restoredState.RightSplitterDistance);
        ApplyImageGridColumnWidths(_restoredState.ImageGridColumnWidths);
        _deferredLayoutRestorePending = false;
    }

    private void ApplyRestoredFormBounds(MainFormState state)
    {
        if (state.WindowBounds is null)
        {
            return;
        }

        var bounds = state.WindowBounds.ToRectangle();
        if (!IsUsableWindowBounds(bounds))
        {
            return;
        }

        StartPosition = FormStartPosition.Manual;
        WindowState = FormWindowState.Normal;
        Bounds = bounds;
    }

    private void ApplyRestoredWindowState(MainFormState state)
    {
        if (string.IsNullOrWhiteSpace(state.WindowState) ||
            !Enum.TryParse<FormWindowState>(state.WindowState, true, out var windowState))
        {
            return;
        }

        WindowState = windowState == FormWindowState.Minimized
            ? FormWindowState.Normal
            : windowState;
    }

    private void ApplyRestoredControlState(MainFormState state)
    {
        _txtModelPath.Text = state.ModelPath ?? string.Empty;
        _txtClassifierModelPath.Text = string.IsNullOrWhiteSpace(state.ClassifierModelPath)
            ? GetDefaultClassifierModelPath()
            : state.ClassifierModelPath;
        _txtImagePath.Text = state.ImagePath ?? string.Empty;
        _txtFolderPath.Text = state.FolderPath ?? string.Empty;

        SetNumericUpDownValue(_numWarmup, state.WarmupRuns);
        SetNumericUpDownValue(_numConf, state.ConfThreshold);
        _chkAutoInfer.Checked = state.AutoInferEnabled;
        SetNumericUpDownValue(_numAutoIntervalMs, state.AutoIntervalMs);

        _chkShowBoxes.Checked = state.ShowBoxes;
        _chkShowLabels.Checked = state.ShowLabels;
        _chkShowScores.Checked = state.ShowScores;
        _chkShowAnnotations.Checked = state.ShowAnnotations;

        SetTrackBarValue(_trkBoxThickness, state.BoxThicknessScalePercent);
        SetTrackBarValue(_trkLabelFontSize, state.LabelFontScalePercent);
        SetTrackBarValue(_trkScoreFontSize, state.ScoreFontScalePercent);
        SetTrackBarValue(_trkAnnotationThickness, state.AnnotationBoxThicknessScalePercent);
        SetTrackBarValue(_trkAnnotationFontSize, state.AnnotationFontScalePercent);
        UpdateStyleScaleLabel(_lblBoxThicknessValue, "粗细", _trkBoxThickness.Value);
        UpdateStyleScaleLabel(_lblLabelFontSizeValue, "字号", _trkLabelFontSize.Value);
        UpdateStyleScaleLabel(_lblScoreFontSizeValue, "字号", _trkScoreFontSize.Value);
        UpdateStyleScaleLabel(_lblAnnotationThicknessValue, "粗细", _trkAnnotationThickness.Value);
        UpdateStyleScaleLabel(_lblAnnotationFontSizeValue, "字号", _trkAnnotationFontSize.Value);

        _boxColor = SafeColorFromArgb(state.BoxColorArgb, DefaultBoxColor);
        _labelColor = SafeColorFromArgb(state.LabelColorArgb, DefaultLabelColor);
        _scoreColor = SafeColorFromArgb(state.ScoreColorArgb, DefaultScoreColor);
        _annotationColor = SafeColorFromArgb(state.AnnotationColorArgb, DefaultAnnotationColor);
        ConfigureColorButton(_btnBoxColor, _btnBoxColor.Text, _boxColor);
        ConfigureColorButton(_btnLabelColor, _btnLabelColor.Text, _labelColor);
        ConfigureColorButton(_btnScoreColor, _btnScoreColor.Text, _scoreColor);
        ConfigureColorButton(_btnAnnotationColor, _btnAnnotationColor.Text, _annotationColor);

        if (!string.IsNullOrWhiteSpace(state.SortColumn) && _gridImages.Columns.Contains(state.SortColumn))
        {
            _currentSortColumn = state.SortColumn;
        }

        _sortAscending = state.SortAscending;
        UpdateImageGridSortGlyph();
    }

    private void RestoreImageContext(MainFormState state)
    {
        var folderPath = SafeNormalizeOptionalPath(state.FolderPath);
        if (!string.IsNullOrWhiteSpace(folderPath) && Directory.Exists(folderPath))
        {
            try
            {
                var images = CollectImageItems(folderPath);
                if (images.Count > 0)
                {
                    _txtFolderPath.Text = folderPath;
                    SetImageItems(images, true);
                    RestoreCurrentImageSelection(state.ImagePath);
                    return;
                }
            }
            catch
            {
                // Ignore invalid historical folder state and keep the UI responsive.
            }
        }

        var imagePath = SafeNormalizeOptionalPath(state.ImagePath);
        if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
        {
            return;
        }

        try
        {
            _txtFolderPath.Text = string.Empty;
            SetImageItems([CreateImageFileItem(new FileInfo(imagePath))], true);
        }
        catch
        {
            // Ignore invalid historical image state and keep the UI responsive.
        }
    }

    private void RestoreCurrentImageSelection(string? savedImagePath)
    {
        var imagePath = SafeNormalizeOptionalPath(savedImagePath);
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            return;
        }

        var index = _imageItems.FindIndex(x => PathEquals(x.FullPath, imagePath));
        if (index < 0 || index == _currentIndex)
        {
            return;
        }

        _currentIndex = index;
        ShowCurrentImagePreview(clearDetections: true);
    }

    private void SaveUiState()
    {
        MainFormStateStore.Save(CaptureUiState());
    }

    private MainFormState CaptureUiState()
    {
        var normalBounds = WindowState == FormWindowState.Normal ? Bounds : RestoreBounds;
        var safeWindowState = WindowState == FormWindowState.Minimized
            ? FormWindowState.Normal
            : WindowState;

        return new MainFormState
        {
            Engine = GetSelectedEngine().ToString(),
            ModelPath = _txtModelPath.Text,
            ClassifierModelPath = _txtClassifierModelPath.Text,
            ModelInfoPath = null,
            ImagePath = TryGetCurrentImage(out var currentImagePath) ? currentImagePath : _txtImagePath.Text,
            FolderPath = _txtFolderPath.Text,
            WarmupRuns = Decimal.ToInt32(_numWarmup.Value),
            ConfThreshold = _numConf.Value,
            AutoInferEnabled = _chkAutoInfer.Checked,
            AutoIntervalMs = Decimal.ToInt32(_numAutoIntervalMs.Value),
            ShowBoxes = _chkShowBoxes.Checked,
            ShowLabels = _chkShowLabels.Checked,
            ShowScores = _chkShowScores.Checked,
            ShowAnnotations = _chkShowAnnotations.Checked,
            BoxThicknessScalePercent = _trkBoxThickness.Value,
            LabelFontScalePercent = _trkLabelFontSize.Value,
            ScoreFontScalePercent = _trkScoreFontSize.Value,
            AnnotationBoxThicknessScalePercent = _trkAnnotationThickness.Value,
            AnnotationFontScalePercent = _trkAnnotationFontSize.Value,
            BoxColorArgb = _boxColor.ToArgb(),
            LabelColorArgb = _labelColor.ToArgb(),
            ScoreColorArgb = _scoreColor.ToArgb(),
            AnnotationColorArgb = _annotationColor.ToArgb(),
            SortColumn = _currentSortColumn,
            SortAscending = _sortAscending,
            WindowBounds = WindowBoundsState.FromRectangle(normalBounds),
            WindowState = safeWindowState.ToString(),
            MainSplitterDistance = _splitMain.IsHandleCreated ? _splitMain.SplitterDistance : null,
            RightSplitterDistance = _splitRight.IsHandleCreated ? _splitRight.SplitterDistance : null,
            ImageGridColumnWidths = CaptureImageGridColumnWidths()
        };
    }

    private Dictionary<string, int> CaptureImageGridColumnWidths()
    {
        var widths = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (DataGridViewColumn column in _gridImages.Columns)
        {
            widths[column.Name] = column.Width;
        }

        return widths;
    }

    private void ApplyImageGridColumnWidths(Dictionary<string, int>? widths)
    {
        if (widths is null)
        {
            return;
        }

        foreach (var pair in widths)
        {
            if (pair.Value < 40 || !_gridImages.Columns.Contains(pair.Key))
            {
                continue;
            }

            var column = _gridImages.Columns[pair.Key];
            if (column is not null)
            {
                column.Width = pair.Value;
            }
        }
    }

    private static void ApplySplitterDistance(SplitContainer splitContainer, int? savedDistance)
    {
        if (savedDistance is null || savedDistance <= 0)
        {
            return;
        }

        var totalLength = splitContainer.Orientation == Orientation.Vertical
            ? splitContainer.ClientSize.Width
            : splitContainer.ClientSize.Height;
        var minDistance = splitContainer.Panel1MinSize;
        var maxDistance = totalLength - splitContainer.SplitterWidth - splitContainer.Panel2MinSize;
        if (maxDistance <= minDistance)
        {
            return;
        }

        splitContainer.SplitterDistance = Math.Clamp(savedDistance.Value, minDistance, maxDistance);
    }

    private static bool IsUsableWindowBounds(Rectangle bounds)
    {
        if (bounds.Width < 900 || bounds.Height < 600)
        {
            return false;
        }

        return Screen.AllScreens.Any(screen => screen.WorkingArea.IntersectsWith(bounds));
    }

    private static void SetNumericUpDownValue(NumericUpDown input, decimal value)
    {
        input.Value = Math.Clamp(value, input.Minimum, input.Maximum);
    }

    private static void SetNumericUpDownValue(NumericUpDown input, int value)
    {
        SetNumericUpDownValue(input, (decimal)value);
    }

    private static void SetTrackBarValue(TrackBar trackBar, int value)
    {
        trackBar.Value = Math.Clamp(value, trackBar.Minimum, trackBar.Maximum);
    }

    private static Color SafeColorFromArgb(int argb, Color fallback)
    {
        try
        {
            return Color.FromArgb(argb);
        }
        catch
        {
            return fallback;
        }
    }

    private static string? SafeNormalizeOptionalPath(string? rawPath)
    {
        try
        {
            return NormalizeOptionalPath(rawPath);
        }
        catch
        {
            return null;
        }
    }

    private void RefreshSuggestedModelPath()
    {
        var current = NormalizeOptionalPath(_txtModelPath.Text);
        if (!string.IsNullOrWhiteSpace(current) && File.Exists(current))
        {
            return;
        }

        _txtModelPath.Text = AppConfig.GetDefaultModelPath(GetSelectedEngine(), null);
    }

    private static string GetDefaultClassifierModelPath()
    {
        var candidate = Path.Combine(AppConfig.ProjectRoot, "runs", "export_openvino", "classifier_resnet34_last", "model.xml");
        if (File.Exists(candidate))
        {
            return candidate;
        }

        var root = Path.Combine(AppConfig.ProjectRoot, "runs", "export_openvino");
        if (!Directory.Exists(root))
        {
            return string.Empty;
        }

        return Directory.EnumerateFiles(root, "model.xml", SearchOption.AllDirectories)
            .Where(path => path.Contains("classifier", StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .FirstOrDefault() ?? string.Empty;
    }

    private async Task LoadModelAsync()
    {
        if (_busy || _autoInferRunning)
        {
            return;
        }

        var modelPath = NormalizeOptionalPath(_txtModelPath.Text);
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
        {
            MessageBox.Show(this, "模型文件不存在，请先选择有效的模型路径。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        var config = new AppConfig
        {
            Engine = InferenceEngine.OpenVino,
            ModelPath = modelPath,
            ModelInfoPath = null,
            ClassNamesPath = AppConfig.GetDefaultClassNamesPath(),
            WarmupRuns = Math.Max(1, (int)_numWarmup.Value)
        };

        AppConfig? classifierConfig = null;
        var classifierPath = NormalizeOptionalPath(_txtClassifierModelPath.Text);
        if (!string.IsNullOrWhiteSpace(classifierPath))
        {
            if (!File.Exists(classifierPath))
            {
                MessageBox.Show(this, "分类模型文件不存在，请先选择有效的 OpenVINO XML。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            classifierConfig = new AppConfig
            {
                Engine = InferenceEngine.OpenVino,
                ModelPath = classifierPath,
                ModelInfoPath = null,
                ClassNamesPath = AppConfig.GetDefaultClassNamesPath(),
                WarmupRuns = Math.Max(1, (int)_numWarmup.Value)
            };
        }

        SetBusy(true, "模型加载与预热中...");
        try
        {
            await _inferenceService.LoadModelAsync(config, classifierConfig);
            var mode = classifierConfig is null ? "检测" : "检测+分类级联";
            SetStatus($"模型加载完成 | {mode} | {Path.GetFileName(config.ModelPath)}");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "模型加载失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            SetStatus("模型加载失败");
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async Task HandleInferButtonClickAsync()
    {
        if (_autoInferRunning)
        {
            StopAutoInference("正在停止自动轮询...");
            return;
        }

        if (_chkAutoInfer.Checked)
        {
            await RunAutoInferLoopAsync();
            return;
        }

        await InferCurrentAsync();
    }

    private void HandleAutoInferCheckedChanged()
    {
        if (!_chkAutoInfer.Checked && _autoInferRunning)
        {
            StopAutoInference("自动轮询已停止");
        }

        UpdateUiState();
    }

    private async Task RunAutoInferLoopAsync()
    {
        if (_busy || _autoInferRunning)
        {
            return;
        }

        if (!_inferenceService.IsModelLoaded)
        {
            MessageBox.Show(this, "请先加载模型。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        if (_imageItems.Count == 0)
        {
            MessageBox.Show(this, "列表中没有可推理的图片。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        if (_currentIndex < 0 || _currentIndex >= _imageItems.Count)
        {
            _currentIndex = 0;
        }

        var intervalMs = (int)_numAutoIntervalMs.Value;
        _autoInferCts = new CancellationTokenSource();
        var token = _autoInferCts.Token;
        _autoInferRunning = true;
        UpdateUiState();
        SetStatus($"自动轮询开始（间隔 {intervalMs} ms）");

        try
        {
            for (var i = _currentIndex; i < _imageItems.Count; i++)
            {
                token.ThrowIfCancellationRequested();

                _currentIndex = i;
                ShowCurrentImagePreview(clearDetections: true);
                UpdateUiState();
                await InferCurrentAsync();

                if (i < _imageItems.Count - 1 && intervalMs > 0)
                {
                    await Task.Delay(intervalMs, token);
                }
            }

            SetStatus("自动轮询完成");
        }
        catch (OperationCanceledException)
        {
            SetStatus("自动轮询已停止");
        }
        finally
        {
            _autoInferRunning = false;
            _autoInferCts?.Dispose();
            _autoInferCts = null;
            UpdateUiState();
        }
    }

    private async Task InferCurrentAsync()
    {
        if (_busy)
        {
            return;
        }

        if (!_inferenceService.IsModelLoaded)
        {
            MessageBox.Show(this, "请先加载模型。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        if (!TryGetCurrentImage(out var imagePath))
        {
            MessageBox.Show(this, "请先选择图片或图片目录。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        var conf = (float)_numConf.Value;

        SetBusy(true, $"正在推理: {Path.GetFileName(imagePath)}");
        try
        {
            var (result, elapsedMs) = await _inferenceService.PredictAsync(imagePath, conf);
            try
            {
                _previewDetections.Clear();
                _previewDetections.AddRange(result.Detections);
                _hasInferencePreview = true;

                var groundTruths = LoadGroundTruthAnnotations(imagePath, result.OriginalImage.Width, result.OriginalImage.Height);
                using var canvas = ResultRenderer.DrawDetections(result, BuildRenderOptions(), groundTruths);
                ReplacePreviewImage(ResultRenderer.ToBitmap(canvas));

                BindDetections(result.Detections);
                SetStatus($"推理完成 | {Path.GetFileName(imagePath)} | 检测数={result.Detections.Count} | {elapsedMs} 毫秒");
            }
            finally
            {
                result.OriginalImage.Dispose();
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "推理失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            SetStatus("推理失败");
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void BrowseModel()
    {
        if (_busy || _autoInferRunning)
        {
            return;
        }

        using var dialog = new OpenFileDialog
        {
            Filter = "OpenVINO XML (*.xml)|*.xml|所有文件 (*.*)|*.*",
            CheckFileExists = true
        };

        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _txtModelPath.Text = dialog.FileName;
            UpdateUiState();
        }
    }

    private void BrowseClassifierModel()
    {
        if (_busy || _autoInferRunning)
        {
            return;
        }

        using var dialog = new OpenFileDialog
        {
            Filter = "OpenVINO XML (*.xml)|*.xml|所有文件 (*.*)|*.*",
            CheckFileExists = true
        };

        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _txtClassifierModelPath.Text = dialog.FileName;
            UpdateUiState();
        }
    }

    private void BrowseSingleImage()
    {
        if (_busy)
        {
            return;
        }

        using var dialog = new OpenFileDialog
        {
            Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp|所有文件 (*.*)|*.*",
            CheckFileExists = true
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        var info = new FileInfo(dialog.FileName);
        if (!info.Exists)
        {
            MessageBox.Show(this, "所选图片文件不存在。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        _txtFolderPath.Text = string.Empty;
        SetImageItems([CreateImageFileItem(info)], true);
        SetStatus($"已加载单张图片: {info.Name}");
    }

    private void BrowseFolderImages()
    {
        if (_busy)
        {
            return;
        }

        using var dialog = new FolderBrowserDialog
        {
            Description = "选择图片目录（将递归读取目录中的图片）"
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        List<ImageFileItem> images;
        try
        {
            images = CollectImageItems(dialog.SelectedPath);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "读取目录失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        if (images.Count == 0)
        {
            MessageBox.Show(this, "目录中未找到支持的图片文件。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        _txtFolderPath.Text = dialog.SelectedPath;
        SetImageItems(images, true);
        SetStatus($"目录加载完成: 共 {images.Count} 张图片");
    }

    private static List<ImageFileItem> CollectImageItems(string folderPath)
    {
        var files = Directory.EnumerateFiles(folderPath, "*.*", SearchOption.AllDirectories)
            .Where(path => SupportedImageExtensions.Contains(Path.GetExtension(path)))
            .Select(path => new FileInfo(path))
            .Where(info => info.Exists)
            .Select(CreateImageFileItem)
            .ToList();

        return files;
    }

    private static ImageFileItem CreateImageFileItem(FileInfo info)
    {
        var type = string.IsNullOrWhiteSpace(info.Extension) ? "文件" : info.Extension.TrimStart('.').ToLowerInvariant();
        return new ImageFileItem(
            info.FullName,
            info.Name,
            type,
            info.Length,
            info.LastWriteTime);
    }

    private void SetImageItems(List<ImageFileItem> items, bool selectFirst)
    {
        _imageItems.Clear();
        _imageItems.AddRange(items);

        _currentIndex = selectFirst && _imageItems.Count > 0 ? 0 : -1;
        SortImageItems(keepCurrentSelection: true);
        ShowCurrentImagePreview(clearDetections: true);
        UpdateUiState();
    }

    private void SortImageItems(bool keepCurrentSelection)
    {
        var currentPath = keepCurrentSelection && TryGetCurrentImage(out var selectedPath) ? selectedPath : null;

        IOrderedEnumerable<ImageFileItem> ordered = _currentSortColumn switch
        {
            ImageColType => _sortAscending
                ? _imageItems.OrderBy(x => x.Type, StringComparer.OrdinalIgnoreCase).ThenBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
                : _imageItems.OrderByDescending(x => x.Type, StringComparer.OrdinalIgnoreCase).ThenBy(x => x.Name, StringComparer.OrdinalIgnoreCase),
            ImageColSize => _sortAscending
                ? _imageItems.OrderBy(x => x.SizeBytes).ThenBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
                : _imageItems.OrderByDescending(x => x.SizeBytes).ThenBy(x => x.Name, StringComparer.OrdinalIgnoreCase),
            ImageColModified => _sortAscending
                ? _imageItems.OrderBy(x => x.LastWriteTime).ThenBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
                : _imageItems.OrderByDescending(x => x.LastWriteTime).ThenBy(x => x.Name, StringComparer.OrdinalIgnoreCase),
            _ => _sortAscending
                ? _imageItems.OrderBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
                : _imageItems.OrderByDescending(x => x.Name, StringComparer.OrdinalIgnoreCase)
        };

        var sorted = ordered.ToList();
        _imageItems.Clear();
        _imageItems.AddRange(sorted);

        if (!string.IsNullOrWhiteSpace(currentPath))
        {
            _currentIndex = _imageItems.FindIndex(x => PathEquals(x.FullPath, currentPath));
        }

        if (_imageItems.Count == 0)
        {
            _currentIndex = -1;
        }
        else if (_currentIndex < 0 || _currentIndex >= _imageItems.Count)
        {
            _currentIndex = 0;
        }

        BindImageGrid();
        UpdateImageGridSortGlyph();
    }

    private void BindImageGrid()
    {
        _suppressImageSelectionEvent = true;
        try
        {
            _gridImages.Rows.Clear();
            foreach (var item in _imageItems)
            {
                var rowIndex = _gridImages.Rows.Add(item.Name, item.Type, item.SizeBytes, item.LastWriteTime);
                _gridImages.Rows[rowIndex].Tag = item.FullPath;
            }

            SelectCurrentImageRow();
        }
        finally
        {
            _suppressImageSelectionEvent = false;
        }
    }

    private void SelectCurrentImageRow()
    {
        foreach (DataGridViewRow row in _gridImages.Rows)
        {
            row.Selected = false;
        }

        if (_currentIndex < 0 || _currentIndex >= _imageItems.Count)
        {
            return;
        }

        var currentPath = _imageItems[_currentIndex].FullPath;
        for (var i = 0; i < _gridImages.Rows.Count; i++)
        {
            var row = _gridImages.Rows[i];
            if (row.Tag is string path && PathEquals(path, currentPath))
            {
                row.Selected = true;
                _gridImages.CurrentCell = row.Cells[0];
                break;
            }
        }
    }

    private void HandleImageGridHeaderClick(DataGridViewCellMouseEventArgs e)
    {
        if (e.ColumnIndex < 0 || e.ColumnIndex >= _gridImages.Columns.Count)
        {
            return;
        }

        var clickedColumn = _gridImages.Columns[e.ColumnIndex].Name;
        if (clickedColumn == _currentSortColumn)
        {
            _sortAscending = !_sortAscending;
        }
        else
        {
            _currentSortColumn = clickedColumn;
            _sortAscending = true;
        }

        SortImageItems(keepCurrentSelection: true);
        ShowCurrentImagePreview(clearDetections: true);
        UpdateUiState();
    }

    private void HandleImageGridSelectionChanged()
    {
        if (_suppressImageSelectionEvent || _busy || _autoInferRunning)
        {
            return;
        }

        if (_gridImages.SelectedRows.Count == 0)
        {
            return;
        }

        var selectedPath = _gridImages.SelectedRows[0].Tag as string;
        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            return;
        }

        var index = _imageItems.FindIndex(x => PathEquals(x.FullPath, selectedPath));
        if (index < 0 || index == _currentIndex)
        {
            return;
        }

        _currentIndex = index;
        ShowCurrentImagePreview(clearDetections: true);
        UpdateUiState();
    }

    private static void HandleImageGridCellFormatting(DataGridViewCellFormattingEventArgs e)
    {
        if (e.Value is long sizeBytes)
        {
            e.Value = FormatFileSize(sizeBytes);
            e.FormattingApplied = true;
            return;
        }

        if (e.Value is DateTime dt)
        {
            e.Value = dt.ToString("yyyy-MM-dd HH:mm:ss");
            e.FormattingApplied = true;
        }
    }

    private static string FormatFileSize(long bytes)
    {
        const double kb = 1024.0;
        const double mb = kb * 1024.0;
        const double gb = mb * 1024.0;

        if (bytes >= gb)
        {
            return $"{bytes / gb:F2} GB";
        }

        if (bytes >= mb)
        {
            return $"{bytes / mb:F2} MB";
        }

        if (bytes >= kb)
        {
            return $"{bytes / kb:F2} KB";
        }

        return $"{bytes} B";
    }

    private void UpdateImageGridSortGlyph()
    {
        foreach (DataGridViewColumn col in _gridImages.Columns)
        {
            col.HeaderCell.SortGlyphDirection = SortOrder.None;
        }

        if (_gridImages.Columns.Contains(_currentSortColumn))
        {
            var col = _gridImages.Columns[_currentSortColumn];
            if (col is not null)
            {
                col.HeaderCell.SortGlyphDirection =
                    _sortAscending ? SortOrder.Ascending : SortOrder.Descending;
            }
        }
    }

    private void ShowCurrentImagePreview(bool clearDetections)
    {
        if (!TryGetCurrentImage(out var imagePath))
        {
            if (clearDetections)
            {
                ClearInferencePreviewState();
            }

            ReplacePreviewImage(null);
            _txtImagePath.Text = string.Empty;
            if (clearDetections)
            {
                _gridDetections.Rows.Clear();
            }

            UpdateUiState();
            return;
        }

        _txtImagePath.Text = imagePath;
        SelectCurrentImageRow();

        if (clearDetections)
        {
            ClearInferencePreviewState();
        }

        try
        {
            ReplacePreviewImage(RenderPreviewBitmap(imagePath));

            if (clearDetections)
            {
                _gridDetections.Rows.Clear();
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "图片加载失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private bool TryGetCurrentImage(out string imagePath)
    {
        if (_currentIndex >= 0 && _currentIndex < _imageItems.Count)
        {
            imagePath = _imageItems[_currentIndex].FullPath;
            return true;
        }

        imagePath = string.Empty;
        return false;
    }

    private void BindDetections(IReadOnlyCollection<Detection> detections)
    {
        _gridDetections.Rows.Clear();
        foreach (var d in detections)
        {
            _gridDetections.Rows.Add(
                d.ClassName,
                d.Confidence.ToString("F4"),
                d.ClassifierClassName ?? "",
                d.ClassifierConfidence.HasValue ? d.ClassifierConfidence.Value.ToString("F4") : "",
                d.BoundingBox.Left,
                d.BoundingBox.Top,
                d.BoundingBox.Width,
                d.BoundingBox.Height);
        }
    }

    private void ReplacePreviewImage(Image? image, bool resetView = true)
    {
        var old = _imageViewer.DisplayImage;
        _imageViewer.SetImage(image, resetView);
        old?.Dispose();
        _btnResetPreview.Enabled = image is not null;
    }

    private InferenceEngine GetSelectedEngine()
    {
        return InferenceEngine.OpenVino;
    }

    private static string? NormalizeOptionalPath(string? rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
        {
            return null;
        }

        var trimmed = rawPath.Trim().Trim('"');
        if (string.IsNullOrWhiteSpace(trimmed))
        {
            return null;
        }

        return Path.GetFullPath(trimmed);
    }

    private static bool PathEquals(string a, string b)
    {
        return string.Equals(
            Path.GetFullPath(a),
            Path.GetFullPath(b),
            StringComparison.OrdinalIgnoreCase);
    }

    private void StopAutoInference(string statusText)
    {
        if (!_autoInferRunning)
        {
            return;
        }

        _autoInferCts?.Cancel();
        SetStatus(statusText);
    }

    private void SetBusy(bool busy, string? statusText = null)
    {
        _busy = busy;
        if (!string.IsNullOrWhiteSpace(statusText))
        {
            SetStatus(statusText);
        }

        UpdateUiState();
    }

    private void SetStatus(string message)
    {
        _statusLabel.Text = message;
    }

    private void UpdateUiState()
    {
        var hasImage = _currentIndex >= 0 && _currentIndex < _imageItems.Count;
        var hasPreview = _imageViewer.DisplayImage is not null;
        var canInfer = !_busy && _inferenceService.IsModelLoaded && hasImage;
        var canEditInput = !_busy && !_autoInferRunning;
        var canAdjustPreview = !_busy;

        _btnLoadModel.Enabled = canEditInput;
        _btnBrowseModel.Enabled = canEditInput;
        _btnBrowseClassifierModel.Enabled = canEditInput;
        _btnBrowseImage.Enabled = canEditInput;
        _btnBrowseFolder.Enabled = canEditInput;
        _numWarmup.Enabled = canEditInput;
        _numAutoIntervalMs.Enabled = canEditInput;
        _chkAutoInfer.Enabled = !_busy || _autoInferRunning;
        _chkShowBoxes.Enabled = canAdjustPreview;
        _chkShowLabels.Enabled = canAdjustPreview;
        _chkShowScores.Enabled = canAdjustPreview;
        _chkShowAnnotations.Enabled = canAdjustPreview;
        _trkBoxThickness.Enabled = canAdjustPreview;
        _trkLabelFontSize.Enabled = canAdjustPreview;
        _trkScoreFontSize.Enabled = canAdjustPreview;
        _trkAnnotationThickness.Enabled = canAdjustPreview;
        _trkAnnotationFontSize.Enabled = canAdjustPreview;
        _btnBoxColor.Enabled = canAdjustPreview;
        _btnLabelColor.Enabled = canAdjustPreview;
        _btnScoreColor.Enabled = canAdjustPreview;
        _btnAnnotationColor.Enabled = canAdjustPreview;
        _btnResetPreview.Enabled = hasPreview;

        if (_autoInferRunning)
        {
            _btnInferCurrent.Text = "停止轮询";
            _btnInferCurrent.Enabled = true;
        }
        else
        {
            _btnInferCurrent.Text = _chkAutoInfer.Checked ? "开始轮询" : "推理当前";
            _btnInferCurrent.Enabled = canInfer;
        }

        _gridImages.Enabled = canEditInput;

        _lblImageIndex.Text = hasImage
            ? $"当前图片: {_currentIndex + 1} / {_imageItems.Count}"
            : "当前图片: 0 / 0";
    }

    private void RefreshInferencePreview()
    {
        if (_busy || !TryGetCurrentImage(out var imagePath))
        {
            return;
        }

        if (!_hasInferencePreview && !_chkShowAnnotations.Checked)
        {
            return;
        }

        try
        {
            ReplacePreviewImage(RenderPreviewBitmap(imagePath), resetView: false);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "预览刷新失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private DetectionRenderOptions BuildRenderOptions()
    {
        return new DetectionRenderOptions
        {
            ShowBoxes = _chkShowBoxes.Checked,
            ShowLabels = _chkShowLabels.Checked,
            ShowScores = _chkShowScores.Checked,
            ShowAnnotations = _chkShowAnnotations.Checked,
            BoxColor = _boxColor,
            LabelColor = _labelColor,
            ScoreColor = _scoreColor,
            AnnotationColor = _annotationColor,
            BoxThicknessScalePercent = _trkBoxThickness.Value,
            LabelFontScalePercent = _trkLabelFontSize.Value,
            ScoreFontScalePercent = _trkScoreFontSize.Value,
            AnnotationBoxThicknessScalePercent = _trkAnnotationThickness.Value,
            AnnotationFontScalePercent = _trkAnnotationFontSize.Value
        };
    }

    private Bitmap RenderPreviewBitmap(string imagePath)
    {
        using var mat = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (mat.Empty())
        {
            throw new IOException($"无法读取图片: {imagePath}");
        }

        var renderOptions = BuildRenderOptions();
        var groundTruths = LoadGroundTruthAnnotations(imagePath, mat.Width, mat.Height);

        if (_hasInferencePreview)
        {
            using var canvas = ResultRenderer.DrawDetections(mat, _previewDetections, renderOptions, groundTruths);
            return ResultRenderer.ToBitmap(canvas);
        }

        if (renderOptions.ShowAnnotations && groundTruths.Count > 0)
        {
            using var canvas = ResultRenderer.DrawGroundTruths(mat, groundTruths, renderOptions);
            return ResultRenderer.ToBitmap(canvas);
        }

        return ResultRenderer.ToBitmap(mat);
    }

    private List<GroundTruthAnnotation> LoadGroundTruthAnnotations(string imagePath, int imageWidth, int imageHeight)
    {
        if (!_chkShowAnnotations.Checked)
        {
            return [];
        }

        var classNames = ResolveDisplayClassNames();
        return LabelMeAnnotationLoader.LoadForImage(imagePath, classNames, imageWidth, imageHeight);
    }

    private string[] ResolveDisplayClassNames()
    {
        try
        {
            return ModelInfo.Load(new AppConfig
            {
                ClassNamesPath = AppConfig.GetDefaultClassNamesPath(),
                ModelPath = NormalizeOptionalPath(_txtModelPath.Text) ?? string.Empty,
                ModelInfoPath = null
            }).ClassNames;
        }
        catch
        {
            return ModelInfo.Load(new AppConfig
            {
                ClassNamesPath = AppConfig.GetDefaultClassNamesPath(),
                ModelInfoPath = null
            }).ClassNames;
        }
    }

    private void ClearInferencePreviewState()
    {
        _previewDetections.Clear();
        _hasInferencePreview = false;
    }

    private void HandleStyleSliderChanged(Label valueLabel, string prefix, int value)
    {
        UpdateStyleScaleLabel(valueLabel, prefix, value);
        RefreshInferencePreview();
    }

    private void ChooseRenderColor(Button button, string colorName, Action<Color> onColorChanged, Color currentColor)
    {
        if (_busy)
        {
            return;
        }

        _colorDialog.Color = currentColor;
        if (_colorDialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        onColorChanged(_colorDialog.Color);
        ConfigureColorButton(button, button.Text, _colorDialog.Color);
        RefreshInferencePreview();
        SetStatus($"{colorName}已更新");
    }

    private static void ConfigureAccentButton(Button button, Color backColor)
    {
        button.FlatStyle = FlatStyle.Flat;
        button.FlatAppearance.BorderSize = 0;
        button.UseVisualStyleBackColor = false;
        button.BackColor = backColor;
        button.ForeColor = Color.White;
    }

    private static void ConfigureSecondaryButton(Button button)
    {
        button.FlatStyle = FlatStyle.Flat;
        button.FlatAppearance.BorderSize = 1;
        button.FlatAppearance.BorderColor = Color.FromArgb(203, 213, 225);
        button.UseVisualStyleBackColor = false;
        button.BackColor = Color.White;
        button.ForeColor = Color.FromArgb(30, 41, 59);
    }

    private static void ConfigureColorButton(Button button, string text, Color color)
    {
        button.Text = text;
        button.Dock = DockStyle.Fill;
        button.FlatStyle = FlatStyle.Flat;
        button.FlatAppearance.BorderSize = 1;
        button.FlatAppearance.BorderColor = Color.FromArgb(203, 213, 225);
        button.UseVisualStyleBackColor = false;
        button.BackColor = color;
        button.ForeColor = GetReadableButtonTextColor(color);
    }

    private static void UpdateStyleScaleLabel(Label label, string prefix, int value)
    {
        label.Text = $"{prefix} {value}%";
    }

    private static Color GetReadableButtonTextColor(Color background)
    {
        var luminance = (0.299 * background.R) + (0.587 * background.G) + (0.114 * background.B);
        return luminance >= 165 ? Color.Black : Color.White;
    }

    private sealed record ImageFileItem(
        string FullPath,
        string Name,
        string Type,
        long SizeBytes,
        DateTime LastWriteTime);
}
