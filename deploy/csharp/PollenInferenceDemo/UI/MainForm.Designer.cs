namespace PollenInferenceDemo.UI;

internal sealed partial class MainForm
{
    private TextBox _txtModelPath = null!;
    private TextBox _txtClassifierModelPath = null!;
    private NumericUpDown _numWarmup = null!;
    private Button _btnBrowseModel = null!;
    private Button _btnBrowseClassifierModel = null!;
    private Button _btnLoadModel = null!;

    private TextBox _txtImagePath = null!;
    private TextBox _txtFolderPath = null!;
    private Button _btnBrowseImage = null!;
    private Button _btnBrowseFolder = null!;
    private Label _lblImageIndex = null!;

    private NumericUpDown _numConf = null!;
    private CheckBox _chkShowBoxes = null!;
    private CheckBox _chkShowLabels = null!;
    private CheckBox _chkShowScores = null!;
    private CheckBox _chkShowAnnotations = null!;
    private TrackBar _trkBoxThickness = null!;
    private TrackBar _trkLabelFontSize = null!;
    private TrackBar _trkScoreFontSize = null!;
    private Label _lblBoxThicknessValue = null!;
    private Label _lblLabelFontSizeValue = null!;
    private Label _lblScoreFontSizeValue = null!;
    private Button _btnBoxColor = null!;
    private Button _btnLabelColor = null!;
    private Button _btnScoreColor = null!;
    private TrackBar _trkAnnotationThickness = null!;
    private TrackBar _trkAnnotationFontSize = null!;
    private Label _lblAnnotationThicknessValue = null!;
    private Label _lblAnnotationFontSizeValue = null!;
    private Button _btnAnnotationColor = null!;

    private CheckBox _chkAutoInfer = null!;
    private NumericUpDown _numAutoIntervalMs = null!;
    private Button _btnInferCurrent = null!;

    private DataGridView _gridImages = null!;
    private ZoomableImageViewer _imageViewer = null!;
    private Button _btnResetPreview = null!;
    private DataGridView _gridDetections = null!;
    private ToolStripStatusLabel _statusLabel = null!;
    private SplitContainer _splitMain = null!;
    private SplitContainer _splitRight = null!;

    private void InitializeComponent()
    {
        SuspendLayout();

        _txtModelPath = new TextBox
        {
            Dock = DockStyle.Fill
        };
        _txtClassifierModelPath = new TextBox
        {
            Dock = DockStyle.Fill
        };
        _numWarmup = new NumericUpDown
        {
            Minimum = 1,
            Maximum = 200,
            DecimalPlaces = 0,
            Value = 10,
            Dock = DockStyle.Fill
        };
        _btnBrowseModel = new Button { Text = "浏览模型..." };
        _btnBrowseClassifierModel = new Button { Text = "浏览分类..." };
        _btnLoadModel = new Button { Text = "加载模型并预热" };

        _txtImagePath = new TextBox
        {
            Dock = DockStyle.Fill,
            ReadOnly = true
        };
        _txtFolderPath = new TextBox
        {
            Dock = DockStyle.Fill,
            ReadOnly = true
        };
        _btnBrowseImage = new Button { Text = "选择图片..." };
        _btnBrowseFolder = new Button { Text = "选择目录..." };
        _lblImageIndex = new Label
        {
            AutoSize = true,
            Text = "当前图片: 0 / 0"
        };

        _numConf = new NumericUpDown
        {
            Minimum = 0.00m,
            Maximum = 1.00m,
            DecimalPlaces = 2,
            Increment = 0.01m,
            Value = 0.25m,
            Dock = DockStyle.Fill
        };
        _chkShowBoxes = new CheckBox
        {
            Text = "显示目标框",
            AutoSize = true,
            Checked = true
        };
        _chkShowLabels = new CheckBox
        {
            Text = "显示标签",
            AutoSize = true,
            Checked = true
        };
        _chkShowScores = new CheckBox
        {
            Text = "显示分数",
            AutoSize = true,
            Checked = true
        };
        _chkShowAnnotations = new CheckBox
        {
            Text = "显示标注（真值）",
            AutoSize = true
        };
        _trkBoxThickness = CreateStyleTrackBar();
        _trkLabelFontSize = CreateStyleTrackBar();
        _trkScoreFontSize = CreateStyleTrackBar();
        _lblBoxThicknessValue = CreateStyleValueLabel();
        _lblLabelFontSizeValue = CreateStyleValueLabel();
        _lblScoreFontSizeValue = CreateStyleValueLabel();
        _btnBoxColor = new Button();
        _btnLabelColor = new Button();
        _btnScoreColor = new Button();
        _trkAnnotationThickness = CreateStyleTrackBar();
        _trkAnnotationFontSize = CreateStyleTrackBar();
        _lblAnnotationThicknessValue = CreateStyleValueLabel();
        _lblAnnotationFontSizeValue = CreateStyleValueLabel();
        _btnAnnotationColor = new Button();

        _chkAutoInfer = new CheckBox
        {
            Text = "自动轮询列表",
            AutoSize = true
        };
        _numAutoIntervalMs = new NumericUpDown
        {
            Minimum = 0,
            Maximum = 60000,
            DecimalPlaces = 0,
            Increment = 100,
            Value = 1000,
            Dock = DockStyle.Fill
        };
        _btnInferCurrent = new Button
        {
            Text = "推理当前"
        };

        _gridImages = new DataGridView
        {
            Dock = DockStyle.Fill,
            ReadOnly = true,
            AllowUserToAddRows = false,
            AllowUserToDeleteRows = false,
            AllowUserToResizeRows = false,
            AllowUserToResizeColumns = true,
            SelectionMode = DataGridViewSelectionMode.FullRowSelect,
            MultiSelect = false,
            RowHeadersVisible = false,
            AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None
        };
        _imageViewer = new ZoomableImageViewer
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(17, 24, 39),
            Margin = new Padding(0, 10, 0, 0)
        };
        _btnResetPreview = new Button
        {
            Text = "复位视图",
            AutoSize = true
        };
        _gridDetections = new DataGridView
        {
            Dock = DockStyle.Fill,
            ReadOnly = true,
            AllowUserToAddRows = false,
            AllowUserToDeleteRows = false,
            AllowUserToResizeRows = false,
            SelectionMode = DataGridViewSelectionMode.FullRowSelect,
            MultiSelect = false,
            RowHeadersVisible = false,
            AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill
        };
        _statusLabel = new ToolStripStatusLabel("就绪");

        AutoScaleDimensions = new System.Drawing.SizeF(7F, 17F);
        AutoScaleMode = AutoScaleMode.Font;
        BackColor = Color.FromArgb(244, 246, 248);
        Font = new Font("Microsoft YaHei UI", 9F, FontStyle.Regular, GraphicsUnit.Point);
        MinimumSize = new System.Drawing.Size(1320, 820);
        Name = "MainForm";
        StartPosition = FormStartPosition.CenterScreen;
        Text = "花粉 OpenVINO 推理演示";
        Size = new System.Drawing.Size(1500, 920);

        BuildLayout();

        ResumeLayout(false);
        PerformLayout();
    }

    private void BuildLayout()
    {
        _splitMain = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Vertical,
            SplitterDistance = 520,
            SplitterWidth = 8
        };
        Controls.Add(_splitMain);
        _splitMain.Panel1.Padding = new Padding(0, 0, 12, 0);

        var leftHost = new Panel
        {
            Dock = DockStyle.Fill,
            AutoScroll = true
        };
        _splitMain.Panel1.Controls.Add(leftHost);

        var leftLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            ColumnCount = 1,
            Padding = new Padding(0, 0, 4, 0)
        };
        leftLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        leftHost.Controls.Add(leftLayout);

        leftLayout.Controls.Add(CreateModelGroup(), 0, 0);
        leftLayout.Controls.Add(CreateInputGroup(), 0, 1);
        leftLayout.Controls.Add(CreateParamsGroup(), 0, 2);
        leftLayout.Controls.Add(CreateDisplayGroup(), 0, 3);
        leftLayout.Controls.Add(CreateInferGroup(), 0, 4);
        leftLayout.Controls.Add(CreateImageListGroup(), 0, 5);

        _splitRight = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Horizontal,
            SplitterDistance = 620,
            SplitterWidth = 8
        };
        _splitMain.Panel2.Controls.Add(_splitRight);
        _splitRight.Panel1.Padding = new Padding(0, 0, 0, 8);
        _splitRight.Panel2.Padding = new Padding(0, 8, 0, 0);
        _splitRight.Panel1.Controls.Add(CreatePreviewGroup());
        _splitRight.Panel2.Controls.Add(CreateDetectionsGroup());

        var statusStrip = new StatusStrip();
        statusStrip.Items.Add(_statusLabel);
        Controls.Add(statusStrip);
    }

    private GroupBox CreateModelGroup()
    {
        var group = new GroupBox
        {
            Text = "模型",
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(10),
            Margin = new Padding(0, 0, 0, 10)
        };

        var table = CreateThreeColumnTable(3);
        table.Controls.Add(new Label { Text = "检测 XML", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 0);
        table.Controls.Add(_txtModelPath, 1, 0);
        table.Controls.Add(_btnBrowseModel, 2, 0);

        table.Controls.Add(new Label { Text = "分类 XML", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 1);
        table.Controls.Add(_txtClassifierModelPath, 1, 1);
        table.Controls.Add(_btnBrowseClassifierModel, 2, 1);

        table.Controls.Add(new Label { Text = "预热次数", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 2);
        table.Controls.Add(_numWarmup, 1, 2);
        table.Controls.Add(_btnLoadModel, 2, 2);

        group.Controls.Add(table);
        return group;
    }

    private GroupBox CreateInputGroup()
    {
        var group = new GroupBox
        {
            Text = "输入",
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(10),
            Margin = new Padding(0, 0, 0, 10)
        };

        var table = CreateThreeColumnTable(3);
        table.Controls.Add(new Label { Text = "图片", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 0);
        table.Controls.Add(_txtImagePath, 1, 0);
        table.Controls.Add(_btnBrowseImage, 2, 0);

        table.Controls.Add(new Label { Text = "目录", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 1);
        table.Controls.Add(_txtFolderPath, 1, 1);
        table.Controls.Add(_btnBrowseFolder, 2, 1);

        table.Controls.Add(_lblImageIndex, 0, 2);
        table.SetColumnSpan(_lblImageIndex, 3);

        group.Controls.Add(table);
        return group;
    }

    private GroupBox CreateParamsGroup()
    {
        var group = new GroupBox
        {
            Text = "推理参数",
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(10),
            Margin = new Padding(0, 0, 0, 10)
        };

        var table = CreateThreeColumnTable(1);
        table.Controls.Add(new Label { Text = "置信度", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 0);
        table.Controls.Add(_numConf, 1, 0);
        table.SetColumnSpan(_numConf, 2);

        group.Controls.Add(table);
        return group;
    }

    private GroupBox CreateDisplayGroup()
    {
        var group = new GroupBox
        {
            Text = "显示样式",
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(10),
            Margin = new Padding(0, 0, 0, 10)
        };

        var table = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoSize = true,
            ColumnCount = 3,
            RowCount = 7
        };
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 120F));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 96F));

        var hint = new Label
        {
            Text = "每行可分别控制显示、粗细或字号，以及对应颜色。",
            AutoSize = true,
            ForeColor = Color.FromArgb(100, 116, 139),
            Margin = new Padding(0, 0, 0, 8)
        };
        table.Controls.Add(hint, 0, 0);
        table.SetColumnSpan(hint, 3);

        table.Controls.Add(_chkShowBoxes, 0, 1);
        table.Controls.Add(CreateStyleSliderHost(_trkBoxThickness, _lblBoxThicknessValue), 1, 1);
        table.Controls.Add(_btnBoxColor, 2, 1);
        table.Controls.Add(_chkShowLabels, 0, 2);
        table.Controls.Add(CreateStyleSliderHost(_trkLabelFontSize, _lblLabelFontSizeValue), 1, 2);
        table.Controls.Add(_btnLabelColor, 2, 2);
        table.Controls.Add(_chkShowScores, 0, 3);
        table.Controls.Add(CreateStyleSliderHost(_trkScoreFontSize, _lblScoreFontSizeValue), 1, 3);
        table.Controls.Add(_btnScoreColor, 2, 3);
        table.Controls.Add(_chkShowAnnotations, 0, 4);
        table.Controls.Add(CreateStyleSliderHost(_trkAnnotationThickness, _lblAnnotationThicknessValue), 1, 4);
        table.Controls.Add(_btnAnnotationColor, 2, 4);
        table.Controls.Add(new Label(), 0, 5);
        table.Controls.Add(CreateStyleSliderHost(_trkAnnotationFontSize, _lblAnnotationFontSizeValue), 1, 5);

        var annotationHint = new Label
        {
            Text = "如存在同名 LabelMe JSON，则叠加显示真值框和标签。",
            AutoSize = true,
            ForeColor = Color.FromArgb(100, 116, 139),
            Anchor = AnchorStyles.Left
        };
        table.Controls.Add(annotationHint, 0, 6);
        table.SetColumnSpan(annotationHint, 3);

        group.Controls.Add(table);
        return group;
    }

    private GroupBox CreateInferGroup()
    {
        var group = new GroupBox
        {
            Text = "推理",
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(10),
            Margin = new Padding(0, 0, 0, 10)
        };

        var table = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoSize = true,
            ColumnCount = 4,
            RowCount = 2
        };
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 45F));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 15F));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 20F));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 20F));

        _chkAutoInfer.Dock = DockStyle.Fill;
        _numAutoIntervalMs.Dock = DockStyle.Fill;
        _btnInferCurrent.Dock = DockStyle.Fill;

        table.Controls.Add(_chkAutoInfer, 0, 0);
        table.SetColumnSpan(_chkAutoInfer, 2);
        table.Controls.Add(new Label { Text = "间隔(ms)", AutoSize = true, Anchor = AnchorStyles.Left }, 2, 0);
        table.Controls.Add(_numAutoIntervalMs, 3, 0);

        table.Controls.Add(_btnInferCurrent, 0, 1);
        table.SetColumnSpan(_btnInferCurrent, 4);

        group.Controls.Add(table);
        return group;
    }

    private GroupBox CreateImageListGroup()
    {
        var group = new GroupBox
        {
            Text = "图片列表（点击表头可按名称/类型/大小/时间排序）",
            Dock = DockStyle.Top,
            Height = 310,
            Padding = new Padding(10),
            Margin = new Padding(0, 0, 0, 10)
        };
        group.Controls.Add(_gridImages);
        return group;
    }

    private GroupBox CreatePreviewGroup()
    {
        var group = new GroupBox
        {
            Text = "结果预览",
            Dock = DockStyle.Fill,
            Padding = new Padding(10)
        };

        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 2
        };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));

        var tipLabel = new Label
        {
            Text = "滚轮缩放 | 左键拖拽 | 双击复位",
            AutoSize = true,
            ForeColor = Color.FromArgb(100, 116, 139),
            Anchor = AnchorStyles.Left
        };

        layout.Controls.Add(tipLabel, 0, 0);
        layout.Controls.Add(_btnResetPreview, 1, 0);
        layout.Controls.Add(_imageViewer, 0, 1);
        layout.SetColumnSpan(_imageViewer, 2);

        group.Controls.Add(layout);
        return group;
    }

    private GroupBox CreateDetectionsGroup()
    {
        var group = new GroupBox
        {
            Text = "检测结果",
            Dock = DockStyle.Fill,
            Padding = new Padding(10)
        };
        group.Controls.Add(_gridDetections);
        return group;
    }

    private static TableLayoutPanel CreateStyleSliderHost(TrackBar trackBar, Label valueLabel)
    {
        var host = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoSize = true,
            ColumnCount = 2,
            Margin = new Padding(0, 0, 8, 0)
        };
        host.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        host.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 74F));
        host.Controls.Add(trackBar, 0, 0);
        host.Controls.Add(valueLabel, 1, 0);
        return host;
    }

    private static TableLayoutPanel CreateThreeColumnTable(int rowCount)
    {
        var table = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoSize = true,
            ColumnCount = 3,
            RowCount = rowCount
        };

        table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 80F));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 130F));

        for (var i = 0; i < rowCount; i++)
        {
            table.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        }

        return table;
    }

    private static TrackBar CreateStyleTrackBar()
    {
        return new TrackBar
        {
            Minimum = 50,
            Maximum = 300,
            Value = 100,
            SmallChange = 5,
            LargeChange = 20,
            TickFrequency = 10,
            TickStyle = TickStyle.None,
            AutoSize = false,
            Height = 28,
            Dock = DockStyle.Fill,
            Margin = new Padding(0)
        };
    }

    private static Label CreateStyleValueLabel()
    {
        return new Label
        {
            Dock = DockStyle.Fill,
            Width = 74,
            TextAlign = ContentAlignment.MiddleRight,
            ForeColor = Color.FromArgb(71, 85, 105),
            Margin = new Padding(0)
        };
    }
}
