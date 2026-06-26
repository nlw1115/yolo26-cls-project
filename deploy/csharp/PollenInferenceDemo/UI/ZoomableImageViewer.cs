using System.Drawing.Drawing2D;

namespace PollenInferenceDemo.UI;

internal sealed class ZoomableImageViewer : Control
{
    private const float ZoomStep = 1.1f;
    private const float MinZoom = 0.05f;
    private const float MaxZoom = 20.0f;

    private Image? _image;
    private float _zoom = 1.0f;
    private PointF _pan = PointF.Empty;
    private bool _isPanning;
    private Point _lastMousePoint;
    private bool _keepCustomView;

    public ZoomableImageViewer()
    {
        DoubleBuffered = true;
        TabStop = true;
        BackColor = Color.Black;
        Cursor = Cursors.Hand;

        SetStyle(
            ControlStyles.AllPaintingInWmPaint |
            ControlStyles.OptimizedDoubleBuffer |
            ControlStyles.ResizeRedraw |
            ControlStyles.UserPaint,
            true);
    }

    public Image? DisplayImage => _image;

    public void SetImage(Image? image, bool resetView = true)
    {
        _image = image;
        _isPanning = false;

        if (image is null)
        {
            _zoom = 1.0f;
            _pan = PointF.Empty;
            _keepCustomView = false;
            Invalidate();
            return;
        }

        if (resetView)
        {
            ResetView();
            return;
        }

        ClampPan();
        Invalidate();
    }

    public void ResetView()
    {
        if (_image is null || ClientSize.Width <= 0 || ClientSize.Height <= 0)
        {
            _zoom = 1.0f;
            _pan = PointF.Empty;
            _keepCustomView = false;
            Invalidate();
            return;
        }

        _zoom = CalculateFitZoom();
        _pan = GetCenteredPan(_zoom);
        _keepCustomView = false;
        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.Clear(BackColor);

        if (_image is null)
        {
            DrawEmptyHint(e.Graphics);
            return;
        }

        e.Graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
        e.Graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
        e.Graphics.SmoothingMode = SmoothingMode.HighQuality;

        var destRect = new RectangleF(
            _pan.X,
            _pan.Y,
            _image.Width * _zoom,
            _image.Height * _zoom);

        e.Graphics.DrawImage(_image, destRect);
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);

        if (_image is null)
        {
            return;
        }

        if (_keepCustomView)
        {
            ClampPan();
            Invalidate();
            return;
        }

        ResetView();
    }

    protected override void OnMouseEnter(EventArgs e)
    {
        base.OnMouseEnter(e);
        Focus();
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);

        if (e.Button != MouseButtons.Left || _image is null)
        {
            return;
        }

        Focus();
        _isPanning = true;
        _lastMousePoint = e.Location;
        Cursor = Cursors.SizeAll;
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);

        if (!_isPanning || _image is null)
        {
            return;
        }

        var dx = e.X - _lastMousePoint.X;
        var dy = e.Y - _lastMousePoint.Y;

        _pan = new PointF(_pan.X + dx, _pan.Y + dy);
        _lastMousePoint = e.Location;
        _keepCustomView = true;
        ClampPan();
        Invalidate();
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);

        if (e.Button != MouseButtons.Left)
        {
            return;
        }

        _isPanning = false;
        Cursor = Cursors.Hand;
    }

    protected override void OnMouseLeave(EventArgs e)
    {
        base.OnMouseLeave(e);

        if (!_isPanning)
        {
            return;
        }

        _isPanning = false;
        Cursor = Cursors.Hand;
    }

    protected override void OnMouseDoubleClick(MouseEventArgs e)
    {
        base.OnMouseDoubleClick(e);

        if (e.Button == MouseButtons.Left)
        {
            ResetView();
        }
    }

    protected override void OnMouseWheel(MouseEventArgs e)
    {
        base.OnMouseWheel(e);

        if (_image is null)
        {
            return;
        }

        Focus();

        var oldZoom = _zoom;
        var zoomFactor = e.Delta > 0 ? ZoomStep : 1.0f / ZoomStep;
        var newZoom = Math.Clamp(oldZoom * zoomFactor, MinZoom, MaxZoom);

        if (Math.Abs(newZoom - oldZoom) < 0.0001f)
        {
            return;
        }

        var imagePoint = new PointF(
            (e.X - _pan.X) / oldZoom,
            (e.Y - _pan.Y) / oldZoom);

        _zoom = newZoom;
        _pan = new PointF(
            e.X - imagePoint.X * _zoom,
            e.Y - imagePoint.Y * _zoom);

        _keepCustomView = true;
        ClampPan();
        Invalidate();
    }

    private void DrawEmptyHint(Graphics graphics)
    {
        using var brush = new SolidBrush(Color.FromArgb(180, Color.White));
        using var format = new StringFormat
        {
            Alignment = StringAlignment.Center,
            LineAlignment = StringAlignment.Center
        };

        graphics.DrawString(
            "暂无预览图像",
            Font,
            brush,
            ClientRectangle,
            format);
    }

    private float CalculateFitZoom()
    {
        if (_image is null || _image.Width <= 0 || _image.Height <= 0)
        {
            return 1.0f;
        }

        var zoomX = (float)ClientSize.Width / _image.Width;
        var zoomY = (float)ClientSize.Height / _image.Height;
        var fitZoom = Math.Min(zoomX, zoomY);

        if (float.IsNaN(fitZoom) || float.IsInfinity(fitZoom) || fitZoom <= 0)
        {
            return 1.0f;
        }

        return fitZoom;
    }

    private PointF GetCenteredPan(float zoom)
    {
        if (_image is null)
        {
            return PointF.Empty;
        }

        var scaledWidth = _image.Width * zoom;
        var scaledHeight = _image.Height * zoom;

        return new PointF(
            (ClientSize.Width - scaledWidth) / 2.0f,
            (ClientSize.Height - scaledHeight) / 2.0f);
    }

    private void ClampPan()
    {
        if (_image is null)
        {
            return;
        }

        var scaledWidth = _image.Width * _zoom;
        var scaledHeight = _image.Height * _zoom;

        if (scaledWidth <= ClientSize.Width)
        {
            _pan.X = (ClientSize.Width - scaledWidth) / 2.0f;
        }
        else
        {
            _pan.X = Math.Clamp(_pan.X, ClientSize.Width - scaledWidth, 0.0f);
        }

        if (scaledHeight <= ClientSize.Height)
        {
            _pan.Y = (ClientSize.Height - scaledHeight) / 2.0f;
        }
        else
        {
            _pan.Y = Math.Clamp(_pan.Y, ClientSize.Height - scaledHeight, 0.0f);
        }
    }
}
