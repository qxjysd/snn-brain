package io.brainx.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

/**
 * 大脑视觉视图 —— 电信号大脑解码可视化 (完整 5120 维)。
 *
 * 高度与相机预览一致 (4:3), 布局无文字重叠:
 *   - 标题独占顶部一行 → 感受野方块从标题下方开始 (不重叠)
 *   - 感受野方块说明文字画在方块内部顶部 (半透明底)
 *   - 竖屏 (w < h): 标题 → 感受野方块(全宽优先) → 全宽方向条带(行高弹性) → 解码文本
 *   - 横屏 (w ≥ h): 标题 → 左:感受野方块(全高) + 右:方向条带列(全高) → 解码文本
 *   - 感受野网格 (64×64=4096): 视网膜中心-周边拮抗脉冲 → 青白亮点
 *   - 方向网格 (4方向×4尺度×64=1024): V1 方向选择性脉冲 → 橙红亮点
 */
public class BrainVisionView extends View {
    /** 感受野网格 (与编码器同步: 64) */
    private static final int RF_GRID = io.brainx.core.VisualNeuralEncoder.RF_GRID;
    /** 方向尺度 (4) */
    private static final int DIR_SCALES = io.brainx.core.VisualNeuralEncoder.DIR_SCALES;
    /** 方向数 */
    private static final int DIR_COUNT = 4;

    private double[] features = new double[io.brainx.core.VisualNeuralEncoder.OUTPUT_DIM];
    private String label = "等待电信号...";
    private double confidence = 0;
    private String cameraDir = "";
    private boolean hasFrame = false;

    private final Paint cellPaint = new Paint();
    private final Paint textPaint = new Paint();
    private final Paint borderPaint = new Paint();
    private final Paint bgPaint = new Paint();
    private final Paint captionBgPaint = new Paint();
    private final String[] DIR_NAMES = {"水平", "垂直", "对角\\", "对角/"};
    private final String[] SCALE_NAMES = {"细", "中", "粗", "极粗"};

    public BrainVisionView(Context context) {
        super(context);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(26f);
        textPaint.setAntiAlias(true);
        borderPaint.setColor(Color.parseColor("#26C6DA"));
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(2f);
        bgPaint.setColor(Color.parseColor("#0A1628"));
        captionBgPaint.setColor(Color.parseColor("#AA0A1628"));
    }

    /** 更新大脑电信号 (完整 5120 维) */
    public void setVisual(double[] features, String label, double confidence, String cameraDir) {
        if (features != null) {
            this.features = features.clone();
            this.hasFrame = true;
        }
        this.label = label;
        this.confidence = confidence;
        this.cameraDir = cameraDir;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth(), h = getHeight();
        canvas.drawRect(0, 0, w, h, bgPaint);

        // 标题独占顶部一行 (下方即方块区, 不重叠) — 双语
        textPaint.setTextSize(30f);
        textPaint.setColor(Color.parseColor("#26C6DA"));
        canvas.drawText(io.brainx.app.Lang.t(
                        "⚡ 大脑视觉电信号 (" + io.brainx.core.VisualNeuralEncoder.OUTPUT_DIM + "维解码)",
                        "⚡ Brain Visual (" + io.brainx.core.VisualNeuralEncoder.OUTPUT_DIM + "d)"),
                20, 40, textPaint);

        int pad = 16;
        int titleH = 60;          // 标题区高度 (方块从下方开始)
        int gap = 12;
        int decodeH = 96;
        int dirRows = DIR_COUNT * DIR_SCALES;      // 16 行

        if (w < h) {
            // ===== 竖屏: 上下排布 (RF 全宽优先, 方向条行高弹性) =====
            int dirRowH = 26;
            int dirH = dirRows * dirRowH;
            int minRfW = (int) ((w - 2 * pad) * 0.60);   // RF 至少 60% 宽
            int rfSide = Math.min(w - 2 * pad, h - titleH - gap - dirH - decodeH);
            if (rfSide < minRfW) {
                // 高度不足: 压缩方向条行高 (最小 5px)
                dirRowH = Math.max(5, (h - titleH - gap - decodeH - minRfW) / dirRows);
                dirH = dirRows * dirRowH;
                rfSide = Math.min(w - 2 * pad, h - titleH - gap - dirH - decodeH);
            }
            if (rfSide < minRfW) {
                // 仍不足: 再压缩解码文本区
                decodeH = 60;
                dirRowH = Math.max(5, (h - titleH - gap - decodeH - minRfW) / dirRows);
                dirH = dirRows * dirRowH;
                rfSide = Math.min(w - 2 * pad, h - titleH - gap - dirH - decodeH);
            }
            if (rfSide < 40) rfSide = 40;
            int rfX = (w - rfSide) / 2;
            int rfY = titleH;
            int dirY = rfY + rfSide + gap;

            // 感受野脉冲网格
            drawRF(canvas, rfX, rfY, rfSide, rfSide / (float) RF_GRID, rfSide / (float) RF_GRID);

            // 方向脉冲条带 (全宽)
            int dirLabelW = 92;
            float dCellW = (w - 2 * pad - dirLabelW) / (float) RF_GRID;
            drawDir(canvas, pad, dirY, w - 2 * pad, dirH, dirRows, dirRowH, dirLabelW, dCellW);

            // 解码结果
            drawDecode(canvas, w, dirY + dirH + 34, decodeH);
        } else {
            // ===== 横屏/方形: 左右分栏 (RF 全高 + 方向列全高) =====
            int availH = h - titleH - gap - decodeH;
            if (availH < 40) availH = 40;
            int rfSide = Math.min(availH, (int) ((w - 2 * pad) * 0.62));
            if (rfSide < 40) rfSide = 40;
            int rfX = pad;
            int rfY = titleH;
            int dirX = rfX + rfSide + gap;
            int dirW = Math.max(60, w - pad - dirX);
            float dCellH = availH / (float) dirRows;

            drawRF(canvas, rfX, rfY, rfSide, rfSide / (float) RF_GRID, rfSide / (float) RF_GRID);

            int dirLabelW = 56;
            float dCellW = (dirW - dirLabelW) / (float) RF_GRID;
            drawDir(canvas, dirX, rfY, dirW, availH, dirRows, dCellH, dirLabelW, dCellW);

            drawDecode(canvas, w, h - decodeH + 26, decodeH);
        }
    }

    /** 感受野脉冲网格 (青白亮点, 暗底); 说明文字画在方块内部顶部 (半透明底, 不与外部文字重叠) */
    private void drawRF(Canvas canvas, int x, int y, int side, float cellW, float cellH) {
        for (int gy = 0; gy < RF_GRID; gy++) {
            for (int gx = 0; gx < RF_GRID; gx++) {
                int idx = gy * RF_GRID + gx;
                double v = hasFrame && idx < features.length ? features[idx] : 0;
                int bright = (int) (Math.pow(v, 0.6) * 255);
                cellPaint.setColor(Color.rgb(bright, bright, Math.min(255, bright + 40)));
                canvas.drawRect(x + gx * cellW, y + gy * cellH,
                        x + (gx + 1) * cellW, y + (gy + 1) * cellH, cellPaint);
            }
        }
        canvas.drawRect(x, y, x + side, y + side, borderPaint);
        // 说明文字: 方块内部顶部半透明底 (不与标题/方向条重叠)
        String caption = io.brainx.app.Lang.t(
                "视网膜感受野拮抗 (" + RF_GRID + "×" + RF_GRID + "=" + (RF_GRID * RF_GRID) + ")",
                "Retinal RF (" + RF_GRID + "×" + RF_GRID + "=" + (RF_GRID * RF_GRID) + ")");
        textPaint.setTextSize(20f);
        textPaint.setColor(Color.parseColor("#4DD0E1"));
        float cw = textPaint.measureText(caption) + 16;
        canvas.drawRoundRect(new RectF(x + 6, y + 6, x + 6 + cw, y + 34), 6, 6, captionBgPaint);
        canvas.drawText(caption, x + 14, y + 27, textPaint);
    }

    /** 方向脉冲条带/列 (橙红, V1 方向选择性); 行标签在左侧、细胞在右侧, 互不重叠 */
    private void drawDir(Canvas canvas, int x, int y, int w, int h,
                         int dirRows, float rowH, int labelW, float cellW) {
        for (int d = 0; d < DIR_COUNT; d++) {
            for (int s = 0; s < DIR_SCALES; s++) {
                int row = d * DIR_SCALES + s;
                float rowY = y + row * rowH;
                textPaint.setTextSize(18f);
                textPaint.setColor(Color.parseColor("#FFAB40"));
                canvas.drawText(DIR_NAMES[d] + SCALE_NAMES[s], x + 4, rowY + rowH - 8, textPaint);
                for (int pos = 0; pos < RF_GRID; pos++) {
                    int idx = RF_GRID * RF_GRID + (d * DIR_SCALES + s) * RF_GRID + pos;
                    double v = hasFrame && idx < features.length ? features[idx] : 0;
                    int bright = (int) (Math.pow(v, 0.6) * 255);
                    cellPaint.setColor(Color.rgb(255, bright / 2, 40));
                    canvas.drawRect(x + labelW + pos * cellW, rowY,
                            x + labelW + (pos + 1) * cellW, rowY + rowH, cellPaint);
                }
            }
        }
        canvas.drawRect(x, y, x + w, y + h, borderPaint);
    }

    /** 解码结果文本 */
    private void drawDecode(Canvas canvas, int w, int ty, int decodeH) {
        textPaint.setTextSize(28f);
        if (hasFrame) {
            textPaint.setColor(Color.parseColor("#A5D6A7"));
            canvas.drawText("大脑解码: " + label, 20, ty, textPaint);
            textPaint.setColor(Color.parseColor("#FFF59D"));
            canvas.drawText("置信度: " + String.format("%.0f%%", confidence * 100),
                    20, ty + 36, textPaint);
            textPaint.setColor(Color.parseColor("#B39DDB"));
            canvas.drawText("📷 " + cameraDir, w - 200, ty, textPaint);
        } else {
            textPaint.setColor(Color.parseColor("#546E7A"));
            canvas.drawText("⚡ 等待电信号... 摄像头观察中", 20, ty, textPaint);
        }
    }
}
