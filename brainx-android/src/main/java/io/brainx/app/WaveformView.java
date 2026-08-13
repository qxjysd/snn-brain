package io.brainx.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

/**
 * 音波可视化 —— 声音输入/输出的实时波形显示。
 *
 * 显示:
 *   - 输入音波 (麦克风电信号 → 耳蜗解析的原始 PCM)
 *   - 输出音波 (大脑 → 模拟声带振动 → 合成 PCM)
 *   - 滚动波形 + 模式标签 (🎤 输入 / 🗣️ 输出)
 */
public class WaveformView extends View {
    /** 波形缓冲 (环形, 显示最近样本 — 16kHz 下 500ms) */
    private final float[] buffer;
    private int writePos = 0;
    private int count = 0;
    /** 当前模式 */
    private String mode = "🎤 输入";
    private int modeColor = Color.parseColor("#4FC3F7");

    private final Paint wavePaint = new Paint();
    private final Paint midPaint = new Paint();
    private final Paint textPaint = new Paint();
    private final Paint bgPaint = new Paint();
    private final Paint cellPaint2 = new Paint();

    public WaveformView(Context context) {
        super(context);
        buffer = new float[8000];
        wavePaint.setColor(Color.parseColor("#4FC3F7"));
        wavePaint.setStrokeWidth(3f);
        wavePaint.setAntiAlias(true);
        midPaint.setColor(Color.parseColor("#37474F"));
        midPaint.setStrokeWidth(1f);
        textPaint.setColor(Color.parseColor("#B0BEC5"));
        textPaint.setTextSize(22f);
        bgPaint.setColor(Color.parseColor("#10203A"));
    }

    /** 设为输入模式 (录音音波) */
    public void setInputMode() {
        mode = io.brainx.app.Lang.t("🎤 输入", "🎤 In");
        modeColor = Color.parseColor("#4FC3F7");
        wavePaint.setColor(modeColor);
        postInvalidate();
    }

    /** 设为输出模式 (声带合成音波) */
    public void setOutputMode() {
        mode = io.brainx.app.Lang.t("🗣️ 输出", "🗣️ Out");
        modeColor = Color.parseColor("#FFAB91");
        wavePaint.setColor(modeColor);
        postInvalidate();
    }

    /** 追加一批 PCM 样本 (录音/合成) */
    public void push(short[] samples, int len) {
        int n = Math.min(len, samples.length);
        synchronized (buffer) {
            for (int i = 0; i < n; i++) {
                buffer[writePos] = samples[i] / 32768.0f;
                writePos = (writePos + 1) % buffer.length;
                if (count < buffer.length) count++;
            }
        }
        postInvalidate();
    }

    /** 追加单样本 */
    public void pushSample(short s) {
        synchronized (buffer) {
            buffer[writePos] = s / 32768.0f;
            writePos = (writePos + 1) % buffer.length;
            if (count < buffer.length) count++;
        }
        postInvalidate();
    }

    /** 显示耳蜗频带频谱 (声音输入: 24频带能量条) */
    private double[] spectrum = null;
    private String spectrumLabel = "";

    public void showSpectrum(double[] features, String label) {
        this.spectrum = features != null ? features.clone() : null;
        this.spectrumLabel = label;
        postInvalidate();
    }

    /** 清空 */
    public void clear() {
        synchronized (buffer) {
            count = 0;
            writePos = 0;
            spectrum = null;
        }
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth(), h = getHeight();
        canvas.drawRect(0, 0, w, h, bgPaint);

        // 模式标签
        textPaint.setColor(modeColor);
        canvas.drawText(mode + (spectrum != null ? " 波形+频谱" : " 音波"), 16, 34, textPaint);
        textPaint.setColor(Color.parseColor("#546E7A"));
        canvas.drawText(spectrumLabel.isEmpty() ? String.format("%d 样本", count) : spectrumLabel,
                w - 180, 34, textPaint);

        // ===== 波形区 (上部 60%) — 实时输入波形, AGC 自动增益 =====
        int waveBottom = (int) (h * 0.62f);
        int midY = (34 + waveBottom) / 2;
        canvas.drawLine(16, midY, w - 16, midY, midPaint);
        synchronized (buffer) {
            int visible = Math.min(count, w - 32);
            if (visible >= 2) {
                // AGC: 用窗口最大幅度自适应 (弱音放大可见)
                float peak = 0.05f;
                int startIdx = (writePos - visible + buffer.length) % buffer.length;
                for (int i = 0; i < visible; i++) {
                    float s = buffer[(startIdx + i) % buffer.length];
                    peak = Math.max(peak, Math.abs(s));
                }
                float stepX = (w - 32) / (float) (visible - 1);
                float amp = (midY - 40) / Math.max(0.05f, peak);  // AGC 归一化
                float prevX = 16, prevY = midY;
                boolean first = true;
                for (int i = 0; i < visible; i++) {
                    float sample = buffer[(startIdx + i) % buffer.length];
                    float x = 16 + i * stepX;
                    float y = midY - sample * amp;
                    if (first) { prevX = x; prevY = y; first = false; continue; }
                    canvas.drawLine(prevX, prevY, x, y, wavePaint);
                    prevX = x;
                    prevY = y;
                }
            }
        }

        // ===== 频谱区 (下部 40%) — 128 频带聚合为 32 根清晰显示 =====
        if (spectrum != null && spectrum.length > 0) {
            int bands = spectrum.length;
            // 聚合: 128 → 32 根 (每 4 频带取最大)
            int agg = 32;
            float barW = (w - 40) / (float) agg;
            float baseY = h - 30;
            float maxH = (h - waveBottom) - 40;
            int perAgg = Math.max(1, bands / agg);
            for (int b = 0; b < agg; b++) {
                double v = 0;
                for (int k = 0; k < perAgg; k++) {
                    int idx = b * perAgg + k;
                    if (idx < bands) v = Math.max(v, spectrum[idx]);
                }
                float bh = (float) (Math.pow(v, 0.7) * maxH);
                int g = 255, r = (int) (v * 255);
                int col = Color.rgb(Math.min(255, r + 100), g, 60);
                cellPaint2.setColor(col);
                canvas.drawRect(16 + b * barW, baseY - bh, 16 + (b + 1) * barW - 2, baseY, cellPaint2);
            }
            textPaint.setColor(Color.parseColor("#66BB6A"));
            canvas.drawText("耳蜗频谱 (低频←→高频)", 16, baseY + 18, textPaint);
        }
    }
}
