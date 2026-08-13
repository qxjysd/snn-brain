package io.brainx.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * 脑图可视化 View —— Canvas 绘制大脑活动。
 * 上半区: 神经元发放热图 (每列=神经元, 每行=时间, 亮点=发放)
 * 下半区: 发放率实时曲线 (模拟 EEG 波形)
 */
public class BrainView extends View {
    private static final int NEURONS = 48;
    private static final int TRAIL = 64;

    private final Paint cellPaint = new Paint();
    private final Paint wavePaint = new Paint();
    private final Paint gridPaint = new Paint();
    private final Paint textPaint = new Paint();

    private final boolean[][] raster = new boolean[TRAIL][NEURONS];
    private final Deque<Float> waveHistory = new ArrayDeque<>();
    private float currentRate = 0;
    private String statusText = "大脑待机中...";
    private int fireCount = 0;

    public BrainView(Context ctx) {
        super(ctx);
        cellPaint.setColor(Color.parseColor("#00E5FF"));
        wavePaint.setColor(Color.parseColor("#4CAF50"));
        wavePaint.setStrokeWidth(3f);
        gridPaint.setColor(Color.parseColor("#1A2B4A"));
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(28f);
    }

    /** 每帧调用: 记录神经元发放模式 */
    public void updateRaster(boolean[] firing, float rate, String status) {
        System.arraycopy(firing, 0, raster[0], 0, Math.min(firing.length, NEURONS));
        // 右移历史
        for (int r = TRAIL - 1; r >= 1; r--) {
            System.arraycopy(raster[r - 1], 0, raster[r], 0, NEURONS);
        }
        currentRate = rate;
        statusText = status;
        waveHistory.addFirst(rate);
        while (waveHistory.size() > 256) waveHistory.removeLast();
        fireCount = 0;
        for (int i = 0; i < Math.min(firing.length, NEURONS); i++) if (firing[i]) fireCount++;
        postInvalidateOnAnimation();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth(), h = getHeight();

        // 背景
        canvas.drawColor(Color.parseColor("#0B1026"));

        // 上半: 发放热图
        int rasterH = h / 2;
        float cellW = (float) w / NEURONS;
        float cellH = (float) rasterH / TRAIL;
        for (int r = 0; r < TRAIL; r++) {
            for (int n = 0; n < NEURONS; n++) {
                if (raster[r][n]) {
                    // 亮点: 越新越亮
                    float alpha = 1f - (float) r / TRAIL;
                    cellPaint.setAlpha((int) (alpha * 255));
                    canvas.drawRect(n * cellW, r * cellH, (n + 1) * cellW, (r + 1) * cellH, cellPaint);
                }
            }
        }
        // 网格线
        gridPaint.setAlpha(40);
        for (int n = 0; n <= NEURONS; n += 8) {
            canvas.drawLine(n * cellW, 0, n * cellW, rasterH, gridPaint);
        }

        // 下半: EEG 波形
        int waveTop = rasterH + 10;
        int waveH = h - rasterH - 60;
        wavePaint.setColor(Color.parseColor("#4CAF50"));
        float[] pts = new float[waveHistory.size() * 4];
        int i = 0;
        float prevX = w, prevY = waveTop + waveH / 2f;
        for (float v : waveHistory) {
            float x = prevX - 4;
            float y = waveTop + waveH / 2f - v * waveH * 3f;
            if (i > 0) {
                pts[i * 4 - 4] = prevX; pts[i * 4 - 3] = prevY;
                pts[i * 4 - 2] = x;     pts[i * 4 - 1] = y;
            }
            prevX = x; prevY = y;
            i++;
        }
        if (i > 1) canvas.drawLines(pts, 0, (i - 1) * 4, wavePaint);

        // 状态文字
        textPaint.setColor(Color.WHITE);
        canvas.drawText(statusText, 20, h - 30, textPaint);

        // 发放计数
        textPaint.setColor(Color.parseColor("#FF9800"));
        canvas.drawText("⚡ " + fireCount + " spikes | 速率 " + String.format("%.1f", currentRate), 20, rasterH - 12, textPaint);
    }
}
