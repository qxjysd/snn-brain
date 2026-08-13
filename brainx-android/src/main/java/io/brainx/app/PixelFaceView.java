package io.brainx.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

import io.brainx.core.EduTrainer;

/**
 * 像素表情 View —— 用像素网格绘制大脑的"脸"，随情绪变化。
 * 类似老式 8-bit 游戏像素脸：眼睛/眉毛/嘴巴由像素块构成。
 * 情绪 → 表情: 开心(^ ^)/沮丧(v v)/好奇(? ?)/兴奋(★ ★)/平静(- -)等
 */
public class PixelFaceView extends View {
    private static final int GRID = 16;   // 16x16 像素网格

    private final Paint pxPaint = new Paint();
    private final Paint bgPaint = new Paint();
    private final Paint borderPaint = new Paint();

    private EduTrainer.Emotion emotion = EduTrainer.Emotion.平静;
    private boolean[][] pixels = new boolean[GRID][GRID];

    public PixelFaceView(Context ctx) {
        super(ctx);
        pxPaint.setColor(Color.parseColor("#FFE082"));  // 皮肤色
        borderPaint.setColor(Color.parseColor("#1A2B4A"));
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(3f);
        renderFace();
    }

    public void setEmotion(EduTrainer.Emotion e) {
        this.emotion = e;
        renderFace();
        postInvalidate();
    }

    /** 根据情绪渲染像素脸 */
    private void renderFace() {
        for (int y = 0; y < GRID; y++)
            for (int x = 0; x < GRID; x++) pixels[x][y] = false;

        // 脸轮廓 (圆)
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                double dx = (x - 7.5) / 7.0, dy = (y - 8.5) / 8.0;
                if (dx * dx + dy * dy < 1.0) pixels[x][y] = true;
            }
        }
        // 腮红/眼白去掉底色部分绘制眼睛嘴巴
        drawEyesAndMouth();
    }

    private void drawEyesAndMouth() {
        // 根据情绪画眼睛
        switch (emotion) {
            case 开心: case 兴奋: case 骄傲:
                drawEye(5, 6, true);   // ^ 眼
                drawEye(10, 6, true);
                break;
            case 沮丧:
                drawEye(5, 8, false);  // v 眼
                drawEye(10, 8, false);
                break;
            case 好奇:
                drawEye(5, 6, true);
                drawEye(10, 6, true);
                break;
            case 惊讶:
                drawDot(5, 7); drawDot(10, 7);  // 圆眼
                break;
            default:  // 平静/困惑
                drawEye(5, 7, true);
                drawEye(10, 7, true);
                break;
        }
        // 嘴巴
        switch (emotion) {
            case 开心: case 兴奋: case 骄傲:
                drawSmile(); break;
            case 沮丧:
                drawFrown(); break;
            case 好奇:
                drawO(); break;       // 小 o 嘴
            case 惊讶:
                drawBigO(); break;    // 大 O 嘴
            case 困惑:
                drawWry(); break;     // 歪嘴
            default:
                drawFlat(); break;    // 平嘴
        }
    }

    /** 眉毛/眼: up=true 弯向上 (开心), false 弯向下 (沮丧) */
    private void drawEye(int cx, int cy, boolean up) {
        if (up) {
            pixels[cx-1][cy-1] = true;
            pixels[cx][cy] = true;
            pixels[cx+1][cy-1] = true;
        } else {
            pixels[cx-1][cy+1] = true;
            pixels[cx][cy] = true;
            pixels[cx+1][cy+1] = true;
        }
    }

    private void drawDot(int x, int y) {
        pixels[x][y] = true;
    }

    private void drawSmile() {
        pixels[5][11] = true; pixels[6][12] = true; pixels[7][12] = true;
        pixels[8][12] = true; pixels[9][12] = true; pixels[10][11] = true;
    }

    private void drawFrown() {
        pixels[5][13] = true; pixels[6][12] = true; pixels[7][12] = true;
        pixels[8][12] = true; pixels[9][12] = true; pixels[10][13] = true;
    }

    private void drawO() {
        pixels[7][12] = true; pixels[8][12] = true;
    }

    private void drawBigO() {
        pixels[6][12] = true; pixels[7][11] = true; pixels[8][11] = true; pixels[9][12] = true;
        pixels[6][13] = true; pixels[9][13] = true;
    }

    private void drawWry() {
        pixels[5][12] = true; pixels[6][12] = true; pixels[7][12] = true;
        pixels[8][13] = true; pixels[9][13] = true;
    }

    private void drawFlat() {
        pixels[6][12] = true; pixels[7][12] = true; pixels[8][12] = true; pixels[9][12] = true;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth(), h = getHeight();
        int cell = Math.min(w, h) / GRID;
        int offsetX = (w - cell * GRID) / 2;
        int offsetY = (h - cell * GRID) / 2;

        // 背景圆
        bgPaint.setColor(emotion == EduTrainer.Emotion.沮丧 ? Color.parseColor("#3A2A4A")
                : emotion == EduTrainer.Emotion.兴奋 ? Color.parseColor("#2A3A5A")
                : Color.parseColor("#16233F"));
        canvas.drawRect(0, 0, w, h, bgPaint);

        // 像素脸
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                if (pixels[x][y]) {
                    canvas.drawRect(offsetX + x * cell, offsetY + y * cell,
                            offsetX + (x + 1) * cell, offsetY + (y + 1) * cell, pxPaint);
                }
            }
        }
    }
}
