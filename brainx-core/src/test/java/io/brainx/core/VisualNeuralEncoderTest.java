package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 视觉神经编码验证: 图像直接转为神经信号 (视网膜感受野+方向选择性)。
 */
public class VisualNeuralEncoderTest {

    @Test
    void outputDimension() {
        VisualNeuralEncoder enc = new VisualNeuralEncoder();
        // 任意尺寸灰度图 → 96 维神经信号
        double[] img = new double[64 * 64];
        for (int i = 0; i < img.length; i++) img[i] = 128;
        double[] signal = enc.encode(img, 64, 64);
        assertEquals(VisualNeuralEncoder.OUTPUT_DIM, signal.length, "输出"+VisualNeuralEncoder.OUTPUT_DIM+"维");
        // 所有值在 0-1 (发放率)
        for (double v : signal) {
            assertTrue(v >= 0 && v <= 1.0, "神经信号应在0-1: " + v);
        }
    }

    @Test
    void uniformImageWeakRF() {
        VisualNeuralEncoder enc = new VisualNeuralEncoder();
        // 均匀灰度图 → 感受野拮抗响应弱 (无边缘对比)
        double[] img = new double[64 * 64];
        for (int i = 0; i < img.length; i++) img[i] = 100;
        double[] signal = enc.encode(img, 64, 64);
        // 感受野部分 (前 RF_GRID×RF_GRID) 响应应低 (均匀无对比)
        double rfSum = 0;
        for (int i = 0; i < VisualNeuralEncoder.RF_GRID * VisualNeuralEncoder.RF_GRID; i++) rfSum += signal[i];
        assertTrue(rfSum < 10, "均匀图感受野响应应弱, sum=" + rfSum);
    }

    @Test
    void edgeImageStrongRF() {
        VisualNeuralEncoder enc = new VisualNeuralEncoder();
        // 一半黑一半白 (强边缘) → 感受野拮抗响应强
        double[] img = new double[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                img[y * 64 + x] = x < 32 ? 0 : 255;  // 垂直边缘
            }
        }
        double[] signal = enc.encode(img, 64, 64);
        double rfSum = 0;
        for (int i = 0; i < VisualNeuralEncoder.RF_GRID * VisualNeuralEncoder.RF_GRID; i++) rfSum += signal[i];
        assertTrue(rfSum > 1, "边缘图感受野响应应强, sum=" + rfSum);
        // 方向特征: 垂直边缘 → 垂直方向 (dir=1) 能量高
        double vert = 0, horiz = 0;
        for (int i = 0; i < VisualNeuralEncoder.RF_GRID; i++) {
            horiz += signal[VisualNeuralEncoder.RF_GRID*VisualNeuralEncoder.RF_GRID + 0*VisualNeuralEncoder.DIR_SCALES*VisualNeuralEncoder.RF_GRID + i];  // dir0 水平边缘
            vert += signal[VisualNeuralEncoder.RF_GRID*VisualNeuralEncoder.RF_GRID + 1*VisualNeuralEncoder.DIR_SCALES*VisualNeuralEncoder.RF_GRID + i];   // dir1 垂直边缘
        }
        assertTrue(vert > horiz, "垂直边缘应垂直方向能量高: v=" + vert + " h=" + horiz);
    }

    @Test
    void horizontalEdgeDetected() {
        VisualNeuralEncoder enc = new VisualNeuralEncoder();
        double[] img = new double[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                img[y * 64 + x] = y < 32 ? 0 : 255;  // 水平边缘
            }
        }
        double[] signal = enc.encode(img, 64, 64);
        double vert = 0, horiz = 0;
        for (int i = 0; i < VisualNeuralEncoder.RF_GRID; i++) {
            horiz += signal[VisualNeuralEncoder.RF_GRID*VisualNeuralEncoder.RF_GRID + 0*VisualNeuralEncoder.DIR_SCALES*VisualNeuralEncoder.RF_GRID + i];
            vert += signal[VisualNeuralEncoder.RF_GRID*VisualNeuralEncoder.RF_GRID + 1*VisualNeuralEncoder.DIR_SCALES*VisualNeuralEncoder.RF_GRID + i];
        }
        assertTrue(horiz > vert, "水平边缘应水平方向能量高: h=" + horiz + " v=" + vert);
    }

    @Test
    void scaleInvariance() {
        VisualNeuralEncoder enc = new VisualNeuralEncoder();
        // 同一图案不同尺寸 → 神经信号近似 (编码器尺度不变性)
        double[] small = new double[32 * 32];
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                small[y * 32 + x] = x < 16 ? 0 : 255;
            }
        }
        double[] big = new double[128 * 128];
        for (int y = 0; y < 128; y++) {
            for (int x = 0; x < 128; x++) {
                big[y * 128 + x] = x < 64 ? 0 : 255;
            }
        }
        double[] s = enc.encode(small, 32, 32);
        double[] b = enc.encode(big, 128, 128);
        // 方向特征应相似 (都是垂直边缘)
        double sV = 0, bV = 0;
        for (int i = 0; i < VisualNeuralEncoder.RF_GRID; i++) {
            sV += s[VisualNeuralEncoder.RF_GRID*VisualNeuralEncoder.RF_GRID + VisualNeuralEncoder.DIR_SCALES*VisualNeuralEncoder.RF_GRID + i];
            bV += b[VisualNeuralEncoder.RF_GRID*VisualNeuralEncoder.RF_GRID + VisualNeuralEncoder.DIR_SCALES*VisualNeuralEncoder.RF_GRID + i];
        }
        assertEquals(0, Math.abs(sV - bV) > 0.3 ? 1 : 0, "尺度变化方向特征应近似: s=" + sV + " b=" + bV);
    }

    @Test
    void brainAcceptsNeuralSignal() {
        Brain brain = Brain.simpleBrain();
        assertEquals(VisualNeuralEncoder.OUTPUT_DIM, brain.visualCortexSize(),
                "视觉皮层应匹配" + VisualNeuralEncoder.OUTPUT_DIM + "维神经信号");
        // 用神经信号学习+识别
        VisualNeuralEncoder enc = new VisualNeuralEncoder();
        double[] imgA = new double[64 * 64];
        double[] imgB = new double[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                imgA[y * 64 + x] = x < 32 ? 0 : 255;   // 垂直边缘
                imgB[y * 64 + x] = y < 32 ? 0 : 255;   // 水平边缘
            }
        }
        double[] sigA = enc.encode(imgA, 64, 64);
        double[] sigB = enc.encode(imgB, 64, 64);
        for (int e = 0; e < 5; e++) {
            brain.learnVisualWord(sigA, 0);
            brain.learnVisualWord(sigB, 1);
        }
        String r = brain.recognizeVisual(sigA);
        assertTrue(r.equals("你好") || r.equals("苹果"), "神经信号识别应工作, got=" + r);
    }
}
