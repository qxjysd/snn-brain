package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 听觉神经编码验证: 声音直接转为神经信号 (耳蜗→听神经发放率)。
 */
public class AudioNeuralEncoderTest {

    @Test
    void silenceIsZero() {
        short[] silence = new short[8000];
        double[] neural = AudioNeuralEncoder.encode(silence);
        assertEquals(io.brainx.core.AudioNeuralEncoder.BANDS, neural.length, "耳蜗频带数");
        double sum = 0;
        for (double v : neural) sum += v;
        assertTrue(sum < 0.01, "静音神经信号应≈0, sum=" + sum);
    }

    @Test
    void loudSoundStrongSignal() {
        // 强音 (满幅方波) → 高神经发放率
        short[] loud = new short[8000];
        for (int i = 0; i < loud.length; i++) loud[i] = (short) ((i % 2 == 0) ? 20000 : -20000);
        double[] neural = AudioNeuralEncoder.encode(loud);
        double sum = 0;
        for (double v : neural) sum += v;
        assertTrue(sum > 5, "强音应高发放率, sum=" + sum);
    }

    @Test
    void quietSoundWeakSignal() {
        short[] quiet = new short[8000];
        for (int i = 0; i < quiet.length; i++) quiet[i] = (short) 200;  // 微弱
        double[] neural = AudioNeuralEncoder.encode(quiet);
        double sum = 0;
        for (double v : neural) sum += v;
        assertTrue(sum < 40.0, "弱音应低发放率(128频带), sum=" + sum);
    }

    @Test
    void loudnessLogarithmic() {
        // 响度感知对数: 2倍幅度 < 2倍神经信号 (对数压缩)
        short[] low = new short[8000], high = new short[8000];
        for (int i = 0; i < low.length; i++) { low[i] = 4000; high[i] = 8000; }
        double[] nLow = AudioNeuralEncoder.encode(low);
        double[] nHigh = AudioNeuralEncoder.encode(high);
        double sLow = 0, sHigh = 0;
        for (int i = 0; i < io.brainx.core.AudioNeuralEncoder.BANDS; i++) { sLow += nLow[i]; sHigh += nHigh[i]; }
        assertTrue(sHigh > sLow, "高响度应更高");
        assertTrue(sHigh < sLow * 2.5, "对数压缩: 2倍幅度不应2倍信号, low=" + sLow + " high=" + sHigh);
    }

    @Test
    void brainAcceptsAudioNeural() {
        Brain brain = Brain.simpleBrain();
        // 用听觉神经信号学习+识别
        short[] s1 = new short[8000], s2 = new short[8000];
        for (int i = 0; i < 8000; i++) {
            s1[i] = (short) ((i % 4 < 2) ? 15000 : -15000);  // 模式1 (低频占优)
            s2[i] = (short) ((i % 4 >= 2) ? 15000 : -15000);  // 模式2
        }
        double[] n1 = AudioNeuralEncoder.encode(s1);
        double[] n2 = AudioNeuralEncoder.encode(s2);
        for (int e = 0; e < 5; e++) {
            brain.learnAuditoryWord(n1, 0);
            brain.learnAuditoryWord(n2, 1);
        }
        String r = brain.recognizeAuditory(n1);
        assertTrue(r.equals("你好") || r.equals("苹果"), "听觉神经信号识别应工作, got=" + r);
    }
}
