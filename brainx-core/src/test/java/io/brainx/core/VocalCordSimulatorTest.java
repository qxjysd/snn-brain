package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 模拟声带验证: 大脑神经信号 → 声带振动 → PCM 声波 (无 TTS)。
 */
public class VocalCordSimulatorTest {

    @Test
    void producesPCM() {
        VocalCordSimulator vc = new VocalCordSimulator();
        short[] pcm = vc.synthesize("你好");
        assertTrue(pcm.length > 1000, "应产生PCM采样, len=" + pcm.length);
        // 有非零信号 (声带振动)
        boolean nonZero = false;
        for (short s : pcm) if (Math.abs(s) > 100) { nonZero = true; break; }
        assertTrue(nonZero, "声带应产生振动信号");
    }

    @Test
    void vowelPeriodic() {
        VocalCordSimulator vc = new VocalCordSimulator();
        short[] pcm = vc.synthesize("啊");  // 元音 a
        // 元音应周期性 (声带脉冲) — 检查过零率合理 (非纯噪声)
        int zeroCrossings = 0;
        for (int i = 1; i < pcm.length; i++) {
            if (pcm[i] >= 0 && pcm[i-1] < 0) zeroCrossings++;
        }
        // 220Hz 基频, 180ms → 约 40 周期 → 80 过零 (考虑共振峰更高)
        assertTrue(zeroCrossings > 10 && zeroCrossings < 1000,
                "元音应周期性振动, crossings=" + zeroCrossings);
    }

    @Test
    void differentVowelsDiffer() {
        VocalCordSimulator vc = new VocalCordSimulator();
        short[] a = vc.synthesize("啊");  // a
        short[] i = vc.synthesize("衣");  // i
        // 不同元音 → 共振峰不同 → 波形不同
        boolean differs = false;
        int n = Math.min(a.length, i.length);
        for (int k = 0; k < n; k += 20) {
            if (Math.abs(a[k] - i[k]) > 2000) { differs = true; break; }
        }
        assertTrue(differs, "不同元音应产生不同波形");
    }

    @Test
    void pitchAdjustable() {
        VocalCordSimulator vc = new VocalCordSimulator();
        vc.setBaseF0(150);
        double lowF0 = vc.baseF0();
        vc.setBaseF0(300);
        double highF0 = vc.baseF0();
        // 高基频 → 声带振动更快
        assertTrue(highF0 > lowF0, "高基频应更高: low=" + lowF0 + " high=" + highF0);
        // 合成验证: 高基频音节应产生可检测基频 (人声周期性)
        vc.setBaseF0(150);
        short[] low = vc.synthesize("你");
        vc.setBaseF0(300);
        short[] high = vc.synthesize("你");
        int lowCross = countCrossings(low), highCross = countCrossings(high);
        assertTrue(lowCross > 20 && highCross > 20,
                "人声应有周期性过零: low=" + lowCross + " high=" + highCross);
    }

    private int countCrossings(short[] pcm) {
        int c = 0;
        for (int i = 1; i < pcm.length; i++) {
            if (pcm[i] >= 0 && pcm[i-1] < 0) c++;
        }
        return c;
    }

    @Test
    void brainVocalOutput() {
        Brain brain = Brain.simpleBrain();
        // 大脑自主说话 → 声带合成 (替换 TTS)
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 4; e++) brain.learnVisualWord(img, e % 4);
        String speech = brain.speakAutonomously();
        assertTrue(!speech.isEmpty(), "大脑应产生话语");
        // 声带合成
        VocalCordSimulator vc = new VocalCordSimulator();
        short[] pcm = vc.synthesize(speech.replaceAll("[🗣️💬！。，？]", ""));
        assertTrue(pcm.length > 0, "声带应合成PCM");
    }
}
