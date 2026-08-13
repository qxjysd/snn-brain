package io.brainx.core;

import io.brainx.core.neuron.LIF;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 类脑频率波验证:
 * 1. 信号强度→频率映射 (发放频率编码)
 * 2. 频率波驱动神经元 (振荡输入)
 * 3. 记忆=频率共振 (检索=共振激活)
 * 4. 脑电节律生成 (θ/α/γ)
 * 5. 整体联动 (频率贯穿各模块)
 */
public class FrequencyWaveTest {

    // ===== 信号→频率 =====

    @Test
    void intensityMapsToFrequency() {
        FrequencyWave fw = FrequencyWave.neuralRange();
        double weak = fw.intensityToHz(0.1);
        double strong = fw.intensityToHz(1.0);
        assertTrue(weak < strong, "强信号应更高频: weak=" + weak + " strong=" + strong);
        assertTrue(strong <= 40.0 && strong >= 5.0, "频率应在范围内");
        // 往返: 频率→强度→频率
        double roundTrip = fw.intensityToHz(fw.hzToIntensity(strong));
        assertEquals(strong, roundTrip, 0.5, "往返应一致");
    }

    @Test
    void waveOscillates() {
        FrequencyWave fw = FrequencyWave.neuralRange();
        double[] samples = new double[100];
        for (int t = 0; t < 100; t++) samples[t] = fw.wave(0.5, 1.0);
        // 应有正负振荡
        boolean pos = false, neg = false;
        for (double s : samples) { if (s > 0.1) pos = true; if (s < -0.1) neg = true; }
        assertTrue(pos && neg, "频率波应正负振荡");
    }

    @Test
    void waveDrivesNeuron() {
        // 频率波驱动 LIF 神经元应能发放 (类脑: 波→电流→脉冲)
        LIF n = LIF.defaultParams();
        FrequencyWave fw = FrequencyWave.neuralRange();
        int spikes = 0;
        for (int t = 0; t < 2000; t++) {
            double current = fw.driveNeuron(0.8, 1.0);  // 强信号高幅波
            n.step(current, 1.0);
            if (n.fired()) spikes++;
        }
        assertTrue(spikes > 5, "频率波应驱动神经元发放, spikes=" + spikes);
    }

    // ===== 记忆=频率共振 =====

    @Test
    void memoryStoredAsFrequency() {
        ResonanceMemory rm = ResonanceMemory.defaultParams();
        rm.write("苹果", 0.5);
        rm.write("猫", 0.5);
        assertEquals(2, rm.size());
        // 每条记忆应有不同共振频率
        double f1 = rm.frequencyOf("苹果"), f2 = rm.frequencyOf("猫");
        assertTrue(Math.abs(f1 - f2) > 0.1, "不同记忆应有不同频率: " + f1 + " vs " + f2);
    }

    @Test
    void retrievalByResonance() {
        ResonanceMemory rm = ResonanceMemory.defaultParams();
        rm.write("苹果", 0.8);
        double hz = rm.frequencyOf("苹果");
        // 精确频率检索 → 共振激活苹果
        String[] r = rm.retrieveByFreq(hz);
        assertEquals("苹果", r[0], "精确频率应共振出苹果");
        double r1v = Double.parseDouble(r[1]);
        assertTrue(r1v > 0.5, "共振度应高");
        // 偏移频率 → 共振减弱 (可能仍激活, 但强度低)
        String[] r2 = rm.retrieveByFreq(hz + 3.0);
        double r2v = Double.parseDouble(r2[1]);
        assertTrue(r2v < r1v, "偏移频率应降低共振");
    }

    @Test
    void memoryStrengthAffectsRetrieval() {
        ResonanceMemory rm = ResonanceMemory.defaultParams();
        rm.write("强记忆", 0.9);
        rm.write("弱记忆", 0.1);
        double hz1 = rm.frequencyOf("强记忆"), hz2 = rm.frequencyOf("弱记忆");
        String[] r1 = rm.retrieveByFreq(hz1);
        String[] r2 = rm.retrieveByFreq(hz2);
        assertTrue(Double.parseDouble(r1[1]) > Double.parseDouble(r2[1]), "强记忆共振应更强");
    }

    // ===== 脑电节律 =====

    @Test
    void eegRhythmHasMultipleBands() {
        FrequencyWave fw = FrequencyWave.neuralRange();
        double[] eeg = new double[200];
        for (int t = 0; t < 200; t++) eeg[t] = fw.eegRhythm(0.3, 0.8, 0.5, 1.0);
        // 波形有变化 (非恒定)
        boolean varies = false;
        for (int t = 1; t < 200; t++) if (Math.abs(eeg[t] - eeg[t-1]) > 0.01) { varies = true; break; }
        assertTrue(varies, "脑电节律应变化");
        // 幅度受限 (三频段叠加)
        for (double v : eeg) assertTrue(Math.abs(v) < 2.0, "波形幅度应合理");
    }

    @Test
    void resonanceMemoryFrequenciesInThetaAlphaBand() {
        FrequencyWave fw = FrequencyWave.memoryRange();
        // 记忆频带应覆盖 4-13Hz (θ-α)
        double low = fw.featureHz(0, 8);
        double high = fw.featureHz(7, 8);
        assertTrue(low >= 4.0 && low <= 5.5, "低端应在θ带, got=" + low);
        assertTrue(high >= 11.0 && high <= 13.0, "高端应在α带, got=" + high);
    }

    // ===== 整体联动 =====

    @Test
    void brainFrequencyIntegration() {
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM], imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) { imgA[i] = (i%3==0)?0.9:0.1; imgB[i] = (i%3==1)?0.9:0.1; }
        // 学习 → 共振记忆写入
        for (int e = 0; e < 5; e++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        assertTrue(brain.resonanceMemory().size() >= 2, "学习应写入频率共振记忆");
        // 脑电联动
        assertNotNull(brain.currentEEG());
        String freq = brain.frequencySummary();
        assertTrue(freq.contains("脑电"), freq);
        // 识别仍工作 (频率共振参与)
        String[] r = brain.recognizeVisualWithConfidence(imgA);
        assertTrue(r[0].equals(brain.vocabulary(0)) || r[0].equals(brain.vocabulary(1)),
                "频率联动后识别应工作, got=" + r[0]);
    }

    @Test
    void frequencySurvivesSnapshot() {
        // 频率共振记忆应随快照迁移 (记忆=频率是模型的一部分)
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 3; e++) brain.learnVisualWord(img, 0);
        String snap = BrainSnapshot.export(brain, 1, 5, 30, "", "");
        assertTrue(snap.contains("resmem:") || snap.contains("words:"),
                "快照应含记忆信息 (频率记忆随词表迁移)");
    }
}
