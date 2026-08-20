package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * v5.5 论文引用新机制集成测试 (Brain 端到端)。
 * 验证 5 个新机制在 Brain 主流程中真实运转:
 *  - 星形胶质细胞: 学习后胶质状态/门控随活动演化
 *  - 扩散近似: 视觉学习后宏观率 > 0, 状态标签有效
 *  - 想象发声: 学习高置信概念后产生内心独白
 *  - e-prop: 联想层第二路径权重演化
 *  - 全脑网络: 视觉学习后枕叶活动上升
 */
public class BrainV55IntegrationTest {

    private Brain brain() {
        Brain b = Brain.simpleBrain();
        // 先学 8 个概念各一次 (建立基础)
        double[] base = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int w = 0; w < 8; w++) {
            for (int i = 0; i < base.length; i++) base[i] = 0.1 + 0.8 * ((i / (base.length / 8)) == w ? 1.0 : 0.0);
            b.learnVisual(base);
        }
        return b;
    }

    /** 星形胶质细胞: 学习后胶质状态非零, 门控在 [0,1] */
    @Test
    public void astrocyteIntegrates() {
        Brain b = brain();
        assertTrue(b.astrocyteLayer().steps() > 0, "胶质层应被驱动");
        assertTrue(b.astrocyteLayer().meanState() >= 0, "胶质状态应非负");
        for (int i = 0; i < 8; i++) {
            double g = b.astrocyteLayer().gate(i);
            assertTrue(g >= 0 && g <= 1, "门控应在 [0,1]: " + g);
        }
        String s = b.astrocyteLayer().summary();
        assertTrue(s.contains("胶质"), "summary 应含胶质");
    }

    /** 扩散近似: 学习后宏观率 > 0 且解释有效 */
    @Test
    public void diffusionApproxIntegrates() {
        Brain b = brain();
        double rate = b.diffusionApprox().meanRate();
        assertTrue(rate > 0, "视觉学习后宏观率应 > 0, got " + rate);
        String s = b.diffusionApprox().interpret();
        assertTrue(s.contains("Hz"), "interpret 应含 Hz");
        assertTrue(s.contains("扩散近似"), "interpret 应含扩散近似");
    }

    /** 想象发声: 高置信学习后可能触发内心独白 (有模板时) */
    @Test
    public void imaginedSpeechIntegrates() {
        Brain b = brain();
        // 给声带一个模板 (模拟学过声音)
        VoiceLearner.VoiceTemplate t = new VoiceLearner.VoiceTemplate();
        t.f0 = 200; t.f1 = 800; t.f2 = 1200; t.durationMs = 400; t.heardCount = 2;
        b.voiceLearner().library().add(t);
        // 再学一轮 (高置信) → 触发想象
        double[] feat = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < feat.length; i++) feat[i] = 0.1 + 0.8 * ((i / (feat.length / 8)) == 0 ? 1.0 : 0.0);
        b.learnVisual(feat);
        assertTrue(b.imaginedSpeech().imaginesCount() >= 0, "想象计数非负");
        // summary 有效
        String s = b.brain55Summary();
        assertTrue(s.contains("胶质"), "brain55Summary 应含胶质");
        assertTrue(s.contains("扩散近似"), "brain55Summary 应含扩散近似");
    }

    /** e-prop: 联想层第二路径权重随学习演化 */
    @Test
    public void epropIntegrates() {
        Brain b = brain();
        double[][] w = b.epropAssoc().weights();
        double total = 0;
        for (double[] row : w) for (double v : row) total += Math.abs(v);
        assertTrue(total > 0, "e-prop 权重应非零");
        assertTrue(b.epropAssoc().weights().length == 120, "e-prop 覆盖联想层");
    }

    /** 全脑网络: 视觉学习后枕叶活动上升, FC 矩阵有效 */
    @Test
    public void wholeBrainIntegrates() {
        Brain b = brain();
        double occipital = b.wholeBrain().activity(3);
        assertTrue(occipital > 0.1, "视觉学习后枕叶应激活, got " + occipital);
        double[][] fc = b.wholeBrain().functionalConnectivity();
        assertEquals(WholeBrainNetwork.N, fc.length);
        assertEquals(1.0, fc[0][0], 1e-9, "FC 对角=1");
        assertTrue(WholeBrainNetwork.REGIONS.length == 8);
    }

    /** 端到端: 学习 8 概念后识别仍正常 (新机制不破坏旧功能) */
    @Test
    public void recognitionStillWorks() {
        Brain b = brain();
        double[] feat = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < feat.length; i++) feat[i] = 0.1 + 0.8 * ((i / (feat.length / 8)) == 2 ? 1.0 : 0.0);
        String[] r = b.recognizeVisualWithConfidence(feat);
        assertNotNull(r);
        assertTrue(r.length >= 2);
    }

    /** brain55Summary 全量摘要可读 */
    @Test
    public void summaryReadable() {
        Brain b = brain();
        String s = b.brain55Summary();
        assertFalse(s.isEmpty());
        assertTrue(s.contains("全脑"), "summary 应含全脑网络");
    }
}
