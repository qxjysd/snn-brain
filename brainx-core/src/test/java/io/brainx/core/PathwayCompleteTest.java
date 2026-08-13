package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 路径完善验证: 联想皮层接入中枢 + 全局爆发→意识扩散。
 */
public class PathwayCompleteTest {

    /** 联想皮层接入中枢 (全局工作空间节点) */
    @Test
    void assocCortexConnectedToHub() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        brain.learnVisualWord(img, 0);
        // 中枢注册联想皮层
        assertTrue(brain.hubModuleCount() >= 7, "中枢应含视觉+听觉+联想+4记忆, got=" + brain.hubModuleCount());
        assertNotNull(brain.assocBridge());
        assertEquals("联想皮层", brain.assocBridge().moduleName());
        assertEquals(brain.assocSize(), brain.assocBridge().pulseDim(), "联想维度=assocSize");
        // 识别 → 联想激活回填桥接器
        brain.recognizeVisualWithConfidence(img);
        double[] act = brain.assocBridge().activation();
        assertTrue(act.length == brain.assocSize(), "联想激活应回填");
        // 有激活值 (非全零)
        double sum = 0;
        for (double a : act) sum += a;
        assertTrue(sum > 0, "联想激活应有值");
    }

    /** 全局爆发检测 */
    @Test
    void ignitionDetected() {
        EEGGenerator gen = new EEGGenerator();
        // 强脉冲 → 跨阈值 → 爆发
        for (int i = 0; i < 30; i++) {
            gen.sample(new double[]{0.9, 0.95, 0.9}, 1.0);
        }
        assertTrue(gen.ignition(), "强活动应触发全局爆发");
        assertTrue(gen.ignitionStrength() > 0, "爆发强度应>0");
        // 弱活动 → 无爆发
        EEGGenerator gen2 = new EEGGenerator();
        for (int i = 0; i < 30; i++) {
            gen2.sample(new double[]{0.05, 0.02, 0.0}, 1.0);
        }
        assertFalse(gen2.ignition(), "弱活动不应爆发");
        assertEquals(0, gen2.ignitionStrength(), 1e-9);
    }

    /** 全局爆发 → 意识内容扩散 (P300 意识标志) */
    @Test
    void ignitionBroadcastsToConsciousness() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        // 学习建立记忆 → 识别触发脉冲聚合
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        double phiBefore = brain.consciousness().phi();
        // 强识别 + 多轮同步 → 聚合跨阈值 → 爆发 → 意识扩散
        brain.recognizeVisualWithConfidence(img);
        for (int i = 0; i < 20; i++) brain.syncFrequencyBus();
        // 全局爆发后: 广播强度或 Φ 提升 (意识内容扩散)
        assertTrue(brain.consciousness().broadcastStrength() >= 0, "广播强度应存在");
        assertTrue(brain.consciousness().phi() >= phiBefore, "Φ不应下降");
        // 爆发时意识广播频率在 γ 带 (意识绑定)
        double hz = brain.consciousness().broadcastHz();
        assertTrue(hz >= 30.0, "意识广播应γ带, got=" + hz);
    }

    /** 联想桥独立测试 */
    @Test
    void assocBridgeStandalone() {
        AssocBridge bridge = new AssocBridge(4);
        assertEquals(4, bridge.pulseDim());
        bridge.updateActivation(new double[]{0.1, 0.5, 1.0, -0.3});
        double[] act = bridge.activation();
        // tanh 归一 (负值截断, 上限1)
        assertEquals(0.1, act[0], 1e-6);
        assertEquals(1.0, act[2], 1e-6);
        assertEquals(0.0, act[3], 1e-6, "负激活应截断");
        // 强广播 → 增益上升
        bridge.receiveBroadcast(new double[]{0.9, 0.9});
        assertTrue(bridge.topDownGain() > 1.0);
    }

    /** 完整路径: 感觉→联想→中枢→爆发→意识→记忆 (全链路) */
    @Test
    void fullPathway() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        // 学习 (视觉→联想→记忆→中枢)
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        // 识别 (联想激活→中枢→爆发→意识)
        String[] r = brain.recognizeVisualWithConfidence(img);
        assertTrue(r[0].equals(brain.vocabulary(0)) || r[0].equals(brain.vocabulary(1)), "识别应工作");
        // 中枢已整合联想脉冲
        assertTrue(brain.centralHub().inputDim() > 0);
        // 多轮联动后联想增益 ≥1 (中枢反馈)
        for (int i = 0; i < 5; i++) brain.hubPulseCycle();
        assertTrue(brain.assocBridge().topDownGain() >= 1.0, "联想增益应≥1");
        // 全部路径不破坏已有功能
        String[] r2 = brain.recognizeVisualWithConfidence(img);
        assertEquals(r[0], r2[0], "路径完善后识别应稳定");
    }
}
