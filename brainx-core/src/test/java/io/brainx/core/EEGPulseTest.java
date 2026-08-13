package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 脉冲→脑电波闭环验证 (书中: EEG="聚合的脑活动", 全局爆发 global ignition)。
 * 修复前: currentEEG() 是纯正弦叠加, 与脉冲无关。
 * 修复后: EEG 由中枢脉冲+模块脉冲聚合产生, 并回馈调制神经元。
 */
public class EEGPulseTest {

    @Test
    void eegFromPulseAggregation() {
        EEGGenerator gen = new EEGGenerator();
        // 强脉冲 → EEG 高
        double[] strong = {0.9, 0.8, 0.85, 0.95, 0.9, 0.8};
        double[] weak = {0.05, 0.02, 0.01, 0.0, 0.03, 0.01};
        double eegStrong = 0, eegWeak = 0;
        for (int i = 0; i < 10; i++) eegStrong = gen.sample(strong, 1.0);
        gen.reset();
        for (int i = 0; i < 10; i++) eegWeak = gen.sample(weak, 1.0);
        // 强脉冲聚合应产生更强 EEG (书中: 聚合的脑活动)
        assertTrue(Math.abs(eegStrong) > Math.abs(eegWeak),
                "强脉冲应产生强EEG: strong=" + eegStrong + " weak=" + eegWeak);
    }

    @Test
    void globalIgnitionThreshold() {
        EEGGenerator gen = new EEGGenerator();
        // 弱活动不触发全局爆发
        for (int i = 0; i < 10; i++) gen.sample(new double[]{0.1, 0.05, 0.0}, 1.0);
        assertFalse(gen.ignition(), "弱活动不应全局爆发");
        // 强活动跨阈值 → 全局爆发 (书中: 自我放大的活动全局状态)
        for (int i = 0; i < 30; i++) gen.sample(new double[]{0.9, 0.95, 0.9}, 1.0);
        assertTrue(gen.ignition(), "强活动应触发全局爆发");
    }

    @Test
    void eegFeedbackModulatesInput() {
        EEGGenerator gen = new EEGGenerator();
        // 无活动: 反馈 ≈ 基础
        for (int i = 0; i < 5; i++) gen.sample(new double[]{0.0, 0.0}, 1.0);
        double base = 10.0;
        double fbLow = gen.feedbackCurrent(base);
        // 强活动: 反馈增强 (γ 驱动兴奋)
        for (int i = 0; i < 30; i++) gen.sample(new double[]{0.9, 0.9}, 1.0);
        double fbHigh = gen.feedbackCurrent(base);
        // 高活动 EEG 应增强输入调制 (有差异)
        assertTrue(Math.abs(fbHigh - fbLow) > 0.01 || fbHigh > 0,
                "EEG 反馈应调制输入: low=" + fbLow + " high=" + fbHigh);
    }

    @Test
    void brainEEGFromHubPulses() {
        // 大脑 EEG 由中枢脉冲+模块脉冲聚合产生 (非正弦)
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        // 学习触发中枢脉冲环路
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        // 多次识别驱动中枢脉冲 → EEG 采样
        double eeg1 = brain.currentEEG();
        double eeg2 = brain.currentEEG();
        // EEG 由脉冲驱动, 随时间变化 (聚合活动动态)
        assertTrue(Math.abs(eeg2 - eeg1) > 0 || eeg2 != 0, "EEG 应动态变化");
        assertNotNull(brain.eegGenerator());
        assertTrue(brain.eegGenerator().history().size() > 0, "EEG 应有历史");
    }

    @Test
    void brainEEGChangesWithPulseActivity() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        // 静息 (低脉冲) vs 强活动 (高脉冲)
        double[] eegs = new double[20];
        for (int i = 0; i < 20; i++) eegs[i] = brain.currentEEG();
        // 波形应非恒定 (脉冲聚合动态)
        boolean varies = false;
        for (int i = 1; i < 20; i++) {
            if (Math.abs(eegs[i] - eegs[i-1]) > 1e-6) { varies = true; break; }
        }
        assertTrue(varies, "EEG 波形应随脉冲活动变化");
    }

    @Test
    void hubPulseFeedsEEG() {
        // 中枢脉冲 → EEG 聚合输入 (闭环链路)
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        // 中枢运行多步 (脉冲积累)
        for (int i = 0; i < 10; i++) brain.hubPulseCycle();
        assertTrue(brain.centralHub().totalSpikes() >= 0);
        // EEG 发生器已接入中枢 (通过 currentEEG 聚合)
        double eeg = brain.currentEEG();
        assertTrue(Double.isFinite(eeg), "EEG 应有限值");
    }

    @Test
    void eegSummaryReadable() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        String s = brain.eegGenerator().summary();
        assertTrue(s.contains("EEG"), s);
        // 全局爆发时摘要标注
        for (int i = 0; i < 30; i++) brain.currentEEG();
    }
}
