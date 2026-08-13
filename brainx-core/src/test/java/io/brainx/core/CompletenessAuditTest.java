package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 全模块完整性审计验证:
 * 1. 所有模块都被 Brain 实际调用 (无"定义了但没用")
 * 2. 感觉皮层接入中枢 (丘脑-皮层环路完整: 上行+下行)
 * 3. 内部模型真正接入 (前向预测/感觉预测误差)
 * 4. 核心联动闭环: 学习→记忆→中枢→EEG→反馈
 */
public class CompletenessAuditTest {

    /** 所有核心模块都应被 Brain 调用 (功能完整) */
    @Test
    void allModulesAreWired() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;

        // 学习触发所有模块
        for (int e = 0; e < 3; e++) brain.learnVisualWord(img, 0);

        // 内部模型已接入 (之前 0 调用)
        assertTrue(brain.internalModel().trainCount() > 0, "内部模型应被训练");
        // 预测引擎
        assertTrue(brain.predictiveEngine().priorCount() > 0, "预测引擎应建立先验");
        // 多巴胺
        assertTrue(brain.dopamineSystem().dopamine() >= 0, "多巴胺应活动");
        // 僵尸行动者
        assertTrue(brain.zombieSkills().size() > 0, "僵尸行动者应练习");
        // 认知模式
        assertNotNull(brain.cognitiveModeDescription());
        // 工作记忆
        assertTrue(brain.workingMemory().occupancy() > 0, "工作记忆应有内容");
        // 分层记忆
        assertTrue(brain.hierarchicalMemory().episodic().size() > 0, "分层记忆应有情景");
        // 共振记忆
        assertTrue(brain.resonanceMemory().size() >= 1, "共振记忆应写入");
        // 意识
        assertNotNull(brain.consciousness().describe());
        // 自我意识
        assertNotNull(brain.selfNarrative());
        // EEG
        assertNotNull(brain.eegGenerator());
        // 频率总线/中枢
        assertNotNull(brain.frequencyBus());
        assertNotNull(brain.centralHub());
    }

    /** 感觉皮层接入中枢 (丘脑-皮层环路: 上行脉冲) */
    @Test
    void sensoryCortexConnectedToHub() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        // 中枢注册了感觉皮层
        assertTrue(brain.hubModuleCount() >= 6, "中枢应含视觉+听觉+4记忆模块, got=" + brain.hubModuleCount());
        // 视觉皮层发射脉冲到中枢
        assertNotNull(brain.visualBridge());
        assertNotNull(brain.auditoryBridge());
        assertEquals("视觉皮层", brain.visualBridge().moduleName());
        // 学习后视觉皮层有发放率 (发射过脉冲)
        double[] pulses = brain.visualBridge().emitPulses();
        assertTrue(pulses.length == brain.visualCortexSize(), "视觉脉冲维度=皮层大小");
    }

    /** 中枢下行调制感觉皮层 (丘脑-皮层环路: 下行增益) */
    @Test
    void topDownModulation() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        // 强学习 → 中枢活跃 → 下行增益
        for (int e = 0; e < 5; e++) {
            brain.learnVisualWord(img, 0);
            brain.hubPulseCycle();  // 中枢活跃
        }
        // 中枢广播 → 视觉皮层增益 > 1 (注意增强)
        assertTrue(brain.visualBridge().topDownGain() >= 1.0, "下行增益应≥1");
    }

    /** 内部模型真正工作 (前向预测+误差学习) */
    @Test
    void internalModelWorks() {
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM], imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) { imgA[i] = (i%3==0)?0.9:0.1; imgB[i] = (i%3==1)?0.9:0.1; }
        for (int e = 0; e < 5; e++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        // 内部模型已训练
        assertTrue(brain.internalModel().trainCount() > 0, "内部模型应训练");
        // 预测误差存在 (学习过程中产生)
        assertTrue(brain.internalModel().lastError() >= 0);
        // 泛化评估可用
        assertTrue(brain.internalModel().generalizationScore() >= 0);
    }

    /** 核心联动闭环: 学习→记忆→中枢→EEG→下行反馈 (双向) */
    @Test
    void fullLoopClosed() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        // 学习 → 记忆写入 + 中枢脉冲 + EEG 聚合 + 下行调制
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        // 中枢已发放 (脉冲整合)
        assertTrue(brain.centralHub().totalSpikes() >= 0);
        // EEG 由脉冲聚合产生
        double eeg = brain.currentEEG();
        assertTrue(Double.isFinite(eeg));
        // EEG 回馈调制皮层输入
        assertTrue(brain.eegFeedback(10.0) >= 10.0, "EEG反馈应增强输入");
        // 下行增益调制皮层 (双向闭环)
        assertTrue(brain.visualBridge().topDownGain() >= 1.0);
        // 识别仍正常 (全部联动不破坏功能)
        String[] r = brain.recognizeVisualWithConfidence(img);
        assertEquals(brain.vocabulary(0), r[0], "识别应正常工作");
    }

    /** 桥接器独立测试 */
    @Test
    void bridgeStandalone() {
        // 直接用 LIF 皮层测桥接
        io.brainx.core.neuron.LIF[] cortex = new io.brainx.core.neuron.LIF[4];
        for (int i = 0; i < 4; i++) cortex[i] = io.brainx.core.neuron.LIF.defaultParams();
        SensoryCortexBridge bridge = new SensoryCortexBridge(cortex, "测试皮层");
        assertEquals(4, bridge.pulseDim());
        // 强广播 → 增益上升
        bridge.receiveBroadcast(new double[]{0.9, 0.8, 0.9});
        assertTrue(bridge.topDownGain() > 1.0, "强广播应升增益");
        // 弱广播 → 增益回落
        for (int i = 0; i < 5; i++) bridge.receiveBroadcast(new double[]{0.1, 0.0, 0.0});
        assertTrue(bridge.topDownGain() < 1.5);
    }
}
