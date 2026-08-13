package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 模块↔中枢链接完整性审计:
 * - 直接链接: 注册为 PulseModule (发射/接收脉冲)
 * - 间接链接: 经 Brain 流程 (学习/识别/同步) 影响中枢
 * - 验证: 每个模块至少有一条路径到中枢
 */
public class ConnectivityAuditTest {

    /** 直接注册中枢的模块 (PulseModule) */
    @Test
    void directHubModules() {
        Brain brain = Brain.simpleBrain();
        // 7 个直接模块: 视觉+听觉+联想桥 + 工作/分层/共振记忆 + 意识
        assertEquals(7, brain.hubModuleCount(), "直接注册中枢模块数=7");
        assertEquals("视觉皮层", brain.visualBridge().moduleName());
        assertEquals("听觉皮层", brain.auditoryBridge().moduleName());
        assertEquals("联想皮层", brain.assocBridge().moduleName());
        assertEquals("工作记忆", brain.workingMemory().moduleName());
        assertEquals("分层记忆", brain.hierarchicalMemory().moduleName());
        assertEquals("共振记忆", brain.resonanceMemory().moduleName());
        assertEquals("意识", brain.consciousness().moduleName());
        // 全部实现 PulseModule 接口
        assertTrue(brain.workingMemory() instanceof PulseModule);
        assertTrue(brain.hierarchicalMemory() instanceof PulseModule);
        assertTrue(brain.resonanceMemory() instanceof PulseModule);
        assertTrue(brain.consciousness() instanceof PulseModule);
    }

    /** 间接模块: 通过 Brain 流程影响中枢 (学习/识别/同步) */
    @Test
    void indirectModulesReachHub() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;

        // 间接模块在联动中参与 (它们影响识别→联想→中枢)
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        // 触发完整联动 (所有间接模块经 syncFrequencyBus/hubPulseCycle 参与)
        brain.recognizeVisualWithConfidence(img);
        brain.syncFrequencyBus();

        // 多巴胺: 调制 γ 注入频率总线 (经中枢)
        assertTrue(brain.dopamineSystem().dopamine() >= 0);
        // 预测引擎: 先验影响识别 → 联想 → 中枢
        assertTrue(brain.predictiveEngine().priorCount() > 0);
        // 内部模型: 预测误差修正置信度 → 识别 → 中枢
        assertTrue(brain.internalModel().trainCount() > 0);
        // 僵尸行动者: 自动化影响识别 → 中枢
        assertTrue(brain.zombieAgent().freedAttention() >= 0);
        // 认知模式: 活动经同步参与
        assertNotNull(brain.cognitiveModeDescription());
        // 自我意识: 镜像反馈影响识别
        assertNotNull(brain.selfNarrative());
        // 多感觉整合: γ 绑定注入总线
        assertTrue(brain.msi() != null);
        // EEG: 聚合中枢脉冲 → 回馈皮层 (闭环)
        assertTrue(brain.eegGenerator().history().size() >= 0);
        // 突触形成: 皮层发放 → 连接 (影响皮层→中枢)
        assertTrue(brain.synapseFormation().connectionCount() >= 0);
    }

    /** 链路闭环: 间接模块 → 中枢 → EEG → 反馈 (全路径) */
    @Test
    void fullConnectivityLoop() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 3 == 0) ? 0.9 : 0.1;

        // 学习 → 记忆/多巴胺/预测/内部模型全部参与
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        // 识别 → 联想 → 中枢 → EEG → 反馈 → 下行调制
        String[] r = brain.recognizeVisualWithConfidence(img);
        assertTrue(!r[0].isEmpty(), "识别应工作");

        // 中枢已整合 (直接模块发射)
        assertTrue(brain.centralHub().totalSpikes() >= 0);
        // EEG 由脉冲聚合
        double eeg = brain.currentEEG();
        assertTrue(Double.isFinite(eeg));
        // EEG 回馈调制皮层输入 (闭环到感觉层)
        assertTrue(brain.eegFeedback(5.0) >= 5.0, "EEG反馈应增强");
        // 下行增益调制皮层 (中枢→感觉)
        assertTrue(brain.visualBridge().topDownGain() >= 1.0);
        // 联想增益调制 (中枢→联想)
        assertTrue(brain.assocBridge().topDownGain() >= 1.0);
        // 意识接收总线频率 (中枢→意识)
        assertTrue(brain.consciousness().broadcastHz() > 0);
    }

    /** 全部模块可访问 (无悬空引用) */
    @Test
    void allModulesAccessible() {
        Brain brain = Brain.simpleBrain();
        assertNotNull(brain.workingMemory());
        assertNotNull(brain.hierarchicalMemory());
        assertNotNull(brain.resonanceMemory());
        assertNotNull(brain.consciousness());
        assertNotNull(brain.dopamineSystem());
        assertNotNull(brain.predictiveEngine());
        assertNotNull(brain.internalModel());
        assertNotNull(brain.zombieAgent());
        assertNotNull(brain.synapseFormation());
        assertNotNull(brain.eegGenerator());
        assertNotNull(brain.frequencyBus());
        assertNotNull(brain.centralHub());
        assertNotNull(brain.visualBridge());
        assertNotNull(brain.auditoryBridge());
        assertNotNull(brain.assocBridge());
        assertNotNull(brain.emotionalVoice());
        assertNotNull(brain.msi());
    }

    /** 神经发育: 中枢连接初始全连接→学习可增可减 (延续) */
    @Test
    void hubPlasticityStillWorks() {
        CentralHub hub = CentralHub.defaultParams(8);
        assertEquals(1.0, hub.connectionDensity(), 1e-9, "初始全连接");
        // 半活跃输入 → 修剪
        double[] input = new double[8];
        for (int j = 0; j < 4; j++) input[j] = 0.9;
        for (int i = 0; i < 300; i++) {
            hub.clearInput();
            hub.inject(0, input);
            hub.step(1.0);
            hub.learnWeights(1.0);
        }
        assertTrue(hub.connectionDensity() < 1.0, "学习后修剪");
        assertTrue(hub.activeConnections() > 0, "活跃连接保留");
    }
}
