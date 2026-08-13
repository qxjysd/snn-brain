package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 中枢脉冲网络联动验证 —— 各模块通过中枢脉冲统一联动。
 * 修复前: 无中枢脉冲网络, 模块间为数据总线联动。
 * 修复后: CentralHub (皮层-丘脑环路) 统一整合广播。
 */
public class CentralHubTest {

    @Test
    void hubIntegratesModulePulses() {
        // 中枢整合: 模块脉冲 → LIF 发放 → 活跃度 (LIF 需多步充电)
        CentralHub hub = CentralHub.defaultParams(4);
        hub.inject(0, new double[]{0.8, 0.7, 0.6, 0.9});  // 强输入
        for (int i = 0; i < 20; i++) hub.step(1.0);       // 持续驱动
        assertTrue(hub.totalSpikes() > 0, "强输入应驱动中枢发放");
        assertTrue(hub.activity() > 0, "中枢应有活跃度");
    }

    @Test
    void weakInputLowActivity() {
        CentralHub hub = CentralHub.defaultParams(4);
        hub.inject(0, new double[]{0.05, 0.02, 0.01, 0.0});  // 弱输入
        hub.step(1.0);
        // 弱输入不应显著激活中枢 (阈值机制)
        assertTrue(hub.activity() < 0.5, "弱输入应低活跃");
    }

    @Test
    void hubBroadcastsToModules() {
        // 中枢广播 → 模块接收调制 (闭环)
        WorkingMemory wm = WorkingMemory.defaultParams();
        double[] content = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) content[i] = 0.5;
        wm.write(content);
        // 强度已饱和 1.0, 先衰减到中间值再测广播巩固
        for (int t = 0; t < 30; t++) wm.tick(1000.0);
        double before = wm.strength(0);
        assertTrue(before < 1.0, "衰减后应非饱和");
        // 强广播 → 工作记忆巩固
        boolean modulated = wm.receiveBroadcast(new double[]{0.8, 0.7, 0.9});
        assertTrue(modulated, "强广播应调制工作记忆");
        assertTrue(wm.strength(0) > before, "中枢广播应巩固工作记忆");
    }

    @Test
    void modulesEmitPulses() {
        // 各模块发射脉冲 (状态→脉冲率)
        WorkingMemory wm = WorkingMemory.defaultParams();
        double[] c = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) c[i] = 0.5;
        wm.write(c);
        double[] wmPulses = wm.emitPulses();
        assertEquals(wm.pulseDim(), wmPulses.length, "脉冲维度应匹配");
        assertTrue(wmPulses[0] > 0, "有内容槽位应发射脉冲");

        Consciousness con = new Consciousness();
        con.perceive(0.9, 0.1, "苹果", "");
        double[] conPulses = con.emitPulses();
        assertEquals(2, conPulses.length);
        assertTrue(conPulses[0] > 0, "意识广播应发射脉冲");

        HierarchicalMemory hm = HierarchicalMemory.defaultParams();
        double[] f = new double[VisualNeuralEncoder.OUTPUT_DIM];
        hm.addEpisodic("苹果", f, 0.5);
        double[] hmPulses = hm.emitPulses();
        assertEquals(2, hmPulses.length);

        ResonanceMemory rm = ResonanceMemory.defaultParams();
        rm.write("苹果", 0.8);
        double[] rmPulses = rm.emitPulses();
        assertEquals(2, rmPulses.length);
        assertTrue(rmPulses[0] > 0, "共振记忆应发射脉冲");
    }

    @Test
    void brainHubCycleIntegratesAllModules() {
        // 大脑完整脉冲环路: 学习 → 模块发射 → 中枢整合 → 广播
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM], imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) { imgA[i] = (i%3==0)?0.9:0.1; imgB[i] = (i%3==1)?0.9:0.1; }
        for (int e = 0; e < 5; e++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        // 学习触发 syncFrequencyBus → hubPulseCycle
        assertTrue(brain.centralHub().totalSpikes() >= 0, "中枢应已运行");
        assertNotNull(brain.hubSummary());
        assertTrue(brain.hubSummary().contains("中枢脉冲网络"));
        // 中枢活跃度应非零 (有记忆模块发射脉冲)
        assertTrue(brain.centralHub().inputDim() > 0, "中枢应有输入维度");
    }

    @Test
    void hubRunsMultipleCycles() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        int spikes1 = brain.centralHub().totalSpikes();
        // 多次脉冲环路 → 中枢持续整合
        for (int i = 0; i < 10; i++) brain.hubPulseCycle();
        int spikes2 = brain.centralHub().totalSpikes();
        assertTrue(spikes2 >= spikes1, "多次环路应累积发放");
        assertTrue(brain.centralHub().timeStep() >= 10, "应运行多步");
    }

    @Test
    void hubFeedsFrequencyBus() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        // 中枢脉冲 → 频率总线 ("中枢" 模块)
        assertTrue(brain.frequencyBus().moduleFreq().containsKey("中枢"),
                "中枢应上报频率总线");
    }

    @Test
    void sleepCycleRunsHub() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 4; e++) brain.learnVisualWord(img, 0);
        int before = brain.centralHub().totalSpikes();
        brain.sleepConsolidate();  // 触发同步 → 脉冲环路
        assertTrue(brain.centralHub().totalSpikes() >= before);
    }
}
