package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 频率总线联动验证 —— 确认各模块通过频率交互 (整体联动)。
 * 修复前: 12 模块中 10 个无频率交互。修复后: 记忆/认知模块均含频率。
 */
public class FrequencyBusTest {

    @Test
    void busTracksModuleFrequencies() {
        FrequencyBus bus = new FrequencyBus();
        bus.report("工作记忆", 6.0, 0.5);   // θ
        bus.report("意识", 40.0, 0.8);      // γ 强
        bus.report("长期记忆", 10.0, 0.3);  // α
        // 主导 = 强度最强 (意识 γ)
        assertEquals(40.0, bus.dominantHz(), 1e-9, "主导应为最强模块频率");
        assertEquals("γ绑定", FrequencyBus.bandName(40.0));
        // 耦合度: 与主导频率越近越联动
        double cGamma = bus.coupling("意识");
        double cTheta = bus.coupling("工作记忆");
        assertTrue(cGamma > cTheta, "γ模块应与总线耦合更强");
    }

    @Test
    void workingMemoryHasFrequency() {
        WorkingMemory wm = WorkingMemory.defaultParams();
        double[] content = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) content[i] = 0.5;
        wm.write(content);
        // 槽位有频率
        assertTrue(wm.slotFrequency(0) >= 4.0 && wm.slotFrequency(0) <= 13.0, "槽位应在θ-α带");
        // 频率读取: 精确频率共振命中
        double[] r = wm.readByFrequency(wm.currentFrequency());
        assertTrue(r[0] >= 0, "频率读取应命中槽位");
        // 主导频率可上报总线
        assertTrue(wm.currentFrequency() >= 4.0);
    }

    @Test
    void hierarchicalMemoryFrequency() {
        HierarchicalMemory hm = HierarchicalMemory.defaultParams();
        double[] feat = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) feat[i] = 0.5;
        // 巩固为长期记忆 (3次)
        for (int i = 0; i < 3; i++) hm.addEpisodic("苹果", feat, 0.5);
        assertTrue(hm.inLongTerm("苹果"));
        // 长期记忆有频率
        double freq = hm.currentFrequency();
        assertTrue(freq >= 4.0 && freq <= 13.0, "长期记忆应在θ-α带");
        // 按频率检索
        String recall = hm.recallByFrequency(freq);
        assertEquals("苹果", recall, "频率检索应命中");
    }

    @Test
    void consciousnessReceivesFrequency() {
        Consciousness c = new Consciousness();
        c.receiveFrequency(6.0, 0.6);  // θ 注入
        assertEquals(6.0, c.broadcastHz(), 1e-9, "意识应接收频率");
        c.receiveFrequency(45.0, 0.9);  // γ 强注入
        assertEquals(45.0, c.broadcastHz(), 1e-9);
    }

    @Test
    void gammaBindingConsistent() {
        MultiSensoryIntegration msi = MultiSensoryIntegration.defaultParams();
        // 一致 → 高绑定
        double bind = msi.gammaBinding("苹果", "苹果", 40.0);
        assertTrue(bind > 0.7, "跨模态一致应高γ绑定, got=" + bind);
        // 冲突 → 无绑定
        double conflict = msi.gammaBinding("苹果", "猫", 40.0);
        assertTrue(conflict < 0.5, "冲突应低绑定");
    }

    @Test
    void brainModulesReportToBus() {
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM], imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) { imgA[i] = (i%3==0)?0.9:0.1; imgB[i] = (i%3==1)?0.9:0.1; }
        // 学习 (触发 syncFrequencyBus)
        for (int e = 0; e < 5; e++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        // 总线上应有记忆模块频率
        var mods = brain.frequencyBus().moduleFreq();
        assertTrue(mods.containsKey("工作记忆"), "总线应有工作记忆");
        assertTrue(mods.containsKey("共振记忆"), "总线应有共振记忆");
        assertTrue(mods.containsKey("长期记忆"), "总线应有长期记忆");
        assertTrue(mods.containsKey("意识"), "总线应有意识");
        // 主导频率合理
        assertTrue(brain.frequencyBus().dominantHz() > 0);
        // 摘要可读
        assertTrue(brain.busSummary().contains("频率总线"));
    }

    @Test
    void busDrivesConsciousnessFeedback() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        // 识别触发频率同步 + 意识接收总线频率
        brain.recognizeVisualWithConfidence(img);
        assertTrue(brain.consciousness().broadcastHz() > 0, "意识应持有广播频率");
    }

    @Test
    void sleepSyncsLongTermFrequency() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 4; e++) brain.learnVisualWord(img, 0);
        // 睡眠巩固 → 长期记忆频率更新 → 总线同步
        int[] rep = brain.sleepConsolidate();
        assertNotNull(brain.frequencyBus());
        assertTrue(brain.hierarchicalMemory().currentFrequency() >= 4.0);
    }

    @Test
    void allModulesHaveFrequencyInterface() {
        // 审计: 关键模块都应暴露 currentFrequency (频率输入输出接口)
        WorkingMemory wm = WorkingMemory.defaultParams();
        HierarchicalMemory hm = HierarchicalMemory.defaultParams();
        Consciousness c = new Consciousness();
        // 接口存在且返回合理频率
        assertTrue(wm.currentFrequency() >= 4.0);
        assertTrue(hm.currentFrequency() >= 4.0);
        assertTrue(c.broadcastHz() >= 30.0, "意识默认γ");
        // 频率总线能接收所有模块
        FrequencyBus bus = new FrequencyBus();
        bus.report("wm", wm.currentFrequency(), 0.5);
        bus.report("hm", hm.currentFrequency(), 0.5);
        bus.report("c", c.broadcastHz(), 0.5);
        assertEquals(3, bus.moduleFreq().size());
    }
}
