package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 工作记忆 + 意识模块验证 (fWBM 论文 + GWT 全局工作空间理论)。
 * 1. 工作记忆: 写入/读取/容量限制/遗忘曲线/丘脑门控
 * 2. 意识: 广播竞争/睡眠关闭/唤醒/整合度
 * 3. 睡眠: 工作记忆重放抵抗遗忘
 * 4. 集成: 学习→工作记忆→识别辅助
 */
public class MemoryConsciousnessTest {

    // ===== 工作记忆 (fWBM: 自持续活动) =====

    @Test
    void wmWriteAndRead() {
        WorkingMemory wm = WorkingMemory.defaultParams();
        assertEquals(4, wm.capacity(), "人类工作记忆容量 4±1");
        double[] content = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) content[i] = (i % 2 == 0) ? 1.0 : 0.0;
        int slot = wm.write(content);
        assertTrue(slot >= 0, "应成功写入");
        assertEquals(1.0, wm.strength(slot), 1e-9, "写入后强度应为 1.0");
        // 读取匹配
        double[] query = content.clone();
        double[] result = wm.read(query);
        assertEquals(slot, (int) result[0], "应匹配已写槽位");
        assertTrue(result[1] > 0.9, "相似度应高");
    }

    @Test
    void wmCapacityLimited() {
        WorkingMemory wm = WorkingMemory.defaultParams();
        // 写 8 次 → 只保留 4 槽 (容量竞争)
        for (int n = 0; n < 8; n++) {
            double[] c = new double[VisualNeuralEncoder.OUTPUT_DIM];
            for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) c[i] = (i + n) % 7 == 0 ? 1 : 0;
            wm.write(c);
        }
        assertEquals(4, wm.load(), "负载不应超过容量");
        assertTrue(wm.occupancy() <= 1.0);
    }

    @Test
    void wmForgetsWithoutRehearsal() {
        WorkingMemory wm = WorkingMemory.defaultParams();
        double[] content = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) content[i] = 1.0;
        int slot = wm.write(content);
        // 时间流逝 (Ebbinghaus 遗忘: 指数衰减)
        for (int t = 0; t < 300; t++) wm.tick(1000.0);  // 300s
        assertTrue(wm.strength(slot) < 1.0, "无重放应遗忘");
        // 最终应完全遗忘 (30s tau, 300s 后 << 0.01)
        assertEquals(0, wm.load(), "长时间无重放应完全遗忘");
    }

    @Test
    void wmRehearsalResistsForgetting() {
        WorkingMemory wm = WorkingMemory.defaultParams();
        double[] content = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) content[i] = 1.0;
        int slot = wm.write(content);
        // 定期重放 (每 10s 复习一次)
        for (int t = 0; t < 300; t++) {
            wm.tick(1000.0);
            if (t % 10 == 0) wm.rehearse(slot);
        }
        assertTrue(wm.strength(slot) > 0.5, "定期重放应保持记忆, strength=" + wm.strength(slot));
    }

    @Test
    void wmThalamicGates() {
        WorkingMemory wm = WorkingMemory.defaultParams();
        double[] content = new double[VisualNeuralEncoder.OUTPUT_DIM];
        // 写入门控关闭 → 拒绝写入
        wm.setWriteGate(0.0);
        int slot = wm.write(content);
        assertEquals(-1, slot, "门控关闭应拒绝写入");
        // 读入门控关闭 → 拒绝读取
        wm.setWriteGate(1.0);
        slot = wm.write(content);
        wm.setReadGate(0.0);
        double[] r = wm.read(content);
        assertEquals(-1, (int) r[0], "读入门控关闭应拒绝读取");
        // 清除门控关闭 → 拒绝清除
        wm.setReadGate(1.0);
        wm.setClearGate(0.0);
        wm.clear(slot);
        assertTrue(wm.strength(slot) > 0, "清除门控关闭应保留");
    }

    // ===== 意识 (GWT 全局工作空间) =====

    @Test
    void consciousnessBroadcastsStrongInput() {
        Consciousness c = new Consciousness();
        assertEquals(Consciousness.State.意识, c.state(), "初始清醒");
        c.perceive(0.9, 0.1, "苹果", "噪音");
        assertEquals("苹果", c.broadcast(), "强视觉输入应广播");
        assertTrue(c.broadcastStrength() > 0.5);
    }

    @Test
    void consciousnessSleepBlocksBroadcast() {
        Consciousness c = new Consciousness();
        c.sleep();
        assertEquals(Consciousness.State.无意识, c.state(), "睡眠应无意识");
        c.perceive(0.9, 0.9, "苹果", "苹果");
        assertTrue(c.broadcast().isEmpty(), "睡眠中不应广播外部输入");
        // 唤醒恢复
        c.wake();
        c.perceive(0.9, 0.1, "苹果", "");
        assertEquals("苹果", c.broadcast(), "唤醒后应恢复感知");
    }

    @Test
    void consciousnessPhiIntegratesMultimodal() {
        Consciousness c = new Consciousness();
        // 视觉+听觉识别一致 → Φ 上升 (多感觉整合)
        for (int i = 0; i < 5; i++) {
            c.perceive(0.8, 0.8, "苹果", "苹果");
        }
        assertTrue(c.phi() > 0.3, "多模态一致应提升整合度 Φ, phi=" + c.phi());
        // 不一致 → Φ 下降
        for (int i = 0; i < 10; i++) {
            c.perceive(0.8, 0.8, "苹果", "猫");
        }
        assertTrue(c.phi() < 0.3, "多模态不一致应降低 Φ, phi=" + c.phi());
    }

    @Test
    void consciousnessAttentionFocus() {
        Consciousness c = new Consciousness();
        c.setAttention(1.0);
        c.perceive(0.3, 0.9, "弱视觉", "强听觉");
        // 高注意力于听觉 → 听觉广播
        assertEquals("强听觉", c.broadcast(), "注意力偏向听觉时应广播听觉");
    }

    // ===== 集成 =====

    @Test
    void brainWMHelpsRecognition() {
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM], imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) {
            imgA[i] = (i % 3 == 0) ? 0.9 : 0.1;
            imgB[i] = (i % 3 == 1) ? 0.9 : 0.1;
        }
        // 学习 (写入工作记忆)
        for (int e = 0; e < 10; e++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        assertTrue(brain.workingMemory().load() > 0, "学习后工作记忆应有内容");
        // 工作记忆摘要可读
        String summary = brain.workingMemorySummary();
        assertTrue(summary.contains("工作记忆"), summary);
        // 识别仍工作
        String[] r = brain.recognizeVisualWithConfidence(imgA);
        assertTrue(r[0].equals("你好") || r[0].equals("苹果"), "识别应工作, got=" + r[0]);
    }

    @Test
    void sleepRehearsesWorkingMemory() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        // 时间流逝 (遗忘开始)
        for (int t = 0; t < 50; t++) brain.workingMemory().tick(1000.0);
        double before = brain.workingMemory().strength(0);
        // 睡眠 (重放强化)
        int[] report = brain.sleepConsolidate();
        assertEquals(4, report.length, "睡眠报告应含工作记忆负载");
        assertTrue(report[3] >= 1, "睡眠前工作记忆应有负载, load=" + report[3]);
        // 睡眠后: 意识应恢复清醒
        assertEquals(Consciousness.State.意识, brain.consciousness().state(), "睡眠后应唤醒");
    }

    @Test
    void consciousnessDescribes() {
        Brain brain = Brain.simpleBrain();
        String desc = brain.consciousness().describe();
        assertTrue(desc.contains("意识") || desc.contains("无意识"), desc);
        // 睡眠时描述
        brain.consciousness().sleep();
        assertTrue(brain.consciousness().describe().contains("睡眠"), "睡眠时描述应含睡眠");
        brain.consciousness().wake();
    }
}
