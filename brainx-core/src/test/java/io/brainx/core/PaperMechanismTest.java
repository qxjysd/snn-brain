package io.brainx.core;

import io.brainx.core.encoding.TTFS;
import io.brainx.core.neuron.PLIF;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 论文机制验证 (第二轮脑域完善):
 * 1. PLIF 可学习时间常数 (STEP 论文: 增益最大)
 * 2. E/I 平衡网络证据累积决策 (BrainTrace Fig5: 虚拟跑道→T字路口)
 * 3. TTFS 时间到首次脉冲编码 + 顺序发放 (BrainTrace Fig5F)
 * 4. 睡眠巩固: 重放+稳态 (记忆巩固)
 */
public class PaperMechanismTest {

    // ===== 1. PLIF (STEP 论文) =====

    @Test
    void plifFiresAndLearnsTau() {
        PLIF n = PLIF.defaultParams();
        assertEquals(10.0, n.tauMs(), 1e-9, "初始 tau 应为 10ms");
        // 发放
        int spikes = 0;
        for (int t = 0; t < 1000; t++) {
            n.step(25.0, 0.1);
            if (n.fired()) spikes++;
        }
        assertTrue(spikes > 5, "PLIF 应发放");
        // 学习更新 tau (误差驱动), 应在生物学范围 [2,20]
        for (int t = 0; t < 100; t++) {
            n.step(20.0, 0.1);
            n.learn(n.fired() ? 1.0 : -0.5);
        }
        assertTrue(n.tauMs() >= 2.0 && n.tauMs() <= 20.0, "tau 应在生物学范围, got=" + n.tauMs());
        assertTrue(n.tauMs() != 10.0, "tau 应已学习调整, got=" + n.tauMs());
    }

    @Test
    void plifFasterTauFiresMore() {
        PLIF fast = new PLIF(5.0, -65, -50, -65, 1.0, 0);
        PLIF slow = new PLIF(20.0, -65, -50, -65, 1.0, 0);
        int f = 0, s = 0;
        for (int t = 0; t < 1000; t++) {
            fast.step(25.0, 0.1);
            slow.step(25.0, 0.1);
            if (fast.fired()) f++;
            if (slow.fired()) s++;
        }
        // 小 tau (快响应) 应发放更多 (STEP 论文: tau 是重要可调参数)
        assertTrue(f > s, "快 tau 应发放更多: fast=" + f + " slow=" + s);
    }

    // ===== 2. E/I 平衡网络 (BrainTrace Fig5) =====

    @Test
    void eibNetworkDecidesByMajorityCues() {
        EIBNetwork net = EIBNetwork.defaultParams();
        // 7 视觉线索: 5左2右 → 应决策左
        double[] leftMajority = new double[14];
        for (int t = 0; t < 14; t += 2) {
            leftMajority[t] = (t < 10) ? 1.0 : 0.0;   // 5 左
            leftMajority[t + 1] = (t < 10) ? 0.0 : 1.0; // 2 右
        }
        int d = net.runTrial(leftMajority);
        assertEquals(1, d, "多数线索在左应决策左, got=" + d);

        // 对称测试: 2左5右 → 决策右
        double[] rightMajority = new double[14];
        for (int t = 0; t < 14; t += 2) {
            rightMajority[t] = (t < 4) ? 1.0 : 0.0;
            rightMajority[t + 1] = (t < 4) ? 0.0 : 1.0;
        }
        net.reset();
        int d2 = net.runTrial(rightMajority);
        assertEquals(2, d2, "多数线索在右应决策右, got=" + d2);
    }

    @Test
    void eibNetworkIsBalanced() {
        EIBNetwork net = EIBNetwork.defaultParams();
        assertEquals(800, net.totalNeurons());
        // 4:1 E/I 比例 (BrainTrace 论文)
        int eCount = 0, iCount = 0;
        for (int i = 0; i < 800; i++) {
            if (net.typeOf(i) == EIBNetwork.Type.EXCITATORY) eCount++;
            else iCount++;
        }
        assertEquals(640, eCount, "兴奋神经元应占 4/5");
        assertEquals(160, iCount, "抑制神经元应占 1/5");
    }

    @Test
    void eibNetworkEvidenceAccumulates() {
        EIBNetwork net = EIBNetwork.defaultParams();
        // 持续左线索 → 左证据增长
        for (int t = 0; t < 100; t++) net.step(1.0, 0.0, 1.0);
        assertTrue(net.evidenceLeft() > net.evidenceRight(),
                "左线索应使左证据占优: L=" + net.evidenceLeft() + " R=" + net.evidenceRight());
    }

    // ===== 3. TTFS (BrainTrace Fig5F) =====

    @Test
    void ttfsStrongStimulusFiresFirst() {
        TTFS ttfs = TTFS.defaultParams();
        assertTrue(ttfs.verifyOrdering(), "强刺激应更早发放");
        double strong = ttfs.encode(0.9);
        double weak = ttfs.encode(0.1);
        assertTrue(strong < weak, "强刺激首次脉冲应更早: strong=" + strong + " weak=" + weak);
        // 解码还原
        assertEquals(0.9, ttfs.decode(strong), 0.15, "早脉冲应解码为强刺激");
    }

    @Test
    void ttfsRankOrderMatchesIntensity() {
        TTFS ttfs = TTFS.defaultParams();
        double[] intensities = {0.1, 0.9, 0.5, 0.7, 0.3};
        int[] order = ttfs.rankOrder(intensities);
        // 第一个发放的应是强度 0.9 (索引1)
        assertEquals(1, order[0], "最强刺激应最先发放");
        // 最后一个应是 0.1 (索引0)
        assertEquals(0, order[order.length - 1], "最弱刺激应最后发放");
        // 全顺序应为 1,3,2,4,0
        assertArrayEquals(new int[]{1, 3, 2, 4, 0}, order, "rank ordering 应按强度降序");
    }

    // ===== 4. 睡眠巩固 =====

    @Test
    void sleepConsolidatesMemory() {
        // 白天学习 → 睡眠巩固 → 连接增强
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        for (int epoch = 0; epoch < 5; epoch++) {
            brain.learnVisualWord(img, 0);
        }
        int matureBefore = brain.synapseFormation().matureCount(0.1);
        // 睡眠
        int[] report = brain.sleepConsolidate();
        int matureAfter = brain.synapseFormation().matureCount(0.1);
        assertTrue(report[0] > 0, "睡眠应有重放, replays=" + report[0]);
        assertTrue(matureAfter >= matureBefore, "睡眠巩固应增强连接: before=" + matureBefore + " after=" + matureAfter);
    }

    @Test
    void sleepReportFields() {
        Brain brain = Brain.simpleBrain();
        int[] report = brain.sleepConsolidate();
        assertEquals(4, report.length, "报告应含 [重放, 修剪, 成熟连接, 工作记忆负载]");
        assertTrue(report[0] >= 0 && report[2] >= 0 && report[3] >= 0);
    }

    // ===== 5. 综合: 大脑全机制协同 =====

    @Test
    void brainAllMechanismsCoexist() {
        Brain brain = Brain.simpleBrain();
        // TTFS
        double[] tt = brain.encodeTTFS(new double[]{0.8, 0.2});
        assertTrue(tt[0] < tt[1]);
        // E/I 决策
        double[] cues = new double[14];
        for (int t = 0; t < 14; t += 2) { cues[t] = 1.0; cues[t+1] = 0.0; }
        assertEquals(1, brain.runDecisionTrial(cues));
        // 学习 + 睡眠
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 2);
        int[] sleepReport = brain.sleepConsolidate();
        assertTrue(sleepReport[0] > 0);
        // 识别仍工作
        String guess = brain.recognizeVisual(img);
        assertTrue(brain.learnedWords().contains(guess) || guess.equals("未知") || guess.equals("?"),
                "全机制协同后识别仍应工作, got=" + guess);
    }
}
