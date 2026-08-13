package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 脉冲大脑连接健康度回归测试 (v5.4 连接审计)。
 *
 * 背景: 原 SynapseFormation 只"增强候选池内连接"——真实规模 (5488 神经元)
 * 下随机候选命中率 ~0.07%, 连接从不成熟 (matureCount 恒 0, 睡眠重放无效)。
 * 修复: coactivate 未命中候选时直接建立新连接 (Hebbian synaptogenesis, 8N 上限)。
 */
public class ConnectionHealthTest {

    @Test
    void realScaleLearningFormsMatureConnections() {
        // 真实规模大脑: 学习 4 类模式后, 自发突触形成必须产生成熟连接 (原恒 0)
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        double[][] pats = new double[4][V];
        for (int c = 0; c < 4; c++)
            for (int i = 0; i < V; i++) pats[c][i] = ((i / (V / 8)) == c) ? 0.9 : 0.1;
        int before = brain.synapseFormation().matureCount(0.1);
        for (int e = 0; e < 30; e++)
            for (int c = 0; c < 4; c++) brain.learnVisualWord(pats[c], c);
        int after = brain.synapseFormation().matureCount(0.1);
        assertTrue(after > before,
                "学习应建立成熟连接: before=" + before + " after=" + after);
        assertTrue(after > 0, "成熟连接应>0 (Hebbian 突触形成生效)");
    }

    @Test
    void sleepConsolidationStrengthensConnections() {
        // 睡眠重放共激活 → 连接不减少 (巩固)
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        int before = brain.synapseFormation().matureCount(0.1);
        brain.sleepConsolidate();
        int after = brain.synapseFormation().matureCount(0.1);
        assertTrue(after >= before,
                "睡眠巩固不应削弱连接: before=" + before + " after=" + after);
        assertTrue(before > 0, "学习后应有成熟连接供巩固");
    }

    @Test
    void synaptogenesisRespectsCap() {
        // 大量共激活 → 连接数受 8N 上限约束 (防无界爆炸)
        SynapseFormation sf = new SynapseFormation(64, 8, 1, 0.05, 100000);
        for (int t = 0; t < 20000; t++) {
            sf.coactivate(t % 64, (t * 7 + 3) % 64);
        }
        assertTrue(sf.candidateCount() <= 64 * 8,
                "连接数应受 8N 上限约束, got=" + sf.candidateCount());
        // 重复共激活增强权重 (Hebbian 增强仍工作)
        sf.coactivate(1, 2);
        sf.coactivate(1, 2);
        boolean strong = false;
        for (double[] c : sf.exportConnections()) {
            if (c[0] == 1 && c[1] == 2 && c[2] >= 0.15) strong = true;
            if (c[0] == 2 && c[1] == 1 && c[2] >= 0.15) strong = true;
        }
        assertTrue(strong, "重复共激活应增强既有连接权重");
    }

    @Test
    void hubPrunesWithLearning() {
        // 中枢脉冲网络: 学习后密度下降 (用进废退修剪)
        Brain brain = Brain.simpleBrain();
        double d0 = brain.centralHub().connectionDensity();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 10; e++) {
            brain.learnVisualWord(img, 0);
            brain.syncFrequencyBus();
        }
        double d1 = brain.centralHub().connectionDensity();
        assertTrue(d1 <= d0 + 1e-9, "中枢密度不应上升: " + d0 + " → " + d1);
        assertTrue(d0 >= 0.9 && d0 <= 1.0001, "初始应为近全连接: " + d0);
    }
}
