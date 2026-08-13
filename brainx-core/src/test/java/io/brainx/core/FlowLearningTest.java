package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 学习流完整性回归测试 (v5.4 端到端审计):
 *   数据转换链 / 模块连接 / 网络连接 — 都须随学习真实演化。
 *
 * 背景: 审计发现 visualToAssoc/auditoryToAssoc 只 forward 不 update (权重冻结),
 *   联想学习全靠 associativeMemory。修复: pp-prop 误差驱动接入 (目标=该词关联列,
 *   信号=目标-当前)。坑: ① 自巩固信号 (tanh 激活) 灾难性漂移 → 全识别偏向最后学的词;
 *   ② lr=0.01 破坏泛化 (过拟合变体, variant 0.5→0.3), lr=0.0005 平衡。
 */
public class FlowLearningTest {

    private static double[] pat(int c, int V) {
        double[] p = new double[V];
        for (int i = 0; i < V; i++) p[i] = ((i / (V / 8)) == c) ? 0.9 : 0.1;
        return p;
    }

    @Test
    void ppPropConnectionLearns() {
        // 网络连接: 视觉→联想 pp-prop 权重随学习演化 (原冻结 Δ=0)
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        double[][] before = clone2d(brain.visualToAssocWeights());
        for (int e = 0; e < 20; e++)
            for (int c = 0; c < 4; c++) brain.learnVisualWord(pat(c, V), c);
        double[][] after = brain.visualToAssocWeights();
        double delta = 0;
        for (int i = 0; i < after.length; i++)
            for (int j = 0; j < after[i].length; j++) delta += Math.abs(after[i][j] - before[i][j]);
        assertTrue(delta > 1.0, "pp-prop 连接应随学习演化, Δ=" + delta);
    }

    @Test
    void learningFlowRecognizesAll() {
        // 数据转换链 + 识别: 学习 4 词 → 全识别 (端到端畅通)
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        for (int e = 0; e < 30; e++)
            for (int c = 0; c < 4; c++) brain.learnVisualWord(pat(c, V), c);
        String[] vocab = {"概念#1", "概念#2", "概念#3", "概念#4"};
        int hit = 0;
        for (int c = 0; c < 4; c++) {
            if (brain.recognizeVisual(pat(c, V)).equals(vocab[c])) hit++;
        }
        assertTrue(hit >= 3, "学习流应识别至少 3/4, got " + hit + "/4");
        // 数据转换: 编码器输出维度 = 皮层维度
        assertEquals(V, new VisualNeuralEncoder().encode(new double[128 * 128], 128, 128).length,
                "编码器输出维度应匹配皮层");
    }

    @Test
    void moduleConnectionsActivateThroughLearning() {
        // 模块连接: 学习 → 中枢/记忆/多巴胺/突触全部联动
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        int hubBefore = brain.centralHub().totalSpikes();
        int synBefore = brain.synapseFormation().matureCount(0.1);
        double dopBefore = brain.dopamineSystem().dopamine();
        int ltBefore = brain.longTermLabels().length();
        for (int e = 0; e < 5; e++)
            for (int c = 0; c < 4; c++) brain.learnVisualWord(pat(c, V), c);
        for (int c = 0; c < 4; c++) brain.recognizeVisualWithConfidence(pat(c, V));
        brain.syncFrequencyBus();
        assertTrue(brain.centralHub().totalSpikes() > hubBefore, "中枢应通过学习产生脉冲");
        assertTrue(brain.synapseFormation().matureCount(0.1) > synBefore, "学习应建立突触连接");
        assertTrue(brain.dopamineSystem().dopamine() != dopBefore, "学习应触发多巴胺 RPE");
        assertTrue(brain.longTermLabels().length() > ltBefore, "学习应沉淀长期记忆");
        // 联想权重分化: 学过的词 > 未学
        double[][] aw = brain.assocWeights();
        double learned = 0, unlearned = 0;
        for (double[] row : aw) { learned += row[0]; unlearned += row[7]; }
        assertTrue(learned > unlearned, "学过的词联想权重应显著>未学: " + learned + " vs " + unlearned);
    }

    @Test
    void auditoryConnectionLearns() {
        // 听觉→联想 pp-prop 连接同样演化
        Brain brain = Brain.simpleBrain();
        int B = AudioNeuralEncoder.BANDS;
        double[][] before = clone2d(brain.auditoryToAssocWeights());
        double[] aud = new double[B];
        for (int i = 0; i < B; i++) aud[i] = 0.5 + 0.4 * Math.sin(i);
        for (int e = 0; e < 10; e++) brain.learnAuditoryWord(aud, 1);
        double[][] after = brain.auditoryToAssocWeights();
        double delta = 0;
        for (int i = 0; i < after.length; i++)
            for (int j = 0; j < after[i].length; j++) delta += Math.abs(after[i][j] - before[i][j]);
        assertTrue(delta > 0.1, "听觉 pp-prop 连接应随学习演化, Δ=" + delta);
        assertEquals(brain.vocabulary(1), brain.recognizeAuditory(aud), "听觉识别应指向学的概念");
    }

    private static double[][] clone2d(double[][] m) {
        double[][] c = new double[m.length][];
        for (int i = 0; i < m.length; i++) c[i] = m[i].clone();
        return c;
    }
}
