package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 预测验证与纠正闭环验证 (v5.4 用户"预测补足→性能验证→纠正连接"):
 *   - 匀速运动: 预测误差小 (前向模型正确, 连接稳定)
 *   - 场景突变: 误差大 → 预测器重置纠正 → 快速恢复
 *   - 内部联结: 学习驱动突触/联想/pp-prop 连接演化
 */
public class PredictionLoopTest {

    /** 中速平移模式: 高空间频率正弦以 2 采样/帧匀速右移 (缓慢移动场景 — 预测可靠区间) */
    private static double[] movingFrame(int t, int dim) {
        double[] f = new double[dim];
        for (int i = 0; i < dim; i++) f[i] = 0.5 + 0.4 * Math.sin((i - t * 2) * 0.4);
        return f;
    }

    @Test
    void verificationLowErrorOnUniformMotion() {
        // 匀速运动: 预测验证误差应小 (预测准)
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        for (int t = 0; t < 10; t++) {
            double err = brain.verifyVisualPrediction(movingFrame(t, V));
            // t=0 是初始化帧 (无预测可验证, err=1.0 符合设计); 之后误差应小
            if (t > 0) assertTrue(err < 0.25, "匀速运动预测误差应小, t=" + t + " err=" + err);
        }
        assertTrue(brain.lastVisualPredErr() < 0.25, "预测误差应记录");
    }

    @Test
    void correctionAfterSuddenChange() {
        // 场景突变: 误差骤增 → 速度 EMA 翻转纠正 → 之后预测恢复
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        for (int t = 0; t < 10; t++) brain.verifyVisualPrediction(movingFrame(t, V));
        // 突变: 反向运动 (速度 EMA 从 +0.18 翻转到 -0.18, 约需 5 帧)
        double errAtChange = brain.verifyVisualPrediction(movingFrame(-10, V));
        double errAfter = 1.0;
        for (int t = 10; t < 24; t++) {
            errAfter = brain.verifyVisualPrediction(movingFrame(-t, V));
        }
        assertTrue(errAfter < 0.25,
                "突变后速度翻转应恢复预测: atChange=" + errAtChange + " after=" + errAfter);
    }

    @Test
    void internalConnectionsDriveLearning() {
        // 内部联结检查: 学习流驱动各连接演化 (突触/中枢/联想/pp-prop)
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        double[][] before = clone2d(brain.visualToAssocWeights());
        int synBefore = brain.synapseFormation().matureCount(0.1);
        int hubBefore = brain.centralHub().totalSpikes();
        for (int e = 0; e < 5; e++)
            for (int c = 0; c < 4; c++) brain.learnVisualWord(movingFrame(c * 5, V), c);
        brain.syncFrequencyBus();
        // 突触成熟连接增长 (学习建连接)
        assertTrue(brain.synapseFormation().matureCount(0.1) > synBefore,
                "突触连接应随学习增长: " + synBefore + " → " + brain.synapseFormation().matureCount(0.1));
        // 中枢联动 (脉冲增加)
        assertTrue(brain.centralHub().totalSpikes() > hubBefore,
                "中枢应随学习产生脉冲: " + hubBefore + " → " + brain.centralHub().totalSpikes());
        // pp-prop 视觉→联想连接演化
        double[][] after = brain.visualToAssocWeights();
        double delta = 0;
        for (int i = 0; i < after.length; i++)
            for (int j = 0; j < after[i].length; j++) delta += Math.abs(after[i][j] - before[i][j]);
        assertTrue(delta > 0.1, "pp-prop 连接应演化, Δ=" + delta);
        // 联想权重分化 (学过概念 > 未学)
        double[][] aw = brain.assocWeights();
        double learned = 0, unlearned = 0;
        for (double[] row : aw) { learned += row[0]; unlearned += row[7]; }
        assertTrue(learned > unlearned,
                "学过的概念联想权重应显著>未学: " + learned + " vs " + unlearned);
        // 预测器可工作 (验证闭环已接入)
        assertTrue(brain.visualPredictionConfidence() >= 0, "预测器置信度可读");
    }

    private static double[][] clone2d(double[][] m) {
        double[][] c = new double[m.length][];
        for (int i = 0; i < m.length; i++) c[i] = m[i].clone();
        return c;
    }
}
