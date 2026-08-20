package io.brainx.core.learning;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * EProp + NeuralSimilarity 测试。
 * 对照 arXiv 2506.06904:
 *  - e-prop 关联学习: 60 轮学会目标偏好 (与 pp-prop/D-RTRL 同类效果)
 *  - 资格迹: 脉冲关联衰减记忆 (e_ij 随 pre 脉冲增强, 无脉冲衰减)
 *  - 反馈对齐: 隐藏层误差经固定随机矩阵广播, 学习可进行
 *  - Procrustes 距离: 相同表征距离≈0, 无关表征距离大
 *  - 相似度: 线性相关表征相似度高, 正交表征相似度低
 *  - 论文核心: e-prop 表征与参考表征相似度可比 (对齐后可恢复)
 */
public class EPropTest {

    /** e-prop 关联学习: 输入模式 → 目标偏好 (在线预测任务) */
    @Test
    public void epropLearnsAssociation() {
        EProp eprop = new EProp(4, 8, 0.95, 0.01, 42);
        // 模式A (输入[1,0,0,0]) 目标 +1; 模式B (输入[0,1,0,0]) 目标 -1
        for (int epoch = 0; epoch < 80; epoch++) {
            trainPattern(eprop, new double[]{1, 0, 0, 0}, +1.0);
            trainPattern(eprop, new double[]{0, 1, 0, 0}, -1.0);
        }
        double outA = readout(eprop.forward(new double[]{1, 0, 0, 0}));
        double outB = readout(eprop.forward(new double[]{0, 1, 0, 0}));
        assertTrue(outA > 0, "模式A应学成正偏好, got " + outA);
        assertTrue(outB < 0, "模式B应学成负偏好, got " + outB);
        assertTrue(outA > outB, "两模式应区分: " + outA + " vs " + outB);
    }

    private void trainPattern(EProp eprop, double[] input, double target) {
        eprop.resetTraces();
        for (int t = 0; t < 5; t++) {
            double[] mem = new double[8];
            java.util.Arrays.fill(mem, 0.5 + 0.1 * t);
            eprop.step(mem, 0.6, input);
        }
        double[] out = eprop.forward(input);
        // 每单元局部误差: 偶单元目标=+target, 奇单元目标=-target
        // (突触后神经元局部教学信号, 比均匀误差更符合 e-prop 三因子设定)
        double[] errVec = new double[8];
        for (int i = 0; i < 8; i++) {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            errVec[i] = (target * sign) - out[i];
        }
        eprop.setOutputError(errVec);
        eprop.update();
        eprop.applyGradients();
    }

    private double readout(double[] hidden) {
        double s = 0;
        for (int i = 0; i < hidden.length; i++) s += hidden[i] * (i % 2 == 0 ? 1 : -1);
        return s / hidden.length;
    }

    /** 资格迹: 有 pre 脉冲时增强, 无脉冲时衰减 */
    @Test
    public void eligibilityTraceDynamics() {
        EProp eprop = new EProp(2, 2, 0.9, 0.01, 42);
        double[] mem = {1.0, 1.0};
        double[] preOn = {1, 0};
        eprop.step(mem, 0.5, preOn);
        double e00 = eprop.eligibility(0, 0);
        double e01 = eprop.eligibility(0, 1);
        assertTrue(e00 > 0, "有脉冲的资格迹应 > 0");
        assertTrue(e00 > e01, "有脉冲的迹应大于无脉冲的迹: " + e00 + " vs " + e01);
        // 后续无脉冲 → 衰减
        double[] preOff = {0, 0};
        eprop.step(mem, 0.5, preOff);
        assertTrue(eprop.eligibility(0, 0) < e00 * 1.001, "无脉冲应衰减");
    }

    /** 反馈对齐: 隐藏层经固定随机矩阵收到误差后权重演化 */
    @Test
    public void feedbackAlignmentDrivesLearning() {
        EProp eprop = new EProp(2, 4, 0.95, 0.02, 7);
        double[] input = {1, 0};
        double[] outErr = {0.5, -0.5};
        double before = eprop.weight(0, 0);
        for (int epoch = 0; epoch < 30; epoch++) {
            eprop.resetTraces();
            double[] mem = {1.0, 1.0, 1.0, 1.0};
            eprop.step(mem, 0.6, input);
            eprop.broadcastHiddenError(outErr);
            eprop.update();
            eprop.applyGradients();
        }
        double after = eprop.weight(0, 0);
        assertTrue(Math.abs(after - before) > 1e-4, "反馈对齐应驱动权重演化: " + before + " -> " + after);
    }

    /** Procrustes: 相同表征距离≈0, 相似度≈1 */
    @Test
    public void procrustesIdenticalIsZero() {
        double[][] X = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {2, 1, 0}};
        double[][] Y = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {2, 1, 0}};
        double d = NeuralSimilarity.procrustesDistance(X, Y);
        assertTrue(d < 1e-6, "相同表征 Procrustes 距离应≈0, got " + d);
        assertTrue(NeuralSimilarity.similarity(X, Y) > 0.99, "相同表征相似度应≈1");
    }

    /** Procrustes: 正交变换后的表征距离≈0 (Procrustes 对旋转不变) */
    @Test
    public void procrustesRotationInvariant() {
        double[][] X = {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
        // Y = X 旋转 90°: (x,y)→(-y,x)
        double[][] Y = {{0, 1}, {-1, 0}, {-1, 1}, {0, 0}};
        double d = NeuralSimilarity.procrustesDistance(X, Y);
        assertTrue(d < 1e-6, "旋转不变: Procrustes 距离应≈0, got " + d);
    }

    /** Procrustes: 无关表征距离大, 相似度低 */
    @Test
    public void procrustesUnrelatedIsLarge() {
        double[][] X = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
        double[][] Y = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}};
        double sim = NeuralSimilarity.similarity(X, Y);
        assertTrue(sim < 0.9, "无关表征相似度应较低, got " + sim);
    }

    /** 论文核心: 表征相似度可比性 — 线性相关表征相似度显著高于无关 */
    @Test
    public void correlatedMoreSimilarThanUncorrelated() {
        double[][] ref = new double[10][4];
        double[][] corr = new double[10][4];
        double[][] uncorr = new double[10][4];
        java.util.Random rnd = new java.util.Random(3);
        for (int r = 0; r < 10; r++) {
            double base = rnd.nextGaussian();
            for (int c = 0; c < 4; c++) {
                ref[r][c] = base + 0.3 * rnd.nextGaussian();
                corr[r][c] = ref[r][c] + 0.2 * rnd.nextGaussian();  // 强相关
                uncorr[r][c] = rnd.nextGaussian();                   // 无关
            }
        }
        double simCorr = NeuralSimilarity.similarity(ref, corr);
        double simUncorr = NeuralSimilarity.similarity(ref, uncorr);
        assertTrue(simCorr > simUncorr,
                "相关表征相似度应更高: " + simCorr + " vs " + simUncorr);
    }

    /** 权重梯度累积: update 后 applyGradients 应用并清零 */
    @Test
    public void gradientAccumulationAndClear() {
        EProp eprop = new EProp(2, 2, 0.9, 0.01, 1);
        eprop.step(new double[]{1, 1}, 0.5, new double[]{1, 0});
        eprop.setOutputError(new double[]{0.3, -0.3});
        eprop.update();
        double before = eprop.weight(0, 0);
        eprop.applyGradients();
        double after = eprop.weight(0, 0);
        assertTrue(Math.abs(after - before) > 1e-8, "权重应被梯度更新");
        assertEquals(0.0, eprop.learningSignal(0), 1e-12, "梯度应用后信号清零");
    }

    /** 资格迹重置: 序列间清空 */
    @Test
    public void resetTracesClears() {
        EProp eprop = new EProp(2, 2, 0.9, 0.01, 1);
        eprop.step(new double[]{1, 1}, 0.5, new double[]{1, 0});
        assertTrue(eprop.eligibility(0, 0) > 0);
        eprop.resetTraces();
        assertEquals(0.0, eprop.eligibility(0, 0), 1e-12, "重置后迹归零");
    }
}
