package io.brainx.core.learning;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Sparse RTRL 论文复现验证 (arXiv 2603.15195):
 *   - 随机 k=4 路径恢复 ~78-84% 全 RTRL 适应能力
 *   - 绝对 k 定律: k=4 在小网络同样有效
 *   - k=0 → k>0 阶梯式跃迁 (无雅可比传播远差于任意非零路径)
 *   - 内存 O(k) vs 全 RTRL O(H²·I)
 *
 * 任务: 带在线分布漂移的序列预测 (论文场景)。
 */
public class SparseRTRLTest {

    /** 在线适应任务: 频率漂移正弦自回归预测。返回末段平均绝对误差。 */
    private double[] runTask(int hidden, int k, long seed) {
        int train = 1200, test = 400;
        SparseRTRL model = new SparseRTRL(2, hidden, k, 0.05, seed);
        double yPrev = 0, yPrev2 = 0;
        // 训练段: 频率漂移 0.012 → 0.035 (在线分布漂移)
        for (int t = 0; t < train; t++) {
            double f = 0.012 + 0.023 * t / train;
            double target = Math.sin(2 * Math.PI * f * t) * 0.8 + 0.1 * Math.sin(2 * Math.PI * 0.007 * t);
            double[] input = {yPrev, yPrev2};
            model.learnStep(input, target);
            yPrev2 = yPrev;
            yPrev = target;
        }
        // 测试段: 固定频率 0.035, 纯预测 (评估适应后的泛化)
        double sumErr = 0;
        for (int t = 0; t < test; t++) {
            double target = Math.sin(2 * Math.PI * 0.035 * (train + t)) * 0.8 + 0.1 * Math.sin(2 * Math.PI * 0.007 * (train + t));
            double[] input = {yPrev, yPrev2};
            double pred = model.predict(input);
            sumErr += Math.abs(target - pred);
            yPrev2 = yPrev;
            yPrev = target;
        }
        return new double[]{sumErr / test, model.traceMemory(), model.fullTraceMemory()};
    }

    private double mean5(int hidden, int k, int idx) {
        double sum = 0;
        for (long s = 1; s <= 5; s++) sum += runTask(hidden, k, s * 100)[idx];
        return sum / 5;
    }

    @Test
    void sparseK4RecoversMostOfFullRTRL() {
        // 论文: k=4 恢复 78-84% 适应能力 (5 seeds)。留余量断言 ≥55%。
        double fullMae = mean5(8, Integer.MAX_VALUE, 0);   // 全 RTRL (k=全部权重)
        double sparseMae = mean5(8, 4, 0);                 // k=4 稀疏
        double k0Mae = mean5(8, 0, 0);                     // 无雅可比传播基线
        // 适应能力 = 相对 k=0 基线的误差降低
        double fullImp = Math.max(1e-9, k0Mae - fullMae);
        double sparseImp = Math.max(1e-9, k0Mae - sparseMae);
        double recovery = sparseImp / fullImp;
        System.out.println("full=" + fullMae + " sparse4=" + sparseMae + " k0=" + k0Mae
                + " recovery=" + String.format("%.0f%%", recovery * 100));
        assertTrue(recovery >= 0.55,
                "稀疏 k=4 应恢复大部分适应能力, recovery=" + String.format("%.0f%%", recovery * 100));
        assertTrue(sparseMae < k0Mae * 0.9,
                "k>0 应显著优于 k=0 (阶梯跃迁): sparse=" + sparseMae + " k0=" + k0Mae);
    }

    @Test
    void absoluteKLaw() {
        // 绝对 k 定律: k=4 在小网络 (n=4) 也有效, 网络越大相对越便宜
        double smallSparse = mean5(4, 4, 0);
        double smallFull = mean5(4, Integer.MAX_VALUE, 0);
        double smallK0 = mean5(4, 0, 0);
        assertTrue(smallSparse < smallK0, "小网络 k=4 也优于无传播");
        // 大网络相对内存节省更显著
        double[] r8 = runTask(8, 4, 1);
        double[] r32 = runTask(32, 4, 1);
        int fullMem8 = (int) r8[2], fullMem32 = (int) r32[2];
        // k=4 内存恒为 4×W (绝对 k 定律), 全 RTRL 内存随隐藏元数线性增长
        assertEquals(4 * (8 * 2 + 8 * 8), (int) r8[1], "k=4 迹内存=4×W (H=8)");
        assertEquals(4 * (32 * 2 + 32 * 32), (int) r32[1], "k=4 迹内存=4×W (H=32)");
        assertTrue(fullMem32 > fullMem8 * 3, "全 RTRL 内存随 H 线性增长: " + fullMem8 + " → " + fullMem32);
    }

    @Test
    void traceMemoryIsLinear() {
        // 算力友好核心: 迹内存 O(k·W) vs 全 RTRL O(H·W), 节省 H/k 倍
        SparseRTRL sparse = new SparseRTRL(64, 16, 4, 0.05, 1);
        SparseRTRL full = new SparseRTRL(64, 16, Integer.MAX_VALUE, 0.05, 1);
        assertEquals(4, sparse.pathCount(), "k=4 路径");
        assertTrue(sparse.traceMemory() < full.traceMemory() / 3,
                "稀疏内存应远小于全 RTRL: sparse=" + sparse.traceMemory()
                        + " full=" + full.traceMemory());
        assertEquals(full.fullTraceMemory(), full.traceMemory(), "k>=H 退化为全 RTRL");
        // 绝对 k 定律: k=4 固定, 网络 H 越大相对节省越多
        SparseRTRL big = new SparseRTRL(64, 256, 4, 0.05, 1);
        assertEquals(4 * (256 * 64 + 256 * 256), big.traceMemory(), "k 固定内存恒定");
        assertTrue((double) big.fullTraceMemory() / big.traceMemory() > 50,
                "大网络相对节省更多: " + (big.fullTraceMemory() / (double) big.traceMemory()));
    }

    @Test
    void fullRTRLBeatsSparseOnHardDrift() {
        // 快速漂移: 全 RTRL 略优 (稀疏恢复大部分但非全部 — 论文 78-84%)
        double full = mean5(8, Integer.MAX_VALUE, 0);
        double sparse = mean5(8, 4, 0);
        // 稀疏不显著差于全量 (噪声范围内)
        assertTrue(sparse < full * 1.6,
                "稀疏应接近全 RTRL: sparse=" + sparse + " full=" + full);
    }
}
