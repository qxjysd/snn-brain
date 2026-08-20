package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * v5.5 流畅度基准测试 — 验证 5 个新机制未拖慢核心循环。
 * 对照 v5.4 基线 (JIT 优化后 learnVisualWord 29.2ms):
 *  - learn 单次 < 60ms (含全部新机制, 帧耗时阈值 80ms 的 75%)
 *  - recognize < 25ms
 *  - encode < 15ms
 *  - 新机制单步开销: astrocyte O(N) / wholeBrain 8节点 / diffusion O(N)
 *  - e-prop 联想层前向 (120×5248 最大热点) 有预算约束
 */
public class V55PerformanceTest {

    private Brain newBrain() {
        Brain b = Brain.simpleBrain();
        // 预热: 学 2 个概念 (JIT/缓冲分配)
        double[] f = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int w = 0; w < 2; w++) {
            for (int i = 0; i < f.length; i++) f[i] = 0.1 + 0.8 * ((i / (f.length / 8)) == w ? 1.0 : 0.0);
            b.learnVisual(f);
        }
        return b;
    }

    private long timeIt(Runnable r, int reps) {
        long best = Long.MAX_VALUE;
        for (int k = 0; k < reps; k++) {
            long t0 = System.nanoTime();
            r.run();
            long dt = System.nanoTime() - t0;
            best = Math.min(best, dt);
        }
        return best / 1_000_000;  // ms
    }

    /** 学习耗时: learnVisualWord 单独 (含5新机制) 应 ≈ v5.4 基线 29.2ms */
    @Test
    public void learnWithinBudget() {
        Brain b = newBrain();
        double[] f = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < f.length; i++) f[i] = 0.1 + 0.8 * ((i / (f.length / 8)) == 3 ? 1.0 : 0.0);
        long ms = timeIt(() -> b.learnVisualWord(f, 3), 5);
        System.out.println("[perf] learnVisualWord = " + ms + "ms (v5.4基线 29.2ms, 预算 60ms)");
        assertTrue(ms < 60, "learnVisualWord 应 < 60ms, got " + ms + "ms (含5新机制)");
    }

    /** 识别耗时 < 25ms */
    @Test
    public void recognizeWithinBudget() {
        Brain b = newBrain();
        double[] f = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < f.length; i++) f[i] = 0.1 + 0.8 * ((i / (f.length / 8)) == 1 ? 1.0 : 0.0);
        b.learnVisual(f);
        long ms = timeIt(() -> b.recognizeVisual(f), 5);
        System.out.println("[perf] recognizeVisual = " + ms + "ms (预算 25ms)");
        assertTrue(ms < 25, "recognize 应 < 25ms, got " + ms + "ms");
    }

    /** 新机制单步: 星形胶质细胞 5368 神经元 O(N) < 2ms */
    @Test
    public void astrocyteStepCheap() {
        Brain b = newBrain();
        AstrocyteLayer a = b.astrocyteLayer();
        long ms = timeIt(() -> a.step(null, null, new boolean[5368], 1.0), 50);
        System.out.println("[perf] astrocyteStep(5368) = " + ms + "ms");
        assertTrue(ms < 2, "胶质单步应 < 2ms, got " + ms + "ms");
    }

    /** 全脑网络单步 8 节点 < 1ms */
    @Test
    public void wholeBrainStepCheap() {
        Brain b = newBrain();
        WholeBrainNetwork w = b.wholeBrain();
        double[] stim = new double[WholeBrainNetwork.N];
        stim[3] = 4.0;
        long ms = timeIt(() -> w.step(stim, 1.0), 100);
        System.out.println("[perf] wholeBrainStep = " + ms + "ms");
        assertTrue(ms < 1, "全脑单步应 < 1ms, got " + ms + "ms");
    }

    /** e-prop 前向 (120×5248 热点) — 学习内已调用, 单独验证不失控 */
    @Test
    public void epropForwardBudget() {
        Brain b = newBrain();
        var e = b.epropAssoc();
        double[] in = new double[VisualNeuralEncoder.OUTPUT_DIM + 128];
        java.util.Arrays.fill(in, 0.1);
        long ms = timeIt(() -> e.forward(in), 20);
        System.out.println("[perf] epropForward(5248→120) = " + ms + "ms");
        assertTrue(ms < 15, "e-prop 前向应 < 15ms, got " + ms + "ms");
    }

    /** 连续学习 10 概念: 总时长 < 1.2s (含一切开销, 防累积劣化) */
    @Test
    public void tenLearnTotalBudget() {
        Brain b = newBrain();
        long t0 = System.nanoTime();
        for (int w = 0; w < 10; w++) {
            double[] f = new double[VisualNeuralEncoder.OUTPUT_DIM];
            for (int i = 0; i < f.length; i++) f[i] = 0.1 + 0.8 * ((i / (f.length / 8)) == (w % 8) ? 1.0 : 0.0);
            b.learnVisual(f);
        }
        long total = (System.nanoTime() - t0) / 1_000_000;
        System.out.println("[perf] 10x learnVisual = " + total + "ms (预算 1200ms)");
        assertTrue(total < 1200, "10 次学习应 < 1.2s, got " + total + "ms");
    }

    /** 内存: 新机制不引入大数组 (astrocyte 2×5368 double ≈ 86KB) */
    @Test
    public void memoryFootprintSmall() {
        Brain b = newBrain();
        long mb = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        System.out.println("[perf] JVM 已用内存 = " + mb / 1024 + "KB");
        assertTrue(mb < 512 * 1024 * 1024, "JVM 内存应 < 512MB, got " + mb / 1024 / 1024 + "MB");
    }
}
