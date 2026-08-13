package io.brainx.core;

import io.brainx.core.neuron.HH;
import io.brainx.core.neuron.LIF;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 算法效果验证 (v5.4 效果审计):
 *   - BrainFuse: HH 噪声鲁棒性优于 LIF (发放率编码 d' 分离度, 强噪声下差距最明显)
 *   - SparseRTRL: 稳定任务下全 RTRL 最优 (k=H 退化正确)
 */
public class AlgorithmEffectTest {

    /** 发放率编码: 500ms 内发放数, 输入 = 信号(有/无) + 噪声 */
    private static double rate(io.brainx.core.Neuron n, double sig, double noiseAmp, boolean hasSig, long seed) {
        java.util.Random r = new java.util.Random(seed);
        n.reset();
        int c = 0;
        for (int i = 0; i < 500; i++) {
            double noise = (r.nextDouble() - 0.5) * 2 * noiseAmp;
            n.step((hasSig ? sig : 0) + noise, 1.0);
            if (n.fired()) c++;
        }
        return c;
    }

    /** d': 有信号 vs 无信号发放率分离度 */
    private static double dPrime(io.brainx.core.Neuron n, double sig, double noiseAmp) {
        int N = 30;
        double[] a = new double[N], b = new double[N];
        for (int i = 0; i < N; i++) {
            a[i] = rate(n, sig, noiseAmp, true, i + 1);
            b[i] = rate(n, sig, noiseAmp, false, i + 1);
        }
        double ma = 0, mb = 0;
        for (int i = 0; i < N; i++) { ma += a[i]; mb += b[i]; }
        ma /= N; mb /= N;
        double va = 0, vb = 0;
        for (int i = 0; i < N; i++) { va += (a[i] - ma) * (a[i] - ma); vb += (b[i] - mb) * (b[i] - mb); }
        va /= N; vb /= N;
        return Math.abs(ma - mb) / Math.sqrt((va + vb) / 2);
    }

    @Test
    void hhNoiseRobustnessBeatsLIF() {
        // BrainFuse 论文实证: HH 噪声鲁棒性优于 LIF。
        // 发放率编码信号检测: 强噪声 (0.6-1.0×阈值) 下 HH d' 应显著高于 LIF。
        double hhHigh = dPrime(new HH(), 0.65 * 0.75, 0.65 * 1.0);
        double lifHigh = dPrime(LIF.defaultParams(), 15 * 0.75, 15 * 1.0);
        assertTrue(Double.isFinite(lifHigh), "LIF 高噪声 d' 应有限");
        assertTrue(hhHigh > lifHigh,
                "HH 噪声鲁棒性应优于 LIF (BrainFuse): HH d'=" + hhHigh + " LIF d'=" + lifHigh);
        // 中等噪声同样成立
        double hhMid = dPrime(new HH(), 0.65 * 0.75, 0.65 * 0.6);
        double lifMid = dPrime(LIF.defaultParams(), 15 * 0.75, 15 * 0.6);
        assertTrue(hhMid > lifMid, "中等噪声 HH 仍应占优: " + hhMid + " vs " + lifMid);
    }

    @Test
    void sparseRTRLFullIsOptimalOnStableTask() {
        // 稳定任务 (无漂移): 全 RTRL (k=H) 应最优 — 退化正确性
        double full = meanStable(8, Integer.MAX_VALUE);
        double sparse = meanStable(8, 4);
        assertTrue(full <= sparse * 1.2,
                "稳定任务全 RTRL 应最优: full=" + full + " sparse=" + sparse);
    }

    private static double meanStable(int hidden, int k) {
        double sum = 0;
        for (int seed = 1; seed <= 3; seed++) sum += stableMae(hidden, k, seed);
        return sum / 3;
    }

    private static double stableMae(int hidden, int k, long seed) {
        io.brainx.core.learning.SparseRTRL m = new io.brainx.core.learning.SparseRTRL(2, hidden, k, 0.05, seed);
        int train = 800, test = 300;
        double y1 = 0, y2 = 0;
        for (int t = 0; t < train; t++) {
            double target = Math.sin(2 * Math.PI * 0.02 * t) * 0.8;
            m.learnStep(new double[]{y1, y2}, target);
            y2 = y1; y1 = target;
        }
        double sum = 0;
        for (int t = 0; t < test; t++) {
            double target = Math.sin(2 * Math.PI * 0.02 * (train + t)) * 0.8;
            sum += Math.abs(target - m.predict(new double[]{y1, y2}));
            y2 = y1; y1 = target;
        }
        return sum / test;
    }
}
