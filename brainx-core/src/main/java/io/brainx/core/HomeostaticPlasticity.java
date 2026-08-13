package io.brainx.core;

/**
 * 稳态可塑性 (Homeostatic Plasticity) —— 神经元发放率稳态调节。
 *
 * 神经科学依据 (Turrigiano 1998, 皮层稳态):
 *   - 神经元有目标发放率区间; 长期过高 → 突触/兴奋性下调 (防暴走/癫痫)
 *   - 长期过低 → 兴奋性上调 (防静默/死亡)
 *   - 与 Hebbian (正反馈) 互补: Hebbian 放大差异, 稳态收敛回目标 → 网络稳定
 *
 * 实现: 每神经元滑动平均发放率 (τ≈10s) 与目标比较 → 逐神经元输入增益微调。
 * 增益范围 [0.2, 5.0], 每步调节 ±0.02 (缓慢, 不破坏学习)。
 */
public class HomeostaticPlasticity {
    /** 目标发放率 (Hz) */
    private static final double TARGET_RATE = 2.0;
    /** 滑动平均时间常数 (ms) */
    private static final double TAU_MS = 10000.0;
    /** 增益范围 */
    private static final double MIN_GAIN = 0.2, MAX_GAIN = 5.0;
    /** 每步调节步长 */
    private static final double STEP_SIZE = 0.02;

    private final int n;
    private final double[] avgRate;   // 每神经元滑动平均发放率 (Hz)
    private final double[] gain;      // 每神经元稳态增益

    public HomeostaticPlasticity(int n) {
        this.n = Math.max(1, n);
        this.avgRate = new double[n];
        this.gain = new double[n];
        for (int i = 0; i < n; i++) gain[i] = 1.0;
    }

    /**
     * 每步更新: 滑动平均发放率 → 稳态增益微调。
     * @param firing 神经元发放状态
     * @param dtMs 步长 (ms)
     * @return 本步调节的神经元数 (诊断)
     */
    public int step(boolean[] firing, double dtMs) {
        int adjusted = 0;
        double alpha = Math.min(1.0, dtMs / TAU_MS);
        double instRate = dtMs > 0 ? 1000.0 / dtMs : 0;
        for (int i = 0; i < n; i++) {
            double r = firing != null && i < firing.length && firing[i] ? instRate : 0;
            avgRate[i] += alpha * (r - avgRate[i]);
            if (avgRate[i] > TARGET_RATE * 1.5 && gain[i] > MIN_GAIN) {
                gain[i] = Math.max(MIN_GAIN, gain[i] - STEP_SIZE);   // 防暴走: 下调
                adjusted++;
            } else if (avgRate[i] < TARGET_RATE * 0.5 && gain[i] < MAX_GAIN) {
                gain[i] = Math.min(MAX_GAIN, gain[i] + STEP_SIZE);   // 防静默: 上调
                adjusted++;
            }
        }
        return adjusted;
    }

    /** 神经元稳态增益 (输入调制系数) */
    public double gain(int i) {
        return (i >= 0 && i < n) ? gain[i] : 1.0;
    }

    /** 增益分布统计: [上调数, 下调数, 保持数] */
    public int[] gainStats() {
        int up = 0, down = 0;
        for (int i = 0; i < n; i++) {
            if (gain[i] > 1.01) up++;
            else if (gain[i] < 0.99) down++;
        }
        return new int[]{up, down, n - up - down};
    }

    /** 摘要 */
    public String summary() {
        int[] s = gainStats();
        return String.format("⚖️ 稳态可塑性: 上调%d/下调%d/保持%d (目标%.1fHz)",
                s[0], s[1], s[2], TARGET_RATE);
    }
}
