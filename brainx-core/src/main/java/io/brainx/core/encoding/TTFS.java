package io.brainx.core.encoding;

/**
 * TTFS (Time-To-First-Spike) 时间到首次脉冲编码。
 * 来源: BrainTrace 论文 (NC 2026) Fig.5F 实证发现:
 *   "Neurons in each population predominantly utilized a time-to-first-spike
 *    coding strategy... neuronal responses displayed clear rank ordering,
 *    with neurons firing sequentially in response to stimuli"
 *   (神经元主要用首次发放时间编码; 响应呈清晰顺序排列, 神经元按序发放)
 *
 * 编码原理: 刺激越强 → 发放越早 (首次脉冲时间越短)。
 *   t_first = T_max * exp(-lambda * intensity)
 * 解码: 脉冲时间顺序 = 强度排序 (rank ordering)。
 */
public class TTFS {
    private final double tMaxMs;    // 最大延迟 (弱刺激)
    private final double lambda;    // 强度-延迟斜率
    private final double tMinMs;    // 最小延迟 (强刺激)

    public TTFS(double tMaxMs, double lambda, double tMinMs) {
        this.tMaxMs = tMaxMs;
        this.lambda = lambda;
        this.tMinMs = tMinMs;
    }

    public static TTFS defaultParams() { return new TTFS(100.0, 3.0, 1.0); }

    /** 强度 (0-1) → 首次脉冲时间 (ms)。强刺激早发放。 */
    public double encode(double intensity) {
        double clamped = Math.max(0, Math.min(1, intensity));
        double t = tMaxMs * Math.exp(-lambda * clamped);
        return Math.max(tMinMs, t);
    }

    /** 时间 (ms) → 解码强度 (0-1)。早发放 = 强刺激。 */
    public double decode(double spikeTimeMs) {
        if (spikeTimeMs <= tMinMs) return 1.0;
        if (spikeTimeMs >= tMaxMs) return 0.0;
        return -Math.log(spikeTimeMs / tMaxMs) / lambda;
    }

    /**
     * 对一组强度编码: 返回每个神经元的首次脉冲时间数组。
     * 顺序 = 强度排序 (rank ordering, BrainTrace Fig5F)。
     */
    public double[] encodeAll(double[] intensities) {
        double[] out = new double[intensities.length];
        for (int i = 0; i < intensities.length; i++) out[i] = encode(intensities[i]);
        return out;
    }

    /** 发放顺序: 按首次脉冲时间排序的神经元索引 (顺序编码) */
    public int[] rankOrder(double[] intensities) {
        double[] times = encodeAll(intensities);
        Integer[] idx = new Integer[times.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(times[a], times[b]));
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) out[i] = idx[i];
        return out;
    }

    /** 生成脉冲序列: 每个神经元只在首次脉冲时间发放一次 (TTFS 编码) */
    public int[][] generateSpikeTrains(double[] intensities, int durationSteps, double dtMs) {
        double[] times = encodeAll(intensities);
        int[][] trains = new int[intensities.length][durationSteps];
        for (int i = 0; i < intensities.length; i++) {
            int step = (int) Math.round(times[i] / dtMs);
            if (step < durationSteps) trains[i][step] = 1;
        }
        return trains;
    }

    /** 验证: 强刺激的首次脉冲应显著早于弱刺激 */
    public boolean verifyOrdering() {
        double strong = encode(0.9);
        double weak = encode(0.1);
        return strong < weak;  // 强刺激应更早
    }
}
