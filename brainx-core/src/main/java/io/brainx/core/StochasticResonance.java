package io.brainx.core;

/**
 * 随机共振 (Stochastic Resonance) —— 噪声帮助弱信号检测。
 *
 * 神经科学依据: 神经元接收到的子阈值刺激本身不足以引发放电，
 * 但叠加合适幅度的随机噪声后，膜电位可偶尔越过阈值 →
 * 噪声"帮助"信号被检测 (Benzi et al. 1981; 大脑中普遍存在)。
 *
 * 模型: 单阈值检测器
 *   y(t) = sign( x(t) + noise(t) - threshold )
 *   有噪声时弱信号检测率提升; 噪声过大反而淹没信号 (倒U型曲线)。
 */
public class StochasticResonance {
    private final double threshold;
    private final double dtMs;
    private final NoiseSource noise;
    private final java.util.Random rnd;

    public StochasticResonance(double threshold, double noiseAmplitude, double dtMs, long seed) {
        this.threshold = threshold;
        this.dtMs = dtMs;
        this.rnd = new java.util.Random(seed);
        this.noise = NoiseSource.gaussian(noiseAmplitude, dtMs);
    }

    /** 单次检测: 子阈值信号 x + 噪声 是否过阈值 */
    public boolean detect(double signal) {
        return signal + noise.sample() >= threshold;
    }

    /**
     * 测试不同噪声幅度下的检测有效性 (倒U型曲线)。
     * 有效性 = P(检测|有信号) - P(检测|无信号) [误报率扣除]
     * 噪声过小时信号过不了阈值(无效), 噪声过大时真假难辨(有效性趋0) →
     * 中间存在最优噪声 (随机共振核心结论)。
     */
    public static double[] detectionCurve(double signal, double threshold, double dtMs, int trials) {
        double[] noiseLevels = {0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0};
        double[] validity = new double[noiseLevels.length];
        for (int n = 0; n < noiseLevels.length; n++) {
            StochasticResonance sr = new StochasticResonance(threshold, noiseLevels[n], dtMs, 42);
            int hits = 0, falseAlarms = 0;
            for (int t = 0; t < trials; t++) {
                if (sr.detect(signal)) hits++;          // 有信号检测
                if (sr.detect(0.0)) falseAlarms++;      // 无信号误报
            }
            double det = (double) hits / trials;
            double fa = (double) falseAlarms / trials;
            validity[n] = det - fa;   // 噪声→∞时 det→0.5, fa→0.5, 差值→0
        }
        return validity;
    }

    /** 最佳噪声幅度 (检测率最高) */
    public static double optimalNoise(double signal, double threshold, double dtMs, int trials) {
        double[] levels = {0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8};
        double best = 0, bestRate = 0;
        for (double lv : levels) {
            StochasticResonance sr = new StochasticResonance(threshold, lv, dtMs, 42);
            int hits = 0;
            for (int t = 0; t < trials; t++) if (sr.detect(signal)) hits++;
            double rate = (double) hits / trials;
            if (rate > bestRate) { bestRate = rate; best = lv; }
        }
        return best;
    }
}
