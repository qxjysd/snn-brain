package io.brainx.core.mass;

/**
 * Kuramoto 相位振荡器网络（同步现象）。
 * 方程: dθ_i/dt = ω_i + K/N * Σ_j sin(θ_j - θ_i)
 * 对应 BrainMass 的 KuramotoNetwork + brain_zoo 的 kuramoto(8相位)。
 * 用于: 神经振荡同步/脑节律研究。
 */
public class Kuramoto {
    private final int n;
    private final double[] phases;     // 相位
    private final double[] freqs;      // 固有频率
    private final double coupling;     // 耦合强度 K
    private final double[] adj;        // 邻接矩阵 (n*n)

    public Kuramoto(int n, double coupling, long seed) {
        this.n = n;
        this.coupling = coupling;
        this.phases = new double[n];
        this.freqs = new double[n];
        this.adj = new double[n * n];
        java.util.Random rnd = new java.util.Random(seed);
        for (int i = 0; i < n; i++) {
            phases[i] = rnd.nextDouble() * 2 * Math.PI;
            freqs[i] = 0.8 + rnd.nextDouble() * 0.4;  // ~1 Hz 附近
        }
        // 默认全连接单位权重
        for (int i = 0; i < n * n; i++) adj[i] = 1.0;
        for (int i = 0; i < n; i++) adj[i * n + i] = 0;
    }

    public static Kuramoto defaultParams() { return new Kuramoto(8, 0.5, 42); }

    /** 前进一步 */
    public void step(double dtMs) {
        double[] dPhase = new double[n];
        double dt = dtMs / 1000.0;
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += adj[i * n + j] * Math.sin(phases[j] - phases[i]);
            }
            dPhase[i] = freqs[i] * 2 * Math.PI + coupling / n * sum;
        }
        for (int i = 0; i < n; i++) {
            phases[i] += dPhase[i] * dt;
        }
    }

    /** 同步序参量 R (0..1): 1=完全同步 */
    public double orderParameter() {
        double sumCos = 0, sumSin = 0;
        for (double p : phases) { sumCos += Math.cos(p); sumSin += Math.sin(p); }
        return Math.sqrt(sumCos * sumCos + sumSin * sumSin) / n;
    }

    public double phase(int i) { return phases[i]; }
    public int size() { return n; }
}
