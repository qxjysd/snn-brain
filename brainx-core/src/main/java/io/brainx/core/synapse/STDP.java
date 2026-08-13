package io.brainx.core.synapse;

/**
 * STDP (Spike-Timing-Dependent Plasticity) 学习规则。
 * 对应 brain.state 的 STDP 可塑性模型：
 *   Δw = A+ * exp(-Δt/τ+)  若 前发放 → 后发放 (potentiation)
 *   Δw = -A- * exp(Δt/τ-)  若 后发放 → 前发放 (depression)
 * 实现: 每个突触维护前后脉冲时间，配对更新。
 */
public final class STDP {
    private final double aPlus, aMinus;   // 增强/抑制幅度
    private final double tauPlus, tauMinus; // 时间窗口常数 (ms)
    private final double wMin, wMax;       // 权重边界

    public STDP(double aPlus, double aMinus, double tauPlus, double tauMinus, double wMin, double wMax) {
        this.aPlus = aPlus;
        this.aMinus = aMinus;
        this.tauPlus = tauPlus;
        this.tauMinus = tauMinus;
        this.wMin = wMin;
        this.wMax = wMax;
    }

    public static STDP defaultParams() {
        return new STDP(0.01, 0.012, 20.0, 20.0, 0.0, 1.0);
    }

    /** 前神经元发放时调用：检查之前是否有后发放 (Δt = t_post - t_pre < 0) → 抑制 */
    public double onPreSpike(Synapse s, double currentTimeMs) {
        double dt = s.lastPostSpikeTime - currentTimeMs;  // Δt = t_post_prev - t_pre < 0 (反因果)
        double dw = 0;
        if (dt < 0 && dt > -50.0) {  // 后发放在前发放之前 50ms 内 → 抑制
            dw = -aMinus * Math.exp(dt / tauMinus);
        }
        s.lastPreSpikeTime = currentTimeMs;
        return apply(s, dw);
    }

    /** 后神经元发放时调用：检查之前是否有前发放 (Δt = t_post - t_pre > 0) → 增强 */
    public double onPostSpike(Synapse s, double currentTimeMs) {
        double dt = currentTimeMs - s.lastPreSpikeTime;  // Δt = t_post - t_pre_prev > 0 (因果)
        double dw = 0;
        if (dt > 0 && dt < 50.0) {  // 前发放先于后发放 50ms 内 → 增强
            dw = aPlus * Math.exp(-dt / tauPlus);
        }
        s.lastPostSpikeTime = currentTimeMs;
        return apply(s, dw);
    }

    private double apply(Synapse s, double dw) {
        s.weight = Math.max(wMin, Math.min(wMax, s.weight + dw));
        return dw;
    }
}
