package io.brainx.core.synapse;

/**
 * 突触模型（对应 brain.state 的 Synapse 类）。
 * Delta: 脉冲到达即刻传递 (无时间常数)
 * Exp: 指数衰减突触电流 (tau_syn)
 */
public final class Synapse {
    /** 突触类型 */
    public enum Type { DELTA, EXP, COBA_AMPA, COBA_GABA }

    public final int preNeuron;    // 前神经元索引
    public final int postNeuron;   // 后神经元索引
    public final Type type;
    public double weight;          // 突触权重 (nA 或 nS)
    public double tauSynMs;        // 突触时间常数
    public double delayMs;         // 传导延迟
    public boolean excitatory;     // 兴奋/抑制

    // 可塑性状态 (STDP)
    public double lastPreSpikeTime = -1e9;
    public double lastPostSpikeTime = -1e9;

    public Synapse(int pre, int post, Type type, double weight, double tauSynMs, double delayMs) {
        this.preNeuron = pre;
        this.postNeuron = post;
        this.type = type;
        this.weight = weight;
        this.tauSynMs = tauSynMs;
        this.delayMs = delayMs;
        this.excitatory = weight >= 0;
    }

    public static Synapse delta(int pre, int post, double weight) {
        return new Synapse(pre, post, Type.DELTA, weight, 0, 0);
    }

    public static Synapse exp(int pre, int post, double weight, double tauMs) {
        return new Synapse(pre, post, Type.EXP, weight, tauMs, 0);
    }

    public static Synapse coba(int pre, int post, double weight, double tauMs, boolean isGaba) {
        return new Synapse(pre, post, isGaba ? Type.COBA_GABA : Type.COBA_AMPA, weight, tauMs, 0);
    }
}
