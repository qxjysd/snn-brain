package io.brainx.core.neuron;

import io.brainx.core.Neuron;

/**
 * PLIF 神经元 (Parametric LIF) —— 可学习膜时间常数的 LIF。
 * 来源: STEP 论文 (arXiv 2505.11151) 系统性消融实证结论:
 *   "PLIF delivers the largest improvement — surpassing even architectural
 *    upgrades on Spikformer — yet it adds only one scalar parameter."
 *   (PLIF 增益最大, 超越架构升级, 但只增加一个标量参数)
 *
 * 与 LIF 唯一区别: tau_ms 是可训练参数, 学习规则:
 *   d(tau)/dL = -dV/dt * dV/dtau * grad... 简化在线更新:
 *   Δtau = lr * error * (dV/dt 方向)
 */
public class PLIF implements Neuron {
    private final double vRest, vThreshold, vReset, resistance;
    private double tauMs;          // 可学习参数 (初始=论文标准 10ms)
    private final double tauMin = 2.0, tauMax = 20.0;  // 生物学范围约束
    private final double learningRate;

    private double v;
    private double lastDv;         // 记录 dv/dt 供梯度学习
    private boolean firedFlag;

    public PLIF(double tauMs, double vRest, double vThreshold, double vReset,
                double resistance, double learningRate) {
        this.tauMs = tauMs;
        this.vRest = vRest;
        this.vThreshold = vThreshold;
        this.vReset = vReset;
        this.resistance = resistance;
        this.learningRate = learningRate;
        this.v = vRest;
    }

    public static PLIF defaultParams() {
        return new PLIF(10.0, -65.0, -50.0, -65.0, 1.0, 0.001);
    }

    @Override public void reset() { v = vRest; firedFlag = false; lastDv = 0; }

    @Override
    public void step(double inputCurrent, double dtMs) {
        // dv/dt = (-(v-vRest) + R*I) / tau
        lastDv = (-(v - vRest) + resistance * inputCurrent) / tauMs;
        v += lastDv * dtMs;
        firedFlag = (v >= vThreshold);
        if (firedFlag) v = vReset;
    }

    /**
     * PLIF 在线学习: 用误差信号更新 tau。
     * 梯度: ∂v/∂tau = -dv/dt (因 tau 在分母, 增大 tau 减慢响应)
     * 简化规则 (STEP 论文的启发): 误差驱动的 tau 调节,
     * 误差为正(需更强响应)→ 减小 tau (更快); 误差为负 → 增大 tau。
     */
    public void learn(double error) {
        double gradTau = -lastDv / (tauMs * tauMs);  // d(dv/dt)/dtau
        tauMs -= learningRate * error * gradTau;
        tauMs = Math.max(tauMin, Math.min(tauMax, tauMs));
    }

    public double tauMs() { return tauMs; }
    public void setTau(double t) { this.tauMs = Math.max(tauMin, Math.min(tauMax, t)); }

    @Override public boolean fired() { return firedFlag; }
    @Override public double membranePotential() { return v; }
    @Override public int size() { return 1; }
    @Override public int stateDim() { return 1; }
    @Override public double state(int i) { return v; }
    @Override public double inputDerivative(int i) { return resistance / tauMs; }
}
