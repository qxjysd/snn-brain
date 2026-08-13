package io.brainx.core.neuron;

import io.brainx.core.Neuron;

/**
 * Leaky Integrate-and-Fire 神经元 (LIF)。
 * 方程: tau_m * dV/dt = -(V - V_rest) + R * I
 * 发放条件: V >= V_threshold → 发放 + 重置到 V_reset
 * 对应 brain.state 的 LIF 模型（brain eLife 2023 基础模型）。
 */
public class LIF implements Neuron {
    // 参数
    private final double tauMs;       // 膜时间常数 (ms)
    private final double vRest;       // 静息电位 (mV)
    private final double vThreshold;  // 阈值 (mV)
    private final double vReset;      // 重置电位 (mV)
    private final double resistance;  // 膜电阻 (GOhm = nA/mV)

    // 状态
    private double v;
    private boolean firedFlag;

    public LIF(double tauMs, double vRest, double vThreshold, double vReset, double resistance) {
        this.tauMs = tauMs;
        this.vRest = vRest;
        this.vThreshold = vThreshold;
        this.vReset = vReset;
        this.resistance = resistance;
        this.v = vRest;
    }

    public static LIF defaultParams() {
        return new LIF(10.0, -65.0, -50.0, -65.0, 1.0);
    }

    @Override public void reset() { v = vRest; firedFlag = false; }

    @Override
    public void step(double inputCurrent, double dtMs) {
        // 欧拉积分: dv/dt = -(v - vRest)/tau + R*I/tau
        double dv = (-(v - vRest) + resistance * inputCurrent) / tauMs;
        v += dv * dtMs;
        firedFlag = (v >= vThreshold);
        if (firedFlag) v = vReset;
    }

    @Override public boolean fired() { return firedFlag; }
    @Override public double membranePotential() { return v; }
    @Override public int size() { return 1; }
    @Override public int stateDim() { return 1; }
    @Override public double state(int i) { return v; }
    @Override public double inputDerivative(int i) { return resistance / tauMs; } // dV/dI = R/tau
}
