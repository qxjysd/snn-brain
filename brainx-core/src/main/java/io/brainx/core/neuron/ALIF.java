package io.brainx.core.neuron;

import io.brainx.core.Neuron;

/**
 * Adaptive LIF 神经元 (ALIF)，带慢适应电流。
 * 方程:
 *   tau_m * dV/dt = -(V - V_rest) + R*I - R*a
 *   tau_a * da/dt = -a
 *   发放时: a += delta_a（适应增强）
 * 对应 BrainTrace 论文 (NC 2026) 中实验用的 ALIF 神经元，
 * 以及 brain.state 的 ALIF 模型。
 */
public class ALIF implements Neuron {
    private final double tauMs, vRest, vThreshold, vReset, resistance;
    private final double tauAdaptMs;   // 适应时间常数
    private final double deltaAdapt;   // 每次发放的适应增量

    private double v;
    private double a;          // 适应变量
    private boolean firedFlag;

    public ALIF(double tauMs, double vRest, double vThreshold, double vReset, double resistance,
                double tauAdaptMs, double deltaAdapt) {
        this.tauMs = tauMs;
        this.vRest = vRest;
        this.vThreshold = vThreshold;
        this.vReset = vReset;
        this.resistance = resistance;
        this.tauAdaptMs = tauAdaptMs;
        this.deltaAdapt = deltaAdapt;
        this.v = vRest;
        this.a = 0;
    }

    public static ALIF defaultParams() {
        return new ALIF(10.0, -65.0, -50.0, -65.0, 0.1, 100.0, 0.5);
    }

    @Override public void reset() { v = vRest; a = 0; firedFlag = false; }

    @Override
    public void step(double inputCurrent, double dtMs) {
        // 电压: dv/dt = (-(v-vRest) + R*I - R*a)/tau
        double dv = (-(v - vRest) + resistance * inputCurrent - resistance * a) / tauMs;
        v += dv * dtMs;
        // 适应: da/dt = -a/tau_a
        a += (-a / tauAdaptMs) * dtMs;
        firedFlag = (v >= vThreshold);
        if (firedFlag) {
            v = vReset;
            a += deltaAdapt;
        }
    }

    @Override public boolean fired() { return firedFlag; }
    @Override public double membranePotential() { return v; }
    @Override public int size() { return 1; }
    @Override public int stateDim() { return 2; }  // v, a
    @Override public double state(int i) { return i == 0 ? v : a; }
    @Override public double inputDerivative(int i) { return i == 0 ? resistance / tauMs : 0.0; }
}
