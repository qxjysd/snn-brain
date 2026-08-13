package io.brainx.core.neuron;

import io.brainx.core.Neuron;

/**
 * Hodgkin-Huxley 神经元（1952，生物物理最精确的点神经元模型）。
 * 方程:
 *   Cm * dV/dt = I_ext - g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L)
 *   门控: dm/dt = alpha_m*(1-m) - beta_m*m 等
 * 对应 BrainFuse 论文的 HH 模型（噪声鲁棒性优于 LIF，BrainFuse 已实证）。
 */
public class HH implements Neuron {
    // 参数 (标准 HH 值)
    private final double cm = 1.0;        // uF/cm^2
    private final double gNa = 120.0, gK = 36.0, gL = 0.3;
    private final double eNa = 50.0, eK = -77.0, eL = -54.4;

    // 状态
    private double v;   // mV
    private double m, h, n;  // 门控变量 0-1
    private boolean firedFlag;

    public HH() { reset(); }

    public static HH defaultParams() { return new HH(); }

    @Override public void reset() {
        v = -65.0;
        m = 0.05; h = 0.6; n = 0.32;
        firedFlag = false;
    }

    // 速率函数
    private static double alphaM(double v) { return 0.1 * (v + 40.0) / (1.0 - Math.exp(-(v + 40.0) / 10.0)); }
    private static double betaM(double v)  { return 4.0 * Math.exp(-(v + 65.0) / 18.0); }
    private static double alphaH(double v) { return 0.07 * Math.exp(-(v + 65.0) / 20.0); }
    private static double betaH(double v)  { return 1.0 / (1.0 + Math.exp(-(v + 35.0) / 10.0)); }
    private static double alphaN(double v) { return 0.01 * (v + 55.0) / (1.0 - Math.exp(-(v + 55.0) / 10.0)); }
    private static double betaN(double v)  { return 0.125 * Math.exp(-(v + 65.0) / 80.0); }

    @Override
    public void step(double inputCurrent, double dtMs) {
        // 电流密度转换: 输入 nA → uA/cm^2 (假设 1 cm^2 膜面积，简化)
        double iExt = inputCurrent * 10.0;

        // 膜电位
        double iIon = gNa * m * m * m * h * (v - eNa) + gK * n * n * n * n * (v - eK) + gL * (v - eL);
        double dv = (iExt - iIon) / cm;
        v += dv * dtMs * 0.01;  // 时间尺度校正 (dt 单位)

        // 门控 (欧拉)
        m += (alphaM(v) * (1 - m) - betaM(v) * m) * dtMs * 0.01;
        h += (alphaH(v) * (1 - h) - betaH(v) * h) * dtMs * 0.01;
        n += (alphaN(v) * (1 - n) - betaN(v) * n) * dtMs * 0.01;

        firedFlag = (v > 0.0);  // 动作电位过零
    }

    @Override public boolean fired() { return firedFlag; }
    @Override public double membranePotential() { return v; }
    @Override public int size() { return 1; }
    @Override public int stateDim() { return 4; }  // v, m, h, n
    @Override public double state(int i) { return i == 0 ? v : i == 1 ? m : i == 2 ? h : n; }
    @Override public double inputDerivative(int i) { return i == 0 ? 10.0 / cm : 0.0; }
}
