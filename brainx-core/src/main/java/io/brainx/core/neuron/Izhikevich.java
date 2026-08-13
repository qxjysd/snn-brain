package io.brainx.core.neuron;

import io.brainx.core.Neuron;

/**
 * Izhikevich 神经元（2003，生物学丰富的脉冲模式）。
 * 方程:
 *   dv/dt = 0.04*v^2 + 5*v + 140 - u + I
 *   du/dt = a*(b*v - u)
 *   发放时: v = c; u = u + d
 * 参数组: 规则发放 (RS), 快速发放 (FS), 丘脑皮层 (TC), 共振 (RZ) 等。
 * 对应 brain.state 的 Izhikevich 模型。
 */
public class Izhikevich implements Neuron {
    // 参数
    private final double a, b, c, d;
    // 状态
    private double v, u;
    private boolean firedFlag;

    public Izhikevich(double a, double b, double c, double d) {
        this.a = a; this.b = b; this.c = c; this.d = d;
        this.v = c; this.u = b * c;
    }

    /** 规则发放 (regular spiking) 皮层锥体细胞 */
    public static Izhikevich regularSpiking() { return new Izhikevich(0.02, 0.2, -65, 8); }
    /** 快速发放 (fast spiking) 抑制性中间神经元 */
    public static Izhikevich fastSpiking() { return new Izhikevich(0.1, 0.2, -65, 2); }
    /** 内在爆发 (intrinsically bursting) */
    public static Izhikevich intrinsicallyBursting() { return new Izhikevich(0.02, 0.2, -55, 4); }
    /** 低阈值发放 (low-threshold spiking) */
    public static Izhikevich lowThresholdSpiking() { return new Izhikevich(0.02, 0.25, -65, 2); }

    @Override public void reset() { v = c; u = b * c; firedFlag = false; }

    @Override
    public void step(double inputCurrent, double dtMs) {
        // 注意: Izhikevich 原始方程用 ms 和 mV，I 单位换算成与 v^2 项兼容（10倍因子）
        double dv = 0.04 * v * v + 5.0 * v + 140.0 - u + inputCurrent * 10.0;
        double du = a * (b * v - u);
        v += dv * dtMs;
        u += du * dtMs;
        firedFlag = (v >= 30.0);
        if (firedFlag) {
            v = c;
            u = u + d;
        }
    }

    @Override public boolean fired() { return firedFlag; }
    @Override public double membranePotential() { return v; }
    @Override public int size() { return 1; }
    @Override public int stateDim() { return 2; }
    @Override public double state(int i) { return i == 0 ? v : u; }
    @Override public double inputDerivative(int i) { return i == 0 ? 10.0 : 0.0; }
}
