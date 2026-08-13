package io.brainx.core.mass;

/**
 * Wilson-Cowan 神经群模型（兴奋-抑制双群体）。
 * 方程:
 *   tau_E * dE/dt = -E + S(w_EE*E - w_IE*I + P)
 *   tau_I * dI/dt = -I + S(w_EI*E - w_II*I + Q)
 *   S(x) = 1/(1+exp(-x))  (sigmoid 激活)
 * 对应 BrainMass 的 WilsonCowanStep 模型。
 */
public class WilsonCowan implements NeuralMass {
    private final double tauE = 8.0, tauI = 8.0;      // 时间常数 (ms)
    private final double wEE = 12.0, wEI = 4.0, wIE = 13.0, wII = 11.0;  // 耦合
    private final double theta = 1.0;   // sigmoid 陡度
    private double e = 0.1, i = 0.1;    // 活动水平

    public WilsonCowan() { reset(); }
    public static WilsonCowan defaultParams() { return new WilsonCowan(); }

    private double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-theta * x)); }

    @Override
    public double step(double externalInput, double dtMs) {
        double de = (-e + sigmoid(wEE * e - wIE * i + externalInput)) / tauE;
        double di = (-i + sigmoid(wEI * e - wII * i)) / tauI;
        e += de * dtMs;
        i += di * dtMs;
        if (e < 0) e = 0;
        if (i < 0) i = 0;
        return e;
    }

    @Override public void reset() { e = i = 0.1; }
    @Override public double activity() { return e; }
    @Override public int stateDim() { return 2; }
    @Override public double state(int i) { return i == 0 ? e : this.i; }
}
