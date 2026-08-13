package io.brainx.core.mass;

/**
 * Wong-Wang 决策模型（双稳态吸引子网络）。
 * 来源: brain 决策分析教程 + BrainMass (WongWangStep)，
 * 论文: Wong & Wang 2006, "A recurrent network mechanism for time
 * integration in perceptual decisions".
 *
 * 两个兴奋性群体竞争（通过共享抑制池），用于感知决策。
 *   r_i = F(I_i) = (a*I_i - b) / (1 - exp(-d*(a*I_i - b)))
 *   dS_i/dt = -S_i/tau_s + (1-S_i)*gamma*F(I_i)   [F 单位 Hz → 秒域]
 *
 * 注意: F-I 曲线输出 Hz，积分在毫秒域需 /1000（否则一步饱和）。
 */
public class WongWang implements NeuralMass {
    // Wong-Wang 参数 (brain 教程)
    private final double gamma = 0.641;
    private final double tau = 100.0;     // NMDA 时间常数 (ms)
    private final double a = 270.0, b = 108.0, d = 0.154;
    private final double i0 = 0.3255;     // 背景电流 (nA)
    private final double je = 0.2609;     // 自耦合
    private final double ji = -0.0497;    // 交叉耦合 (抑制)
    private final double jaExt = 0.00052; // 外部刺激强度

    // 状态: 两个群体的突触门控变量 S1, S2
    private double s1 = 0.0, s2 = 0.0;

    public WongWang() { reset(); }
    public static WongWang defaultParams() { return new WongWang(); }

    private double f(double current) {
        double x = a * current - b;
        if (x < -50) return 0;
        // 数学上 x/(1-e^{-dx}) 在 x→0 趋于 1/d，负 x 给小的正值（F-I 曲线尾部）
        return x / (1.0 - Math.exp(-d * x));
    }

    /**
     * @param externalInput 相干性 c' (-1..1)，两群体刺激强度相反
     * @return 群体1发放率 (Hz)
     */
    @Override
    public double step(double coherence, double dtMs) {
        double mu = 20.0;  // 刺激强度
        double i1 = je * s1 + ji * s2 + i0 + jaExt * mu * (1.0 + coherence);
        double i2 = je * s2 + ji * s1 + i0 + jaExt * mu * (1.0 - coherence);
        double r1 = f(i1);
        double r2 = f(i2);
        // dS/dt (ms 域) = -S/tau + (1-S)*gamma*F/1000  [F 是 Hz]
        s1 += (-s1 / tau + (1 - s1) * gamma * r1 / 1000.0) * dtMs;
        s2 += (-s2 / tau + (1 - s2) * gamma * r2 / 1000.0) * dtMs;
        if (s1 < 0) s1 = 0;
        if (s2 < 0) s2 = 0;
        return r1;
    }

    /** 决策结果: 1 或 2 (谁占优) */
    public int decision() { return s1 >= s2 ? 1 : 2; }
    public double s1() { return s1; }
    public double s2() { return s2; }

    @Override public void reset() { s1 = s2 = 0; }
    @Override public double activity() { return s1 - s2; }
    @Override public int stateDim() { return 2; }
    @Override public double state(int i) { return i == 0 ? s1 : s2; }
}
