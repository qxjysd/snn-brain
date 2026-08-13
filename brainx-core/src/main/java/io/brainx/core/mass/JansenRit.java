package io.brainx.core.mass;

/**
 * Jansen-Rit 神经群模型（皮层柱, α节律/EEG 生成）。
 * 标准方程 (Jansen & Rit 1995 / David & Friston 版):
 *   y0'' + 2a·y0' + a²·y0 = A·a·S(y1 - y2)              (锥体 ← 中间神经元)
 *   y1'' + 2a·y1' + a²·y1 = A·a·(P + C2·S(C1·y0))       (兴奋中间 ← 外部+锥体)
 *   y2'' + 2b·y2' + b²·y2 = B·b·(C4·S(C3·y0))           (抑制中间 ← 锥体)
 *   S(x) = 2e0 / (1 + exp(r·(v0 - x)))                    (sigmoid, 发放率 s⁻¹)
 * 时间单位秒，积分在 ms 域时 dt 需 /1000。
 */
public class JansenRit implements NeuralMass {
    // 标准参数
    private final double a = 100.0, b = 50.0;      // 群体时间常数倒数 (s⁻¹)
    private final double A = 3.25, B = 22.0;       // 突触增益 (mV)
    private final double v0 = 6.0, e0 = 2.5, r = 0.56;  // sigmoid 参数
    private final double c1 = 135.0, c2 = 108.0, c3 = 33.75, c4 = 33.75;  // 连接数
    private double p = 220.0;                       // 外部输入 (脉冲率 s⁻¹)

    // 状态: 三个群体的 PSP 及其导数 (mV, mV/s)
    private double y0, y1, y2;
    private double dy0, dy1, dy2;

    public JansenRit() { reset(); }
    public static JansenRit defaultParams() { return new JansenRit(); }
    public void setExternalInput(double p) { this.p = p; }

    @Override public void reset() {
        y0 = y1 = y2 = 0;
        dy0 = dy1 = dy2 = 0;
    }

    private double sigmoid(double v) {
        return 2.0 * e0 / (1.0 + Math.exp(r * (v0 - v)));
    }

    @Override
    public double step(double externalInput, double dtMs) {
        double dt = dtMs / 1000.0;  // 秒

        // 输入到各群体 (基底外部输入 p + 额外输入)
        double s0 = sigmoid(y1 - y2);
        double s1 = p + externalInput + c2 * sigmoid(c1 * y0);
        double s2 = c4 * sigmoid(c3 * y0);

        // 二阶 ODE: y'' = A·a·S - 2a·y' - a²·y
        double ddy0 = A * a * s0 - 2 * a * dy0 - a * a * y0;
        double ddy1 = A * a * s1 - 2 * a * dy1 - a * a * y1;
        double ddy2 = B * b * s2 - 2 * b * dy2 - b * b * y2;

        // 欧拉积分 (秒域)
        dy0 += ddy0 * dt;
        dy1 += ddy1 * dt;
        dy2 += ddy2 * dt;
        y0 += dy0 * dt;
        y1 += dy1 * dt;
        y2 += dy2 * dt;

        return y0;  // EEG = PYR 群体 PSP
    }

    @Override public double activity() { return y0; }
    @Override public int stateDim() { return 6; }
    @Override public double state(int i) {
        return switch (i) {
            case 0 -> y0; case 1 -> y1; case 2 -> y2;
            case 3 -> dy0; case 4 -> dy1; default -> dy2;
        };
    }
}
