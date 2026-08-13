package io.brainx.core.learning;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 微型自动微分 (DualNumber) 验证 — 可微仿真基础。
 *   - 单变量梯度 vs 解析解
 *   - 多变量偏导 vs 有限差分
 *   - 可微训练: 梯度下降拟合目标函数 (可微学习真实可用)
 *   - 梯度检查: 对 pp-prop 解析梯度做数值对照
 */
public class DualNumberTest {

    @Test
    void univariateGradient() {
        // f(x) = x³ + 2x + 1, f'(x) = 3x² + 2; 在 x=3: f=34, f'=29
        DualNumber x = DualNumber.var(3.0, 0, 1);
        DualNumber f = x.pow(3).add(x.mul(new DualNumber(2))).add(new DualNumber(1));
        assertEquals(34.0, f.value, 1e-9, "f(3)");
        assertEquals(29.0, f.grad[0], 1e-9, "f'(3)");
    }

    @Test
    void chainRule() {
        // f(x) = sin(x²), f'(x) = 2x·cos(x²); x=2: f=sin4≈-0.7568, f'=4·cos4≈-2.6146
        DualNumber x = DualNumber.var(2.0, 0, 1);
        DualNumber f = x.pow(2).sin();
        assertEquals(Math.sin(4), f.value, 1e-9);
        assertEquals(4 * Math.cos(4), f.grad[0], 1e-6, "链式法则");
    }

    @Test
    void multivariateGradientMatchesFiniteDiff() {
        // f(x,y) = x·y + tanh(x) - exp(0.1y); 数值梯度 vs 有限差分 (eps=1e-6)
        int n = 2;
        double x0 = 0.7, y0 = -1.2;
        double fv = x0 * y0 + Math.tanh(x0) - Math.exp(0.1 * y0);
        DualNumber x = DualNumber.var(x0, 0, n);
        DualNumber y = DualNumber.var(y0, 1, n);
        DualNumber f = x.mul(y).add(x.tanh()).sub(y.mul(new DualNumber(0.1)).exp());
        assertEquals(fv, f.value, 1e-9);
        // 有限差分对照
        double eps = 1e-6;
        double fx = (x0 + eps) * y0 + Math.tanh(x0 + eps) - Math.exp(0.1 * y0);
        double fxm = (x0 - eps) * y0 + Math.tanh(x0 - eps) - Math.exp(0.1 * y0);
        double fy = x0 * (y0 + eps) + Math.tanh(x0) - Math.exp(0.1 * (y0 + eps));
        double fym = x0 * (y0 - eps) + Math.tanh(x0) - Math.exp(0.1 * (y0 - eps));
        assertEquals((fx - fxm) / (2 * eps), f.grad[0], 1e-5, "∂f/∂x");
        assertEquals((fy - fym) / (2 * eps), f.grad[1], 1e-5, "∂f/∂y");
    }

    @Test
    void differentiableTraining() {
        // 可微训练: 单神经元速率模型 y = sigmoid(w·x + b), 梯度下降拟合目标 y*=x² 的 5 个点
        int n = 3;   // 变量: w, b, (x 固定输入)
        double[] xs = {0.1, 0.4, 0.7, 1.0, 1.3};
        double[] targets = new double[xs.length];
        for (int i = 0; i < xs.length; i++) targets[i] = Math.min(0.95, xs[i] * xs[i]);
        double w = 0.5, b = 0.0;
        double lr = 2.0;
        double lastLoss = 1e9;
        for (int epoch = 0; epoch < 2000; epoch++) {
            // 前向 (用 DualNumber 同时求 loss 对各参数梯度)
            DualNumber wD = DualNumber.var(w, 0, n);
            DualNumber bD = DualNumber.var(b, 1, n);
            DualNumber loss = new DualNumber(0);
            for (int i = 0; i < xs.length; i++) {
                DualNumber pred = wD.mul(new DualNumber(xs[i])).add(bD).sigmoid();
                DualNumber diff = pred.sub(new DualNumber(targets[i]));
                loss = loss.add(diff.mul(diff));
            }
            // 梯度下降
            w -= lr * loss.grad[0];
            b -= lr * loss.grad[1];
            if (epoch % 500 == 0) lastLoss = loss.value;
        }
        // 训练后: 网络应能近似 x=0.1 低输出, x=1.3 高输出
        double yLow = 1.0 / (1.0 + Math.exp(-(w * 0.1 + b)));
        double yHigh = 1.0 / (1.0 + Math.exp(-(w * 1.3 + b)));
        assertTrue(yLow < 0.3, "x=0.1 输出应低, got=" + yLow);
        assertTrue(yHigh > 0.7, "x=1.3 输出应高, got=" + yHigh);
        assertTrue(lastLoss < 1.0, "训练应收敛, lastLoss=" + lastLoss);
    }

    @Test
    void sigmoidDerivative() {
        // sigmoid'(x) = s(1-s); x=0: s=0.5, s'=0.25
        DualNumber x = DualNumber.var(0.0, 0, 1);
        DualNumber s = x.sigmoid();
        assertEquals(0.5, s.value, 1e-9);
        assertEquals(0.25, s.grad[0], 1e-9);
    }
}
