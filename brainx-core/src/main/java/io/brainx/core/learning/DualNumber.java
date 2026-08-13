package io.brainx.core.learning;

import java.util.Arrays;

/**
 * DualNumber — 微型自动微分 (前向模式, 对应 brain ICLR 2024 可微仿真路线)。
 *
 * 每个数携带 值 + 梯度向量 (对所有变量的偏导), 运算时同步传播梯度 (链式法则)。
 * 一次前向同时得到函数值和全部梯度 — 与 brain 的 JAX 可微核心同原理 (Java 实现)。
 *
 * 用途:
 *   - 可微仿真: 对任意脑动力学表达式求真梯度 (替代手工推导近似)
 *   - 梯度检查: 验证 pp-prop 等解析梯度算法的正确性 (数值梯度对照)
 *   - 可微训练: 速率/脉冲网络用梯度下降端到端学习
 */
public final class DualNumber {
    public final double value;
    public final double[] grad;      // 对每个变量的偏导 (长度 = 变量数; 常数可空)

    public DualNumber(double v) {
        this.value = v;
        this.grad = new double[0];
    }

    public DualNumber(double v, double[] g) {
        this.value = v;
        this.grad = g;
    }

    /** 创建变量 (第 idx 个变量, 共 n 个) */
    public static DualNumber var(double v, int idx, int n) {
        double[] g = new double[n];
        g[idx] = 1.0;
        return new DualNumber(v, g);
    }

    public boolean isConst() { return grad.length == 0; }

    private double[] addGrad(double[] a, double[] b) {
        if (a.length == 0) return b;
        if (b.length == 0) return a;
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
        return out;
    }

    private double[] scaleGrad(double[] g, double s) {
        if (g.length == 0) return g;
        double[] out = new double[g.length];
        for (int i = 0; i < g.length; i++) out[i] = g[i] * s;
        return out;
    }

    public DualNumber add(DualNumber o) {
        return new DualNumber(value + o.value, addGrad(grad, o.grad));
    }

    public DualNumber sub(DualNumber o) {
        return new DualNumber(value - o.value, addGrad(grad, scaleGrad(o.grad, -1)));
    }

    public DualNumber mul(DualNumber o) {
        // (uv)' = u'v + uv'
        double[] g;
        if (grad.length == 0 && o.grad.length == 0) g = new double[0];
        else if (grad.length == 0) g = scaleGrad(o.grad, value);
        else if (o.grad.length == 0) g = scaleGrad(grad, o.value);
        else {
            g = new double[grad.length];
            for (int i = 0; i < grad.length; i++) g[i] = grad[i] * o.value + value * o.grad[i];
        }
        return new DualNumber(value * o.value, g);
    }

    public DualNumber div(DualNumber o) {
        // (u/v)' = (u'v - uv')/v²
        double v2 = o.value * o.value;
        double[] g;
        if (grad.length == 0 && o.grad.length == 0) g = new double[0];
        else if (o.grad.length == 0) g = scaleGrad(grad, 1.0 / o.value);
        else {
            g = new double[Math.max(grad.length, o.grad.length)];
            double u = value, v = o.value;
            double[] ug = grad.length == 0 ? new double[g.length] : grad;
            double[] vg = o.grad.length == 0 ? new double[g.length] : o.grad;
            for (int i = 0; i < g.length; i++) g[i] = (ug[i] * v - u * vg[i]) / v2;
        }
        return new DualNumber(value / o.value, g);
    }

    public DualNumber sin() {
        return new DualNumber(Math.sin(value), scaleGrad(grad, Math.cos(value)));
    }

    public DualNumber cos() {
        return new DualNumber(Math.cos(value), scaleGrad(grad, -Math.sin(value)));
    }

    public DualNumber exp() {
        double e = Math.exp(value);
        return new DualNumber(e, scaleGrad(grad, e));
    }

    public DualNumber tanh() {
        double t = Math.tanh(value);
        return new DualNumber(t, scaleGrad(grad, 1 - t * t));
    }

    /** sigmoid 激活 (神经元发放率近似) */
    public DualNumber sigmoid() {
        double s = 1.0 / (1.0 + Math.exp(-value));
        return new DualNumber(s, scaleGrad(grad, s * (1 - s)));
    }

    public DualNumber pow(double p) {
        return new DualNumber(Math.pow(value, p), scaleGrad(grad, p * Math.pow(value, p - 1)));
    }

    @Override
    public String toString() {
        return "Dual(v=" + value + ", g=" + Arrays.toString(grad) + ")";
    }
}
