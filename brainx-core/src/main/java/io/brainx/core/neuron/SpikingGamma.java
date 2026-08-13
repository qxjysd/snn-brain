package io.brainx.core.neuron;

import io.brainx.core.Neuron;

/**
 * SpikingGamma 神经元（SpikingGamma 论文, arXiv 2602.01978）。
 *
 * 核心机制：
 * 1. 自适应递归记忆 (Gamma 模型): 每个突触输入通过 K 个"桶"(bucket)，
 *    每个桶 κ^k 实现不同延迟的平滑滤波 —— 历史以延迟表示，无自递归。
 * 2. Sigma-Delta 脉冲编码: 内部信号 y 与脉冲触发不应累积 y_hat 之差
 *    超过阈值 theta 即发放；下游从脉冲串重建 y_hat。
 * 3. 前馈结构 → 不需要 BPTT，直接误差反向传播，无需替代梯度。
 *
 * 本实现：每神经元维护 K 个桶的延迟记忆，输入经加权桶滤波后
 * 与 sigma-delta 阈值比较决定发放。
 */
public class SpikingGamma implements Neuron {
    /** 桶数量（延迟分辨率） */
    private final int numBuckets;
    /** 桶时间常数 (ms) —— 指数平滑，越深的桶延迟越大 */
    private final double tauBucketMs;
    /** sigma-delta 阈值 */
    private final double theta;
    /** 可训练的桶权重（对外可访问，用于学习） */
    public final double[] bucketWeights;

    // 状态
    private final double[] buckets;   // 各桶的滤波值 y^k
    private double y;                 // 内部信号 = sum(w_k * y^k)
    private double yHat;              // 脉冲触发不应累积
    private boolean firedFlag;

    public SpikingGamma(int numBuckets, double tauBucketMs, double theta) {
        this.numBuckets = numBuckets;
        this.tauBucketMs = tauBucketMs;
        this.theta = theta;
        this.bucketWeights = new double[numBuckets];
        // 初始化: 权重指数衰减（近桶强，远桶弱）
        for (int k = 0; k < numBuckets; k++) {
            bucketWeights[k] = Math.exp(-k / (double) numBuckets * 2.0);
        }
        this.buckets = new double[numBuckets];
        reset();
    }

    /** 默认: 8 桶, tau=5ms, 阈值 0.5 */
    public static SpikingGamma defaultParams() { return new SpikingGamma(8, 5.0, 0.5); }

    @Override public void reset() {
        java.util.Arrays.fill(buckets, 0.0);
        y = 0; yHat = 0; firedFlag = false;
    }

    @Override
    public void step(double input, double dtMs) {
        // 1. 更新桶: 每个桶对输入做不同延迟的指数平滑
        //    y^k(t) = y^k(t-1) * alpha_k + input * (1 - alpha_k)
        //    不同 k 用不同时间常数 -> 不同延迟
        for (int k = 0; k < numBuckets; k++) {
            double tauK = tauBucketMs * (1.0 + k);  // 越深延迟越大
            double alphaK = Math.exp(-dtMs / tauK);
            buckets[k] = alphaK * buckets[k] + (1.0 - alphaK) * input;
        }
        // 2. 内部信号 = 桶加权和
        y = 0;
        for (int k = 0; k < numBuckets; k++) y += bucketWeights[k] * buckets[k];
        // 3. Sigma-Delta 编码: 近似误差超阈值即发放
        double err = y - yHat;
        firedFlag = (Math.abs(err) >= theta);
        if (firedFlag) {
            // 发放: 添加不应响应（近似 y），带符号
            yHat += Math.signum(err) * theta;
        }
    }

    /** 前馈误差反向传播：误差直接流回桶权重（无替代梯度） */
    public void backpropError(double error, double learningRate) {
        for (int k = 0; k < numBuckets; k++) {
            bucketWeights[k] += learningRate * error * buckets[k];
        }
    }

    @Override public boolean fired() { return firedFlag; }
    @Override public double membranePotential() { return y; }
    @Override public int size() { return 1; }
    @Override public int stateDim() { return numBuckets + 2; }  // buckets + y + yHat
    @Override public double state(int i) {
        if (i < numBuckets) return buckets[i];
        if (i == numBuckets) return y;
        return yHat;
    }
    @Override public double inputDerivative(int i) { return 1.0; }
}
