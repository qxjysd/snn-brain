package io.brainx.core;

import io.brainx.core.neuron.LIF;

/**
 * 中枢脉冲网络 (Central Hub SNN) —— 全模块统一脉冲联动。
 *
 * 神经科学依据 (皮层-丘脑环路 / 全局工作空间 GWT):
 *   - 真实大脑中, 皮层各区域投射到丘脑, 丘脑整合后广播回皮层
 *     (thalamo-cortical loop) —— 这是全局信息整合的神经基础
 *   - 所有模块 (感觉/记忆/认知) 通过这个中枢环路统一联动,
 *     而非各自为政: 模块状态→脉冲→中枢整合→广播→调制各模块
 *
 * 实现:
 *   - 中枢: 一组 LIF 脉冲神经元 (枢纽)
 *   - 输入: 各模块发射的脉冲率向量 → 权重矩阵 → 中枢神经元电流
 *   - 整合: 中枢 LIF 发放动力学 (汇聚+阈值+发放)
 *   - 输出: 中枢发放率 → 广播回各模块 (调制强度/置信度)
 *   - 这就是"各模块通过大脑中枢脉冲网络统一联动"
 */
public class CentralHub {
    /** 中枢神经元数量 (枢纽规模) */
    private final int hubSize;
    /** 输入维度 (各模块脉冲率拼接) */
    private final int inputDim;
    /** 中枢 LIF 神经元 */
    private final LIF[] hubNeurons;
    /** 输入权重矩阵 [hub][input] */
    private final double[][] inputWeights;
    /** 中枢当前发放状态 */
    private final boolean[] firing;
    /** 中枢发放率 (每个神经元滑动平均, 广播用) */
    private final double[] firingRate;
    /** 输入缓冲 (每步从模块收集) */
    private final double[] inputBuffer;
    /** 时间步 */
    private int timeStep = 0;
    /** 中枢总发放数 (活动水平) */
    private int totalSpikes = 0;
    /** 中枢活跃度 (0-1: 发放率归一) */
    private double activity = 0;

    /**
     * @param hubSize   中枢神经元数
     * @param inputDim  输入维度 (模块脉冲数)
     * @param seed      随机种子 (权重初始化)
     */
    public CentralHub(int hubSize, int inputDim, long seed) {
        this.hubSize = hubSize;
        this.inputDim = inputDim;
        this.hubNeurons = new LIF[hubSize];
        this.firing = new boolean[hubSize];
        this.firingRate = new double[hubSize];
        this.inputBuffer = new double[inputDim];
        this.inputWeights = new double[hubSize][inputDim];
        java.util.Random rnd = new java.util.Random(seed);
        for (int i = 0; i < hubSize; i++) {
            hubNeurons[i] = LIF.defaultParams();
            for (int j = 0; j < inputDim; j++) {
                // 【神经发育】初始全连接 (突触过度产生):
                // 婴儿大脑一开始是过度连接, 学习后修剪变稀疏 (书中: 突触形成后随即被破坏)
                inputWeights[i][j] = 0.15 + rnd.nextDouble() * 0.35;  // 全连接小权重
            }
        }
    }

    /** 中枢神经元数 (提升5倍: 24→120, 全局脉冲网络规模扩大) */
    public static final int HUB_SIZE = 120;

    public static CentralHub defaultParams(int inputDim) {
        return new CentralHub(HUB_SIZE, inputDim, 42);
    }

    /** 清除输入缓冲 (每步开始) */
    public void clearInput() {
        java.util.Arrays.fill(inputBuffer, 0);
    }

    /** 注入模块脉冲率 (模块发射 → 中枢) */
    public void inject(int offset, double[] pulseRates) {
        for (int i = 0; i < pulseRates.length && offset + i < inputDim; i++) {
            inputBuffer[offset + i] += pulseRates[i];
        }
    }

    /** 注入单值脉冲率 */
    public void inject(int offset, double rate) {
        if (offset < inputDim) inputBuffer[offset] += rate;
    }

    /**
     * 中枢整合一步: 输入→电流→LIF发放→发放率更新。
     * @param dtMs 时间步 (ms)
     */
    public void step(double dtMs) {
        timeStep++;
        for (int i = 0; i < hubSize; i++) {
            // 加权输入 → 中枢神经元电流
            double current = 0;
            for (int j = 0; j < inputDim; j++) {
                current += inputWeights[i][j] * inputBuffer[j];
            }
            hubNeurons[i].step(current * 40.0 + 3.0, dtMs);  // 基础偏置+输入
            firing[i] = hubNeurons[i].fired();
            if (firing[i]) {
                totalSpikes++;
                firingRate[i] = Math.min(1.0, firingRate[i] + 0.3);
            } else {
                firingRate[i] *= 0.95;  // 衰减
            }
        }
        // 中枢活跃度 (归一)
        double sum = 0;
        for (double r : firingRate) sum += r;
        activity = sum / hubSize;
    }

    /** 中枢发放率向量 (广播回各模块) */
    public double[] broadcastRates() { return firingRate.clone(); }

    /**
     * 中枢突触可塑性 (Hebbian: 可增可减, 书中神经发育)。
     *   - 增强: 模块脉冲强 + 中枢神经元发放 → 该连接增强 (fire together)
     *   - 衰减: 长期无活动 → 权重指数衰减 (用进废退)
     *   - 修剪: 权重 < 阈值 → 置零 (突触修剪, 出生后过度连接被削减)
     */
    public void learnWeights(double dtMs) {
        for (int i = 0; i < hubSize; i++) {
            boolean hubFired = firing[i];
            for (int j = 0; j < inputDim; j++) {
                double w = inputWeights[i][j];
                if (w <= 0) continue;  // 已修剪
                if (hubFired && inputBuffer[j] > 0.2) {
                    // Hebbian 增强: 输入脉冲 + 中枢发放 → 连接强化 (fire together, wire together)
                    inputWeights[i][j] = Math.min(2.0, w + 0.02 * inputBuffer[j]);
                } else {
                    // 用进废退: 无共激活 → 衰减 (τ=200ms 快速修剪, 模拟发育期突触修剪)
                    inputWeights[i][j] = w * Math.exp(-dtMs / 200.0);
                }
                // 突触修剪: 弱连接删除 (出生后过度连接被削减)
                if (inputWeights[i][j] < 0.05) {
                    inputWeights[i][j] = 0;
                }
            }
        }
    }

    /** 连接密度 (0-1): 非零连接比例 —— 初始1.0, 学习后下降 (发育曲线) */
    public double connectionDensity() {
        int active = 0;
        for (int i = 0; i < hubSize; i++) {
            for (int j = 0; j < inputDim; j++) {
                if (inputWeights[i][j] > 0) active++;
            }
        }
        return (double) active / (hubSize * inputDim);
    }

    /** 有效连接数 (非零) */
    public int activeConnections() {
        int active = 0;
        for (int i = 0; i < hubSize; i++) {
            for (int j = 0; j < inputDim; j++) {
                if (inputWeights[i][j] > 0) active++;
            }
        }
        return active;
    }

    /** 总连接数 */
    public int totalConnections() { return hubSize * inputDim; }

    /** 克隆权重矩阵 (测试/导出用) */
    public double[][] cloneWeights() {
        double[][] out = new double[hubSize][];
        for (int i = 0; i < hubSize; i++) out[i] = inputWeights[i].clone();
        return out;
    }

    /** 设置权重 (模型恢复) */
    public void setWeights(double[][] w) {
        for (int i = 0; i < Math.min(hubSize, w.length); i++) {
            for (int j = 0; j < Math.min(inputDim, w[i].length); j++) {
                inputWeights[i][j] = w[i][j];
            }
        }
    }

    /** 中枢发放状态 (可视化) */
    public boolean[] firingState() { return firing.clone(); }

    /** 中枢活跃度 (0-1: 全局整合水平) */
    public double activity() { return activity; }

    /** 中枢总发放 */
    public int totalSpikes() { return totalSpikes; }

    public int hubSize() { return hubSize; }
    public int inputDim() { return inputDim; }
    public int timeStep() { return timeStep; }

    /** 重置 */
    public void reset() {
        for (LIF n : hubNeurons) n.reset();
        java.util.Arrays.fill(firing, false);
        java.util.Arrays.fill(firingRate, 0);
        totalSpikes = 0;
        activity = 0;
        timeStep = 0;
    }

    /** 摘要 (APK 显示) */
    public String summary() {
        return String.format("🔄 中枢脉冲网络: %d神经元 | 活跃度%.0f%% | 连接密度%.0f%%(%d/%d) | 总发放%d",
                hubSize, activity * 100, connectionDensity() * 100,
                activeConnections(), totalConnections(), totalSpikes);
    }
}
