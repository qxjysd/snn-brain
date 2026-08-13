package io.brainx.core;

/**
 * 时序特征预测器 —— 感官前向模型 (人脑预测能力的核心)。
 *
 * 神经科学依据:
 *   - 大脑持续预测下一时刻的感官输入 (predictive coding, Rao & Ballard 1999;
 *     主动推理, Friston free energy)
 *   - 平滑追踪眼动 (smooth pursuit): 视觉延迟 ~100ms, 眼球靠预测补偿延迟
 *   - 小脑前向模型: 动作执行前预测感觉后果, 延迟期用预测填补
 *   - "感官信息获取慢 → 靠预测补偿" (第20篇 运动物理学)
 *
 * 实现: 逐特征维一阶运动外推 + 平滑速度 (防噪声放大)。
 *   - predictNext(current): 更新历史 + 预测下一帧
 *   - extrapolate(): 纯预测 (脑补期间连续外推, 不更新历史)
 * 算力 O(特征维), 与编码器同级。
 */
public class FeaturePredictor {
    private final int dim;
    private double[] prev;          // x_{t-1}
    private double[] velocity;      // 平滑速度 (EMA)
    private boolean initialized = false;

    /** 速度平滑系数 (0-1: 大=跟随快/噪声敏感, 小=平滑/滞后) */
    private static final double VEL_EMA = 0.5;

    public FeaturePredictor(int dim) {
        this.dim = Math.max(1, dim);
        this.velocity = new double[dim];
    }

    /**
     * 更新历史并预测下一帧 (正常感知路径: 看到真实帧 → 大脑预测下一步)。
     * @return 预测的下一帧特征 (0-1)
     */
    public double[] predictNext(double[] current) {
        if (current == null) return extrapolate();
        double[] pred = new double[dim];
        int n = Math.min(dim, current.length);
        if (!initialized) {
            prev = current.clone();
            initialized = true;
            System.arraycopy(current, 0, pred, 0, n);
            return pred;
        }
        for (int i = 0; i < n; i++) {
            double v = current[i] - prev[i];                 // 瞬时速度
            velocity[i] = velocity[i] * (1 - VEL_EMA) + v * VEL_EMA;   // 平滑
            pred[i] = clamp01(current[i] + velocity[i]);     // 运动外推
        }
        prev = current.clone();
        return pred;
    }

    /**
     * 纯外推预测 (脑补: 感官延迟期间预测当前, 不更新历史)。
     * @param steps 延迟帧数 (补几步; 延迟越大预测越不可靠)
     */
    public double[] extrapolate(int steps) {
        double[] pred = new double[dim];
        if (!initialized) return pred;
        int s = Math.max(1, steps);
        for (int i = 0; i < dim; i++) {
            pred[i] = clamp01(prev[i] + velocity[i] * s);
        }
        return pred;
    }

    /** 单步外推 (兼容) */
    public double[] extrapolate() {
        return extrapolate(1);
    }

    /** 预测置信度: 基于速度平滑度 (速度大且变化剧烈 → 预测不可靠) */
    public double confidence() {
        if (!initialized) return 0;
        double vMag = 0;
        for (double v : velocity) vMag += v * v;
        vMag = Math.sqrt(vMag / dim);
        // 速度小 → 场景静止/缓慢 → 预测可靠
        return clamp01(1.0 - vMag * 3.0);
    }

    /** 重置 (场景切换/镜头切换) */
    public void reset() {
        initialized = false;
        prev = null;
        java.util.Arrays.fill(velocity, 0);
    }

    private static double clamp01(double v) {
        return Math.max(0, Math.min(1, v));
    }
}
