package io.brainx.core.learning;

/**
 * pp-prop (pre-post eligibility trace propagation) 在线学习算法。
 * 来源: BrainTrace 论文 (Nature Comm 2026, Eq.6-8)。
 *
 * 资格迹分解:
 *   ε^t = ε_f ⊙ ε_x              (Eq.6, 前迹×后迹)
 *   ε_x^t = α·ε_x^(t-1) + x^t    (Eq.7, 输入迹, O(I))
 *   ε_f^t = α·diag(D^t)⊙ε_f^(t-1) + (1-α)·diag(D_f^t)  (Eq.8, 神经元迹, O(H))
 *
 * 权重更新: Δθ = Σ_t (∂L^t/∂h^t) ⊙ ε^t   (Eq.2)
 * 内存复杂度 O(H) —— 线性内存在线学习。
 */
public class PPProp {
    /** 平滑因子 α (0<α<1), τ = -Δt/ln(α) */
    private final double alpha;
    private final double learningRate;

    /** 前迹 ε_x (输入侧, 维度=输入数) */
    private final double[] epsX;
    /** 后迹 ε_f (神经元侧, 维度=隐藏神经元数) */
    private final double[] epsF;

    /** 网络权重矩阵 (post x pre) 可训练 */
    private final double[][] weights;
    /** 权重梯度累积 */
    private final double[][] gradAccum;

    /**
     * @param numInput   输入神经元数
     * @param numHidden  隐藏神经元数
     * @param alpha      平滑因子 (0.9-0.99)
     * @param learningRate 学习率
     */
    public PPProp(int numInput, int numHidden, double alpha, double learningRate) {
        this.alpha = alpha;
        this.learningRate = learningRate;
        this.epsX = new double[numInput];
        this.epsF = new double[numHidden];
        this.weights = new double[numHidden][numInput];
        this.gradAccum = new double[numHidden][numInput];
        // 初始化权重 (小随机)
        java.util.Random rnd = new java.util.Random(42);
        for (int i = 0; i < numHidden; i++)
            for (int j = 0; j < numInput; j++)
                weights[i][j] = (rnd.nextDouble() - 0.5) * 0.4;
    }

    public static PPProp withDefaultAlpha(int numInput, int numHidden, double lr) {
        return new PPProp(numInput, numHidden, 0.95, lr);
    }

    /**
     * 前向一步（在 SNN 每个时间步后调用）。
     * @param inputSpikes 输入脉冲 (0/1, 长度 numInput)
     * @param hiddenStates 隐藏神经元状态向量
     * @param hiddenInputDeriv 隐藏神经元 inputDerivative (diag(D_f))
     * @param hiddenJacobianDiag 隐藏神经元内在雅可比对角 (diag(D^t))，简化取 0.99
     */
    public void step(double[] inputSpikes, double[] hiddenStates,
                     double[] hiddenInputDeriv, double hiddenJacobianDiag) {
        // Eq.7: ε_x^t = α·ε_x^(t-1) + x^t
        for (int j = 0; j < epsX.length; j++) {
            epsX[j] = alpha * epsX[j] + inputSpikes[j];
        }
        // Eq.8: ε_f^t = α·diag(D)⊙ε_f^(t-1) + (1-α)·diag(D_f)
        for (int i = 0; i < epsF.length; i++) {
            epsF[i] = alpha * hiddenJacobianDiag * epsF[i] + (1.0 - alpha) * hiddenInputDeriv[i];
        }
    }

    /**
     * 学习信号到达时更新权重 (Eq.2: Δθ = Σ (∂L/∂h) ⊙ ε)。
     * @param learningSignal 每隐藏神经元的 ∂L^t/∂h^t
     */
    public void update(double[] learningSignal) {
        for (int i = 0; i < epsF.length; i++) {
            for (int j = 0; j < epsX.length; j++) {
                double eligibility = epsF[i] * epsX[j];  // ε = ε_f ⊗ ε_x
                gradAccum[i][j] += learningSignal[i] * eligibility;
            }
        }
    }

    /** 每个序列/样本结束时应用梯度并清零（带梯度裁剪防爆炸） */
    public void applyGradients() {
        // 梯度裁剪: 限制最大更新幅度
        final double maxGrad = 1.0;
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[i].length; j++) {
                double g = Math.max(-maxGrad, Math.min(maxGrad, gradAccum[i][j]));
                weights[i][j] += learningRate * g;
                gradAccum[i][j] = 0;
            }
    }

    public void resetTraces() {
        java.util.Arrays.fill(epsX, 0);
        java.util.Arrays.fill(epsF, 0);
    }

    public double weight(int post, int pre) { return weights[post][pre]; }
    public double[][] weights() { return weights; }
    public double[] epsX() { return epsX; }
    public double[] epsF() { return epsF; }
    /** 梯度累积 (效果验证: 梯度方向对照) */
    public double[][] gradAccum() { return gradAccum; }

    /** 设置权重 (模型导入/快照恢复) */
    public void setWeight(int post, int pre, double w) {
        if (post >= 0 && post < weights.length && pre >= 0 && pre < weights[0].length) {
            weights[post][pre] = w;
        }
    }

    /** 计算隐藏层输出 = Σ w_ij * input */
    public double[] forward(double[] inputs) {
        double[] out = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double sum = 0;
            for (int j = 0; j < inputs.length; j++) sum += weights[i][j] * inputs[j];
            out[i] = sum;
        }
        return out;
    }
}
