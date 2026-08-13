package io.brainx.core.learning;

/**
 * D-RTRL (Diagonal-approximated Real-Time Recurrent Learning) 在线学习算法。
 * 来源: BrainTrace 论文 (Nature Comm 2026, Eq.4)。
 *
 *   ε^t_D = D^t·ε^(t-1) + diag(D_f^t) ⊗ x^t   (Eq.4)
 *
 * 与 pp-prop 不同，D-RTRL 保留完整的对角化资格迹（不分解），
 * 梯度近似精度更高（余弦相似度 ~0.9+），但内存 O(H²)。
 * 适合中等规模网络追求精度；pp-prop O(H) 适合大规模。
 */
public class DRTRL {
    private final int numInput, numHidden;
    private final double alpha, learningRate;

    /** 资格迹 ε: [hidden][input] */
    private final double[][] eligibility;
    /** 权重矩阵 */
    private final double[][] weights;
    /** 梯度累积 */
    private final double[][] gradAccum;

    public DRTRL(int numInput, int numHidden, double alpha, double learningRate) {
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.alpha = alpha;
        this.learningRate = learningRate;
        this.eligibility = new double[numHidden][numInput];
        this.weights = new double[numHidden][numInput];
        this.gradAccum = new double[numHidden][numInput];
        java.util.Random rnd = new java.util.Random(7);
        for (int i = 0; i < numHidden; i++)
            for (int j = 0; j < numInput; j++)
                weights[i][j] = (rnd.nextDouble() - 0.5) * 0.4;
    }

    /**
     * 前向一步更新资格迹 (Eq.4)。
     * @param inputSpikes 输入 (0/1)
     * @param hiddenInputDeriv diag(D_f): 每个隐藏神经元对输入的导数
     * @param hiddenJacobianDiag diag(D): 内在动力学雅可比对角近似
     */
    public void step(double[] inputSpikes, double[] hiddenInputDeriv, double hiddenJacobianDiag) {
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInput; j++) {
                // ε = D·ε + diag(D_f)⊗x
                eligibility[i][j] = hiddenJacobianDiag * eligibility[i][j]
                        + hiddenInputDeriv[i] * inputSpikes[j];
            }
        }
    }

    /** 学习信号更新 (Eq.2) */
    public void update(double[] learningSignal) {
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInput; j++) {
                gradAccum[i][j] += learningSignal[i] * eligibility[i][j];
            }
        }
    }

    public void applyGradients() {
        for (int i = 0; i < numHidden; i++)
            for (int j = 0; j < numInput; j++) {
                weights[i][j] += learningRate * gradAccum[i][j];
                gradAccum[i][j] = 0;
            }
    }

    public void resetTraces() {
        for (double[] row : eligibility) java.util.Arrays.fill(row, 0);
    }

    public double weight(int post, int pre) { return weights[post][pre]; }
    public double[][] weights() { return weights; }
    public double[][] eligibility() { return eligibility; }
    /** 梯度累积 (效果验证: 梯度方向对照) */
    public double[][] gradAccum() { return gradAccum; }
}
