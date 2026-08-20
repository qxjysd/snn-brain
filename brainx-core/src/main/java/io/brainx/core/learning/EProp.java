package io.brainx.core.learning;

/**
 * e-prop 在线学习规则 (Eligibility Propagation)。
 * 来源: "Can Biologically Plausible Temporal Credit Assignment Rules Match
 * BPTT for Neural Similarity? E-prop as an Example" (arXiv 2506.06904);
 * 原始 e-prop: Bellec et al. 2020.
 *
 * 三因子规则 (论文 Eq.9): ΔW_ij = Σ_t I_i,t · e_ij,t
 *   - I_i,t = ∂L/∂h_i,t: 顶向下教学信号 (输出层=误差; 隐藏层=固定随机
 *     反馈矩阵 B 广播误差 — 反馈对齐 feedback alignment, 无需对称权重)
 *   - e_ij,t: 资格迹 (Hebbian 前后突触因子, 脉冲关联的衰减记忆)
 *     e_ij[t+1] = λe·e_ij[t] + ψ(v_i[t])·s_j[t]
 *     λe = exp(-Δt/τe), ψ 是代理梯度 (surrogate), s_j 是突触前脉冲
 *
 * 生物学对应: 资格迹(突触局部) × 顶向下指导(神经调质) — 论文明确
 * e-prop 是"突触局部资格迹 + 顶向下教学信号"的三因子规则, 与 pp-prop
 * 的区别: pp-prop 用前/后迹张量积 (ε_f⊗ε_x), e-prop 用单迹 × 代理梯度。
 *
 * 论文核心结论 (本实现测试对照): e-prop 在任务准确率匹配 BPTT 时,
 * 神经表征相似度 (Procrustes 距离) 与 BPTT 相当 — 生物合理规则不牺牲
 * 表征质量。
 */
public class EProp {
    /** 资格迹衰减: λe = exp(-Δt/τe), τe≈20ms, Δt=1ms → 0.951 */
    private final double lambdaE;
    private final double learningRate;
    /** 代理梯度斜率 γ (surrogate gradient, 论文 Eq.4 风格) */
    private final double gamma;

    /** 资格迹 e_ij (hidden x input) */
    private final double[][] eligibility;
    /** 权重矩阵 (hidden x input) */
    private final double[][] weights;
    /** 权重梯度累积 */
    private final double[][] gradAccum;
    /** 反馈对齐矩阵 B (固定随机, 方差保持 1/sqrt(Nin)) */
    private final double[][] feedback;

    /** 累积教学信号 I_i (顶向下) */
    private final double[] learningSignalAccum;

    private final int numHidden;

    public EProp(int numInput, int numHidden, double lambdaE, double learningRate, long seed) {
        this.lambdaE = lambdaE;
        this.learningRate = learningRate;
        this.gamma = 1.0;
        this.numHidden = numHidden;
        this.eligibility = new double[numHidden][numInput];
        this.weights = new double[numHidden][numInput];
        this.gradAccum = new double[numHidden][numInput];
        this.feedback = new double[numHidden][numHidden];
        this.learningSignalAccum = new double[numHidden];
        java.util.Random rnd = new java.util.Random(seed);
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInput; j++) {
                weights[i][j] = (rnd.nextDouble() - 0.5) * 0.4;
            }
            // 反馈对齐矩阵: 方差保持 1/sqrt(Nin) (论文: 防门控/学习信号爆炸)
            for (int k = 0; k < numHidden; k++) {
                feedback[i][k] = (rnd.nextDouble() - 0.5) * 2.0 / Math.sqrt(numHidden);
            }
        }
    }

    public static EProp withDefaults(int numInput, int numHidden, double lr) {
        return new EProp(numInput, numHidden, 0.95, lr, 42);
    }

    /**
     * 资格迹更新 + 前向累积 (每个时间步调用)。
     *
     * @param hiddenMembrane 隐藏神经元膜电位 (代理梯度输入)
     * @param hiddenThreshold 隐藏神经元阈值 (代理梯度: 膜电位接近阈值时梯度大)
     * @param preSpikes       突触前脉冲 (0/1, 长度 numInput)
     */
    public void step(double[] hiddenMembrane, double hiddenThreshold,
                     double[] preSpikes) {
        for (int i = 0; i < numHidden; i++) {
            // 代理梯度 ψ(v) = γ·max(0, 1-|v-Z|/δ), δ=1 (论文 Eq.4 精神)
            double psi = gamma * Math.max(0.0, 1.0 - Math.abs(hiddenMembrane[i] - hiddenThreshold));
            for (int j = 0; j < preSpikes.length; j++) {
                // e_ij[t+1] = λe·e_ij[t] + ψ(v_i)·s_j
                eligibility[i][j] = lambdaE * eligibility[i][j] + psi * preSpikes[j];
            }
        }
    }

    /**
     * 输出层教学信号: I_i = y_i - ŷ_i (预测误差)。
     */
    public void setOutputError(double[] predictionError) {
        for (int i = 0; i < numHidden && i < predictionError.length; i++) {
            learningSignalAccum[i] += predictionError[i];
        }
    }

    /**
     * 隐藏层教学信号: I = B·δ (固定随机反馈矩阵广播误差, 反馈对齐)。
     */
    public void broadcastHiddenError(double[] outputError) {
        for (int i = 0; i < numHidden; i++) {
            double s = 0;
            for (int k = 0; k < outputError.length; k++) {
                s += feedback[i][k] * outputError[k];
            }
            learningSignalAccum[i] += s;
        }
    }

    /** 权重更新累积: ΔW_ij += I_i · e_ij (论文 Eq.9) */
    public void update() {
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < eligibility[i].length; j++) {
                gradAccum[i][j] += learningSignalAccum[i] * eligibility[i][j];
            }
        }
    }

    /** 应用梯度 (带裁剪) 并清零累积 */
    public void applyGradients() {
        final double maxGrad = 1.0;
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                double g = gradAccum[i][j];
                if (g > maxGrad) g = maxGrad;
                if (g < -maxGrad) g = -maxGrad;
                weights[i][j] += learningRate * g;
                gradAccum[i][j] = 0;
            }
        }
        java.util.Arrays.fill(learningSignalAccum, 0.0);
    }

    /** 序列间重置资格迹 (新样本不继承旧关联) */
    public void resetTraces() {
        for (double[] row : eligibility) java.util.Arrays.fill(row, 0.0);
        java.util.Arrays.fill(learningSignalAccum, 0.0);
    }

    public double weight(int post, int pre) { return weights[post][pre]; }
    public double[][] weights() { return weights; }
    public double[][] eligibilityTraces() { return eligibility; }
    public double[][] gradAccum() { return gradAccum; }
    public double learningSignal(int i) { return learningSignalAccum[i]; }
    public double eligibility(int post, int pre) { return eligibility[post][pre]; }

    /** 前向: 隐藏激活 (tanh 平滑, 用于预测/关联) */
    public double[] forward(double[] inputs) {
        double[] out = new double[numHidden];
        for (int i = 0; i < numHidden; i++) {
            double acc = 0;
            for (int j = 0; j < inputs.length; j++) acc += weights[i][j] * inputs[j];
            out[i] = Math.tanh(acc);
        }
        return out;
    }
}
