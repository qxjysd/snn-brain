package io.brainx.core;

/**
 * 内部模型 —— 前向预测 + 感觉预测误差 (第20篇, 运动物理学)。
 *
 * 理论依据 (《醉醺醺的脑科学》第20篇 避免碰翻咖啡杯):
 *   - 大脑在动作执行前预测该动作的结果 (前向模型)
 *   - "预测的感官状态和实际感官状态的不匹配" = 感觉预测误差
 *   - 大脑持续监测误差并对后续行为微调 (小脑/下橄榄核纠错)
 *   - 感觉信息获取慢 → 靠预测补偿 (如自己拿书时预判重量)
 *
 * 实现:
 *   - 前向模型: 基于当前状态+动作, 预测结果 (内部模拟)
 *   - 感觉预测误差: 预测 vs 实际感官反馈
 *   - 误差驱动修正: 更新内部模型参数 (学习)
 *   - 泛化: 内部模型学会"规律" → 对未见输入也能预测 (泛化基础)
 */
public class InternalModel {
    /** 内部模型参数: 状态→结果 的线性映射 */
    private double[] modelWeights;
    /** 模型偏置 */
    private double bias = 0;
    /** 学习率 */
    private final double learningRate = 0.1;
    /** 感觉预测误差历史 */
    private final java.util.ArrayDeque<Double> errorHistory = new java.util.ArrayDeque<>();
    /** 上次预测误差 */
    private double lastError = 0;
    /** 训练样本数 (泛化评估) */
    private int trainCount = 0, testCount = 0;
    /** 泛化性能 */
    private double generalizationScore = 0;

    public InternalModel(int inputDim) {
        this.modelWeights = new double[inputDim];
        java.util.Random rnd = new java.util.Random(11);
        for (int i = 0; i < inputDim; i++) modelWeights[i] = (rnd.nextDouble() - 0.5) * 0.2;
    }

    /**
     * 前向预测: 基于状态预测结果。
     * @param state 当前状态向量
     * @return 预测结果
     */
    public double predict(double[] state) {
        double sum = bias;
        int n = Math.min(state.length, modelWeights.length);
        for (int i = 0; i < n; i++) sum += modelWeights[i] * state[i];
        return sum;
    }

    /**
     * 学习一步: 预测 vs 实际 → 感觉预测误差 → 修正模型。
     * @param state   状态向量
     * @param actual  实际感觉结果
     * @return 感觉预测误差
     */
    public double learn(double[] state, double actual) {
        double prediction = predict(state);
        double error = actual - prediction;
        lastError = error;
        errorHistory.addFirst(error);
        if (errorHistory.size() > 100) errorHistory.removeLast();

        // 误差驱动修正 (梯度下降: 最小化预测误差)
        int n = Math.min(state.length, modelWeights.length);
        for (int i = 0; i < n; i++) {
            modelWeights[i] += learningRate * error * state[i];
        }
        bias += learningRate * error;
        trainCount++;
        return error;
    }

    /**
     * 训练 (有监督): 记录训练样本。
     * @return 训练后的平均误差
     */
    public double train(double[][] states, double[] actuals) {
        trainCount += states.length;
        double totalErr = 0;
        for (int i = 0; i < states.length; i++) {
            totalErr += Math.abs(learn(states[i], actuals[i]));
        }
        return totalErr / states.length;
    }

    /**
     * 泛化评估: 用未见过的样本测试。
     * 性能 = 1 - 归一化预测误差 (越高=泛化越好)
     */
    public double evaluateGeneralization(double[][] testStates, double[] testActuals) {
        testCount += testStates.length;
        double totalErr = 0, totalRange = 0;
        double minA = Double.MAX_VALUE, maxA = -Double.MAX_VALUE;
        for (double a : testActuals) {
            minA = Math.min(minA, a);
            maxA = Math.max(maxA, a);
        }
        for (int i = 0; i < testStates.length; i++) {
            totalErr += Math.abs(predict(testStates[i]) - testActuals[i]);
        }
        totalRange = Math.max(1e-9, maxA - minA);
        double normErr = totalErr / testStates.length / totalRange;
        generalizationScore = Math.max(0, 1.0 - normErr);
        return generalizationScore;
    }

    /**
     * 感觉预测误差驱动行为调整 (第20篇: 碰翻杯子后修正)。
     * @return 修正幅度 (误差越大修正越大)
     */
    public double correctionAmount() {
        return Math.min(1.0, Math.abs(lastError) * 2.0);
    }

    public double lastError() { return lastError; }
    public double generalizationScore() { return generalizationScore; }
    public int trainCount() { return trainCount; }
    public int testCount() { return testCount; }
    public java.util.ArrayDeque<Double> errorHistory() { return errorHistory; }
    public double[] modelWeights() { return modelWeights; }

    /** 摘要 (APK 显示) */
    public String summary() {
        return String.format("⚙️ 内部模型: 训练%d/测试%d | 泛化率%.0f%% | 上次误差%+.3f",
                trainCount, testCount, generalizationScore * 100, lastError);
    }
}
