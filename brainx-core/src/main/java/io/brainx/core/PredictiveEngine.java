package io.brainx.core;

/**
 * 预测引擎 —— 大脑=科学家 (第28篇, Yael Niv《醉醺醺的脑科学》)。
 *
 * 三步法 (原文):
 *   第一步 假设: 大脑一直做出预测 ("这杯咖啡有多重?")
 *   第二步 收集数据: 多感官证据整合验证假设
 *   第三步 得出结论: 结合先验解释感官输入 (感知是主观解释)
 *
 * 核心: 错误驱动学习 —— "只有当预测出现失误时, 我们才会学习"
 *
 * 实现:
 *   - predict(): 基于历史先验输出预测
 *   - verify(): 收集证据 (视觉/听觉/记忆) 与预测对比
 *   - conclude(): 输出结论 + 预测误差 → 驱动 DopamineSystem
 *   - 先验更新: 预测误差反向修正先验 (贝叶斯)
 */
public class PredictiveEngine {
    /** 预测先验: 标签 → 先验概率 */
    private final java.util.Map<String, Double> priors = new java.util.HashMap<>();
    /** 预测误差历史 */
    private final java.util.ArrayDeque<Double> errorHistory = new java.util.ArrayDeque<>();
    /** 上次预测 */
    private String lastPrediction = "";
    private double lastError = 0;

    public PredictiveEngine() {}

    /** 第一步: 假设 —— 基于先验预测 (无证据时) */
    public String predict() {
        String best = null;
        double bestP = 0;
        for (java.util.Map.Entry<String, Double> e : priors.entrySet()) {
            if (e.getValue() > bestP) { bestP = e.getValue(); best = e.getKey(); }
        }
        lastPrediction = best != null ? best : "";
        return lastPrediction;
    }

    /** 学习先验 (观察世界) */
    public void observe(String label) {
        priors.merge(label, 0.05, Double::sum);
        // 归一化
        double total = priors.values().stream().mapToDouble(Double::doubleValue).sum();
        priors.replaceAll((k, v) -> v / total);
    }

    /**
     * 第二步+第三步: 验证并得出结论。
     * @param predicted 预测标签
     * @param evidence  证据强度 (0-1, 多感官置信度)
     * @param actual    实际标签
     * @return 预测误差 (0=正确, >0=错误)
     */
    public double verifyAndConclude(String predicted, double evidence, String actual) {
        double error = 0;
        if (actual == null || actual.isEmpty()) {
            error = 1.0 - evidence;  // 证据不足也是误差
        } else if (predicted.equals(actual)) {
            error = 1.0 - evidence;  // 预测对但证据弱
            priors.merge(actual, 0.1, Double::sum);  // 正确强化先验
        } else {
            error = 1.0;  // 预测错误: 最大误差
            priors.merge(actual, 0.6, Double::sum);   // 修正: 强力强化实际标签
            priors.merge(predicted, 0.05, Double::sum);
        }
        // 归一化先验
        double total = priors.values().stream().mapToDouble(Double::doubleValue).sum();
        if (total > 0) priors.replaceAll((k, v) -> v / total);

        lastError = error;
        errorHistory.addFirst(error);
        if (errorHistory.size() > 50) errorHistory.removeLast();
        return error;
    }

    /** 先验概率查询 */
    public double prior(String label) { return priors.getOrDefault(label, 0.0); }
    public int priorCount() { return priors.size(); }
    public double lastError() { return lastError; }
    public String lastPrediction() { return lastPrediction; }
    public java.util.ArrayDeque<Double> errorHistory() { return errorHistory; }

    /** 导出全部先验 (模型快照) */
    public java.util.Map<String, Double> exportPriors() {
        return new java.util.HashMap<>(priors);
    }

    /** 导入先验 (模型恢复) */
    public void importPriors(java.util.Map<String, Double> p) {
        priors.clear();
        priors.putAll(p);
    }

    /** 摘要 (APK 显示) */
    public String summary() {
        return String.format("🔬 预测引擎: 已建立%d个先验 | 上次误差: %.2f | 预测: %s",
                priors.size(), lastError, lastPrediction.isEmpty() ? "无" : lastPrediction);
    }
}
