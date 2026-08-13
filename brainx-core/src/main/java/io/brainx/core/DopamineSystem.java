package io.brainx.core;

/**
 * 多巴胺奖赏系统 —— 错误驱动学习 (error-driven learning)。
 *
 * 理论依据 (《醉醺醺的脑科学》):
 *   - 第28篇 (Yael Niv): "只有当我们的预测出现失误时，我们才会学习"
 *     —— 错误驱动学习: 大脑像科学家一样, 预测→对比现实→学习
 *   - 第9篇 (突触可塑性章节): 腹侧被盖区(VTA)多巴胺能神经元 →
 *     伏隔核(NAc) 构成脑内奖赏回路; 多巴胺释放驱动突触强化
 *   - Schultz 1997 经典发现: 多巴胺信号 = 奖励预测误差 (RPE)
 *
 * 实现:
 *   - 预期奖励: 对每次识别的成功概率估计 (基于历史)
 *   - RPE = 实际奖励 - 预期奖励
 *   - RPE > 0 (意外惊喜) → 强学习信号 (多巴胺激增)
 *   - RPE ≈ 0 (符合预期) → 弱学习 (已学会, 无需强化)
 *   - RPE < 0 (预期落空) → 负向调整 (撤销/修正)
 *   - 多巴胺水平影响: 学习强度 / 动机 / 情绪
 */
public class DopamineSystem {
    /** 预期奖励 (0-1): 对成功的先验估计 */
    private double expectedReward = 0.5;
    /** 当前多巴胺水平 (0-2, 基线1.0) */
    private double dopamine = 1.0;
    /** 学习率 (多巴胺调节) */
    private double learningRate = 0.1;
    /** 奖励历史 (滑动平均) */
    private double rewardHistory = 0.5;
    /** RPE 历史记录 (可视化/诊断) */
    private final java.util.ArrayDeque<Double> rpeHistory = new java.util.ArrayDeque<>();

    public DopamineSystem() {}

    /**
     * 一次学习事件: 输入实际结果, 计算 RPE, 更新预期。
     * @param success     是否成功 (1.0/0.0)
     * @param rewardValue 奖励价值 (0-1, 教育奖励强度)
     * @return RPE (奖励预测误差)
     */
    public double learnEvent(double success, double rewardValue) {
        double actualReward = success * rewardValue;
        double rpe = actualReward - expectedReward;
        rpeHistory.addFirst(rpe);
        if (rpeHistory.size() > 50) rpeHistory.removeLast();

        // 多巴胺: 基线1.0 + RPE 调制 (正向惊喜激增, 负向骤降)
        dopamine = 1.0 + rpe * 2.0;
        dopamine = Math.max(0.1, Math.min(2.5, dopamine));

        // 学习率: RPE 越大学习越快 (错误驱动)
        learningRate = Math.max(0.02, 0.1 + Math.abs(rpe) * 0.5);

        // 预期更新: 向实际结果漂移 (贝叶斯式)
        expectedReward += 0.1 * (actualReward - expectedReward);
        rewardHistory = rewardHistory * 0.9 + actualReward * 0.1;
        return rpe;
    }

    /** 当前 RPE 驱动的学习强度 (0-1): 供学习系统调制 */
    public double learningIntensity() {
        return Math.min(1.0, Math.abs(dopamine - 1.0) * 1.2 + 0.3);
    }

    /** 多巴胺状态描述 (情绪/动机) */
    public String dopamineStatus() {
        if (dopamine > 1.8) return "🔥 多巴胺激增! 意外惊喜, 强烈学习动机";
        if (dopamine > 1.3) return "😄 多巴胺升高, 学习愉悦";
        if (dopamine < 0.5) return "😞 多巴胺骤降, 预期落空";
        if (dopamine < 0.8) return "😐 多巴胺偏低, 动机不足";
        return "😊 多巴胺平稳, 稳定学习";
    }

    public double dopamine() { return dopamine; }
    public double expectedReward() { return expectedReward; }
    public double learningRate() { return learningRate; }
    public double rewardHistory() { return rewardHistory; }
    public double lastRPE() { return rpeHistory.isEmpty() ? 0 : rpeHistory.peekFirst(); }
    public java.util.ArrayDeque<Double> rpeHistory() { return rpeHistory; }

    /** 摘要 (APK 显示) */
    public String summary() {
        return String.format("🧪 多巴胺: %.1f | 预期奖励: %.0f%% | 上次RPE: %+.2f | 学习率: %.2f",
                dopamine, expectedReward * 100, lastRPE(), learningRate);
    }
}
