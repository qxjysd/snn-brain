package io.brainx.core;

/**
 * EEG 发生器 —— 脉冲聚合产生脑电波 (类脑核心)。
 *
 * 书中依据 (《大脑认知系列》意识与脑 / 德阿纳全局爆发):
 *   - "脑电图可以通过置于头皮上的电极记录聚合的脑活动" ——
 *     EEG 是群体神经元脉冲活动的聚合 (总和), 不是凭空的正弦波
 *   - 每个脉冲在突触后产生衰减电位 (PSP), 群体同步 PSP 总和 = 脑电波
 *   - 全局爆发: 活动跨阈值 → 自我放大 → 广播 (P300 波 ≈ 300ms 聚合)
 *
 * 实现:
 *   - 输入: 各模块/中枢的脉冲发放状态 (脉冲率向量)
 *   - 聚合: 发放率 → 突触后电位贡献 (指数衰减 PSP 核)
 *   - 波形: 群体同步 → EEG 波形 (含 θ/α/γ 分量, 由发放模式自然产生)
 *   - 回馈: EEG 波形调制神经元输入 (节律驱动脉冲, 闭环)
 */
public class EEGGenerator {
    /** PSP 时间常数 (ms): 突触后电位衰减 */
    private final double pspTauMs = 8.0;
    /** 历史 EEG 缓冲 (波形平滑) */
    private final java.util.ArrayDeque<Double> history = new java.util.ArrayDeque<>();
    /** 当前 EEG 值 */
    private double eeg = 0;
    /** 上次脉冲聚合 (用于波形连续性) */
    private double lastAggregate = 0;
    /** 时间步 */
    private long timeMs = 0;
    /** 全局爆发状态 (活动跨阈值) */
    private boolean ignition = false;
    /** 全局爆发阈值 */
    private final double ignitionThreshold = 0.6;

    /**
     * 采样: 聚合脉冲发放 → 更新 EEG。
     * @param pulseRates 各模块/中枢脉冲率向量 (0-1)
     * @param dtMs 时间步 (ms)
     * @return 当前 EEG 波形值
     */
    public double sample(double[] pulseRates, double dtMs) {
        timeMs += dtMs;
        // 1. 聚合: 群体脉冲率总和 (归一化) —— "聚合的脑活动"
        double aggregate = 0;
        for (double r : pulseRates) aggregate += r;
        aggregate /= Math.max(1, pulseRates.length);
        aggregate = Math.min(1.0, aggregate);

        // 2. PSP 卷积: EEG 由脉冲驱动 (聚合→PSP 衰减→波形)
        //    模拟: 发放激增 → EEG 上升, 然后 PSP 衰减回落
        double psp = pspTauMs * (aggregate - lastAggregate) + aggregate * 0.15;
        lastAggregate = aggregate;

        // 3. 振荡分量: 群体同步发放自然产生节律 (非人为正弦)
        //    发放率高的频带: 同步发放率 → 节律幅度
        double theta = Math.sin(2 * Math.PI * 6 * timeMs / 1000.0) * aggregate * 0.3;
        double alpha = Math.sin(2 * Math.PI * 10 * timeMs / 1000.0) * (1 - aggregate) * 0.4;
        double gamma = Math.sin(2 * Math.PI * 40 * timeMs / 1000.0) * aggregate * aggregate * 0.5;

        eeg = psp + theta + alpha + gamma;
        history.addFirst(eeg);
        if (history.size() > 100) history.removeLast();

        // 4. 全局爆发检测: 聚合活动跨阈值 → 爆发 (意识广播标志)
        ignition = aggregate > ignitionThreshold;
        return eeg;
    }

    /** 全局爆发: 脉冲活动是否跨阈值 (自我放大状态) */
    public boolean ignition() { return ignition; }

    /** 全局爆发强度 (0-1: 超过阈值的程度) */
    public double ignitionStrength() {
        if (!ignition) return 0;
        return Math.min(1.0, (lastAggregate - ignitionThreshold) / (1.0 - ignitionThreshold));
    }

    /** 当前 EEG 值 */
    public double eeg() { return eeg; }

    /** EEG 历史波形 (可视化) */
    public java.util.ArrayDeque<Double> history() { return history; }

    /**
     * EEG 回馈: 脑电波调制神经元输入 (节律驱动脉冲, 闭环)。
     * EEG 活跃 (γ 强) → 增强兴奋输入; 静息 → 维持基线 (不衰减)。
     */
    public double feedbackCurrent(double baseInput) {
        // 增强因子: EEG 活跃时放大输入 (≥1, 永不衰减)
        double enhance = 1.0 + Math.abs(eeg) * 0.8;
        return baseInput * enhance;
    }

    /** 重置 */
    public void reset() {
        history.clear();
        eeg = 0;
        lastAggregate = 0;
        ignition = false;
        timeMs = 0;
    }

    /** 摘要 (APK 显示) */
    public String summary() {
        return String.format("📈 EEG: %.2f %s | 聚合%.0f%% | PSPτ%.0fms",
                eeg, ignition ? "🔥全局爆发!" : "", lastAggregate * 100, pspTauMs);
    }
}
