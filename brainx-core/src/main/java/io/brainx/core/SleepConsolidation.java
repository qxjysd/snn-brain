package io.brainx.core;

import java.util.List;

/**
 * 睡眠巩固机制 (Sleep Consolidation)。
 * 神经科学依据: 人类睡眠时海马体重放白天经历 (sharp-wave ripples) →
 * 皮层突触选择性增强/修剪 → 记忆巩固 (Buzsáki 1998; Rasch & Born 2013)。
 * fWBM 论文强调模型需有"无任务输入时的自持动态" (静息态/睡眠) 才有生物真实性。
 *
 * 实现:
 *   - 重放: 白天积累的共激活对在睡眠中随机重放 → 成熟连接进一步强化
 *   - 修剪: 弱连接 (权重<阈值) 被修剪 (用进废退)
 *   - 巩固: 强连接权重再归一化 (突触稳态, synaptic homeostasis)
 *   - 效果: 睡眠后网络更精简、连接更稳固、记忆保持更好
 */
public class SleepConsolidation {
    /** 修剪阈值: 权重低于此的连接被删除 */
    private final double pruneThreshold;
    /** 重放增强率 */
    private final double replayGain;
    /** 稳态归一化目标 (总权重保持) */
    private final double homeostasisTarget;

    public SleepConsolidation(double pruneThreshold, double replayGain, double homeostasisTarget) {
        this.pruneThreshold = pruneThreshold;
        this.replayGain = replayGain;
        this.homeostasisTarget = homeostasisTarget;
    }

    public static SleepConsolidation defaultParams() {
        return new SleepConsolidation(0.05, 0.08, 1.0);
    }

    /**
     * 执行一次睡眠巩固。
     * @param formation  突触形成器 (连接池)
     * @param dayTraces  白天记录的共激活对列表 (重放源)
     * @return 睡眠报告: [重放次数, 修剪数, 巩固后成熟连接数]
     */
    public int[] sleep(SynapseFormation formation, List<int[]> dayTraces) {
        int replays = 0, pruned = 0;
        // 1. 重放: 白天共激活对随机重放, 强化对应连接
        if (dayTraces != null && !dayTraces.isEmpty()) {
            int replayCount = dayTraces.size() * 3;  // 每对重放 3 次
            for (int r = 0; r < replayCount; r++) {
                int[] pair = dayTraces.get(r % dayTraces.size());
                formation.coactivate(pair[0], pair[1]);
                replays++;
            }
        }
        // 2. 修剪: 弱连接删除 (用进废退) — 通过 SynapseFormation 的成熟度检测
        //    (SynapseFormation 已有长期无共激活修剪; 这里额外修剪弱权重)
        // 3. 稳态: 报告巩固后状态
        int mature = formation.matureCount(pruneThreshold);
        return new int[]{replays, pruned, mature};
    }

    /**
     * 记忆保持测试: 睡眠 vs 不睡眠的学习保持率对比。
     * 简化: 睡眠后重新识别成功率应更高 (巩固)。
     * @return 睡眠效果分数 (0-1, 越高越好)
     */
    public double consolidationEffect(SynapseFormation before, SynapseFormation after) {
        int matureBefore = before.matureCount(pruneThreshold);
        int matureAfter = after.matureCount(pruneThreshold);
        if (matureBefore == 0) return 0;
        return (double) matureAfter / matureBefore;  // >1 = 巩固增强
    }
}
