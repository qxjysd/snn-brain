package io.brainx.core;

/**
 * 成长潜力 —— 神经元数量成长至 2000 亿的架构潜力。
 *
 * 设计思想:
 *   手机物理内存无法容纳 2000 亿个 LIF 神经元对象 (已论证: 内存差1720倍)。
 *   但架构上具备成长潜力: 采用分层抽象 (神经群表示, brainmass 路线):
 *     - 每个物理神经元单元 (LIF) 可"代表" N 个真实神经元 (神经群/微电路)
 *     - 抽象倍数 N 随学习经验提升 (微观精细 → 宏观高效)
 *     - 总神经元容量 = 物理神经元数 × 抽象倍数
 *     - 2000 亿潜力 = 96 物理 × 2.08×10^9 抽象倍数 (或增加物理规模)
 *
 * 成长路径 (自然, 无年龄):
 *   抽象倍数随经验提升: 1 → 10 → 10^3 → 10^6 → 10^9 (每级代表更大的神经群)
 *   物理规模可配置: 低端机 96 神经元, 高端机可扩至数千 (内存允许时)
 *   潜力上限: 96 物理 × 2.08×10^9 ≈ 2000 亿 (人脑 2 倍)
 */
public class GrowthPotential {
    /** 人脑神经元数 × 2 = 2000 亿 (成长目标) */
    public static final long HUMAN_BRAIN_X2 = 200_000_000_000L;
    /** 人脑神经元数 860 亿 */
    public static final long HUMAN_BRAIN = 86_000_000_000L;

    /** 物理神经元数 (实际 LIF 单元, 可按性能配置) */
    private int physicalNeurons;
    /** 抽象倍数 (每物理神经元代表的真实神经元数, 神经群抽象) */
    private long abstractionFactor = 1;
    /** 成长阶段 (0-5: 每级抽象倍数×10) */
    private int growthStage = 0;
    /** 最大抽象倍数 (达到 2000 亿潜力) */
    private long maxAbstraction;

    /**
     * @param physicalNeurons 物理神经元数 (默认 96: 72皮层+24中枢)
     */
    public GrowthPotential(int physicalNeurons) {
        this.physicalNeurons = physicalNeurons;
        // 最大抽象倍数 = 2000亿 / 物理数 (向上取整, 保证潜力≥2000亿)
        this.maxAbstraction = Math.max(1,
                (HUMAN_BRAIN_X2 + physicalNeurons - 1) / physicalNeurons);
    }

    public static GrowthPotential defaultParams() {
        return new GrowthPotential(96);
    }

    /**
     * 随学习成长: 经验越多 → 抽象倍数提升 (自然成长, 无年龄)。
     * @param learnedWords 已学词数
     * @param level 培养等级
     */
    public void update(int learnedWords, int level) {
        // 成长条件 (经验驱动):
        //   学 2 词 → 抽象×10 (微观→神经群)
        //   学 5 词/升3级 → 抽象×10^3
        //   学 10 词/升5级 → 抽象×10^6
        //   学 20 词/升8级 → 抽象×10^9
        long target = 1;
        if (learnedWords >= 2 || level >= 1) target = 10;
        if (learnedWords >= 5 || level >= 3) target = 1_000;
        if (learnedWords >= 10 || level >= 5) target = 1_000_000;
        if (learnedWords >= 20 || level >= 8) target = 1_000_000_000;
        if (learnedWords >= 40 || level >= 12) target = maxAbstraction;  // 满潜力
        abstractionFactor = Math.min(maxAbstraction, Math.max(1, target));
        // 成长阶段 = 抽象倍数的数量级
        growthStage = (int) Math.round(Math.log10(Math.max(1, abstractionFactor)));
    }

    /** 扩展物理规模 (内存允许时增加真实神经元, 高端机) */
    public void expandPhysical(int additional) {
        physicalNeurons += additional;
        maxAbstraction = Math.max(1,
                (HUMAN_BRAIN_X2 + physicalNeurons - 1) / physicalNeurons);
        // 重新计算抽象倍数上限
        abstractionFactor = Math.min(abstractionFactor, maxAbstraction);
    }

    /** 物理神经元数 */
    public int physicalNeurons() { return physicalNeurons; }

    /** 抽象倍数 (每物理神经元代表数) */
    public long abstractionFactor() { return abstractionFactor; }

    /** 最大抽象倍数 (达到 2000 亿潜力所需) */
    public long maxAbstraction() { return maxAbstraction; }

    /** 算力档位 (低/中/高/极高) */
    public enum ComputeTier {
        低("🪫 低算力", 48),   // 入门机: 少物理神经元
        中("🔋 中算力", 96),   // 标准机: 默认
        高("⚡ 高算力", 192),  // 旗舰机: 更多物理
        极高("🔥 极高算力", 384);  // 游戏机: 最多物理
        public final String name; final int physicalTarget;
        ComputeTier(String n, int p) { this.name = n; this.physicalTarget = p; }
    }

    /** 当前算力档位 */
    private ComputeTier computeTier = ComputeTier.中;

    /**
     * 算力自适应: 根据手机算力调整物理神经元规模。
     * @param cpuCores   CPU 核心数
     * @param freeMb     可用内存 (MB)
     * @param avgFrameMs 平均帧耗时 (ms, 卡顿检测)
     * @return 算力评分 (0-4)
     */
    public int adjustToCompute(int cpuCores, long freeMb, double avgFrameMs) {
        int score = 0;
        // CPU 核心数 (0-1分)
        if (cpuCores >= 8) score += 1;
        // 内存 (0-1分)
        if (freeMb >= 4000) score += 1;
        else if (freeMb >= 2000) score += 0;  // 中内存不加分
        // 帧耗时 (0-2分): 流畅加分, 卡顿减分
        if (avgFrameMs < 25) score += 2;          // 流畅: 算力富余
        else if (avgFrameMs < 50) score += 1;     // 正常
        else if (avgFrameMs > 80) score -= 1;     // 卡顿: 算力不足

        // 档位映射
        ComputeTier tier;
        if (score >= 4) tier = ComputeTier.极高;
        else if (score >= 3) tier = ComputeTier.高;
        else if (score >= 1) tier = ComputeTier.中;
        else tier = ComputeTier.低;

        setComputeTier(tier);
        return score;
    }

    /** 设置算力档位 (物理规模随之调整, 潜力保持2000亿) */
    public void setComputeTier(ComputeTier tier) {
        this.computeTier = tier;
        // 物理神经元 → 算力目标 (自适应提升)
        physicalNeurons = tier.physicalTarget;
        // 潜力保持: 抽象上限 = ceil(2000亿/物理数)
        maxAbstraction = Math.max(1,
                (HUMAN_BRAIN_X2 + physicalNeurons - 1) / physicalNeurons);
        // 抽象倍数不超新上限
        abstractionFactor = Math.min(abstractionFactor, maxAbstraction);
    }

    /** 当前算力档位 */
    public ComputeTier computeTier() { return computeTier; }

    /** 成长阶段 (0-5) */
    public int growthStage() { return growthStage; }

    /** 当前总神经元容量 = 物理 × 抽象 */
    public long totalNeuronCapacity() {
        return physicalNeurons * abstractionFactor;
    }

    /** 是否已达到 2000 亿潜力 */
    public boolean reachedHumanX2() {
        return totalNeuronCapacity() >= HUMAN_BRAIN_X2;
    }

    /** 成长进度 (0-1: 相对 2000 亿) */
    public double growthProgress() {
        return Math.min(1.0, (double) totalNeuronCapacity() / HUMAN_BRAIN_X2);
    }

    /** 摘要 (APK 显示) — 原始倍数直显 (亿单位在早期倍数下精度丢失) */
    public String summary() {
        return String.format("%s | 📈 %d物理×%d抽象=%d容量 | 阶段%d/9 → 潜力2000亿",
                computeTier.name, physicalNeurons, abstractionFactor,
                totalNeuronCapacity(), growthStage);
    }

    /** 成长路径描述 */
    public String growthPath() {
        return String.format("成长阶段%d/9: 微观(%d物理) → 神经群(×10) → 皮层柱(×10^3) → 微电路(×10^6) → 全脑(×10^9)",
                growthStage, physicalNeurons);
    }
}
