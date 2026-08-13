package io.brainx.core;

import java.util.Random;

/**
 * 虚拟神经元层 —— 神经形态等效规模 (每物理神经元代表 K 个虚拟神经元, 神经群抽象)。
 *
 * 脑仿真路线 (brainmass 神经群表示): 物理内存放不下 2000 亿 LIF 对象 (差 ~1700 倍),
 * 但可用"神经群抽象": 一个物理 LIF 单元 = 一个同步发放的微电路 (K 个神经元)。
 * 物理单元发放 → 整个神经群同步发放 (皮层微柱同步振荡) → 虚拟脉冲数 = K × 物理脉冲。
 *
 * 本层让 GrowthPotential 的"抽象倍数"从纯概念变成真实脉冲统计:
 *   - 物理发放 → 虚拟发放 (×K) 计入总脉冲规模 (等效规模真实参与)
 *   - 群体活动率 (归一 0-1) → 中枢/EEG 宏观驱动调制 (群体越大, 宏观信号越强)
 *   - 同步性 (物理发放集中度) → 频率总线 γ 强度 (同步发放 = 高 γ)
 * 计算量 O(物理), 等效规模 = 物理 × K (最高 2000 亿 = 人脑 2 倍)。
 */
public class VirtualNeuronLayer {
    private final Random rnd;
    private long k = 1;                     // 抽象倍数 (每物理神经元代表的虚拟神经元数)
    private int physicalNeurons;
    private long totalVirtualSpikes = 0;    // 累计虚拟脉冲 (等效规模脉冲总量)
    private double virtualActivity = 0;     // 群体活动率 0-1 (本次步进)
    private double synchrony = 0;           // 同步性 0-1 (物理发放集中度)
    private int lastPhysSpikes = 0;
    private double cachedGain = 1.0;        // 宏观增益缓存 (setAbstraction 时更新)

    public VirtualNeuronLayer(int physicalNeurons) {
        this.physicalNeurons = Math.max(1, physicalNeurons);
        this.rnd = new Random(2026);
    }

    /** 同步抽象倍数 (成长: 经验提升 → 每物理单元代表更多虚拟神经元) */
    public void setAbstraction(long k) {
        this.k = Math.max(1, k);
        // 缓存宏观增益 (log10 每次学习循环每神经元调用 → 预热路径优化)
        this.cachedGain = 1.0 + Math.log10(Math.max(1, this.k)) * 0.15;
    }

    /** 更新物理神经元数 (算力自适应) */
    public void setPhysicalNeurons(int n) { this.physicalNeurons = Math.max(1, n); }

    /** 抽象倍数 */
    public long abstraction() { return k; }

    /** 等效虚拟神经元总数 = 物理 × 抽象 (最高 2000 亿) */
    public long virtualNeurons() { return physicalNeurons * k; }

    /**
     * 每步: 物理发放 → 虚拟发放扩展 (神经群同步发放) + 群体统计。
     * @param firingState 物理神经元发放状态 (全脑)
     * @param dtMs 步长 (ms)
     */
    public void step(boolean[] firingState, double dtMs) {
        int spikes = 0;
        for (boolean f : firingState) if (f) spikes++;
        lastPhysSpikes = spikes;
        totalVirtualSpikes += (long) spikes * k;
        // 群体活动率: 物理发放率 (0-1), 虚拟扩展体现在总脉冲规模而非归一值
        virtualActivity = firingState.length > 0 ? Math.min(1.0, (double) spikes / firingState.length) : 0;
        // 同步性: 发放集中度 (1 = 全同步发放, 高 γ 特征)
        synchrony = firingState.length > 0 ? (double) spikes / firingState.length : 0;
    }

    /** 群体活动率 0-1 */
    public double virtualActivity() { return virtualActivity; }

    /** 同步性 0-1 */
    public double synchrony() { return synchrony; }

    /** 本步虚拟脉冲数 = 物理发放 × 抽象倍数 */
    public long lastVirtualSpikes() { return (long) lastPhysSpikes * k; }

    /** 累计虚拟脉冲 (等效规模脉冲总量) */
    public long totalVirtualSpikes() { return totalVirtualSpikes; }

    /** 物理神经元数 */
    public int physicalNeurons() { return physicalNeurons; }

    /** EEG/中枢宏观调制系数: 群体越大宏观信号越强 (1.0 + 对数规模贡献, 缓存) */
    public double macroscopicGain() {
        return cachedGain;
    }

    /** 摘要 */
    public String summary() {
        return String.format("🧬 虚拟层: %d物理×%d抽象=%d等效神经元 | 群体活动%.0f%% | 同步%.0f%%",
                physicalNeurons, k, virtualNeurons(), virtualActivity * 100, synchrony * 100);
    }
}
