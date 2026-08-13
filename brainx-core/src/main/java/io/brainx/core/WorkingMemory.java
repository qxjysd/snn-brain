package io.brainx.core;

import java.util.Arrays;

/**
 * 工作记忆模块 —— 基于吸引子动力学 (attractor dynamics)。
 *
 * 论文依据 (fWBM, arXiv 2605.18118):
 *   1. "working memory capacity maintained through distributed,
 *      self-sustained activity across multiple cortical areas" ——
 *      工作记忆 = 分布式自持续活动 (无外部输入也保持)
 *   2. "thalamic gating" —— 丘脑门控控制信息进出
 *   3. "oscillations, attractors and metastable states" ——
 *      认知功能涌现于吸引子/亚稳态动力学
 *
 * 神经科学依据:
 *   - 前额叶持续性放电 (persistent activity, Goldman-Rakic)
 *   - 人类工作记忆容量 ≈ 4±1 组块 (Miller 1956 神奇数字)
 *   - Ebbinghaus 遗忘曲线: 无重放时记忆指数衰减
 *
 * 实现:
 *   - 固定槽位 (默认 4, 模拟人类容量限制)
 *   - 每个槽位: 内容向量 + 强度 (持续活动水平)
 *   - 丘脑门控: writeGate / readGate / clearGate
 *   - 自持续: 写入后强度缓慢衰减 (无干扰时保持 ~几十秒)
 *   - 遗忘: 指数衰减 + 新写入挤出旧内容 (容量竞争)
 *   - 中枢脉冲联动: 槽位强度→脉冲发射, 接收中枢广播调制 (PulseModule)
 */
public class WorkingMemory implements PulseModule {
    /** 槽位容量 (人类工作记忆 4±1) */
    private final int capacity;
    /** 内容维度 */
    private final int dim;
    /** 每个槽位的内容向量 */
    private final double[][] contents;
    /** 每个槽位的强度 (持续活动水平 0-1) */
    private final double[] strengths;
    /** 槽位年龄 (最近使用时间) */
    private final long[] lastAccess;
    /** 槽位共振频率 (工作记忆=频率标记: 每条记忆关联一个频率) */
    private final double[] slotFreq;
    /** 丘脑门控 (0-1, 默认全开) */
    private double writeGate = 1.0;
    private double readGate = 1.0;
    private double clearGate = 0.0;
    /** 遗忘时间常数 (ms) */
    private final double forgetTauMs = 30000.0;  // ~30s 自持续
    private long timeMs = 0;
    private long nextSlot = 0;  // 轮转写入位置

    public WorkingMemory(int capacity, int dim) {
        this.capacity = Math.max(1, Math.min(8, capacity));  // 人类容量 4±1, 上限8
        this.dim = dim;
        this.contents = new double[capacity][dim];
        this.strengths = new double[capacity];
        this.lastAccess = new long[capacity];
        this.slotFreq = new double[capacity];
        // 槽位默认频率: 均匀分布 θ-α 带 (4-13Hz, 记忆频率)
        for (int i = 0; i < capacity; i++) {
            slotFreq[i] = 4.0 + 9.0 * i / Math.max(1, capacity - 1);
        }
    }

    /** 默认: 4 槽位 (Miller 神奇数字), 32 维内容 */
    public static WorkingMemory defaultParams() {
        return new WorkingMemory(4, 32);
    }

    /** 丘脑门控设置 (0=关闭, 1=全开) */
    public void setWriteGate(double g) { writeGate = Math.max(0, Math.min(1, g)); }
    public void setReadGate(double g) { readGate = Math.max(0, Math.min(1, g)); }
    public void setClearGate(double g) { clearGate = Math.max(0, Math.min(1, g)); }
    public double writeGate() { return writeGate; }
    public double readGate() { return readGate; }

    /**
     * 写入工作记忆 (丘脑门控开启时)。
     * 选择: 空槽位 > 最弱槽位 (容量竞争挤出)。
     * @return 写入的槽位索引
     */
    public int write(double[] content) {
        if (writeGate < 0.5) return -1;  // 门控关闭, 不写入
        // 找空槽 (强度最低的)
        int slot = -1;
        double minStrength = Double.MAX_VALUE;
        for (int i = 0; i < capacity; i++) {
            if (strengths[i] < minStrength) {
                minStrength = strengths[i];
                slot = i;
            }
        }
        // 写入
        System.arraycopy(content, 0, contents[slot], 0, Math.min(dim, content.length));
        strengths[slot] = 1.0;
        lastAccess[slot] = timeMs;
        return slot;
    }

    /**
     * 读取: 返回与查询最匹配的槽位内容 (基于内容相似度 + 频率共振)。
     * @param query 查询向量
     * @return [匹配槽位索引, 相似度] 或 [-1, 0] (门控关/空)
     */
    public double[] read(double[] query) {
        if (readGate < 0.5) return new double[]{-1, 0};
        int best = -1;
        double bestSim = 0;
        for (int i = 0; i < capacity; i++) {
            if (strengths[i] <= 0) continue;
            double sim = cosineSimilarity(contents[i], query);
            if (sim > bestSim) {
                bestSim = sim;
                best = i;
            }
        }
        if (best >= 0) lastAccess[best] = timeMs;
        return new double[]{best, bestSim};
    }

    /**
     * 频率匹配读取: 按查询频率共振检索工作记忆槽位 (记忆=频率)。
     * @param queryHz 查询频率 (Hz)
     * @return [槽位, 共振度] 或 [-1, 0]
     */
    public double[] readByFrequency(double queryHz) {
        if (readGate < 0.5) return new double[]{-1, 0};
        int best = -1;
        double bestR = 0;
        for (int i = 0; i < capacity; i++) {
            if (strengths[i] <= 0) continue;
            double r = FrequencyWave.resonance(queryHz, slotFreq[i], 1.5) * strengths[i];
            if (r > bestR) { bestR = r; best = i; }
        }
        if (best >= 0) lastAccess[best] = timeMs;
        return new double[]{best, bestR};
    }

    /** 槽位共振频率 (工作记忆=频率) */
    public double slotFrequency(int i) { return i >= 0 && i < capacity ? slotFreq[i] : 0; }

    /** 槽位当前主导频率 (工作记忆输出到频率总线) */
    public double currentFrequency() {
        int best = -1;
        double bestS = 0;
        for (int i = 0; i < capacity; i++) {
            if (strengths[i] > bestS) { bestS = strengths[i]; best = i; }
        }
        return best >= 0 ? slotFreq[best] : 6.0;  // 默认 θ
    }

    // ============ PulseModule: 中枢脉冲联动 ============

    @Override public String moduleName() { return "工作记忆"; }
    @Override public int pulseDim() { return capacity; }

    /** 发射脉冲: 槽位强度 → 脉冲率 (0-1) */
    @Override
    public double[] emitPulses() {
        return strengths.clone();
    }

    /** 接收中枢广播: 广播强度调制槽位强度 (中枢整合→工作记忆保持) */
    @Override
    public boolean receiveBroadcast(double[] broadcastRates) {
        if (broadcastRates.length == 0) return false;
        double avg = 0;
        for (double r : broadcastRates) avg += r;
        avg /= broadcastRates.length;
        // 中枢活跃 → 工作记忆巩固 (自持续维持); 低活跃 → 正常遗忘
        if (avg > 0.3) {
            for (int i = 0; i < capacity; i++) {
                if (strengths[i] > 0) {
                    strengths[i] = Math.min(1.0, strengths[i] + avg * 0.02);
                }
            }
            return true;
        }
        return false;
    }

    /** 读取第 i 槽位内容 */
    public double[] readSlot(int i) {
        if (i < 0 || i >= capacity) return null;
        return contents[i].clone();
    }

    /** 清除 (丘脑门控: 清空指定槽位或全部) */
    public void clear(int slot) {
        if (clearGate < 0.5) return;
        if (slot < 0) {
            for (int i = 0; i < capacity; i++) strengths[i] = 0;
        } else if (slot < capacity) {
            strengths[slot] = 0;
        }
    }

    /** 巩固: 重放指定槽位 (睡眠时强化, 抵抗遗忘) */
    public void rehearse(int slot) {
        if (slot >= 0 && slot < capacity) {
            strengths[slot] = Math.min(1.0, strengths[slot] + 0.3);
            lastAccess[slot] = timeMs;
        }
    }

    /**
     * 每时间步: 自持续衰减 (无重放时遗忘, Ebbinghaus 曲线)。
     * @param dtMs 时间步长
     */
    public void tick(double dtMs) {
        timeMs += dtMs;
        for (int i = 0; i < capacity; i++) {
            if (strengths[i] > 0) {
                // 指数遗忘: S(t) = S0 * exp(-t/tau)
                strengths[i] *= Math.exp(-dtMs / forgetTauMs);
                if (strengths[i] < 0.01) strengths[i] = 0;  // 完全遗忘
            }
        }
    }

    /** 工作记忆负载 (活跃槽位数) */
    public int load() {
        int n = 0;
        for (double s : strengths) if (s > 0.1) n++;
        return n;
    }

    /** 占用率 0-1 (容量限制可视化) */
    public double occupancy() {
        return (double) load() / capacity;
    }

    public int capacity() { return capacity; }
    public double strength(int i) { return i >= 0 && i < capacity ? strengths[i] : 0; }
    public double[] strengths() { return strengths.clone(); }
    public long timeMs() { return timeMs; }

    /** 导出工作记忆内容 (模型快照): [槽][内容向量] + 强度 */
    public double[][] exportContents() {
        double[][] out = new double[capacity][];
        for (int i = 0; i < capacity; i++) {
            out[i] = contents[i].clone();
        }
        return out;
    }

    /** 导入工作记忆内容 (模型恢复) */
    public void importContents(double[][] contentsIn, double[] strengthsIn) {
        for (int i = 0; i < Math.min(capacity, contentsIn.length); i++) {
            System.arraycopy(contentsIn[i], 0, contents[i], 0, Math.min(dim, contentsIn[i].length));
            strengths[i] = (i < strengthsIn.length) ? strengthsIn[i] : 0;
        }
    }

    private static double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
