package io.brainx.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 分层记忆系统 —— 短期→中期→长期 (Atkinson-Shiffrin 多存储模型)。
 *
 * 理论依据:
 *   - Atkinson & Shiffrin (1968) 多存储模型:
 *     感觉记忆 → 短期记忆 → 长期记忆, 经注意/复述/编码转移
 *   - fWBM 论文: "between short-term, mid-term and long-term"
 *     (全脑模型需区分短期/中期/长期记忆)
 *   - 巩固理论: 睡眠时短期记忆经重放转化为长期记忆
 *
 * 层级:
 *   - 短期 (working memory): 秒级, 容量小, 已实现 WorkingMemory
 *   - 中期 (episodic): 小时~天级, 事件记录, 睡眠时从短期沉淀
 *   - 长期 (semantic): 天~年级, 稳定知识, 反复巩固后写入
 *
 * 转移规则:
 *   短期 → 中期: 睡眠巩固时 (注意/复述)
 *   中期 → 长期: 反复出现 (出现次数 ≥ 阈值) 或 高情感事件
 */
public class HierarchicalMemory implements PulseModule {
    /** 中期记忆条目 */
    public static class EpisodicItem {
        public final String label;
        public final double[] features;
        public final long timeMs;
        public int occurrences = 1;      // 出现次数 (中期→长期判定)
        public double emotionalValue = 0; // 情感强度 (高情感易巩固)
        public double strength = 1.0;     // 中期记忆强度 (随时间衰减)

        public EpisodicItem(String label, double[] features, long timeMs) {
            this.label = label;
            this.features = features;
            this.timeMs = timeMs;
        }
    }

    /** 长期记忆条目 */
    public static class LongTermItem {
        public final String label;
        public final double[] prototype;   // 原型向量 (多次平均)
        public int consolidation = 1;      // 巩固次数
        public double strength = 1.0;      // 长期强度 (衰减极慢)
        public double frequencyHz = 0;     // 长期记忆=频率: 共振频率 (θ-α带)

        public LongTermItem(String label, double[] prototype, double freqHz) {
            this.label = label;
            this.prototype = prototype;
            this.frequencyHz = freqHz;
        }
    }

    private final List<EpisodicItem> episodic = new ArrayList<>();
    private final Map<String, LongTermItem> longTerm = new HashMap<>();
    private final int dim;
    /** 中期→长期 阈值 (出现次数) */
    private final int consolidationThreshold;
    /** 中期记忆衰减时间常数 (ms) */
    private final double episodicTauMs = 6 * 3600 * 1000.0;  // 6h
    private long timeMs = 0;

    public HierarchicalMemory(int dim, int consolidationThreshold) {
        this.dim = dim;
        this.consolidationThreshold = Math.max(2, consolidationThreshold);
    }

    public static HierarchicalMemory defaultParams() {
        return new HierarchicalMemory(32, 3);
    }

    /** 短期→中期: 学习/经历后沉淀为情景记忆 */
    public int addEpisodic(String label, double[] features, double emotionalValue) {
        // 查找是否已有同标签条目 (合并加强)
        for (EpisodicItem item : episodic) {
            if (item.label.equals(label)) {
                item.occurrences++;
                item.emotionalValue = Math.max(item.emotionalValue, emotionalValue);
                item.strength = 1.0;
                // 出现次数达标 → 尝试巩固为长期
                if (item.occurrences >= consolidationThreshold) {
                    consolidateToLongTerm(item);
                }
                return item.occurrences;
            }
        }
        EpisodicItem item = new EpisodicItem(label, features, timeMs);
        item.emotionalValue = emotionalValue;
        episodic.add(item);
        if (episodic.size() > 100) episodic.remove(0);  // 容量限制
        return 1;
    }

    /** 中期→长期: 反复出现的记忆固化为长期知识 (带共振频率) */
    private void consolidateToLongTerm(EpisodicItem item) {
        LongTermItem lt = longTerm.get(item.label);
        // 长期记忆频率: 基于标签哈希分配稳定频率 (θ-α带 4-13Hz)
        double freq = 4.0 + 9.0 * (item.label.hashCode() & 0xFFFF) / 65535.0;
        if (lt == null) {
            longTerm.put(item.label, new LongTermItem(item.label, item.features.clone(), freq));
        } else {
            // 原型平均
            double[] p = lt.prototype;
            for (int i = 0; i < Math.min(dim, p.length); i++) {
                p[i] = (p[i] * lt.consolidation + item.features[i]) / (lt.consolidation + 1);
            }
            lt.consolidation++;
            lt.strength = 1.0;
        }
    }

    /** 按频率检索长期记忆 (记忆=频率: 查询频率→共振最强者) */
    public String recallByFrequency(double queryHz) {
        String best = null;
        double bestR = 0;
        for (LongTermItem item : longTerm.values()) {
            double r = FrequencyWave.resonance(queryHz, item.frequencyHz, 2.0) * item.strength;
            if (r > bestR) { bestR = r; best = item.label; }
        }
        return bestR > 0.4 ? best : null;
    }

    /** 长期记忆当前主导频率 (输出到频率总线) */
    public double currentFrequency() {
        String best = null;
        double bestS = 0;
        for (LongTermItem item : longTerm.values()) {
            if (item.strength > bestS) { bestS = item.strength; best = item.label; }
        }
        if (best != null) return longTerm.get(best).frequencyHz;
        return 10.0;  // 默认 α
    }

    // ============ PulseModule: 中枢脉冲联动 ============

    @Override public String moduleName() { return "分层记忆"; }
    @Override public int pulseDim() { return 2; }  // [长期强度, 中期负载]

    /** 发射脉冲: [长期记忆总强度, 中期记忆负载] */
    @Override
    public double[] emitPulses() {
        double ltStrength = 0;
        for (LongTermItem item : longTerm.values()) ltStrength += item.strength;
        ltStrength = Math.min(1.0, ltStrength / Math.max(1, longTerm.size()));
        double epLoad = Math.min(1.0, episodic.size() / 20.0);
        return new double[]{ltStrength, epLoad};
    }

    /** 接收中枢广播: 中枢活跃 → 长期记忆巩固 (广播强化语义记忆) */
    @Override
    public boolean receiveBroadcast(double[] broadcastRates) {
        if (broadcastRates.length == 0) return false;
        double avg = 0;
        for (double r : broadcastRates) avg += r;
        avg /= broadcastRates.length;
        if (avg > 0.3 && !longTerm.isEmpty()) {
            // 中枢活跃 → 长期记忆轻微强化 (再巩固)
            for (LongTermItem item : longTerm.values()) {
                item.strength = Math.min(1.0, item.strength + 0.005);
            }
            return true;
        }
        return false;
    }

    /** 睡眠巩固: 全部中期条目重放 → 短期转移 + 高频条目转长期 */
    public void sleepConsolidate(WorkingMemory workingMemory) {
        // 中期条目在睡眠中重放 (转存长期)
        for (EpisodicItem item : new ArrayList<>(episodic)) {
            item.strength = 1.0;
            if (item.occurrences >= 2) consolidateToLongTerm(item);
        }
        // 短期工作记忆内容也沉淀为中期
        for (int i = 0; i < workingMemory.capacity(); i++) {
            if (workingMemory.strength(i) > 0.1) {
                addEpisodic("wm_" + i, workingMemory.readSlot(i), 0.5);
            }
        }
    }

    /** 时间流逝: 中期记忆衰减 (未巩固的遗忘) */
    public void tick(double dtMs) {
        timeMs += dtMs;
        for (int i = episodic.size() - 1; i >= 0; i--) {
            EpisodicItem item = episodic.get(i);
            item.strength *= Math.exp(-dtMs / episodicTauMs);
            if (item.strength < 0.05) episodic.remove(i);  // 中期遗忘
        }
        // 长期记忆衰减极慢 (几乎不遗忘)
        for (LongTermItem item : longTerm.values()) {
            item.strength *= Math.exp(-dtMs / (365 * 24 * 3600 * 1000.0));  // 1年
        }
    }

    /** 检索: 标签在长期记忆中? */
    public boolean inLongTerm(String label) {
        return longTerm.containsKey(label);
    }

    /** 检索: 特征向量最匹配的长期记忆 */
    public String recall(double[] query) {
        String best = null;
        double bestSim = 0;
        for (LongTermItem item : longTerm.values()) {
            double sim = cosine(item.prototype, query);
            if (sim > bestSim) { bestSim = sim; best = item.label; }
        }
        return bestSim > 0.6 ? best : null;
    }

    public int episodicCount() { return episodic.size(); }
    public int longTermCount() { return longTerm.size(); }
    public Map<String, LongTermItem> longTerm() { return longTerm; }
    public List<EpisodicItem> episodic() { return episodic; }

    /** 记忆层级摘要 (APK 显示) */
    public String summary() {
        return String.format("📚 记忆: 情景(中期)%d条 | 语义(长期)%d条",
                episodic.size(), longTerm.size());
    }

    private static double cosine(double[] a, double[] b) {
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
