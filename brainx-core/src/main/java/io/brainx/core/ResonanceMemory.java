package io.brainx.core;

import java.util.HashMap;
import java.util.Map;

/**
 * 频率共振记忆层 —— 记忆以频率形式存储与检索 (类脑核心)。
 *
 * 神经科学依据 (记忆=频率生成):
 *   - 神经元集群以振荡频率编码信息, 记忆存储为突触权重的频率选择性
 *   - 记忆检索 = 注入查询频率模式 → 与存储频率共振 → 激活对应记忆
 *     (类似 Hopfield 网络的神经实现; 海马θ振荡与记忆检索相关)
 *   - 写入记忆 = 建立/强化该记忆的特征频率 (突触权重调谐)
 *   - 频率带宽决定分辨率: 窄带=精确区分, 宽带=泛化
 *
 * 实现:
 *   - 每条记忆 = 一个共振频率 + 强度 (突触调谐)
 *   - 写入: 强化对应频率的共振强度
 *   - 检索: 查询频率谱 → 计算与每条记忆的共振匹配 → 最强激活
 *   - 干扰: 频率相近的记忆互相干扰 (类脑的混淆现象)
 */
public class ResonanceMemory implements PulseModule {
    /** 记忆条目: 标签 → 共振频率 */
    private final Map<String, Double> memoryFreq = new HashMap<>();
    /** 记忆强度: 标签 → 共振强度 (0-1) */
    private final Map<String, Double> memoryStrength = new HashMap<>();
    /** 频率带宽 (Hz): 窄=精确, 宽=泛化 */
    private final double bandwidthHz;
    /** 频率范围 */
    private final FrequencyWave freq;
    /** 最近检索: 标签 → 共振度 (可视化) */
    private final Map<String, Double> lastResonance = new HashMap<>();

    public ResonanceMemory(double bandwidthHz) {
        this.bandwidthHz = bandwidthHz;
        this.freq = FrequencyWave.memoryRange();
    }

    public static ResonanceMemory defaultParams() {
        return new ResonanceMemory(1.5);  // 1.5Hz 带宽
    }

    /** 写入记忆: 分配/强化特征频率 */
    public void write(String label, double intensity) {
        double hz = memoryFreq.getOrDefault(label, freq.featureHz(memoryFreq.size(), 16));
        memoryFreq.put(label, hz);
        double s = memoryStrength.getOrDefault(label, 0.0) + intensity;
        memoryStrength.put(label, Math.min(1.0, s));
    }

    /** 遗忘/削弱记忆 (用进废退) */
    public void forget(String label, double amount) {
        double s = memoryStrength.getOrDefault(label, 0.0) - amount;
        if (s <= 0.02) {
            memoryFreq.remove(label);
            memoryStrength.remove(label);
        } else {
            memoryStrength.put(label, s);
        }
    }

    /**
     * 频率谱检索: 查询强度 → 计算与各记忆的共振。
     * @param queryLabel 查询标签 (其频率来自存储)
     * @param queryIntensity 查询强度 (0-1)
     * @return [最佳标签, 共振度]
     */
    public String[] retrieve(String queryLabel, double queryIntensity) {
        double queryHz = freq.intensityToHz(queryIntensity);
        return retrieveByFreq(queryHz);
    }

    /**
     * 按频率检索: 查询频率 → 最强共振记忆。
     * @param queryHz 查询频率 (Hz)
     * @return [标签, 共振度0-1] 或 ["", 0]
     */
    public String[] retrieveByFreq(double queryHz) {
        String best = "";
        double bestR = 0;
        for (Map.Entry<String, Double> e : memoryFreq.entrySet()) {
            double r = FrequencyWave.resonance(queryHz, e.getValue(), bandwidthHz)
                    * memoryStrength.getOrDefault(e.getKey(), 0.0);
            lastResonance.put(e.getKey(), r);
            if (r > bestR) { bestR = r; best = e.getKey(); }
        }
        return new String[]{best, String.valueOf(bestR)};
    }

    /** 记忆数 */
    public int size() { return memoryFreq.size(); }
    /** 记忆的共振频率 */
    public double frequencyOf(String label) { return memoryFreq.getOrDefault(label, 0.0); }
    /** 记忆强度 */
    public double strengthOf(String label) { return memoryStrength.getOrDefault(label, 0.0); }
    /** 最近检索共振度 (可视化) */
    public Map<String, Double> lastResonance() { return lastResonance; }
    /** 频率源 */
    public FrequencyWave freq() { return freq; }

    /** 摘要 (APK 显示) */
    public String summary() {
        StringBuilder sb = new StringBuilder(String.format("🎵 频率共振记忆: %d条\n", memoryFreq.size()));
        for (Map.Entry<String, Double> e : memoryFreq.entrySet()) {
            sb.append(String.format("  %s → %.1fHz (强度%.0f%%)\n",
                    e.getKey(), e.getValue(), memoryStrength.getOrDefault(e.getKey(), 0.0) * 100));
        }
        return sb.toString();
    }

    // ============ PulseModule: 中枢脉冲联动 ============

    @Override public String moduleName() { return "共振记忆"; }
    @Override public int pulseDim() { return 2; }  // [记忆强度, 检索激活]

    /** 发射脉冲: [记忆总强度, 最近检索共振] (记忆状态→中枢) */
    @Override
    public double[] emitPulses() {
        double total = 0;
        for (double s : memoryStrength.values()) total += s;
        double avgStrength = memoryStrength.isEmpty() ? 0 : total / memoryStrength.size();
        double lastRes = 0;
        for (double r : lastResonance.values()) lastRes = Math.max(lastRes, r);
        return new double[]{avgStrength, lastRes};
    }

    /** 接收中枢广播: 中枢活跃 → 记忆检索增强 (全局整合促进回忆) */
    @Override
    public boolean receiveBroadcast(double[] broadcastRates) {
        if (broadcastRates.length == 0) return false;
        double avg = 0;
        for (double r : broadcastRates) avg += r;
        avg /= broadcastRates.length;
        if (avg > 0.3 && !memoryStrength.isEmpty()) {
            // 中枢活跃 → 记忆强度轻微增强 (回忆促进)
            for (Map.Entry<String, Double> e : memoryStrength.entrySet()) {
                memoryStrength.put(e.getKey(), Math.min(1.0, e.getValue() + 0.005));
            }
            return true;
        }
        return false;
    }
}
