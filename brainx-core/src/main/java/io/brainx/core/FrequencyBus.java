package io.brainx.core;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 频率总线 —— 全脑频率统一联动 (类脑核心)。
 *
 * 设计目标: "任何记忆都是频率生成, 整体联动"
 *   所有认知/记忆模块通过频率交互, 而非孤立数据结构:
 *   - 每条记忆/状态 = 一个共振频率 + 强度
 *   - 模块间传递 = 频率信号 (某模块注入频率 → 其他模块共振响应)
 *   - 脑电节律 (θ/α/γ) 是全局时钟, 同步各模块活动
 *
 * 频率语义约定:
 *   - θ (4-8Hz):   记忆检索/空间导航 (海马)
 *   - α (8-13Hz):  静息/默认网络/回忆
 *   - β (13-30Hz): 警觉/思考
 *   - γ (30-100Hz): 注意绑定/跨模态整合/意识内容
 *
 * 实现:
 *   - 记录每个模块的主导频率
 *   - 频率注入: 模块A发出频率 → 总线广播 → 各模块按共振响应
 *   - 共振强度: 模块频率与总线频率越接近, 耦合越强 (整体联动)
 */
public class FrequencyBus {
    /** 模块频率状态: 模块名 → [当前频率Hz, 强度0-1] */
    private final Map<String, double[]> moduleFreq = new LinkedHashMap<>();
    /** 总线主导频率 (当前全局最强信号) */
    private double dominantHz = 10.0;  // 默认 α
    /** 总线主导强度 */
    private double dominantStrength = 0;
    /** 总线时间 (ms, 波形相位) */
    private long timeMs = 0;

    /** 模块注册并上报频率 */
    public void report(String module, double hz, double strength) {
        moduleFreq.put(module, new double[]{hz, Math.max(0, Math.min(1, strength))});
        updateDominant();
    }

    /** 更新主导频率 (强度加权最强者) */
    private void updateDominant() {
        String best = null;
        double bestS = 0;
        for (Map.Entry<String, double[]> e : moduleFreq.entrySet()) {
            if (e.getValue()[1] > bestS) { bestS = e.getValue()[1]; best = e.getKey(); }
        }
        if (best != null && bestS > 0) {
            dominantHz = moduleFreq.get(best)[0];
            dominantStrength = bestS;
        }
    }

    /** 模块与总线当前频率的共振度 (0-1): 越接近越联动 */
    public double coupling(String module) {
        double[] f = moduleFreq.get(module);
        if (f == null) return 0;
        return FrequencyWave.resonance(dominantHz, f[0], 3.0) * f[1];
    }

    /** 频率注入: 外部信号注入总线 (如意识广播/多巴胺调制) */
    public void inject(double hz, double strength) {
        if (strength > dominantStrength) {
            dominantHz = hz;
            dominantStrength = Math.min(1.0, strength);
        }
    }

    /** 总线主导频率 */
    public double dominantHz() { return dominantHz; }
    public double dominantStrength() { return dominantStrength; }
    public Map<String, double[]> moduleFreq() { return moduleFreq; }

    /** 频率带标签 */
    public static String bandName(double hz) {
        if (hz < 8) return "θ记忆";
        if (hz < 13) return "α静息";
        if (hz < 30) return "β思考";
        return "γ绑定";
    }

    /** 总线状态摘要 (APK 显示) */
    public String summary() {
        StringBuilder sb = new StringBuilder(String.format(
                "🔀 频率总线: 主导 %.1fHz (%s) 强度%.0f%%\n", dominantHz, bandName(dominantHz), dominantStrength * 100));
        for (Map.Entry<String, double[]> e : moduleFreq.entrySet()) {
            sb.append(String.format("  %s: %.1fHz(%.0f%%)\n", e.getKey(), e.getValue()[0], e.getValue()[1] * 100));
        }
        return sb.toString();
    }

    /** 时间推进 (波形同步) */
    public void tick(long dtMs) { timeMs += dtMs; }
    public long timeMs() { return timeMs; }
}
