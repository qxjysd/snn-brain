package io.brainx.core;

import java.util.List;

/**
 * 想象发声 / 内心独白 (Imagined Speech)。
 * 来源: "Imagined Speech Reconstruction with 3D Neural Metabolism and Large
 * Language Model Integration" (bioRxiv 2025.07.30.667805).
 *
 * 论文核心: 想象语音 (ISR) = 不依赖外部听觉输入, 大脑凭内部神经活动
 * (EEG 记录到想象语音音素) 重建语音; 且**代谢成本**是核心约束 —
 * 想象发声的神经代谢活动比实际发声低 (功能成像证据)。
 *
 * 本实现 (机制级复现):
 *   - 内心独白触发: 大脑内部状态 (联想记忆/概念激活/意识广播) 达到阈值时,
 *     从 VoiceLearner 模板库"想象"出语音内容 — 不经过麦克风输入
 *   - 代谢成本模型: 想象发声代谢成本 = 基线 + 强度×单位成本;
 *     实际发声额外叠加声带肌肉成本 → 想象 < 实说 (论文核心论点)
 *   - 能量预算: 大脑按代谢预算分配"想象 vs 实说", 想象优先 (省能量,
 *     对应人脑默认模式网络/内心独白的低代谢特征)
 *   - 与 VoiceLearner 的关系: 模板库是"会说的话", 本类是"想说的话"
 *     (默读/预演), 可升级为实际发声 (speak 决策由 Brain 做)
 */
public class ImaginedSpeech {
    // 代谢成本参数 (相对单位, 论文: 想象 < 实说的代谢差)
    private static final double BASE_METABOLIC = 0.3;    // 静息基线
    private static final double IMAGINE_PER_UNIT = 0.15; // 每单位想象强度成本
    private static final double VOCALIZE_EXTRA = 0.5;    // 实说叠加肌肉成本

    private final VoiceLearner voiceLearner;
    private double metabolicCost = BASE_METABOLIC;
    private double energyBudget = 10.0;      // 代谢预算
    private int imaginesCount = 0;
    private String lastImagined = "";
    private double lastIntensity = 0.0;
    private long cooldownMs = 0;

    public ImaginedSpeech(VoiceLearner voiceLearner) {
        this.voiceLearner = voiceLearner;
    }

    /**
     * 触发一次内心独白: 凭内部概念/联想激活"想象"语音。
     *
     * @param concept        激活的概念 (联想内容, 如 "概念#3")
     * @param activation     激活强度 0..1 (超过阈值才触发)
     * @param nowMs          当前时间 (用于冷却)
     * @return 想象出的语音文本 (空串 = 未触发)
     */
    public String imagine(String concept, double activation, long nowMs) {
        if (nowMs < cooldownMs) return "";
        if (activation < 0.45) return "";           // 强度阈值 (弱激活不触发)
        if (voiceLearner == null || voiceLearner.templateCount() == 0) return "";
        // 从模板库想象语音: 选模板轮转 → 内容=概念 (内心独白)
        List<VoiceLearner.VoiceTemplate> lib = voiceLearner.library();
        VoiceLearner.VoiceTemplate t = lib.get(imaginesCount % lib.size());
        double intensity = Math.min(1.0, activation * (0.7 + 0.3 * t.heardCount / 3.0));
        // 代谢成本: 想象 = 基线 + 强度×单位成本 (论文: 低于实说)
        metabolicCost = BASE_METABOLIC + intensity * IMAGINE_PER_UNIT;
        lastIntensity = intensity;
        lastImagined = concept;
        imaginesCount++;
        cooldownMs = nowMs + 4000 + (long) (3000 * Math.random());  // 4-7s 冷却
        return concept;
    }

    /**
     * 实际发声的代谢成本 (对比: 想象成本 < 实说成本, 论文核心)。
     */
    public double vocalizeCost(double intensity) {
        return BASE_METABOLIC + intensity * IMAGINE_PER_UNIT + VOCALIZE_EXTRA;
    }

    /** 当前内心独白的代谢成本 */
    public double metabolicCost() { return metabolicCost; }

    /** 能量预算内的"想象 vs 实说"决策: true=想象更划算 (默认优先) */
    public boolean shouldImagine(double intensity) {
        return imagineCost(intensity) < vocalizeCost(intensity);
    }

    public double imagineCost(double intensity) {
        return BASE_METABOLIC + intensity * IMAGINE_PER_UNIT;
    }

    /** 设定代谢预算 (外部能量状态, 如睡眠/清醒) */
    public void setEnergyBudget(double budget) { this.energyBudget = budget; }

    public double energyBudget() { return energyBudget; }

    /** 能耗状态: 预算紧张 → 更依赖想象 (省能量, 人脑特征) */
    public double imaginePreference() {
        double costRatio = metabolicCost / Math.max(1e-6, energyBudget);
        return Math.min(1.0, Math.max(0.2, 1.0 - costRatio * 0.5));
    }

    public int imaginesCount() { return imaginesCount; }
    public String lastImagined() { return lastImagined; }
    public double lastIntensity() { return lastIntensity; }

    /** 是否有内心独白内容 (供 UI: 大脑在想什么) */
    public boolean hasInnerSpeech() { return !lastImagined.isEmpty(); }

    /** 摘要: 代谢成本/想象次数/能量偏好 */
    public String summary() {
        return String.format("内心独白: %d次 代谢%.2f 预算%.1f 想象偏好%.0f%%",
                imaginesCount, metabolicCost, energyBudget, imaginePreference() * 100);
    }

    public void reset() {
        metabolicCost = BASE_METABOLIC;
        energyBudget = 10.0;
        imaginesCount = 0;
        lastImagined = "";
        lastIntensity = 0.0;
        cooldownMs = 0;
    }
}
