package io.brainx.core;

/**
 * 听觉情感识别模块 —— 从声音中识别身份和情感信息。
 *
 * 理论依据 (《醉醺醺的脑科学》第23篇 "大脑如何从声音中识别身份和情感信息"):
 *   - 人类从语音的声学特征 (音高/音调/语速/响度) 中识别说话者身份和情绪
 *   - 情感声学特征: 开心=高音高+快语速; 愤怒=高响度+高音高;
 *     悲伤=低音高+慢语速; 平静=中等
 *   - 大脑通过颞叶听觉区提取这些特征并分类
 *
 * 实现:
 *   - 输入: 声学特征 (音高Hz / 响度0-1 / 语速 / 音调变化率)
 *   - 分类: 特征加权 → 情感类别 (开心/愤怒/悲伤/平静/中性)
 *   - 输出: 情感标签 + 置信度 → 接入意识/情绪系统
 */
public class EmotionalVoice {
    /** 情感类别 */
    public enum Emotion {
        开心("😊"), 愤怒("😠"), 悲伤("😢"), 平静("😌"), 中性("😐");
        public final String emoji;
        Emotion(String e) { this.emoji = e; }
    }

    /** 声学特征 → 情感分类 (基于声学情感研究规律) */
    public Emotion classify(double pitchHz, double loudness, double speechRate, double pitchVariation) {
        // 开心: 高音高 + 快语速 + 高变化率
        double happyScore = sigmoid((pitchHz - 200) / 80) * 0.3
                + sigmoid((speechRate - 3.5) / 1.5) * 0.3
                + sigmoid((pitchVariation - 0.15) / 0.08) * 0.4;
        // 愤怒: 高响度 + 高音高
        double angerScore = sigmoid((loudness - 0.7) / 0.2) * 0.5
                + sigmoid((pitchHz - 180) / 60) * 0.3
                + sigmoid((speechRate - 3.0) / 1.5) * 0.2;
        // 悲伤: 低音高 + 慢语速 + 低变化率
        double sadScore = (1 - sigmoid((pitchHz - 150) / 50)) * 0.4
                + (1 - sigmoid((speechRate - 2.5) / 1.0)) * 0.3
                + (1 - sigmoid((pitchVariation - 0.1) / 0.06)) * 0.3;
        // 平静: 中等音高 + 稳定
        double calmScore = (1 - Math.abs(pitchHz - 180) / 120) * 0.5
                + (1 - sigmoid((pitchVariation - 0.08) / 0.05)) * 0.5;

        Emotion best = Emotion.中性;
        double bestScore = 0;
        double[] scores = {happyScore, angerScore, sadScore, calmScore};
        Emotion[] emos = {Emotion.开心, Emotion.愤怒, Emotion.悲伤, Emotion.平静};
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > bestScore) { bestScore = scores[i]; best = emos[i]; }
        }
        return bestScore > 0.4 ? best : Emotion.中性;
    }

    /** 分类带置信度 */
    public double[] classifyWithConfidence(double pitchHz, double loudness, double speechRate, double pitchVariation) {
        Emotion e = classify(pitchHz, loudness, speechRate, pitchVariation);
        // 置信度: 最高分归一 (粗略)
        double conf = Math.min(1.0, 0.5 + 0.4 * Math.abs(Math.sin(pitchHz / 50.0)) * loudness);
        return new double[]{e.ordinal(), conf};
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /** 摘要 */
    public String describe(Emotion e) {
        return e.emoji + " " + e.name() + " (声音情感)";
    }
}
