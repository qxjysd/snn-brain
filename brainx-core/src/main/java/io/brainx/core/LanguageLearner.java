package io.brainx.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 语言学习模块 —— 从模仿到自主语言 (鹦鹉学舌 → 表达自我)。
 *
 * 理论依据 (《醉醺醺的脑科学》第23篇 声音识别 + 发展语言学):
 *   - 婴儿学语: 先模仿听到的语音 (发音模板), 再建立语音↔概念关联,
 *     最后用语言表达内在状态 (自主生成)
 *   - 模仿期: 输出 = 回放听到的发音 (鹦鹉学舌)
 *   - 理解期: 语音模板 ↔ 概念/词关联 (理解他人)
 *   - 自主期: 从内在状态 (记忆/好奇/情绪) 组合已学词生成话语
 *
 * 实现:
 *   - voiceMemory: 词 → 声学模板 (发音记忆)
 *   - 三阶段: 模仿 → 理解 → 自主 (由经验自然推进)
 *   - speak(): 按阶段输出 — 模仿(回放) / 自主(组合生成)
 */
public class LanguageLearner {
    /** 语言阶段 (自然成长, 无年龄) */
    public enum LangStage {
        模仿期("🗣️ 模仿期", 0),   // 鹦鹉学舌: 回放听到的语音
        理解期("👂 理解期", 1),   // 语音↔概念关联: 听懂他人
        自主期("💬 自主期", 2);    // 组合词汇表达内在状态
        public final String name; final int level;
        LangStage(String n, int l) { this.name = n; this.level = l; }
    }

    /** 语音记忆: 词 → 声学特征模板 */
    private final Map<String, double[]> voiceMemory = new HashMap<>();
    /** 听过的语音次数 (经验量) */
    private int heardCount = 0;
    /** 说过的次数 */
    private int spokenCount = 0;
    /** 当前阶段 */
    private LangStage stage = LangStage.模仿期;

    /**
     * 学习语音: 听到语音 + 知道对应词 → 存储发音模板。
     * @param acousticFeatures 声学特征 (音高/频谱等)
     * @param word 对应词
     */
    public void learnSpokenWord(double[] acousticFeatures, String word) {
        heardCount++;
        voiceMemory.put(word, acousticFeatures.clone());
        // 自然成长: 掌握语音越多 → 阶段推进
        updateStage();
    }

    /** 听到语音 (不知道对应词): 建立语音记忆但不关联 */
    public void hearSpeech(double[] acousticFeatures) {
        heardCount++;
        updateStage();
    }

    /** 当前语言阶段 (由经验自然推进) */
    public LangStage stage() { return stage; }

    /**
     * 自主生成话语: 从内在状态组合已学词汇 (成熟后)。
     * @param emotion 当前情绪标签
     * @param curiosity 好奇心 (0-1)
     * @param recentMemory 最近经历
     * @param knownWords 已学词表
     */
    public String generateSpeech(String emotion, double curiosity, String recentMemory, List<String> knownWords) {
        // 自主期: 组合已学词表达内在状态
        if (stage.level >= LangStage.自主期.level && !knownWords.isEmpty()) {
            String w = knownWords.get(0);
            StringBuilder sb = new StringBuilder();
            // 情绪表达
            if (emotion.contains("开心") || emotion.contains("兴奋")) {
                sb.append("我很开心！");
            } else if (emotion.contains("好奇")) {
                sb.append("这是什么？我好想知道！");
            } else if (emotion.contains("困惑")) {
                sb.append("我不太明白...");
            } else if (emotion.contains("沮丧")) {
                sb.append("我有点难过...");
            } else {
                sb.append("我看到").append(w).append("。");
            }
            // 好奇心补充
            if (curiosity > 0.7) sb.append("我想认识更多新东西！");
            // 记忆回顾
            if (recentMemory != null && !recentMemory.isEmpty() && stage.level >= LangStage.自主期.level) {
                sb.append("我记得").append(recentMemory).append("。");
            }
            return sb.toString();
        }
        // 理解期: 简单回应
        if (stage.level >= LangStage.理解期.level && !knownWords.isEmpty()) {
            return "这是" + knownWords.get(0) + "吗？";
        }
        // 模仿期: 无自主 (等待模仿素材)
        return "";
    }

    /** 模仿输出: 回放已学语音 (鹦鹉学舌) */
    public String mimic() {
        if (voiceMemory.isEmpty()) return "";
        // 回放最近学的语音对应词 (模仿发音)
        String last = new ArrayList<>(voiceMemory.keySet()).get(voiceMemory.size() - 1);
        spokenCount++;
        return last;
    }

    /** 说话: 按阶段输出 (模仿→自主) */
    public String speak(String emotion, double curiosity, String recentMemory, List<String> knownWords) {
        spokenCount++;
        if (stage == LangStage.模仿期) {
            String m = mimic();
            if (!m.isEmpty()) return "🗣️ " + m + "！" + m + "！";  // 鹦鹉学舌重复
            return "";
        }
        return generateSpeech(emotion, curiosity, recentMemory, knownWords);
    }

    private void updateStage() {
        // 自然成长: 掌握≥2词+听过多次 → 理解期; 掌握≥4词 → 自主期
        if (voiceMemory.size() >= 4) {
            stage = LangStage.自主期;
        } else if (voiceMemory.size() >= 2 || heardCount >= 5) {
            stage = LangStage.理解期;
        }
    }

    /** 已学语音数 */
    public int voiceCount() { return voiceMemory.size(); }
    public int heardCount() { return heardCount; }
    public int spokenCount() { return spokenCount; }

    /** 摘要 (APK 显示) */
    public String summary() {
        return String.format("🗣️ %s | 语音%d词 | 听过%d次 | 说过%d次",
                stage.name, voiceMemory.size(), heardCount, spokenCount);
    }
}
