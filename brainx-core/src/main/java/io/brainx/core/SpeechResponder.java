package io.brainx.core;

/**
 * 语音对话与逗乐互动引擎 —— 听到语音回复, 可被人逗乐。
 *
 * 设计 (陪伴式交互, 自然触发无按钮):
 *   - 听到语音 → 关键词理解 → 对应回复 (早期模仿, 成熟理解回应)
 *   - 听到笑声/开心语音 → 被逗乐 → 笑 (开心度上升)
 *   - 对话风格随语言阶段演进: 模仿期重复, 理解期回应, 自主期表达
 */
public class SpeechResponder {
    /** 开心度 (0-100, 被逗乐上升, 时间回落) */
    private int happiness = 50;
    /** 大笑计数 */
    private int laughs = 0;
    /** 被逗乐次数 */
    private int teased = 0;
    /** 上次被逗时间 (用于回落) */
    private long lastPlayMs = 0;

    /**
     * 听到语音 → 回复 (关键词理解)。
     * @param heardText 听到的文本 (语音识别结果或关键词)
     * @param knownWords 已学词表
     * @param langStage 语言阶段
     * @param mimicWord 模仿用词 (语言学习器的)
     * @return 回复文本 (空 = 无回应)
     */
    public String respond(String heardText, java.util.List<String> knownWords,
                          LanguageLearner.LangStage langStage, String mimicWord) {
        if (heardText == null || heardText.isEmpty()) return "";
        String t = heardText;

        // 逗乐检测: 笑声/夸赞/玩笑
        if (t.contains("哈") || t.contains("笑") || t.contains("可爱") || t.contains("逗")) {
            return playReact(1.0);
        }

        // 问候
        if (t.contains("你好") || t.contains("嗨") || t.contains("hello")) {
            return "你好！你好！";
        }

        // 叫名字/呼唤 → 回应
        if (t.contains("宝宝") || t.contains("过来") || t.contains("来")) {
            return "来啦！来啦！";
        }

        // 认识的东西 → 理解回应
        for (String w : knownWords) {
            if (t.contains(w)) {
                if (langStage.level >= LanguageLearner.LangStage.理解期.level) {
                    return "是" + w + "！我看到" + w + "了！";
                }
                return w + "！" + w + "！";  // 模仿期重复
            }
        }

        // 提问 → 好奇回应
        if (t.contains("什么") || t.contains("吗") || t.contains("？") || t.contains("?")) {
            return "咦？是什么呢？我好想知道！";
        }

        // 默认: 模仿期重复听到的, 成熟期礼貌回应
        if (langStage.level >= LanguageLearner.LangStage.自主期.level) {
            return "你说" + t + "，我听到了！";
        }
        return "";  // 早期听不懂不回应
    }

    /**
     * 被逗乐: 开心度上升 + 笑反应。
     * @param playStrength 逗乐强度 (0-1: 笑声/鬼脸/触摸)
     * @return 笑反应文本
     */
    public String playReact(double playStrength) {
        teased++;
        happiness = Math.min(100, happiness + (int) (playStrength * 30));
        laughs++;
        lastPlayMs = System.currentTimeMillis();
        if (happiness >= 85) {
            return "哈哈哈哈！好好玩！";
        }
        if (happiness >= 65) {
            return "嘻嘻！你逗我！";
        }
        return "嘿嘿～";
    }

    /** 开心度随时间回落 (自然情绪) */
    public void tick(long dtMs) {
        if (happiness > 50 && System.currentTimeMillis() - lastPlayMs > 60000) {
            happiness = Math.max(50, happiness - 1);
        }
    }

    /** 当前开心度 (0-100) */
    public int happiness() { return happiness; }
    public int laughs() { return laughs; }
    public int teased() { return teased; }

    /** 表情 (按开心度) */
    public String face() {
        if (happiness >= 85) return "😄 大笑";
        if (happiness >= 65) return "😊 开心";
        if (happiness >= 40) return "😌 平静";
        return "😐 无聊";
    }

    /** 摘要 */
    public String summary() {
        return String.format("😊 开心度%d | 被逗%d次 | 笑%d次",
                happiness, teased, laughs);
    }
}
