package io.brainx.core;

import java.util.ArrayList;
import java.util.List;

/**
 * 自我意识模块 —— 元认知 + 镜像自我识别 + 发展阶段 + 自我叙事。
 *
 * 理论依据 (用户提供框架 + 认知科学):
 *   - 一、生理基础: 前额叶皮层负责元认知 (对思考的思考);
 *     多脑区协同整合内感受与外部感知 → 区分"自我"与"外界"
 *   - 二、个体发生: 感知萌芽 → 符号联结 → 反思成熟 (由经验自然推进,
 *     不设年龄 — 一切由实际学习经验自然成长)
 *   - 三、社会建构: 他人评价反馈 → 社会自我 (评价调整自我认知)
 *   - 四、机制解释: 知觉漂移/镜像测试 (Gallup 1970) + 全局整合 (GWT/IIT)
 *
 * 实现:
 *   - 镜像自我: 识别反馈(对/错) → 自我模型更新 (知道自己"是谁/多准")
 *   - 元认知: 监测置信度, 区分"知道/不知道" → 不知道触发好奇/探索
 *   - 发展阶段: 由实际经验自然成长 (学的词数/准确率/元认知 → 推进,
 *     学得快成长快, 学得慢成长慢 — 不预设年龄)
 *   - 自我叙事: 记忆条目串联成"我的经历"文本
 *   - 社会自我: 教育者反馈(奖惩) → 自我评价调整
 */
public class SelfAwareness {
    /**
     * 发展阶段 —— 自然成长 (无年龄预设)。
     * 由实际学习经验推进: 萌芽感知 → 符号联结 → 反思成熟。
     */
    public enum DevStage {
        感知萌芽("🌱 感知萌芽", 0),   // 开始区分自我/外界 (初步镜像感知)
        符号联结("🔗 符号联结", 1),   // 语言符号连接概念, 心理表征形成
        反思成熟("🪞 反思成熟", 2);    // 元认知成熟, 自我反思与叙事
        public final String name; final int level;
        DevStage(String n, int l) { this.name = n; this.level = l; }
    }

    // 元认知状态
    private double selfConfidence = 0.3;   // 自我评价 (知道自己多准)
    private int totalTests = 0, correctTests = 0;
    private int mirrorTests = 0;           // 镜像测试次数 (识别反馈)
    private boolean selfRecognized = false; // 是否通过镜像测试 (认出"自我")

    // 自我叙事 (经历记录)
    private final List<String> narrative = new ArrayList<>();
    private final int narrativeCap = 20;

    public SelfAwareness() {}

    /**
     * 镜像测试/识别反馈: 每次识别后更新自我模型。
     * @param wasCorrect  识别是否正确 (教育者反馈 = 社会评价)
     * @param confidence  自身置信度
     */
    public void mirrorTest(boolean wasCorrect, double confidence) {
        totalTests++;
        mirrorTests++;
        if (wasCorrect) correctTests++;
        // 自我评价更新: 正确率 + 置信度校准
        double accuracy = (double) correctTests / Math.max(1, totalTests);
        selfConfidence = 0.5 * accuracy + 0.5 * confidence;
        // 镜像自我识别: 连续 5 次反馈后建立"自我"概念 (区分自我/外界)
        if (mirrorTests >= 5 && accuracy > 0.5) selfRecognized = true;
        // 自我叙事记录
        addNarrative(wasCorrect
                ? String.format("我认出了它 (置信度%.0f%%)", confidence * 100)
                : String.format("我认错了 (置信度%.0f%%)", confidence * 100));
    }

    /**
     * 元认知: 评估"我知道这个吗" (对思考的思考)。
     * 置信度低 + 未学过 → "不知道" → 触发好奇/探索
     */
    public boolean knows(String label, double confidence) {
        return confidence >= 0.5;
    }

    /** 元认知反思: 输出当前自我认知状态 (自然成长, 无年龄标签) */
    public String metacognition() {
        if (!selfRecognized) return "我还分不清'我'和'世界'... (自我意识萌芽中)";
        return String.format("我知道自己大约有%.0f%%的把握认识东西", selfConfidence * 100);
    }

    // ============ 二阶元认知: 思考"思考的思考" (v5.4) ============

    /**
     * 二阶元认知 — 置信度校准监控 (思考"我对自己思考的判断是否准确")。
     *
     * 神经科学依据 (前额叶元认知/内省):
     *   - 一级元认知: 我知道"我对X有多大把握" (现有 metacognition)
     *   - 二阶元认知: 我知道"我的把握判断本身可不可靠" — 校准监控
     *     (confidence calibration: 我说 90% 把握, 实际是否 90% 对?)
     *   - 校准偏差检测 = 意识到自己过度自信/自信不足 = 真正的递归思考
     */
    public String metaMetacognition() {
        if (totalTests < 8) {
            // 经验不足: 二阶监控尚未建立 (成长中)
            return "我还不太清楚自己的判断准不准... (内省校准中)";
        }
        double accuracy = accuracy();
        // 校准偏差: 自我评价 - 实际正确率 (正=高估, 负=低估)
        double bias = selfConfidence - accuracy;
        if (bias > 0.2) {
            return "我注意到我有时过于自信——说'很有把握'其实没那么准，我要更谨慎";
        } else if (bias < -0.2) {
            return "我有时低估了自己——实际比我想的更准，我要更相信判断";
        }
        return "我对自己把握的判断基本准确——我知道自己知道什么，也知道自己不知道什么";
    }

    /** 反思链 (思考→思考思考→思考思考的思考): 三层递归表达 */
    public String reflectOnThinking(String currentThought) {
        StringBuilder sb = new StringBuilder();
        // 一级: 思考
        sb.append("我在想「").append(currentThought).append("」");
        if (!selfRecognized) return sb.toString();
        // 二级: 思考思考 (元认知)
        sb.append("；我知道我对它有").append(String.format("%.0f", selfConfidence * 100)).append("%的把握");
        if (totalTests < 8) return sb.toString();
        // 三级: 思考思考的思考 (二阶元认知: 校准监控)
        double bias = selfConfidence - accuracy();
        if (bias > 0.2) sb.append("；但我发现我的把握有时偏高，正在校准");
        else if (bias < -0.2) sb.append("；我意识到自己常低估，正在校准");
        else sb.append("；我的把握判断经过检验，是可靠的");
        return sb.toString();
    }

    /**
     * 发展阶段 —— 自然成长 (无年龄预设)。
     * 由实际学习经验推进: 词的掌握 / 识别准确率 / 镜像自我识别。
     * 学得快成长快, 学得慢成长慢 — 一切由经验自然决定。
     * @param learnedWords 已学词数 (经验量)
     * @param accuracy     识别准确率 (0-1, 熟练度)
     * @param selfRecognized 是否通过镜像测试 (自我识别)
     */
    public DevStage devStage(int learnedWords, double accuracy, boolean selfRecognized) {
        // 自然成长条件 (经验驱动, 非年龄):
        // 萌芽: 刚开始积累经验 (哪怕很少)
        // 符号联结: 掌握足够多词 + 有一定准确率 (语言符号连接概念)
        // 反思成熟: 词多 + 准确率高 + 自我识别通过 (元认知成熟)
        if (!selfRecognized) {
            return DevStage.感知萌芽;
        }
        if (learnedWords >= 4 && accuracy >= 0.6) {
            return DevStage.反思成熟;
        }
        if (learnedWords >= 2) {
            return DevStage.符号联结;
        }
        return DevStage.感知萌芽;
    }

    /** 自我叙事: 回顾我的经历 (复盘过去) */
    public String selfNarrative() {
        if (narrative.isEmpty()) return "我还没有属于自己的故事...";
        StringBuilder sb = new StringBuilder("📖 我的故事: ");
        for (String n : narrative) sb.append(n).append("; ");
        return sb.toString();
    }

    /** 社会自我: 教育者评价如何塑造自我认知 */
    public String socialSelf() {
        if (totalTests == 0) return "还没有人评价过我";
        double praise = (double) correctTests / totalTests;
        if (praise > 0.8) return "大家都说我学得快，我是个聪明的孩子";
        if (praise > 0.5) return "我还在进步中，大家说我有潜力";
        return "我学得慢，但我在努力";
    }

    private void addNarrative(String event) {
        narrative.add(0, event);
        if (narrative.size() > narrativeCap) narrative.remove(narrative.size() - 1);
    }

    public double selfConfidence() { return selfConfidence; }
    public boolean selfRecognized() { return selfRecognized; }
    public int totalTests() { return totalTests; }
    public int correctTests() { return correctTests; }
    public double accuracy() { return totalTests == 0 ? 0 : (double) correctTests / totalTests; }
}
