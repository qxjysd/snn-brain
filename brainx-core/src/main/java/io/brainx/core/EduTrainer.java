package io.brainx.core;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * 游戏化教育养成系统 —— 培养大脑模型 + 好奇心驱动探索。
 *
 * 🎮 养成要素:
 *   - XP 经验值 + 等级成长线 (学语期→词汇期→句子期→认知期→对话期)
 *   - 连击 streak: 连续答对加分, 答错清零
 *   - 成就系统: 首次探索/学会5词/10连击/好奇探险家...
 *   - 成长脑图: 等级越高大脑越活跃
 *
 * 🧠 好奇心机制 (对应婴儿探索学习 / 奖励预测误差):
 *   - 遇到"未知"(置信度低) → 好奇值上升 → 探索未知获得新奇奖励
 *   - 好奇心随时间和重复见已知衰减 → 驱动持续探索新事物
 *   - 好奇状态下探索 = 额外经验 + 稀有奖励
 */
public class EduTrainer {
    /** 学习阶段 (等级成长线) */
    public enum Stage { 学语期(0), 词汇期(1), 句子期(2), 认知期(3), 对话期(4);
        final int level; Stage(int l) { this.level = l; } }

    /** 情绪状态 (对应像素表情) */
    public enum Emotion {
        开心("😊", "开心"), 好奇("🤔", "好奇"), 兴奋("🤩", "兴奋"),
        困惑("😕", "困惑"), 沮丧("😢", "沮丧"), 平静("😐", "平静"),
        惊讶("😮", "惊讶"), 骄傲("😎", "骄傲");
        final String emoji; final String name;
        Emotion(String e, String n) { this.emoji = e; this.name = n; }
    }

    // 养成状态
    private int points = 0;
    private int xp = 0;
    private int level = 0;
    private int correctCount = 0;
    private int wrongCount = 0;
    private int streak = 0;
    private int bestStreak = 0;
    private int exploreCount = 0;
    private int newWordCount = 0;

    // 好奇心 (0-100)
    private double curiosity = 20.0;

    // 已学会的词 (未知检测用)
    private final Set<String> knownWords = new LinkedHashSet<>();
    private final Set<String> achievements = new LinkedHashSet<>();

    // 情绪状态
    private Emotion emotion = Emotion.平静;
    private Emotion lastEmotion = Emotion.平静;
    private int emotionHoldSteps = 0;   // 情绪保持帧数 (自然回落)

    private final Random rnd = new Random();
    private final List<String> inventory = new ArrayList<>();
    private final List<String> recentFeedback = new ArrayList<>();

    // 奖励物品池
    private static final String[] REWARDS = {
            "🌟 智慧之星", "🎖️ 学习勋章", "🧩 记忆拼图", "📚 知识书签",
            "🎨 想象力彩笔", "🔭 好奇望远镜", "🎵 旋律音符", "🧠 神经连接徽章"
    };
    // 稀有奖励 (好奇探索时)
    private static final String[] RARE_REWARDS = {
            "💎 认知宝石", "🏆 探索者奖杯", "🦉 智慧猫头鹰", "🌌 星云记忆",
            "🔮 灵感水晶", "🪐 好奇星球", "🎪 惊奇马戏团", "⚡ 突触闪电"
    };
    private static final String[] PRAISE = {
            "太棒了！你答对了！", "真聪明，大脑连接更强了！", "很好！继续加油！",
            "答对啦，奖励你一颗星！", "你真厉害，学会了新词！"
    };
    private static final String[] CURIOUS_LINES = {
            "咦？这是什么？我好想知道！", "没见过的东西！好奇心被点燃了！",
            "哇，新事物！探索它！", "这是什么呀？让我学习一下！"
    };
    private static final String[] CORRECTIONS = {
            "再想想，这个我们学过哦", "不对哦，我们来重新认识它",
            "差一点，再看看这张图", "没关系，大脑需要再练习一次"
    };

    // ============ 奖励/惩罚 ============

    /** 答对: 奖励 (+1点 +XP +连击, 每3次答对升级) */
    public Feedback reward() {
        return rewardWithModifier(1.0, null);
    }

    /** 探索未知: 好奇激励 (经验 x2 + 稀有奖励 + 好奇值上升) */
    public Feedback exploreUnknown(String newThing) {
        exploreCount++;
        if (!knownWords.contains(newThing)) {
            newWordCount++;
            knownWords.add(newThing);
        }
        double mod = 2.0 + curiosity / 100.0;  // 好奇越高奖励越大
        Feedback fb = rewardWithModifier(mod, newThing);
        // 好奇心奖励: 额外 +2 点
        points += 2;
        setEmotion(Emotion.兴奋);
        fb.message = "🔍 探索未知「" + newThing + "」！" + fb.message + " (+2好奇奖励)";
        checkAchievement("好奇探险家", exploreCount >= 5);
        checkAchievement("发现之旅", exploreCount >= 1);
        return fb;
    }

    private Feedback rewardWithModifier(double mod, String newThing) {
        correctCount++;
        streak++;
        bestStreak = Math.max(bestStreak, streak);
        int gain = (int) Math.round(1.0 * mod);
        points += gain;
        xp += (int) Math.round(10 * mod);
        if (newThing != null && !knownWords.isEmpty() && newWordCount >= 5) {
            checkAchievement("词汇大师", true);
        }
        // 升级: 每 30 XP 升一级
        while (xp >= 30 * (level + 1) && level < Stage.values().length - 1) {
            level++;
            xp -= 30 * level;
        }
        String reward = (mod > 1.5 && rnd.nextDouble() < 0.7)
                ? RARE_REWARDS[rnd.nextInt(RARE_REWARDS.length)]
                : REWARDS[rnd.nextInt(REWARDS.length)];
        inventory.add(reward);
        String praise = PRAISE[rnd.nextInt(PRAISE.length)];
        String msg = praise + " (+" + gain + "点, +" + (int)(10*mod) + "XP, 获得" + reward + ")";
        if (streak >= 3) msg += " 🔥连击x" + streak;
        recentFeedback.add(0, msg);
        if (recentFeedback.size() > 6) recentFeedback.remove(recentFeedback.size() - 1);
        checkAchievement("初露锋芒", correctCount >= 5);
        checkAchievement("连击高手", streak >= 5);
        checkAchievement("十连击", streak >= 10);
        // 情绪: 答对→开心, 连击/升级→兴奋
        if (level > 0 && correctCount % 3 == 0) setEmotion(Emotion.兴奋);
        else if (streak >= 5) setEmotion(Emotion.兴奋);
        else setEmotion(Emotion.开心);
        return new Feedback(msg, true, praise, reward);
    }

    /** 答错: 教育性纠正 (连击清零, 扣点) */
    public Feedback punish(String correctAnswer) {
        wrongCount++;
        streak = 0;
        points = Math.max(0, points - 1);
        String correction = CORRECTIONS[rnd.nextInt(CORRECTIONS.length)];
        String msg = correction + " (正确是: " + correctAnswer + ")";
        recentFeedback.add(0, msg);
        if (recentFeedback.size() > 6) recentFeedback.remove(recentFeedback.size() - 1);
        setEmotion(Emotion.沮丧);
        return new Feedback(msg, false, correction, null);
    }

    // ============ 好奇心 ============

    /** 每帧调用: 好奇心随时间衰减 (需要持续探索维持) */
    public void tickCuriosity(double dtSec) {
        curiosity = Math.max(5, curiosity - 0.8 * dtSec);
    }

    /** 看到熟悉事物: 好奇心小幅下降 (熟悉=不新奇) */
    public void seeKnown() {
        curiosity = Math.max(5, curiosity - 3);
    }

    /** 识别结果更新好奇心: 未知→好奇上升, 已知→微降 */
    public void observeRecognition(boolean isUnknown) {
        if (isUnknown) {
            curiosity = Math.min(100, curiosity + 15);
            setEmotion(Emotion.好奇);
        } else {
            curiosity = Math.max(5, curiosity - 2);
            // 只有不在强烈情绪时才回落平静
            if (emotion == Emotion.好奇) setEmotion(Emotion.平静);
        }
    }

    /** 情绪 API */
    public void setEmotion(Emotion e) {
        if (e != emotion) {
            lastEmotion = emotion;
            emotion = e;
            emotionHoldSteps = 25;  // 情绪保持 ~5s (0.2s/帧)
        }
    }
    public Emotion emotion() { return emotion; }
    public String emotionEmoji() { return emotion.emoji; }
    public String emotionName() { return emotion.name; }

    /** 每帧调用: 情绪自然回落 (激动→平静) */
    public void tickEmotion() {
        if (emotion != Emotion.平静 && emotion != Emotion.好奇) {
            emotionHoldSteps--;
            if (emotionHoldSteps <= 0) {
                setEmotion(Emotion.平静);
                emotionHoldSteps = 0;
            }
        }
    }

    /** 好奇心是否处于"高"状态 (探索激励区) */
    public boolean isCurious() { return curiosity >= 50; }
    public double curiosity() { return curiosity; }

    public String curiosityLine() {
        return CURIOUS_LINES[rnd.nextInt(CURIOUS_LINES.length)];
    }

    // ============ 成就 ============

    private void checkAchievement(String name, boolean condition) {
        if (condition) achievements.add(name);
    }

    // ============ 查询 ============

    public int points() { return points; }
    public int xp() { return xp; }
    public int level() { return level; }
    public Stage stage() { return Stage.values()[level]; }
    public int xpToNext() { return 30 * (level + 1); }
    public int correctCount() { return correctCount; }
    public int wrongCount() { return wrongCount; }
    public int streak() { return streak; }
    public int bestStreak() { return bestStreak; }
    public int exploreCount() { return exploreCount; }
    public int newWordCount() { return newWordCount; }
    public Set<String> knownWords() { return knownWords; }
    public Set<String> achievements() { return achievements; }
    public List<String> inventory() { return inventory; }
    public List<String> recentFeedback() { return recentFeedback; }
    public void learnWord(String w) { knownWords.add(w); }

    /** 学习进度 0-1 */
    public double progress() {
        return (double) correctCount / Math.max(1, correctCount + wrongCount);
    }

    /** 成长可视化参数: 0-1 大脑活跃度 (随等级/探索提升) */
    public double brainActivity() {
        return Math.min(1.0, 0.3 + level * 0.12 + exploreCount * 0.02);
    }

    public static class Feedback {
        public String message;
        public final boolean isReward;
        public final String speechText;
        public final String rewardItem;
        public Feedback(String msg, boolean reward, String speech, String item) {
            this.message = msg; this.isReward = reward;
            this.speechText = speech; this.rewardItem = item;
        }
    }
}
