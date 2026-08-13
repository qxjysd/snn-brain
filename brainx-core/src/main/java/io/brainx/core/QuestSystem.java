package io.brainx.core;

import java.util.ArrayList;
import java.util.List;

/**
 * 游戏任务系统 —— 每日目标 + 挑战 (游戏化训练)。
 *
 * 设计: 类似养成游戏的每日任务/成就挑战:
 *   - 每日任务: 学会N个新词 / 探索N次 / 连续答对N次
 *   - 泛化挑战: 用变体测试大脑泛化能力 (训练成果验证)
 *   - 任务奖励: XP/点数加成
 */
public class QuestSystem {
    /** 任务类型 */
    public enum QuestType {
        学习("📚 学会新词", "学习 %d 个新词"),
        探索("🔍 探索未知", "探索 %d 次未知"),
        连击("🔥 连续答对", "连续答对 %d 次"),
        泛化("🎯 泛化挑战", "识别 %d 个变体"),
        睡眠("😴 睡眠巩固", "睡眠巩固 %d 次");

        final String name; final String descTemplate;
        QuestType(String n, String t) { this.name = n; this.descTemplate = t; }
    }

    /** 每日任务 */
    public static class Quest {
        public final QuestType type;
        public final int target;
        public int progress = 0;
        public boolean completed = false;
        public final int xpReward;

        public Quest(QuestType type, int target, int xpReward) {
            this.type = type;
            this.target = target;
            this.xpReward = xpReward;
        }

        public void addProgress(int n) {
            if (completed) return;
            progress += n;
            if (progress >= target) {
                progress = target;
                completed = true;
            }
        }

        public String describe() {
            String d = String.format(type.descTemplate, target);
            return String.format("%s: %s [%d/%d] %s",
                    type.name, d, progress, target, completed ? "✅" : "");
        }
    }

    private final List<Quest> dailyQuests = new ArrayList<>();
    private int completedToday = 0;
    private int day = 1;

    public QuestSystem() {
        generateDailyQuests();
    }

    /** 生成每日任务 (随机 3 个) */
    public void generateDailyQuests() {
        dailyQuests.clear();
        dailyQuests.add(new Quest(QuestType.学习, 3 + (int)(Math.random()*3), 20));
        dailyQuests.add(new Quest(QuestType.探索, 2 + (int)(Math.random()*2), 15));
        dailyQuests.add(new Quest(QuestType.连击, 5, 25));
        completedToday = 0;
    }

    /** 记录事件 → 更新相关任务 */
    public void onEvent(QuestType type, int amount) {
        for (Quest q : dailyQuests) {
            if (q.type == type && !q.completed) {
                q.addProgress(amount);
                if (q.completed) completedToday++;
            }
        }
    }

    /** 每日任务摘要 (APK 显示) */
    public String dailySummary() {
        StringBuilder sb = new StringBuilder(String.format("🎮 第%d天 | 任务完成 %d/3:\n", day, completedToday));
        for (Quest q : dailyQuests) sb.append("  ").append(q.describe()).append("\n");
        return sb.toString();
    }

    public List<Quest> dailyQuests() { return dailyQuests; }
    public int completedToday() { return completedToday; }
    public int day() { return day; }
    public void nextDay() { day++; generateDailyQuests(); }
    public boolean allDone() { return completedToday >= dailyQuests.size(); }

    /** 剩余任务数 */
    public int remaining() {
        int n = 0;
        for (Quest q : dailyQuests) if (!q.completed) n++;
        return n;
    }
}
