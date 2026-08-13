package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 游戏化养成 + 好奇心机制验证。
 * 好奇心 = 未知刺激→好奇上升→探索未知→获得激励 (婴儿学习机制)。
 */
public class CuriosityTest {

    @Test
    void unknownStimulusRaisesCuriosity() {
        EduTrainer t = new EduTrainer();
        double before = t.curiosity();
        // 看到未知 → 好奇上升 + 好奇情绪
        t.observeRecognition(true);
        assertTrue(t.curiosity() > before, "未知刺激应提升好奇心");
        assertEquals(EduTrainer.Emotion.好奇, t.emotion(), "未知应触发好奇情绪");
    }

    @Test
    void emotionsFollowEvents() {
        EduTrainer t = new EduTrainer();
        t.reward();
        assertEquals(EduTrainer.Emotion.开心, t.emotion(), "答对应开心");
        t.punish("苹果");
        assertEquals(EduTrainer.Emotion.沮丧, t.emotion(), "答错应沮丧");
        t.observeRecognition(true);
        assertEquals(EduTrainer.Emotion.好奇, t.emotion(), "未知应好奇");
        t.exploreUnknown("星星");
        assertEquals(EduTrainer.Emotion.兴奋, t.emotion(), "探索未知应兴奋");
    }

    @Test
    void emotionDecaysToCalm() {
        EduTrainer t = new EduTrainer();
        t.reward();
        assertEquals(EduTrainer.Emotion.开心, t.emotion());
        // 情绪保持 ~25 帧后回落平静
        for (int i = 0; i < 30; i++) t.tickEmotion();
        assertEquals(EduTrainer.Emotion.平静, t.emotion(), "情绪应自然回落");
    }

    @Test
    void knownStimulusLowersCuriosity() {
        EduTrainer t = new EduTrainer();
        t.observeRecognition(true);  // 先升
        double high = t.curiosity();
        // 反复看已知 → 好奇下降 (熟悉化)
        for (int i = 0; i < 5; i++) t.observeRecognition(false);
        assertTrue(t.curiosity() < high, "已知刺激应降低好奇心");
    }

    @Test
    void curiosityDecaysOverTime() {
        EduTrainer t = new EduTrainer();
        t.observeRecognition(true);
        double before = t.curiosity();
        // 时间流逝, 无新刺激 → 好奇衰减 (但不低于下限5)
        t.tickCuriosity(30.0);
        assertTrue(t.curiosity() < before, "好奇心应随时间衰减");
        assertTrue(t.curiosity() >= 5, "好奇心不应低于下限");
    }

    @Test
    void exploreUnknownGivesBonusRewards() {
        EduTrainer t = new EduTrainer();
        // 普通答对
        int pointsNormal = t.points();
        t.reward();
        int gainNormal = t.points() - pointsNormal;
        // 探索未知 (高好奇)
        t.observeRecognition(true);
        t.observeRecognition(true);
        t.observeRecognition(true);
        int pointsBefore = t.points();
        EduTrainer.Feedback fb = t.exploreUnknown("苹果");
        int gainExplore = t.points() - pointsBefore;
        // 探索奖励应 > 普通奖励 (好奇激励)
        assertTrue(gainExplore > gainNormal, "探索未知奖励应更大: explore=" + gainExplore + " normal=" + gainNormal);
        assertTrue(fb.message.contains("探索未知"), "应标记为探索");
    }

    @Test
    void streakBuildsAndResets() {
        EduTrainer t = new EduTrainer();
        t.reward();
        t.reward();
        assertEquals(2, t.streak());
        t.punish("苹果");
        assertEquals(0, t.streak(), "答错应清零连击");
        t.reward();
        assertEquals(1, t.streak());
    }

    @Test
    void achievementsUnlock() {
        EduTrainer t = new EduTrainer();
        // 探索 5 次解锁"好奇探险家"
        for (int i = 0; i < 5; i++) {
            t.observeRecognition(true);
            t.exploreUnknown("词" + i);
        }
        assertTrue(t.achievements().contains("好奇探险家"), "探索5次应解锁成就");
        assertTrue(t.achievements().contains("发现之旅"), "首次探索应解锁成就");
        // 10 连击
        for (int i = 0; i < 10; i++) t.reward();
        assertTrue(t.achievements().contains("十连击"), "10连击应解锁成就");
    }

    @Test
    void brainActivityGrowsWithLevel() {
        EduTrainer t = new EduTrainer();
        double early = t.brainActivity();
        // 大量学习提升等级
        for (int i = 0; i < 40; i++) t.reward();
        assertTrue(t.brainActivity() > early, "等级提升应使大脑更活跃");
        assertTrue(t.level() > 0, "应已升级");
    }

    @Test
    void brainConfidenceUnknownDetection() {
        // 未学习模式 → 置信度低 → 判"未知"
        Brain brain = Brain.simpleBrain();
        double[] randomFeatures = new double[VisualNeuralEncoder.OUTPUT_DIM];
        java.util.Random r = new java.util.Random(3);
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) randomFeatures[i] = r.nextDouble();
        String[] result = brain.recognizeVisualWithConfidence(randomFeatures);
        assertTrue(result[0].equals("未知") || Double.parseDouble(result[1]) < 0.5,
                "未学过的输入应低置信度, got " + result[0] + " conf=" + result[1]);
    }
}
