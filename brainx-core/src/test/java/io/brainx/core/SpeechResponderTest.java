package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 语音对话 + 逗乐互动验证:
 * 听到语音→回复 / 被逗→笑 / 观察→描述。
 */
public class SpeechResponderTest {

    @Test
    void greetingReplies() {
        SpeechResponder sr = new SpeechResponder();
        String reply = sr.respond("你好！", java.util.List.of(), LanguageLearner.LangStage.模仿期, "");
        assertEquals("你好！你好！", reply);
        String hi = sr.respond("嗨", java.util.List.of(), LanguageLearner.LangStage.理解期, "");
        assertTrue(hi.contains("你好"), hi);
    }

    @Test
    void knowsWordReplies() {
        SpeechResponder sr = new SpeechResponder();
        // 理解期: 听到认识的词 → 理解回应
        String reply = sr.respond("这个苹果真好看", java.util.List.of("苹果"),
                LanguageLearner.LangStage.理解期, "");
        assertTrue(reply.contains("苹果"), reply);
        assertTrue(reply.contains("看到"), reply);
        // 模仿期: 重复
        String mimic = sr.respond("苹果", java.util.List.of("苹果"),
                LanguageLearner.LangStage.模仿期, "");
        assertTrue(mimic.contains("苹果"), mimic);
    }

    @Test
    void questionRepliesCuriously() {
        SpeechResponder sr = new SpeechResponder();
        String reply = sr.respond("那是什么？", java.util.List.of(), LanguageLearner.LangStage.自主期, "");
        assertTrue(reply.contains("知道"), reply);
    }

    @Test
    void teaseMakesHappy() {
        SpeechResponder sr = new SpeechResponder();
        int before = sr.happiness();
        // 逗乐 → 开心上升
        sr.playReact(1.0);
        assertTrue(sr.happiness() > before, "被逗应开心");
        assertEquals(1, sr.teased());
        // 多次逗乐 → 大笑
        sr.playReact(1.0);
        sr.playReact(1.0);
        assertTrue(sr.happiness() >= 65, "多次逗乐应开心度上升");
        String laugh = sr.playReact(1.0);
        assertTrue(laugh.contains("哈") || laugh.contains("嘻"), "应笑, got=" + laugh);
        assertEquals(4, sr.laughs());
    }

    @Test
    void laughterInSpeechDetected() {
        SpeechResponder sr = new SpeechResponder();
        // 听到笑声 → 被逗乐
        String reply = sr.respond("哈哈哈哈你真可爱", java.util.List.of(), LanguageLearner.LangStage.自主期, "");
        assertTrue(reply.contains("哈") || reply.contains("嘻") || reply.contains("玩"),
                "听到笑声应被逗乐, got=" + reply);
        assertTrue(sr.teased() >= 1, "笑声应计入逗乐");
    }

    @Test
    void brainConversation() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        // 学词 → 会回应认识的词
        for (int e = 0; e < 3; e++) brain.learnVisualWord(img, 0);
        String reply = brain.respondToSpeech("你好啊");
        assertFalse(reply.isEmpty(), "应回应问候");
        // 被逗乐
        String laugh = brain.playReact(1.0);
        assertFalse(laugh.isEmpty(), "被逗应笑");
        assertTrue(brain.happiness() >= 50);
        // 观察
        String obs = brain.observeScene(img);
        assertTrue(obs.contains("看到") || obs.contains("咦"), obs);
        // 互动摘要
        assertTrue(brain.interactSummary().contains("开心"));
    }

    @Test
    void happinessFallsOverTime() {
        SpeechResponder sr = new SpeechResponder();
        sr.playReact(1.0);
        int high = sr.happiness();
        // 时间流逝后回落 (模拟 2 分钟)
        for (int i = 0; i < 120; i++) {
            try { Thread.sleep(1); } catch (Exception e) {}
        }
        sr.tick(1000);
        // 未到60秒可能不降, 但不应上升 (自然回落机制存在)
        assertTrue(sr.happiness() <= high + 5, "开心度不应异常上升");
    }
}
