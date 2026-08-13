package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 资料深度拆解补充验证: 听觉情感识别 (书中23篇)。
 */
public class EmotionalVoiceTest {

    @Test
    void happyVoiceDetected() {
        EmotionalVoice ev = new EmotionalVoice();
        // 开心: 高音高(280Hz) + 快语速(5字/s) + 高变化率
        EmotionalVoice.Emotion e = ev.classify(280, 0.6, 5.0, 0.25);
        assertEquals(EmotionalVoice.Emotion.开心, e, "高音高快语速应识别为开心");
    }

    @Test
    void angryVoiceDetected() {
        EmotionalVoice ev = new EmotionalVoice();
        // 愤怒: 高响度(0.9) + 高音高(250Hz)
        EmotionalVoice.Emotion e = ev.classify(250, 0.9, 4.5, 0.1);
        assertEquals(EmotionalVoice.Emotion.愤怒, e, "高响度高音应识别为愤怒");
    }

    @Test
    void sadVoiceDetected() {
        EmotionalVoice ev = new EmotionalVoice();
        // 悲伤: 低音高(120Hz) + 慢语速(2字/s) + 低变化率
        EmotionalVoice.Emotion e = ev.classify(120, 0.3, 2.0, 0.03);
        assertEquals(EmotionalVoice.Emotion.悲伤, e, "低音高慢语速应识别为悲伤");
    }

    @Test
    void neutralWhenAmbiguous() {
        EmotionalVoice ev = new EmotionalVoice();
        // 中等特征 → 中性
        EmotionalVoice.Emotion e = ev.classify(180, 0.4, 3.0, 0.08);
        assertTrue(e == EmotionalVoice.Emotion.中性 || e == EmotionalVoice.Emotion.平静,
                "模糊特征应中性/平静, got=" + e);
    }

    @Test
    void brainHasEmotionalVoice() {
        Brain brain = Brain.simpleBrain();
        assertNotNull(brain.emotionalVoice());
        // 完整链路: 声音特征 → 情感 → 接入大脑
        EmotionalVoice.Emotion e = brain.emotionalVoice().classify(280, 0.6, 5.0, 0.25);
        assertEquals(EmotionalVoice.Emotion.开心, e);
    }
}
