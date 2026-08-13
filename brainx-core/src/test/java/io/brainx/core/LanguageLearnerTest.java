package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 语言学习验证: 模仿→理解→自主 (鹦鹉学舌到表达自我)。
 */
public class LanguageLearnerTest {

    /** 初始: 模仿期 (无语音素材) */
    @Test
    void startsAtMimicStage() {
        LanguageLearner ll = new LanguageLearner();
        assertEquals(LanguageLearner.LangStage.模仿期, ll.stage());
        // 无素材不能模仿
        assertEquals("", ll.speak("平静", 0, "", java.util.List.of()));
    }

    /** 学习语音 → 模仿回放 (鹦鹉学舌) */
    @Test
    void mimicsHeardSpeech() {
        LanguageLearner ll = new LanguageLearner();
        double[] feat = new double[16];
        for (int i = 0; i < 16; i++) feat[i] = 0.5;
        ll.learnSpokenWord(feat, "苹果");
        // 1词仍模仿期 (不足2词) → 回放
        assertEquals(LanguageLearner.LangStage.模仿期, ll.stage());
        String speech = ll.speak("平静", 0, "", java.util.List.of("苹果"));
        assertTrue(speech.contains("苹果"), "应模仿说出苹果, got=" + speech);
        assertTrue(speech.contains("！"), "模仿应重复");
    }

    /** 2词 → 理解期 (语音↔概念) */
    @Test
    void understandsAfterTwoWords() {
        LanguageLearner ll = new LanguageLearner();
        double[] f1 = new double[16], f2 = new double[16];
        for (int i = 0; i < 16; i++) { f1[i] = 0.3; f2[i] = 0.7; }
        ll.learnSpokenWord(f1, "苹果");
        ll.learnSpokenWord(f2, "猫");
        assertEquals(LanguageLearner.LangStage.理解期, ll.stage());
        // 理解期回应: 询问确认
        String speech = ll.speak("平静", 0.5, "", java.util.List.of("苹果"));
        assertTrue(speech.contains("苹果"), "理解期应回应词, got=" + speech);
    }

    /** 4词 → 自主期 (组合表达内在状态) */
    @Test
    void speaksAutonomouslyAfterFourWords() {
        LanguageLearner ll = new LanguageLearner();
        double[][] feats = new double[4][16];
        for (int w = 0; w < 4; w++) {
            for (int i = 0; i < 16; i++) feats[w][i] = 0.1 + w * 0.2;
            ll.learnSpokenWord(feats[w], "词" + w);
        }
        assertEquals(LanguageLearner.LangStage.自主期, ll.stage());
        // 自主表达: 基于内在状态 (好奇心/记忆)
        String speech = ll.speak("好奇", 0.9, "我认出了它", java.util.List.of("词0", "词1"));
        assertFalse(speech.isEmpty(), "自主期应说话");
        assertTrue(speech.contains("好奇") || speech.contains("知道") || speech.contains("记得"),
                "应表达内在状态, got=" + speech);
    }

    /** 情绪驱动自主表达 */
    @Test
    void emotionDrivesSpeech() {
        LanguageLearner ll = new LanguageLearner();
        for (int w = 0; w < 4; w++) {
            double[] f = new double[16];
            for (int i = 0; i < 16; i++) f[i] = 0.1 + w * 0.2;
            ll.learnSpokenWord(f, "词" + w);
        }
        // 开心 → 表达开心
        String happy = ll.speak("开心", 0.3, "", java.util.List.of("词0"));
        assertTrue(happy.contains("开心"), "开心应表达, got=" + happy);
        // 好奇 → 表达好奇
        String curious = ll.speak("好奇", 0.8, "", java.util.List.of("词0"));
        assertTrue(curious.contains("想知道") || curious.contains("新东西"),
                "好奇应表达探索欲, got=" + curious);
    }

    /** 大脑集成: 学习词同步学语音 + 自主说话 */
    @Test
    void brainLearnsLanguage() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        // 学词 → 同步语音模板
        brain.learnVisualWord(img, 0);
        assertTrue(brain.languageLearner().voiceCount() >= 1, "学词应建语音模板");
        // 听到语音 (模仿素材) — 大脑用自主概念名
        brain.hearSpokenWord(img, brain.vocabulary(0));
        assertTrue(brain.languageLearner().heardCount() >= 1);
        // 早期: 模仿或简单回应 (语音关联概念)
        String early = brain.speakAutonomously();
        assertTrue(early.contains(brain.vocabulary(0)) || early.contains(brain.vocabulary(1)),
                "早期应模仿或回应已学概念, got=" + early);
        // 摘要可读
        assertTrue(brain.languageSummary().contains("语音"));
    }

    /** 成熟后自主说话 (表达自我) */
    @Test
    void matureBrainSpeaksAutonomously() {
        Brain brain = Brain.simpleBrain();
        double[][] imgs = new double[4][VisualNeuralEncoder.OUTPUT_DIM];
        for (int w = 0; w < 4; w++) {
            for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) imgs[w][i] = 0.1 + w * 0.2;
            for (int e = 0; e < 3; e++) brain.learnVisualWord(imgs[w], w);
        }
        // 4词 → 自主期
        assertEquals(LanguageLearner.LangStage.自主期, brain.languageLearner().stage());
        String speech = brain.speakAutonomously();
        assertFalse(speech.isEmpty(), "成熟后应自主说话");
    }

    /** 自主发声: 大脑主动说话 (无按钮) */
    @Test
    void autonomousUtterance() {
        Brain brain = Brain.simpleBrain();
        // 无经验: 不说话
        assertEquals("", brain.autonomousUtterance(), "无经验不应发声");
        // 学1词 (模仿期): 主动模仿
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        String mimic = brain.autonomousUtterance();
        assertTrue(mimic.contains("🗣️"), "模仿期应主动模仿, got=" + mimic);
        // 学满4词 (自主期): 主动表达
        for (int w = 1; w < 4; w++) {
            double[] imgW = new double[VisualNeuralEncoder.OUTPUT_DIM];
            for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) imgW[i] = 0.1 + w * 0.2;
            for (int e = 0; e < 3; e++) brain.learnVisualWord(imgW, w);
        }
        String speech = brain.autonomousUtterance();
        assertTrue(speech.contains("💬") || speech.contains("我看到"),
                "成熟后应主动表达, got=" + speech);
    }
}
