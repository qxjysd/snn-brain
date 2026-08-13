package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 成长链条完整性回归测试 (v5.4 审计):
 *   复读(模仿)能否成长到理解/自主等阶段 — 四条链全查。
 *
 * 背景: 审计发现 selfAwareness.mirrorTest 无调用点 → accuracy 恒 0
 *   → devStage 永卡感知萌芽 (成长断链)。修复: 识别路径接入镜像反馈
 *   (高置信 ≥0.6 = "我认出来了", 自我评估驱动成长)。
 */
public class GrowthChainTest {

    private static double[] pat(int c, int V) {
        double[] p = new double[V];
        for (int i = 0; i < V; i++) p[i] = ((i / (V / 8)) == c) ? 0.9 : 0.1;
        return p;
    }

    @Test
    void languageStageGrowsFromMimicry() {
        // 语言: 模仿(复读) → 理解 → 自主
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        assertEquals(LanguageLearner.LangStage.模仿期, brain.languageLearner().stage(), "初始模仿期");
        brain.learnVisualWord(pat(0, V), 0);
        assertEquals(LanguageLearner.LangStage.模仿期, brain.languageLearner().stage(), "1词仍模仿");
        brain.learnVisualWord(pat(1, V), 1);
        assertTrue(brain.languageLearner().stage().level >= LanguageLearner.LangStage.理解期.level,
                "2词应到理解期: " + brain.languageLearner().stage());
        brain.learnVisualWord(pat(2, V), 2);
        brain.learnVisualWord(pat(3, V), 3);
        assertEquals(LanguageLearner.LangStage.自主期, brain.languageLearner().stage(), "4词应自主期");
        // 自主发声非空
        assertFalse(brain.speakAutonomously().isEmpty(), "自主期应能表达");
    }

    @Test
    void devStageGrowsWithRecognition() {
        // 发展: 感知萌芽 → (识别练习) → 高级阶段
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        for (int e = 0; e < 5; e++)
            for (int c = 0; c < 4; c++) brain.learnVisualWord(pat(c, V), c);
        // 识别练习 → mirrorTest 反馈 → 准确率上升
        for (int e = 0; e < 20; e++)
            for (int c = 0; c < 4; c++) brain.recognizeVisualWithConfidence(pat(c, V));
        double acc = brain.selfAwareness().accuracy();
        assertTrue(acc > 0.5, "识别练习应提升自我评估准确率, acc=" + acc);
        assertTrue(brain.selfAwareness().selfRecognized(), "镜像测试应建立自我识别");
        // 发展阶段应从萌芽推进
        SelfAwareness.DevStage stage = brain.devStage();
        assertTrue(stage.level > 0, "发展应离开感知萌芽, got " + stage.name);
    }

    @Test
    void voiceStageGrowsFromBabbling() {
        // 声带: 咿呀 → 模仿 → 自主 (听到音频 → 模板积累)
        Brain brain = Brain.simpleBrain();
        assertTrue(brain.voiceStage().contains("咿呀"), "初始咿呀期");
        for (int c = 0; c < 5; c++) {
            short[] pcm = new short[16000 / 2];
            double f0 = 120 + c * 40;
            for (int i = 0; i < pcm.length; i++) {
                double ph = (i % (16000.0 / f0)) / (16000.0 / f0);
                pcm[i] = (short) (Math.sin(2 * Math.PI * ph) * 8000);
            }
            brain.learnVoiceFromAudio(pcm);
        }
        assertTrue(brain.voiceStage().contains("自主期"), "模板≥4应自主期: " + brain.voiceStage());
        assertTrue(brain.voiceLearner().templateCount() >= 4, "应积累≥4模板");
    }

    @Test
    void growthPotentialExpands() {
        // 潜力: 抽象倍数随学习成长 + 显示可读 (原亿单位精度丢失)
        Brain brain = Brain.simpleBrain();
        long before = brain.growthPotential().abstractionFactor();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        for (int e = 0; e < 3; e++) brain.learnVisualWord(pat(0, V), 0);
        brain.learnVisualWord(pat(1, V), 1);
        assertTrue(brain.growthPotential().abstractionFactor() >= before,
                "学习应提升抽象倍数: " + before + " → " + brain.growthPotential().abstractionFactor());
        // 摘要可读 (原始倍数直显)
        String s = brain.growthPotential().summary();
        assertTrue(s.contains("抽象"), "摘要应含抽象倍数, got: " + s);
        assertFalse(s.contains("0.0亿"), "不应再有精度丢失显示");
        // 虚拟层同步
        assertEquals(brain.growthPotential().abstractionFactor(), brain.virtualLayer().abstraction(),
                "虚拟层抽象应与成长同步");
    }
}
