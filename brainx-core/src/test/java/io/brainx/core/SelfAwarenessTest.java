package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 第三轮完善验证: 多感觉整合 + 分层记忆 + 自我意识。
 */
public class SelfAwarenessTest {

    // ===== ① 多感觉整合 (Körding 因果推断) =====

    @Test
    void msiFusesReliableSignals() {
        MultiSensoryIntegration msi = MultiSensoryIntegration.defaultParams();
        // 视觉可靠 0.9, 听觉可靠 0.5, 估计相近
        double[] r = msi.integrate(0.8, 0.9, 0.7, 0.5);
        // 融合估计偏向可靠视觉
        assertTrue(r[0] > 0.7 && r[0] < 0.8, "融合应在两估计之间, got=" + r[0]);
        // 同源概率高 (差异小)
        assertTrue(r[1] > 0.5, "差异小应高同源概率, p=" + r[1]);
        assertTrue(r[2] > 0.2, "整合度应>0, phi=" + r[2]);
    }

    @Test
    void msiSeparatesConflictSignals() {
        MultiSensoryIntegration msi = MultiSensoryIntegration.defaultParams();
        // 大差异 (视觉0.9 vs 听觉0.1) → 低同源概率
        double[] r = msi.integrate(0.9, 0.8, 0.1, 0.8);
        assertTrue(r[1] < 0.5, "大差异应低同源概率, p=" + r[1]);
    }

    @Test
    void msiLabelsSameObjectHighPhi() {
        MultiSensoryIntegration msi = MultiSensoryIntegration.defaultParams();
        assertEquals(0.95, msi.integrateLabels("苹果", "苹果"), 0.01, "跨模态一致应高Φ");
        assertTrue(msi.integrateLabels("苹果", "猫") < 0.5, "跨模态冲突应低Φ");
    }

    // ===== ② 分层记忆 (Atkinson-Shiffrin) =====

    @Test
    void episodicToLongTermConsolidation() {
        HierarchicalMemory hm = HierarchicalMemory.defaultParams();
        double[] feat = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) feat[i] = (i % 2 == 0) ? 1 : 0;
        // 反复出现 (3 次 → 长期)
        hm.addEpisodic("苹果", feat, 0.5);
        hm.addEpisodic("苹果", feat, 0.5);
        assertFalse(hm.inLongTerm("苹果"), "2次不应进长期");
        hm.addEpisodic("苹果", feat, 0.5);
        assertTrue(hm.inLongTerm("苹果"), "3次应巩固为长期");
        assertTrue(hm.longTermCount() >= 1);
    }

    @Test
    void episodicDecaysWithoutReinforcement() {
        HierarchicalMemory hm = HierarchicalMemory.defaultParams();
        double[] feat = new double[VisualNeuralEncoder.OUTPUT_DIM];
        hm.addEpisodic("一次性", feat, 0.1);
        assertEquals(1, hm.episodicCount());
        // 24 小时无复习 → 中期遗忘 (τ=6h, 24h 后 e^-4≈0.018 < 0.05)
        for (int t = 0; t < 24 * 3600; t++) hm.tick(1000.0);
        assertEquals(0, hm.episodicCount(), "中期记忆应遗忘");
    }

    @Test
    void longTermRecallByFeatures() {
        HierarchicalMemory hm = HierarchicalMemory.defaultParams();
        double[] feat = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) feat[i] = (i % 3 == 0) ? 1 : 0;
        for (int i = 0; i < 3; i++) hm.addEpisodic("书本", feat, 0.5);
        String recall = hm.recall(feat);
        assertEquals("书本", recall, "特征应召回长期记忆");
    }

    // ===== ③ 自我意识 (镜像/元认知/叙事) =====

    @Test
    void mirrorTestBuildsSelfModel() {
        SelfAwareness sa = new SelfAwareness();
        assertFalse(sa.selfRecognized(), "初始未通过镜像测试");
        // 5 次正确反馈 → 自我识别建立
        for (int i = 0; i < 6; i++) sa.mirrorTest(true, 0.8);
        assertTrue(sa.selfRecognized(), "连续正确应建立自我识别");
        assertTrue(sa.selfConfidence() > 0.5, "自我评价应提升");
    }

    @Test
    void metacognitionReflectsConfidence() {
        SelfAwareness sa = new SelfAwareness();
        String m = sa.metacognition();
        assertNotNull(m);
        assertFalse(sa.knows("陌生词", 0.2), "低置信度应'不知道'");
        assertTrue(sa.knows("熟词", 0.9), "高置信度应'知道'");
    }

    @Test
    void devStagesGrowNaturally() {
        SelfAwareness sa = new SelfAwareness();
        // 自然成长: 未通过镜像测试 = 感知萌芽 (无论词数)
        assertEquals(SelfAwareness.DevStage.感知萌芽, sa.devStage(5, 0.9, false), "未自我识别=萌芽");
        // 通过镜像测试但词少 → 仍萌芽
        assertEquals(SelfAwareness.DevStage.感知萌芽, sa.devStage(1, 0.5, true), "词少=萌芽");
        // 词数≥2 → 符号联结 (语言符号连接概念)
        assertEquals(SelfAwareness.DevStage.符号联结, sa.devStage(2, 0.5, true), "词2=符号联结");
        // 词数≥4 + 准确率≥0.6 + 自我识别 → 反思成熟
        assertEquals(SelfAwareness.DevStage.反思成熟, sa.devStage(4, 0.7, true), "词4准70%=反思成熟");
        // 学得慢: 词少准确率低 → 不成熟 (自然成长, 非年龄硬切)
        assertEquals(SelfAwareness.DevStage.符号联结, sa.devStage(4, 0.3, true), "准确率低=仍符号联结");
    }

    @Test
    void narrativeAndSocialSelf() {
        SelfAwareness sa = new SelfAwareness();
        assertEquals("我还没有属于自己的故事...", sa.selfNarrative());
        sa.mirrorTest(true, 0.8);
        assertTrue(sa.selfNarrative().contains("故事"), "有经历后应有叙事");
        // 1 次全对 → 高评价文案 (社会自我)
        assertTrue(sa.socialSelf().contains("聪明") || sa.socialSelf().contains("潜力"),
                "社会自我应反映评价, got=" + sa.socialSelf());
    }

    // ===== 集成 =====

    @Test
    void brainIntegratesAllNewModules() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        // 学习 → 中期记忆沉淀
        for (int e = 0; e < 4; e++) brain.learnVisualWord(img, 0);
        assertTrue(brain.hierarchicalMemory().episodicCount() > 0, "学习应沉淀中期记忆");
        // 睡眠 → 长期巩固
        brain.sleepConsolidate();
        // 识别 + 镜像反馈
        String[] r = brain.recognizeVisualWithConfidence(img);
        brain.mirrorFeedback(true, Double.parseDouble(r[1]));
        assertTrue(brain.selfAwareness().totalTests() > 0);
        // 跨模态整合
        double phi = brain.crossModalPhi("苹果", "苹果");
        assertEquals(0.95, phi, 0.01);
        // 叙事
        assertTrue(brain.selfNarrative().contains("故事"));
        // 学更多词 (自然成长需要足够经验量)
        double[] img2 = new double[VisualNeuralEncoder.OUTPUT_DIM], img3 = new double[VisualNeuralEncoder.OUTPUT_DIM], img4 = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) {
            img2[i] = (i % 3 == 1) ? 0.9 : 0.1;
            img3[i] = (i % 3 == 2) ? 0.9 : 0.1;
            img4[i] = (i % 4 == 1) ? 0.9 : 0.1;
        }
        for (int e = 0; e < 4; e++) {
            brain.learnVisualWord(img2, 1);
            brain.learnVisualWord(img3, 2);
            brain.learnVisualWord(img4, 3);
        }
        // 自然成长: 多次识别反馈后通过镜像测试 → 反思成熟 (无年龄)
        for (int i = 0; i < 5; i++) {
            String[] r2 = brain.recognizeVisualWithConfidence(img);
            brain.mirrorFeedback(true, Double.parseDouble(r2[1]));
        }
        assertTrue(brain.selfAwareness().selfRecognized(), "5次反馈后应自我识别");
        assertTrue(brain.learnedWords().size() >= 4, "词数应≥4");
        assertEquals(SelfAwareness.DevStage.反思成熟, brain.devStage(),
                "词4+自我识别+准确率 → 反思成熟 (自然成长)");
    }
}
