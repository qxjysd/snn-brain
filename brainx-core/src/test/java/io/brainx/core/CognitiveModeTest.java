package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 认知模式 + 僵尸行动者验证 (《上脑与下脑》Kosslyn + 《意识与脑》Koch)。
 */
public class CognitiveModeTest {

    // ===== 认知模式 (Kosslyn 上脑/下脑) =====

    @Test
    void modesMapToQuadrants() {
        // 均衡: 初始应接近适应者或行动者 (取决于初始值)
        CognitiveMode cm = new CognitiveMode();
        assertNotNull(cm.currentMode());
        // 四个模式定义存在
        assertEquals(4, CognitiveMode.Mode.values().length);
    }

    @Test
    void learningBuildsPerceiver() {
        CognitiveMode cm = new CognitiveMode();
        // 大量学习/识别 (下脑活动) → 趋向感知者 (下脑高)
        for (int i = 0; i < 10; i++) {
            cm.learnActivity(0.1);
            cm.recognizeActivity(0.1);
        }
        assertTrue(cm.lowerBrain() > cm.upperBrain(), "学习应提升下脑使用度");
        CognitiveMode.Mode m = cm.currentMode();
        assertTrue(m == CognitiveMode.Mode.感知者 || m == CognitiveMode.Mode.行动者,
                "下脑高应趋向感知者/行动者, got=" + m.name);
    }

    @Test
    void decidingBuildsStimulator() {
        CognitiveMode cm = new CognitiveMode();
        // 大量决策 (上脑活动为主, 下脑仅0.2增益) → 趋向刺激者 (上脑高)
        for (int i = 0; i < 12; i++) {
            cm.decideActivity(0.08);
        }
        assertTrue(cm.upperBrain() > cm.lowerBrain(), "决策应提升上脑使用度: up=" + cm.upperBrain() + " low=" + cm.lowerBrain());
        CognitiveMode.Mode m = cm.currentMode();
        assertTrue(m == CognitiveMode.Mode.刺激者 || m == CognitiveMode.Mode.行动者,
                "上脑高应趋向刺激者/行动者, got=" + m.name);
    }

    @Test
    void trainingBiasShapesMode() {
        CognitiveMode cm = new CognitiveMode();
        // 纯上脑培养: 决策训练
        cm.setTrainingBias(1.0);
        for (int i = 0; i < 8; i++) cm.decideActivity(0.1);
        assertTrue(cm.upperBrain() > 0.7, "上脑培养应提升上脑使用度");
        // 纯下脑培养
        CognitiveMode cm2 = new CognitiveMode();
        cm2.setTrainingBias(-1.0);
        for (int i = 0; i < 8; i++) cm2.learnActivity(0.1);
        assertTrue(cm2.lowerBrain() > 0.7, "下脑培养应提升下脑使用度");
    }

    @Test
    void sleepSlightlyRelaxes() {
        CognitiveMode cm = new CognitiveMode();
        cm.exploreActivity(0.5);
        double before = cm.upperBrain() + cm.lowerBrain();
        cm.sleepActivity();
        assertTrue(cm.upperBrain() + cm.lowerBrain() < before, "睡眠应微调放松");
    }

    // ===== 僵尸行动者 (Koch 技能自动化) =====

    @Test
    void practiceAutomatesSkill() {
        ZombieAgent za = ZombieAgent.defaultParams();
        assertFalse(za.isAutomated("苹果"));
        // 练习 5 次 → 自动化
        for (int i = 0; i < 5; i++) za.practice("苹果");
        assertTrue(za.isAutomated("苹果"), "5次练习应自动化");
        assertTrue(za.automatedConfidence("苹果") > 0.9, "自动化置信度应稳定高");
        assertEquals(1, za.automatedCount());
    }

    @Test
    void automatedFreesAttention() {
        ZombieAgent za = ZombieAgent.defaultParams();
        assertEquals(0, za.automatedCount());
        double att0 = za.availableAttention();
        // 自动化 3 个技能 → 释放意识
        for (String s : new String[]{"猫", "狗", "水"}) {
            for (int i = 0; i < 5; i++) za.practice(s);
        }
        assertEquals(3, za.automatedCount());
        assertTrue(za.availableAttention() < att0, "自动化应释放意识资源");
        assertTrue(za.freedAttention() > 0);
    }

    @Test
    void automatedProcessesWithoutConsciousness() {
        ZombieAgent za = ZombieAgent.defaultParams();
        for (int i = 0; i < 5; i++) za.practice("书本");
        double[] r = za.process("书本", 0.4);
        // [置信度, 是否自动化, 意识占用]
        assertEquals(1.0, r[1], "应走自动化通道");
        assertEquals(0.0, r[2], "自动化不应占意识");
        assertTrue(r[0] > 0.9, "自动化置信度应高");
        // 非自动化技能需意识
        double[] r2 = za.process("新东西", 0.4);
        assertEquals(0.0, r2[1], "非自动化不应走僵尸通道");
        assertEquals(1.0, r2[2], "非自动化应占意识");
    }

    @Test
    void phiHandlesUnexpected() {
        ZombieAgent za = ZombieAgent.defaultParams();
        // Φ 高 → 意外处理更好 (PHI 理论: 黑天鹅事件)
        double lowPhi = za.handleUnexpected(0.1, "陌生");
        double highPhi = za.handleUnexpected(0.9, "陌生");
        assertTrue(highPhi > lowPhi, "Φ高应更好处理意外: low=" + lowPhi + " high=" + highPhi);
        // 自动化技能无论 Φ 都稳定
        for (int i = 0; i < 5; i++) za.practice("熟词");
        assertEquals(0.95, za.handleUnexpected(0.1, "熟词"), 1e-9);
    }

    // ===== 集成 =====

    @Test
    void brainCognitiveAndZombieIntegration() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        // 学习 5 次 (练习) → 识别 (触发自动化通道)
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        String[] r = brain.recognizeVisualWithConfidence(img);
        // 学习+识别练习 → 已自动化
        assertTrue(brain.zombieAgent().automatedCount() > 0, "学习应产生自动化技能");
        // 认知模式描述可读
        assertTrue(brain.cognitiveModeDescription().contains("模式"));
        assertTrue(brain.zombieSummary().contains("僵尸"));
        // 决策 → 上脑
        double[] cues = new double[14];
        for (int t = 0; t < 14; t += 2) { cues[t] = 1.0; cues[t+1] = 0.0; }
        brain.runDecisionTrial(cues);
        // 探索 → 上脑+下脑
        brain.exploreActivity();
        // 意外处理
        double unexpected = brain.handleUnexpected("全新事物");
        assertTrue(unexpected >= 0.3 && unexpected <= 0.95);
    }
}
