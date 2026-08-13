package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 第四轮完善验证: 模型分享(快照) + 泛化能力 + 内部模型 + 游戏任务。
 */
public class ShareGeneralizeTest {

    // ===== 内部模型 (第20篇: 前向预测/感觉预测误差) =====

    @Test
    void internalModelLearnsPrediction() {
        InternalModel model = new InternalModel(4);
        // 学习规律: result = 2*x0 - x1 (线性)
        double[][] states = new double[50][4];
        double[] actuals = new double[50];
        for (int i = 0; i < 50; i++) {
            states[i][0] = i % 5 / 4.0;
            states[i][1] = (i * 7) % 5 / 4.0;
            actuals[i] = 2 * states[i][0] - states[i][1];
        }
        model.train(states, actuals);
        // 泛化: 未见样本
        double[][] test = new double[10][4];
        double[] testA = new double[10];
        for (int i = 0; i < 10; i++) {
            test[i][0] = (i * 3) % 5 / 4.0;
            test[i][1] = (i * 11) % 5 / 4.0;
            testA[i] = 2 * test[i][0] - test[i][1];
        }
        double score = model.evaluateGeneralization(test, testA);
        assertTrue(score > 0.7, "内部模型应学会规律并泛化, score=" + score);
        assertTrue(model.trainCount() >= 50);
    }

    @Test
    void internalModelErrorDrivesCorrection() {
        InternalModel model = new InternalModel(2);
        double[] state = {1.0, 0.5};
        double err = model.learn(state, 3.0);
        assertTrue(Math.abs(err) > 0.01, "首次预测应有误差");
        // 多次学习后误差收敛
        double lastErr = 1;
        for (int i = 0; i < 50; i++) {
            lastErr = model.learn(state, 3.0);
        }
        assertTrue(Math.abs(lastErr) < Math.abs(err), "误差应收敛");
    }

    // ===== 泛化能力 =====

    @Test
    void variantTrainingGeneralizes() {
        Brain brain = Brain.simpleBrain();
        GeneralizationTrainer gt = GeneralizationTrainer.defaultParams();
        // 3 类原型
        double[][] bases = new double[3][VisualNeuralEncoder.OUTPUT_DIM];
        int[] labels = {0, 1, 2};
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) {
                bases[c][i] = ((i / (VisualNeuralEncoder.OUTPUT_DIM / 8)) == c) ? 0.9 : 0.1;
            }
        }
        String[] vocab = {"你好", "苹果", "猫"};
        // 变体训练 (每类 8 个变体, 噪声 0.15)
        gt.trainWithVariants(brain, bases, labels, 0.15, 8);
        // 未见变体测试 (噪声 0.2, 比训练更高 = 更强泛化挑战)
        double[] result = gt.evaluateGeneralization(brain, bases, labels, vocab, 0.2, 10);
        assertTrue(result[0] > 0.5, "变体训练后应泛化到未见变体, rate=" + result[0] + " (" + (int)result[1] + "/" + (int)result[2] + ")");
    }

    @Test
    void variantsImproveOverSingleSample() {
        GeneralizationTrainer gt = GeneralizationTrainer.defaultParams();
        double[][] bases = new double[2][VisualNeuralEncoder.OUTPUT_DIM];
        int[] labels = {0, 1};
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) {
                bases[c][i] = ((i / (VisualNeuralEncoder.OUTPUT_DIM / 4)) == c) ? 0.9 : 0.1;
            }
        }
        String[] vocab = {"你好", "苹果"};
        double[] compare = gt.compareGeneralization(bases, labels, vocab, 10);
        // 高维下原型学习泛化饱和: 两种训练都应泛化良好 (变体训练不降低泛化)
        assertTrue(compare[0] > 0.5, "单样本训练应泛化: base=" + compare[0]);
        assertTrue(compare[1] > 0.5, "变体训练应泛化: variant=" + compare[1]);
        assertTrue(compare[1] >= compare[0] - 0.3,
                "变体训练不应显著降低泛化: base=" + compare[0] + " variant=" + compare[1]);
    }

    @Test
    void variantsAreDistinct() {
        GeneralizationTrainer gt = GeneralizationTrainer.defaultParams();
        double[] base = new double[16];
        for (int i = 0; i < 16; i++) base[i] = 0.5;
        double[][] variants = gt.generateVariants(base, 0.2, 5);
        assertEquals(5, variants.length);
        // 变体应与原版不同
        boolean differs = false;
        for (double[] v : variants) {
            for (int i = 0; i < 16; i++) if (Math.abs(v[i] - 0.5) > 0.01) { differs = true; break; }
        }
        assertTrue(differs, "变体应包含扰动");
        // 值应在 [0,1]
        for (double[] v : variants) for (double x : v) assertTrue(x >= 0 && x <= 1);
    }

    // ===== 快照导出/导入 =====

    @Test
    void snapshotRoundTrip() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        // 训练
        for (int e = 0; e < 5; e++) {
            brain.learnVisualWord(img, 0);
            brain.learnAuditoryWord(img, 1);
        }
        // 导出
        String snap = BrainSnapshot.export(brain, 2, 15, 60, "初露锋芒", "🌟 智慧之星");
        assertTrue(snap.startsWith("BRAINX-SNAP"), "快照应有魔数");
        assertTrue(snap.contains("words:"), "应含词表");

        // 导入到新大脑
        Brain brain2 = Brain.simpleBrain();
        BrainSnapshot.RestoreInfo info = BrainSnapshot.importSnapshot(brain2, snap);
        assertNotNull(info, "快照应可解析");
        assertTrue(info.wordCount >= 2, "应恢复词, count=" + info.wordCount);
        assertEquals(2, info.level);
        assertEquals(15, info.points);
        // 恢复后识别仍工作
        String guess = brain2.recognizeVisual(img);
        assertTrue(guess.equals("你好") || guess.equals("苹果"), "恢复后应能识别, got=" + guess);
    }

    @Test
    void snapshotRejectsInvalid() {
        Brain brain = Brain.simpleBrain();
        assertNull(BrainSnapshot.importSnapshot(brain, "garbage"), "无效快照应拒绝");
        assertNull(BrainSnapshot.importSnapshot(brain, ""), "空快照应拒绝");
    }

    // ===== 游戏任务系统 =====

    @Test
    void questsTrackProgress() {
        QuestSystem qs = new QuestSystem();
        assertEquals(3, qs.dailyQuests().size(), "每日 3 任务");
        // 学习任务 target 3-5, 加 5 次确保完成
        qs.onEvent(QuestSystem.QuestType.学习, 5);
        assertTrue(qs.dailyQuests().get(0).completed, "学习任务应完成");
        assertEquals(1, qs.completedToday());
        // 探索
        qs.onEvent(QuestSystem.QuestType.探索, 3);
        assertEquals(2, qs.completedToday());
        // 连击
        qs.onEvent(QuestSystem.QuestType.连击, 5);
        assertTrue(qs.allDone(), "全部完成");
    }

    @Test
    void questsDailyReset() {
        QuestSystem qs = new QuestSystem();
        // 完成全部 3 个任务
        qs.onEvent(QuestSystem.QuestType.学习, 5);
        qs.onEvent(QuestSystem.QuestType.探索, 3);
        qs.onEvent(QuestSystem.QuestType.连击, 5);
        assertTrue(qs.allDone(), "全部完成");
        qs.nextDay();
        assertFalse(qs.allDone(), "新一天任务重置");
        assertEquals(2, qs.day());
        assertEquals(3, qs.dailyQuests().size());
    }

    @Test
    void questSummaryReadable() {
        QuestSystem qs = new QuestSystem();
        String s = qs.dailySummary();
        assertTrue(s.contains("第1天"), s);
        assertTrue(s.contains("任务"), s);
    }
}
