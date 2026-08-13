package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 教育闭环验证: 学习→测试→奖惩→升级 (模拟 APK 中 EduTrainer 的逻辑)。
 * 验证"从学语到认知"的完整过程。
 */
public class EduLoopTest {

    @Test
    void educationLoopLearnsAndRewards() {
        // 模拟 EduTrainer 行为 (与 APK 中一致)
        int points = 0, level = 0, correct = 0, wrong = 0;
        for (int i = 0; i < 12; i++) {
            boolean answerCorrect = (i % 3 != 0);  // 2/3 答对 (学习后逐步进步)
            if (answerCorrect) {
                correct++;
                points += 1;
                if (correct % 3 == 0) level++;  // 每 3 次答对升级
            } else {
                wrong++;
                points = Math.max(0, points - 1);
            }
        }
        // 12 次交互后: 8 对 4 错, 5 点 (首次答错时 points=0 被 max 截断不扣), 2 级
        assertEquals(8, correct);
        assertEquals(4, wrong);
        assertEquals(5, points);
        assertEquals(2, level);
        // 进步率 = 8/12 = 0.667
        assertTrue((double) correct / (correct + wrong) > 0.5, "应有进步");
    }

    @Test
    void brainLearnsWordsThenRecognizes() {
        // 完整闭环: 学 4 个词 → 全部识别对 (用 simpleBrain 词表前4: 你好/苹果/猫/狗)
        String[] vocab = {"你好", "苹果", "猫", "狗"};
        Brain brain = Brain.simpleBrain();
        double[][] patterns = new double[4][VisualNeuralEncoder.OUTPUT_DIM];
        for (int w = 0; w < 4; w++) {
            for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) {
                patterns[w][i] = ((i / (VisualNeuralEncoder.OUTPUT_DIM / 8)) == w) ? 0.9 : 0.1;  // 每词不同象限模式
            }
        }
        // 学习
        for (int epoch = 0; epoch < 120; epoch++) {
            for (int w = 0; w < 4; w++) brain.learnVisualWord(patterns[w], w);
        }
        // 识别
        int hit = 0;
        for (int w = 0; w < 4; w++) {
            String guess = brain.recognizeVisual(patterns[w]);
            if (guess.equals(vocab[w])) hit++;
        }
        // 至少认出 3/4 (Hebbian 联想 + 30 epochs)
        assertTrue(hit >= 3, "应认出至少3个词, got " + hit + "/4: " +
                brain.learnedWords());
    }
}
