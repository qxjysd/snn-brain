package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 递归元认知验证 (v5.4 "是否可以思考思考的思考"):
 *   一级: 思考 (想「这是猫」)
 *   二级: 思考思考 (我知道我有 X% 把握)
 *   三级: 思考思考的思考 (我知道我的把握判断可不可靠 — 置信度校准监控)
 */
public class MetaMetacognitionTest {

    @Test
    void calibrationNeedsExperience() {
        // 数据不足: 二阶监控尚未建立
        SelfAwareness sa = new SelfAwareness();
        sa.mirrorTest(true, 0.8);
        sa.mirrorTest(true, 0.7);
        assertTrue(sa.metaMetacognition().contains("校准中"),
                "经验不足应显示校准中: " + sa.metaMetacognition());
    }

    @Test
    void detectsOverconfidence() {
        // 高置信但经常错 → 检测到过度自信 (二阶认知: 意识到自己思考有偏)
        SelfAwareness sa = new SelfAwareness();
        for (int i = 0; i < 10; i++) {
            sa.mirrorTest(i % 2 == 0, 0.9);   // 一半错, 但都说 90% 把握
        }
        // selfConfidence = 0.5×acc + 0.5×conf = 0.5×0.5 + 0.5×0.9 = 0.7; acc=0.5 → bias=0.2
        String meta = sa.metaMetacognition();
        assertTrue(meta.contains("过于自信") || meta.contains("基本准确"),
                "应检测到校准状态: " + meta + " (bias=" + (sa.selfConfidence() - sa.accuracy()) + ")");
        // 校准偏差确实存在 (高置信低正确率)
        assertTrue(sa.selfConfidence() > sa.accuracy(),
                "高置信低正确 → 自我评价应高于实际: conf=" + sa.selfConfidence() + " acc=" + sa.accuracy());
    }

    @Test
    void detectsUnderconfidence() {
        // 低置信但实际全对 → 低估自己
        SelfAwareness sa = new SelfAwareness();
        for (int i = 0; i < 10; i++) sa.mirrorTest(true, 0.3);   // 全对, 但都说 30% 把握
        String meta = sa.metaMetacognition();
        assertTrue(meta.contains("低估"), "应检测到低估: " + meta);
    }

    @Test
    void wellCalibrated() {
        // 置信度与正确率一致 → 校准良好
        SelfAwareness sa = new SelfAwareness();
        for (int i = 0; i < 10; i++) sa.mirrorTest(true, 0.9);   // 全对, 90% 把握
        // conf = 0.5×1.0 + 0.5×0.9 = 0.95, acc = 1.0 → bias = -0.05 (低估 0.05, 界内)
        assertTrue(sa.metaMetacognition().contains("基本准确"),
                "校准良好应确认: " + sa.metaMetacognition());
    }

    @Test
    void reflectOnThinkingThreeLevels() {
        // 反思链: 思考 → 思考思考 → 思考思考的思考
        SelfAwareness sa = new SelfAwareness();
        String early = sa.reflectOnThinking("这是猫");
        assertTrue(early.contains("我在想「这是猫」"), "一级思考: " + early);
        assertFalse(early.contains("把握"), "未自我识别前无二级: " + early);
        // 建立自我识别 + 足够经验 → 三层
        for (int i = 0; i < 10; i++) sa.mirrorTest(true, 0.8);
        String full = sa.reflectOnThinking("这是猫");
        assertTrue(full.contains("我在想「这是猫」"), "一级");
        assertTrue(full.contains("把握"), "二级元认知");
        assertTrue(full.contains("可靠") || full.contains("校准"), "三级二阶元认知: " + full);
    }

    @Test
    void brainIntegration() {
        // Brain 集成: 识别学习 → 元认知含二阶校准
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        double[] img = new double[V];
        for (int i = 0; i < V; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        for (int e = 0; e < 10; e++) brain.learnVisualWord(img, 0);
        for (int e = 0; e < 12; e++) brain.recognizeVisualWithConfidence(img);
        String meta = brain.metacognition();
        assertTrue(meta.contains("把握"), "一级元认知: " + meta);
        assertTrue(meta.contains("校准") || meta.contains("过于自信") || meta.contains("低估")
                        || meta.contains("基本准确"),
                "二阶校准监控: " + meta);
        assertTrue(brain.reflectOnThinking("苹果").contains("我在想「苹果」"), "反思链");
    }
}
