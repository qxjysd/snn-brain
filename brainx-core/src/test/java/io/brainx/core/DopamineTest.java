package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 多巴胺奖赏 (错误驱动学习) + 预测引擎验证。
 * 来源: 《醉醺醺的脑科学》第28篇 (Yael Niv) + 第9篇 (多巴胺奖赏回路)。
 */
public class DopamineTest {

    // ===== 多巴胺系统 (Schultz RPE) =====

    @Test
    void surpriseRaisesDopamine() {
        DopamineSystem ds = new DopamineSystem();
        // 预期 0.5, 实际全奖励 → 正向 RPE (意外惊喜)
        double rpe = ds.learnEvent(1.0, 1.0);
        assertTrue(rpe > 0, "意外奖励应产生正RPE, rpe=" + rpe);
        assertTrue(ds.dopamine() > 1.0, "正向RPE应提升多巴胺");
        assertEquals(0.5, ds.lastRPE(), 0.001);  // 首次 RPE = 1.0 - 0.5(初始预期)
    }

    @Test
    void expectedRewardNoSurprise() {
        DopamineSystem ds = new DopamineSystem();
        // 先学习成功 → 预期升高
        ds.learnEvent(1.0, 1.0);
        ds.learnEvent(1.0, 1.0);
        ds.learnEvent(1.0, 1.0);
        double expectedAfter = ds.expectedReward();
        // 再成功 → RPE 接近 0 (符合预期, 学习减弱)
        double rpe = ds.learnEvent(1.0, 1.0);
        assertTrue(Math.abs(rpe) < Math.abs(1.0 - 0.5), "预期校准后RPE应减小, rpe=" + rpe);
        assertTrue(expectedAfter > 0.6, "预期应向实际漂移");
    }

    @Test
    void predictionFailureDropsDopamine() {
        DopamineSystem ds = new DopamineSystem();
        ds.learnEvent(1.0, 1.0);  // 先建立高预期
        ds.learnEvent(1.0, 1.0);
        double before = ds.dopamine();
        double rpe = ds.learnEvent(0.0, 1.0);  // 预期落空
        assertTrue(rpe < 0, "失败应产生负RPE");
        assertTrue(ds.dopamine() < before, "负RPE应降低多巴胺");
    }

    @Test
    void rpeDrivesLearningRate() {
        DopamineSystem ds = new DopamineSystem();
        double lr0 = ds.learningRate();
        ds.learnEvent(0.0, 1.0);  // 大错误 → 学习率提升 (错误驱动)
        assertTrue(ds.learningRate() > lr0, "错误应提升学习率: " + lr0 + "→" + ds.learningRate());
    }

    @Test
    void dopamineStatusDescribes() {
        DopamineSystem ds = new DopamineSystem();
        assertTrue(ds.dopamineStatus().length() > 0);
        ds.learnEvent(1.0, 1.0);
        assertNotNull(ds.summary());
    }

    // ===== 预测引擎 (大脑=科学家) =====

    @Test
    void predictionBuildsFromObservations() {
        PredictiveEngine pe = new PredictiveEngine();
        // 观察: 苹果出现多
        pe.observe("苹果");
        pe.observe("苹果");
        pe.observe("苹果");
        pe.observe("猫");
        assertEquals("苹果", pe.predict(), "多数先验应成为预测");
        assertTrue(pe.prior("苹果") > pe.prior("猫"), "苹果先验应更高");
    }

    @Test
    void correctPredictionLowError() {
        PredictiveEngine pe = new PredictiveEngine();
        pe.observe("苹果");
        pe.observe("苹果");
        String pred = pe.predict();
        double error = pe.verifyAndConclude(pred, 0.9, "苹果");
        assertTrue(error < 0.2, "正确预测误差应小, error=" + error);
    }

    @Test
    void wrongPredictionHighErrorAndCorrection() {
        PredictiveEngine pe = new PredictiveEngine();
        pe.observe("苹果");
        pe.observe("苹果");
        pe.observe("猫");  // 猫也出现, 但苹果先验高
        String pred = pe.predict();
        double error = pe.verifyAndConclude(pred, 0.9, "猫");
        assertEquals(1.0, error, "错误预测误差应为1");
        // 修正后猫的先验提升
        assertTrue(pe.prior("猫") > 0.3, "错误后应修正先验");
    }

    @Test
    void predictiveEngineSummarizes() {
        PredictiveEngine pe = new PredictiveEngine();
        pe.observe("书本");
        assertNotNull(pe.summary());
        assertTrue(pe.summary().contains("预测引擎"));
    }

    // ===== 集成 =====

    @Test
    void brainDopamineIntegration() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        // 学习 (建立先验)
        for (int e = 0; e < 3; e++) brain.learnVisualWord(img, 0);
        // 识别 (驱动预测+多巴胺)
        String[] r = brain.recognizeVisualWithConfidence(img);
        assertNotNull(brain.dopamineSystem());
        assertNotNull(brain.predictiveEngine());
        assertTrue(brain.predictiveEngine().priorCount() > 0, "学习应建立先验");
        // 教育事件驱动多巴胺
        brain.rewardEvent(true, 1.0);
        assertTrue(brain.dopamineSystem().lastRPE() > -0.5);
        // 摘要可读
        assertTrue(brain.dopamineSummary().contains("多巴胺"));
        assertTrue(brain.predictiveSummary().contains("预测引擎"));
    }
}
