package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 时序预测与延迟脑补验证 (v5.4, 人脑预测能力):
 *   - 匀速运动外推: 预测帧 ≈ 真实下一帧
 *   - 延迟补偿 (脑补): 感官延迟期用预测填补, 感知不跳变
 *   - 噪声鲁棒: 平滑防噪声放大
 *   - 突变纠正: 场景突变后预测快速恢复
 */
public class PredictionTest {

    /** 匀速移动模式: 正弦波缓慢相位移动 (帧 t 的特征) */
    private static double[] frame(int t, int dim) {
        double[] f = new double[dim];
        for (int i = 0; i < dim; i++) f[i] = 0.5 + 0.4 * Math.sin((i + t * 0.5) / 8.0);
        return f;
    }

    private static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return na == 0 || nb == 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    @Test
    void predictsUniformMotion() {
        // 匀速运动: 预测帧应与真实下一帧高度相似
        FeaturePredictor p = new FeaturePredictor(128);
        for (int t = 0; t < 10; t++) p.predictNext(frame(t, 128));   // 学习运动
        double[] pred = p.predictNext(frame(10, 128));
        double[] actual = frame(11, 128);
        assertTrue(cosine(pred, actual) > 0.95,
                "匀速运动预测应接近真实: cos=" + cosine(pred, actual));
    }

    /** 匀速平移模式: 高空间频率正弦以 4 采样/帧匀速右移 (处处变化 + 滞后差异显著) */
    private static double[] movingGradient(int t, int dim) {
        double[] f = new double[dim];
        for (int i = 0; i < dim; i++) {
            f[i] = 0.5 + 0.4 * Math.sin((i - t * 4) * 0.4);   // 周期 ~15.7 采样
        }
        return f;
    }

    @Test
    void delayCompensationBrainFillsGap() {
        // 延迟脑补: 感官延迟 3 帧 → 用预测填补, 感知与真实"当前"差异小
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        // 正常帧流: 大脑持续预测 (匀速移动 — 平滑追踪场景)
        for (int t = 0; t < 10; t++) {
            brain.predictNextVisual(movingGradient(t, V));
        }
        // 延迟发生: 当前真实帧是 t=13, 大脑最后一次看到 t=10 → 脑补 3 步
        double[] filled = brain.perceiveVisualWithDelayCompensation(null, 3);
        double[] truth = movingGradient(13, V);
        assertTrue(cosine(filled, truth) > 0.95,
                "匀速运动脑补应接近真实: cos=" + cosine(filled, truth));
        // 无预测对照: 滞后的 t=10 帧 → 位移 12 采样, 差异显著
        double[] stale = movingGradient(10, V);
        assertTrue(cosine(filled, truth) > cosine(stale, truth) + 0.02,
                "脑补应明显优于滞后帧: fill=" + cosine(filled, truth)
                        + " stale=" + cosine(stale, truth));
    }

    @Test
    void noiseRobust() {
        // 噪声输入下预测仍合理 (平滑防放大)
        FeaturePredictor p = new FeaturePredictor(64);
        java.util.Random r = new java.util.Random(3);
        double[] f0 = frame(0, 64);
        p.predictNext(f0);
        for (int t = 1; t < 6; t++) {
            double[] noisy = frame(t, 64).clone();
            for (int i = 0; i < noisy.length; i++) noisy[i] += (r.nextDouble() - 0.5) * 0.05;  // ±0.05 噪声
            p.predictNext(noisy);
        }
        double[] pred = p.predictNext(frame(6, 64));
        double[] actual = frame(7, 64);
        assertTrue(cosine(pred, actual) > 0.9, "噪声下预测仍应接近: cos=" + cosine(pred, actual));
    }

    @Test
    void recoversAfterSuddenChange() {
        // 场景突变: 预测短暂失准, 但速度 EMA 快速纠正
        FeaturePredictor p = new FeaturePredictor(64);
        for (int t = 0; t < 8; t++) p.predictNext(frame(t, 64));
        // 突变: 切换到相反移动
        for (int t = 8; t < 20; t++) {
            p.predictNext(frame(-t, 64));   // 反向移动
        }
        // 纠正后应能预测反向运动
        double[] pred = p.predictNext(frame(-20, 64));
        double[] actual = frame(-21, 64);
        assertTrue(cosine(pred, actual) > 0.9,
                "突变后应快速纠正: cos=" + cosine(pred, actual));
    }

    @Test
    void confidenceReflectsMotion() {
        FeaturePredictor p = new FeaturePredictor(64);
        assertEquals(0, p.confidence(), "未初始化置信 0");
        p.predictNext(frame(0, 64));
        // 静止场景 → 高置信
        for (int t = 0; t < 5; t++) p.predictNext(frame(0, 64));
        assertTrue(p.confidence() > 0.9, "静止应高置信: " + p.confidence());
        // 快速运动 → 置信下降
        FeaturePredictor fast = new FeaturePredictor(64);
        for (int t = 0; t < 10; t++) fast.predictNext(frame(t * 3, 64));
        assertTrue(fast.confidence() < 0.9, "快速运动置信应降: " + fast.confidence());
    }

    @Test
    void brainPredictionIntegration() {
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM;
        for (int t = 0; t < 5; t++) brain.predictNextVisual(frame(t, V));
        assertTrue(brain.predictionSummary().contains("预测"), "摘要含预测能力");
        assertTrue(brain.visualPredictionConfidence() >= 0, "置信度可读");
        // 重置后重新学习
        brain.resetPredictors();
        assertEquals(0, brain.visualPredictionConfidence(), "重置后置信归零");
    }
}
