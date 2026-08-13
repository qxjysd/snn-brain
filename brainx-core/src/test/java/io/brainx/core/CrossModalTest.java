package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 多模态学习与输出验证 (v5.4):
 *   - 跨模态绑定: 视觉+听觉同时学同一词 → 原型绑定
 *   - 融合识别: 双模态一致 → 融合置信度提升; 不一致 → 分离降级
 *   - 跨模态回忆: 视觉→听觉原型, 听觉→视觉原型 (输出侧)
 */
public class CrossModalTest {

    private static double[] visPat(int c, int V) {
        double[] p = new double[V];
        for (int i = 0; i < V; i++) p[i] = ((i / (V / 8)) == c) ? 0.9 : 0.1;
        return p;
    }

    private static double[] audPat(int c, int B) {
        double[] p = new double[B];
        // 不同频率正弦: 频带能量编码对频率敏感 (相位会被幅度谱丢弃)
        for (int i = 0; i < B; i++) p[i] = 0.5 + 0.4 * Math.sin(i * (0.5 + c * 0.5));
        return p;
    }

    @Test
    void crossModalBindingLearning() {
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM, B = AudioNeuralEncoder.BANDS;
        // 4 词跨模态绑定学习 (看到+听到同一物体)
        for (int e = 0; e < 5; e++)
            for (int c = 0; c < 4; c++) brain.learnCrossModal(visPat(c, V), audPat(c, B), c);
        // 绑定成功
        for (int c = 0; c < 4; c++) {
            assertTrue(brain.crossModalMemory().bindCount(c) > 0, "词" + c + "应绑定 (count>0)");
            assertTrue(brain.crossModalMemory().isBound(c), "词" + c + "应已绑定");
        }
        // 双模态各自识别正确 (概念共享)
        String[] vocab = {"你好", "苹果", "猫", "狗"};
        int hitV = 0, hitA = 0;
        for (int c = 0; c < 4; c++) {
            if (brain.recognizeVisual(visPat(c, V)).equals(vocab[c])) hitV++;
            if (brain.recognizeAuditory(audPat(c, B)).equals(vocab[c])) hitA++;
        }
        assertTrue(hitV >= 3, "视觉识别应 ≥3/4, got " + hitV);
        assertTrue(hitA >= 3, "听觉识别应 ≥3/4, got " + hitA);
    }

    @Test
    void multimodalFusionRecognition() {
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM, B = AudioNeuralEncoder.BANDS;
        for (int e = 0; e < 25; e++) {
            for (int c = 0; c < 2; c++) brain.learnCrossModal(visPat(c, V), audPat(c, B), c);
        }
        // 一致输入: 视觉猫 + 听觉猫 → 融合置信度 ≥ 单模态
        String[] fused = brain.recognizeMultiModal(visPat(1, V), audPat(1, B));
        assertEquals("苹果", fused[0], "一致输入应融合到共同词");
        double fusedConf = Double.parseDouble(fused[1]);
        double singleConf = Double.parseDouble(brain.recognizeVisualWithConfidence(visPat(1, V))[1]);
        assertTrue(fusedConf >= singleConf - 0.05,
                "融合置信度应 ≥ 单模态: fused=" + fusedConf + " single=" + singleConf);
        // Φ 整合度
        assertTrue(Double.parseDouble(fused[2]) > 0.3, "一致输入 Φ 应高: " + fused[2]);
    }

    @Test
    void crossModalRecall() {
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM, B = AudioNeuralEncoder.BANDS;
        double[] vis = visPat(2, V);
        double[] aud = audPat(2, B);
        for (int e = 0; e < 8; e++) brain.learnCrossModal(vis, aud, 2);
        // 视觉 → 回忆听觉原型 (应接近绑定时的听觉特征)
        double[] recalledAud = brain.recallAudioFromVisual(vis);
        assertNotNull(recalledAud, "视觉应能回忆听觉原型");
        double simA = CrossModalMemory.similarity(recalledAud, aud);
        assertTrue(simA > 0.8, "回忆的听觉原型应相似于绑定特征, sim=" + simA);
        // 听觉 → 回忆视觉原型
        double[] recalledVis = brain.recallVisualFromAudio(aud);
        assertNotNull(recalledVis, "听觉应能回忆视觉原型");
        double simV = CrossModalMemory.similarity(recalledVis, vis);
        assertTrue(simV > 0.8, "回忆的视觉原型应相似于绑定特征, sim=" + simV);
        // 摘要可读
        assertTrue(brain.crossModalMemory().summary().contains("跨模态"));
    }

    @Test
    void multimodalConflictDetected() {
        Brain brain = Brain.simpleBrain();
        int V = VisualNeuralEncoder.OUTPUT_DIM, B = AudioNeuralEncoder.BANDS;
        for (int e = 0; e < 10; e++) {
            for (int c = 0; c < 2; c++) brain.learnCrossModal(visPat(c, V), audPat(c, B), c);
        }
        // 冲突输入: 视觉猫 + 听觉苹果 → 分离 (融合置信度打折)
        String[] fused = brain.recognizeMultiModal(visPat(1, V), audPat(0, B));
        double fusedConf = Double.parseDouble(fused[1]);
        double singleConf = Double.parseDouble(brain.recognizeVisualWithConfidence(visPat(1, V))[1]);
        assertTrue(fusedConf < singleConf,
                "冲突输入融合置信度应打折: fused=" + fusedConf + " single=" + singleConf);
        // Φ 应低 (不一致)
        assertTrue(Double.parseDouble(fused[2]) < 0.5, "冲突输入 Φ 应低: " + fused[2]);
    }
}
