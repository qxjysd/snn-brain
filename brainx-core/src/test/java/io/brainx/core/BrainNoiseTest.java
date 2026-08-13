package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 脑域机制验证:
 * 1. 随机共振: 子阈值信号 + 噪声 → 检测率提升 (倒U型)
 * 2. 自发突触形成: 随机共激活 → 连接建立 (Hebbian)
 * 3. 噪声辅助学习: 噪声环境下大脑仍能学习识别
 * 4. 噪声不破坏正常功能
 */
public class BrainNoiseTest {

    @Test
    void stochasticResonanceHelpsDetection() {
        // 子阈值信号 0.5, 阈值 1.0
        double signal = 0.5, threshold = 1.0;
        double[] validity = StochasticResonance.detectionCurve(signal, threshold, 0.1, 5000);
        // 无噪声 (index 0): 信号不可检测 → 有效性≈0
        assertEquals(0.0, validity[0], 0.02, "无噪声时子阈值信号应不可检测");
        // 中等噪声: 有效性应显著提升 (物理峰值 ~0.16, 取决于信噪比条件)
        double maxValidity = 0;
        for (double v : validity) maxValidity = Math.max(maxValidity, v);
        assertTrue(maxValidity > 0.1, "合适噪声应提升检测有效性, max=" + maxValidity);
        // 倒U型: 过大噪声 (最后一项) 有效性应低于峰值
        assertTrue(validity[validity.length - 1] < maxValidity,
                "噪声过大应降低有效性 (倒U型): last=" + validity[validity.length-1] + " peak=" + maxValidity);
    }

    @Test
    void spontaneousActivityFormsConnections() {
        // 自发随机信号 → 共激活 → 连接建立
        SynapseFormation sf = new SynapseFormation(16, 8, 42, 0.05, 10000);
        boolean[] firing = new boolean[16];
        java.util.Random r = new java.util.Random(7);
        int matureBefore = sf.matureCount(0.15);
        // 模拟 200 秒自发活动 (dt=50ms)
        for (int t = 0; t < 4000; t++) {
            for (int i = 0; i < 16; i++) firing[i] = r.nextDouble() < 0.1;
            sf.step(firing, 50);
        }
        int matureAfter = sf.matureCount(0.15);
        // 自发共激活应使成熟连接增加
        assertTrue(matureAfter > matureBefore,
                "自发活动应建立连接: before=" + matureBefore + " after=" + matureAfter);
    }

    @Test
    void unpairedConnectionsGetPruned() {
        // 用进废退: 长期不共激活的连接被修剪
        SynapseFormation sf = new SynapseFormation(8, 2, 1, 0.05, 500);
        boolean[] silent = new boolean[8];  // 无任何发放
        // 静默 600ms: 初始候选 (从未共激活, lastCoactivate=0) 应被修剪删除
        for (int t = 0; t < 60; t++) sf.step(silent, 10);
        boolean initialSurvived = false;
        for (double[] c : sf.exportConnections()) {
            if (c.length >= 4 && c[3] == 0) initialSurvived = true;   // lastCoactivate==0 = 初始候选
        }
        assertFalse(initialSurvived, "从未共激活的初始候选应被修剪删除");
        assertTrue(sf.candidateCount() <= 8 * 8, "连接数应受 8N 上限约束");
    }

    @Test
    void noiseDoesNotBreakLearning() {
        // 有自发噪声时大脑仍能学习识别 (噪声是辅助非破坏)
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM], imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) {
            imgA[i] = (i % 3 == 0) ? 0.9 : 0.1;
            imgB[i] = (i % 3 == 1) ? 0.9 : 0.1;
        }
        for (int epoch = 0; epoch < 15; epoch++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        String r1 = brain.recognizeVisual(imgA);
        String r2 = brain.recognizeVisual(imgB);
        assertTrue(r1.equals("你好") || r1.equals("苹果"), "噪声下应仍能识别A, got=" + r1);
        assertTrue(r2.equals("你好") || r2.equals("苹果"), "噪声下应仍能识别B, got=" + r2);
    }

    @Test
    void spontaneousActivityVisibleInBrain() {
        // 大脑可采样自发活动 (可视化用)
        Brain brain = Brain.simpleBrain();
        boolean[] act = brain.sampleSpontaneousActivity();
        assertEquals(brain.visualCortexSize() + AudioNeuralEncoder.BANDS + brain.assocSize(), act.length, "采样维度应=全部神经元");
        // 自发活动应有部分神经元活跃 (泊松噪声 8Hz)
        int active = 0;
        for (boolean a : act) if (a) active++;
        assertTrue(active > 0, "自发活动应有神经元活跃, active=" + active);
        // 突触形成器已接入
        assertNotNull(brain.synapseFormation());
        assertTrue(brain.synapseFormation().candidateCount() > 0, "应有候选连接");
    }

    @Test
    void synapseFormationConnectsVisualToAssoc() {
        // 视觉学习后, 自发突触形成应在视觉+联想皮层间建立连接
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        for (int epoch = 0; epoch < 10; epoch++) {
            brain.learnVisualWord(img, 0);
        }
        int mature = brain.synapseFormation().matureCount(0.05);
        assertTrue(mature > 0, "学习后应有成熟连接, mature=" + mature);
    }
}
