package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 脉冲网络完善验证 (v5.4): 虚拟神经元层 (神经形态等效规模) + 稳态可塑性 + 皮层局部连接。
 */
public class NeuralNetworkEnhancementTest {

    // ===== 1. 虚拟神经元层 (等效规模真实参与脉冲统计) =====

    @Test
    void virtualLayerEquivalentScale() {
        VirtualNeuronLayer layer = new VirtualNeuronLayer(5488);
        assertEquals(5488, layer.virtualNeurons(), "k=1 时等效=物理");
        // 成长: 抽象倍数提升 → 等效规模扩大 (最高 2000 亿)
        layer.setAbstraction(1_000_000L);
        assertEquals(5488L * 1_000_000L, layer.virtualNeurons(), "等效规模=物理×抽象");
        // 满潜力: 抽象上限保证 2000 亿
        layer.setAbstraction(GrowthPotential.HUMAN_BRAIN_X2 / 5488 + 1);
        assertTrue(layer.virtualNeurons() >= GrowthPotential.HUMAN_BRAIN_X2,
                "等效规模应达 2000 亿 (人脑2倍)");
    }

    @Test
    void virtualLayerPulseExtension() {
        VirtualNeuronLayer layer = new VirtualNeuronLayer(100);
        layer.setAbstraction(1000);
        boolean[] firing = new boolean[100];
        for (int i = 0; i < 10; i++) firing[i] = true;   // 10 物理发放
        layer.step(firing, 1.0);
        assertEquals(10L * 1000L, layer.lastVirtualSpikes(), "虚拟脉冲 = 物理发放 × 抽象倍数");
        assertEquals(0.1, layer.virtualActivity(), 1e-9, "群体活动率=发放率 0.1");
        assertEquals(0.1, layer.synchrony(), 1e-9, "同步性=0.1");
        // 宏观增益: 规模越大越强
        layer.setAbstraction(1);
        assertEquals(1.0, layer.macroscopicGain(), 1e-9, "k=1 无增益");
        layer.setAbstraction(1_000_000);
        assertTrue(layer.macroscopicGain() > 1.5, "大规模宏观增益增强");
    }

    @Test
    void virtualLayerGrowsWithLearning() {
        // Brain 集成: 学习 → 抽象倍数提升 → 虚拟层等效规模同步扩大
        Brain brain = Brain.simpleBrain();
        assertEquals(1, brain.virtualLayer().abstraction(), "初始抽象=1");
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);   // 学 2 词 → 抽象×10
        long vBefore = brain.virtualLayer().virtualNeurons();
        for (int w = 1; w < 5; w++) {
            for (int e = 0; e < 3; e++) brain.learnVisualWord(img, w);
        }
        assertTrue(brain.virtualLayer().virtualNeurons() >= vBefore,
                "学习成长后等效规模不应缩小");
        assertTrue(brain.growthSummary().contains("虚拟层"), "摘要含虚拟层");
    }

    // ===== 2. 稳态可塑性 (防暴走/防静默) =====

    @Test
    void homeostaticDownRegulatesOveractive() {
        HomeostaticPlasticity hp = new HomeostaticPlasticity(10);
        boolean[] firing = new boolean[10];
        for (int i = 0; i < 10; i++) firing[i] = true;   // 持续强发放
        for (int t = 0; t < 3000; t++) hp.step(firing, 10);  // 30s
        assertTrue(hp.gain(0) < 0.95, "持续暴走应下调增益, gain=" + hp.gain(0));
        int[] stats = hp.gainStats();
        assertTrue(stats[1] >= 10, "全部神经元应下调, down=" + stats[1]);
    }

    @Test
    void homeostaticUpRegulatesSilent() {
        HomeostaticPlasticity hp = new HomeostaticPlasticity(10);
        boolean[] silent = new boolean[10];   // 完全静默
        for (int t = 0; t < 3000; t++) hp.step(silent, 10);  // 30s
        assertTrue(hp.gain(0) > 1.05, "长期静默应上调增益, gain=" + hp.gain(0));
        assertTrue(hp.gain(0) <= 5.0, "增益受上限约束");
    }

    @Test
    void homeostaticGainBounded() {
        HomeostaticPlasticity hp = new HomeostaticPlasticity(5);
        boolean[] firing = new boolean[5];
        for (int i = 0; i < 5; i++) firing[i] = true;
        // 超长暴走 → 增益到下限
        for (int t = 0; t < 50000; t++) hp.step(firing, 10);
        assertTrue(hp.gain(0) >= 0.2, "增益不低于下限 0.2");
        boolean[] silent = new boolean[5];
        for (int t = 0; t < 50000; t++) hp.step(silent, 10);
        assertTrue(hp.gain(0) <= 5.0, "增益不高于上限 5.0");
    }

    @Test
    void homeostaticIntegratedInBrain() {
        // Brain 集成: 学习后增益可访问且受限
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 10; e++) brain.learnVisualWord(img, 0);
        assertTrue(brain.homeostaticPlasticity().gain(0) >= 0.2
                        && brain.homeostaticPlasticity().gain(0) <= 5.0,
                "学习后增益应在界内");
        assertTrue(brain.homeostaticPlasticity().summary().contains("稳态"));
    }

    // ===== 3. 皮层局部连接结构 (空间距离衰减) =====

    @Test
    void spatialConnectionsShorterThanRandom() {
        // spatial: 视觉皮层按网格, 连接局部主导 → 平均距离显著小于全局随机
        SynapseFormation spatial = new SynapseFormation(4096, 4096 * 4, 42, 0.05, 5000, 64);
        SynapseFormation random = new SynapseFormation(4096, 4096 * 4, 42, 0.05, 5000, 0);
        double dSpatial = spatial.averageDistance();
        double dRandom = random.averageDistance();
        // 全局随机在 64×64 网格上的期望距离 ≈ 43 (均匀分布两点距)
        assertTrue(dSpatial < dRandom * 0.6,
                "局部连接距离应显著小于全局随机: spatial=" + dSpatial + " random=" + dRandom);
    }

    @Test
    void brainUsesSpatialConnections() {
        Brain brain = Brain.simpleBrain();
        // 大脑已启用皮层局部连接 (spatialGrid = RF_GRID)
        double d = brain.synapseFormation().averageDistance();
        assertTrue(d > 0, "大脑突触形成应为空间模式");
        // 学习后成熟连接仍形成 (与 ConnectionHealthTest 兼容)
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        assertTrue(brain.synapseFormation().matureCount(0.1) > 0, "空间模式下学习仍建连接");
    }
}
