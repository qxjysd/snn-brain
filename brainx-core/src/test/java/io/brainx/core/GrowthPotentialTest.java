package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 成长潜力验证: 神经元数量具备成长至 2000 亿的架构潜力。
 */
public class GrowthPotentialTest {

    @Test
    void startsAtMicroScale() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        assertEquals(96, gp.physicalNeurons());
        assertEquals(1, gp.abstractionFactor());
        assertEquals(96, gp.totalNeuronCapacity(), "初始 96 神经元");
        assertFalse(gp.reachedHumanX2());
    }

    @Test
    void growsWithLearning() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        // 学 2 词 → 抽象×10 (神经群级)
        gp.update(2, 0);
        assertEquals(10, gp.abstractionFactor());
        assertEquals(960, gp.totalNeuronCapacity());
        // 学 5 词 → 抽象×10^3
        gp.update(5, 0);
        assertEquals(1_000, gp.abstractionFactor());
        // 学 10 词 → 抽象×10^6
        gp.update(10, 0);
        assertEquals(1_000_000, gp.abstractionFactor());
        assertEquals(96_000_000, gp.totalNeuronCapacity(), "9.6千万神经元容量");
    }

    @Test
    void reachesHumanX2Potential() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        // 满成长 (40词 或 12级) → 最大抽象倍数
        gp.update(40, 12);
        assertTrue(gp.totalNeuronCapacity() >= GrowthPotential.HUMAN_BRAIN_X2,
                "总容量应≥2000亿, got=" + gp.totalNeuronCapacity());
        assertTrue(gp.reachedHumanX2(), "应达到2000亿潜力");
        assertEquals(1.0, gp.growthProgress(), 0.01);
    }

    @Test
    void physicalExpansionSupported() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        // 高端机: 扩展物理规模
        gp.expandPhysical(400);  // 96 → 496
        assertEquals(496, gp.physicalNeurons());
        // 潜力仍可达 2000 亿 (最大抽象倍数按物理数重算)
        gp.update(40, 12);
        assertTrue(gp.reachedHumanX2(), "扩展物理后仍可达2000亿潜力");
    }

    @Test
    void progressTracksGrowth() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        assertEquals(0, gp.growthProgress(), 0.01);
        gp.update(2, 0);
        assertTrue(gp.growthProgress() > 0, "成长后进度应>0");
        gp.update(40, 12);
        assertEquals(1.0, gp.growthProgress(), 0.01);
        // 摘要可读
        assertTrue(gp.summary().contains("2000亿"), gp.summary());
        assertTrue(gp.growthPath().contains("成长阶段"));
    }

    @Test
    void brainGrowsWithLearning() {
        Brain brain = Brain.simpleBrain();
        // 学 3 个不同词 → 词数 3 → 成长触发
        for (int w = 0; w < 3; w++) {
            double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
            for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.1 + w * 0.3;
            for (int e = 0; e < 3; e++) brain.learnVisualWord(img, w);
        }
        assertTrue(brain.growthPotential().totalNeuronCapacity() > 96,
                "学习后神经元容量应增长, got=" + brain.growthPotential().totalNeuronCapacity());
        assertTrue(brain.growthSummary().contains("神经元"));
    }

    @Test
    void maxAbstractionCalculated() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        // 96 物理 → 最大抽象 = ceil(2000亿/96) ≈ 2.09×10^9
        long expected = (GrowthPotential.HUMAN_BRAIN_X2 + 95) / 96;
        assertEquals(expected, gp.maxAbstraction());
        // 潜力: 96 × maxAbstraction ≥ 2000亿 (向上取整保证)
        assertTrue(96L * gp.maxAbstraction() >= GrowthPotential.HUMAN_BRAIN_X2);
    }
}
