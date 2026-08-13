package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 算力自适应验证: 神经元数量根据手机算力提升。
 */
public class ComputeAdaptiveTest {

    @Test
    void lowComputeSmallNeurons() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        // 低算力: 4核 / 500MB / 帧120ms卡顿 → 低档
        gp.adjustToCompute(4, 500, 120);
        assertEquals(GrowthPotential.ComputeTier.低, gp.computeTier());
        assertEquals(48, gp.physicalNeurons(), "低算力应少物理神经元");
        // 满成长后潜力仍可达2000亿 (抽象倍数补偿)
        gp.update(40, 12);
        assertTrue(gp.reachedHumanX2(), "低算力档满成长后潜力应达2000亿");
    }

    @Test
    void highComputeMoreNeurons() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        // 高算力: 8核 / 8GB / 帧20ms流畅 → 极高
        gp.adjustToCompute(8, 8000, 20);
        assertEquals(GrowthPotential.ComputeTier.极高, gp.computeTier());
        assertEquals(384, gp.physicalNeurons(), "高算力应多物理神经元");
        // 满成长后潜力保持2000亿
        gp.update(40, 12);
        assertTrue(gp.totalNeuronCapacity() >= GrowthPotential.HUMAN_BRAIN_X2,
                "潜力保持2000亿");
    }

    @Test
    void midComputeDefault() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        // 中算力: 6核 / 2GB / 帧40ms → 中
        gp.adjustToCompute(6, 2000, 40);
        assertEquals(GrowthPotential.ComputeTier.中, gp.computeTier());
        assertEquals(96, gp.physicalNeurons());
    }

    @Test
    void dynamicAdjustment() {
        GrowthPotential gp = GrowthPotential.defaultParams();
        // 先低算力 → 后高算力 (自适应提升)
        gp.adjustToCompute(4, 500, 120);
        assertEquals(48, gp.physicalNeurons());
        gp.adjustToCompute(8, 8000, 15);
        assertEquals(384, gp.physicalNeurons(), "算力提升→物理神经元提升");
        // 摘要显示算力档位
        assertTrue(gp.summary().contains("算力"), gp.summary());
    }

    @Test
    void brainAdaptsToCompute() {
        Brain brain = Brain.simpleBrain();
        // 高算力 → 神经元规模提升
        brain.adjustNeuronsToCompute(8, 8000, 15);
        assertEquals(384, brain.growthPotential().physicalNeurons());
        // 低算力 → 缩小
        brain.adjustNeuronsToCompute(4, 500, 120);
        assertEquals(48, brain.growthPotential().physicalNeurons());
        // 学习仍工作
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        assertTrue(brain.recognizeVisual(img).equals(brain.vocabulary(0)));
    }

    @Test
    void potentialPreservedAtAllTiers() {
        for (GrowthPotential.ComputeTier tier : GrowthPotential.ComputeTier.values()) {
            GrowthPotential gp = GrowthPotential.defaultParams();
            gp.setComputeTier(tier);
            gp.update(40, 12);  // 满成长
            assertTrue(gp.reachedHumanX2(),
                    tier.name() + " 档潜力应保持2000亿");
        }
    }
}
