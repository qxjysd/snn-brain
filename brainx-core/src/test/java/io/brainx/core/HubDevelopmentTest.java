package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 中枢脉冲网络神经发育验证 (书中: 突触过度产生→修剪)。
 * 修复前: 中枢初始稀疏随机 (40% 连接)。
 * 修复后: 初始全连接 (100%), 学习后 Hebbian 增强 + 用进废退修剪。
 */
public class HubDevelopmentTest {

    @Test
    void initiallyFullyConnected() {
        // 初始全连接 (突触过度产生: 婴儿大脑过度连接)
        CentralHub hub = CentralHub.defaultParams(8);
        assertEquals(1.0, hub.connectionDensity(), 1e-9, "初始应为全连接 100%");
        assertEquals(hub.totalConnections(), hub.activeConnections(),
                "初始所有连接应有效");
        // 权重为小正值 (非零)
        assertTrue(hub.activeConnections() == hub.totalConnections());
    }

    @Test
    void learningPrunesConnections() {
        CentralHub hub = CentralHub.defaultParams(8);
        // 部分输入活跃 (前4个强): 活跃连接增强保留, 不活跃连接修剪
        for (int i = 0; i < 300; i++) {
            hub.clearInput();
            double[] input = new double[8];
            input[0] = 0.9; input[1] = 0.9; input[2] = 0.9; input[3] = 0.9;
            hub.inject(0, input);
            hub.step(1.0);
            hub.learnWeights(1.0);
        }
        // 学习后: 不活跃输入连接被修剪, 密度下降
        assertTrue(hub.connectionDensity() < 1.0,
                "学习后连接应被修剪, density=" + hub.connectionDensity());
        // 但活跃连接保留 (Hebbian 增强)
        assertTrue(hub.activeConnections() > 0, "应有连接保留");
    }

    @Test
    void inactiveConnectionsPruned() {
        CentralHub hub = CentralHub.defaultParams(4);
        // 全部输入 0 (无活动) → 连接应被修剪 (用进废退)
        hub.inject(0, new double[]{0.0, 0.0, 0.0, 0.0});
        for (int i = 0; i < 1000; i++) {
            hub.step(1.0);
            hub.learnWeights(1.0);  // 无共激活 → 衰减
        }
        // 长时间无活动 → 连接被修剪 (0.15→0.05 需~220步, 1000步全剪)
        assertTrue(hub.connectionDensity() < 0.1,
                "无活动连接应被修剪, density=" + hub.connectionDensity());
    }

    @Test
    void hebbianStrengthensActive() {
        CentralHub hub = CentralHub.defaultParams(4);
        // 记录初始权重
        double[][] before = hub.cloneWeights();
        // 强输入驱动中枢发放
        for (int i = 0; i < 30; i++) {
            hub.clearInput();
            hub.inject(0, new double[]{1.0, 1.0, 1.0, 1.0});
            hub.step(1.0);
            hub.learnWeights(1.0);
        }
        double[][] after = hub.cloneWeights();
        // Hebbian: 共激活连接应增强 (至少一些权重增加)
        boolean anyStronger = false;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (after[i][j] > before[i][j] + 0.01) { anyStronger = true; break; }
            }
        }
        assertTrue(anyStronger, "共激活连接应增强 (fire together, wire together)");
    }

    @Test
    void plasticityBothWays() {
        // 可增可减: 活跃连接增强, 不活跃连接修剪
        CentralHub hub = CentralHub.defaultParams(6);
        double[][] before = hub.cloneWeights();
        // 只有前2个输入活跃
        for (int i = 0; i < 40; i++) {
            hub.clearInput();
            double[] input = new double[6];
            input[0] = 1.0;
            input[1] = 1.0;
            hub.inject(0, input);
            hub.step(1.0);
            hub.learnWeights(1.0);
        }
        double[][] after = hub.cloneWeights();
        // 活跃输入的连接应更强 (平均)
        double activeAvg = 0, inactiveAvg = 0;
        int activeN = 0, inactiveN = 0;
        for (int i = 0; i < 6; i++) {
            activeAvg += after[i][0] + after[i][1];
            activeN += 2;
            for (int j = 2; j < 6; j++) {
                inactiveAvg += after[i][j];
                inactiveN++;
            }
        }
        activeAvg /= activeN;
        inactiveAvg /= inactiveN;
        assertTrue(activeAvg > inactiveAvg,
                "活跃连接应强于不活跃: active=" + activeAvg + " inactive=" + inactiveAvg);
    }

    @Test
    void brainDevelopmentOverLearning() {
        // 大脑层面: 初始全连接, 学习后连接可增可减 (发育)
        Brain brain = Brain.simpleBrain();
        double density0 = brain.centralHub().connectionDensity();
        assertEquals(1.0, density0, 1e-9, "初始全连接");
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        // 大量学习 → 中枢环路多次运行 → 连接可塑
        for (int e = 0; e < 10; e++) {
            brain.learnVisualWord(img, 0);
            brain.learnAuditoryWord(img, 1);
        }
        double densityAfter = brain.centralHub().connectionDensity();
        assertTrue(densityAfter <= density0, "学习后连接密度不应增加");
        // 摘要含密度信息
        assertTrue(brain.hubSummary().contains("连接密度"));
    }
}
