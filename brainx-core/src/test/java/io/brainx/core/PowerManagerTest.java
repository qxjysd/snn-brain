package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 运行功率自适应验证 (防卡死):
 * 1. 卡顿 → 降档 (核心防卡死)
 * 2. 流畅 → 试探升档
 * 3. 内存不足 → 降档
 * 4. 降档后锁定防抖
 * 5. 档位→模型参数生效
 */
public class PowerManagerTest {

    @Test
    void startsAtBalanced() {
        PowerManager pm = new PowerManager();
        assertEquals(PowerManager.Level.标准, pm.level());
        assertEquals(20, pm.profile().learnTimeSteps, "标准档默认20步");
    }

    @Test
    void stutterDowngrades() {
        PowerManager pm = new PowerManager();
        long t = 0;
        // 连续卡顿 (3帧 >80ms)
        for (int i = 0; i < 3; i++) {
            pm.reportFrame(120, t);
            t += 200;
        }
        // 超过锁定时间后评估
        pm.evaluate(t + 6000);
        assertEquals(PowerManager.Level.节能, pm.level(), "卡顿应降档到节能");
        // 节能参数生效: 步数减少
        assertEquals(8, pm.profile().learnTimeSteps);
    }

    @Test
    void smoothUpgrades() {
        PowerManager pm = new PowerManager();
        pm.setLevel(PowerManager.Level.节能, 0);
        long t = 0;
        // 连续流畅帧 (30帧 <25ms)
        for (int i = 0; i < 30; i++) {
            pm.reportFrame(15, t);
            t += 100;
        }
        pm.evaluate(t + 6000);
        assertEquals(PowerManager.Level.标准, pm.level(), "流畅应升档到标准");
    }

    @Test
    void lowMemoryDowngrades() {
        PowerManager pm = new PowerManager();
        long t = 0;
        // 可用内存 <15% → 立即降档
        pm.reportMemory(10, 200, t);  // 5% 可用
        assertTrue(pm.level().rank < PowerManager.Level.标准.rank,
                "内存不足应降档, level=" + pm.level());
    }

    @Test
    void lockPreventsOscillation() {
        PowerManager pm = new PowerManager();
        long t = 0;
        // 卡顿降档
        for (int i = 0; i < 3; i++) {
            pm.reportFrame(120, t);
            t += 200;
        }
        pm.evaluate(t + 6000);
        assertEquals(PowerManager.Level.节能, pm.level());
        // 锁定期内即使流畅也不升档
        for (int i = 0; i < 30; i++) {
            pm.reportFrame(10, t);
            t += 100;
        }
        pm.evaluate(t + 1000);  // 仍在锁定期
        assertEquals(PowerManager.Level.节能, pm.level(), "锁定期不应升档");
        // 锁定期后流畅 → 升档 (等超过 5s 锁定)
        pm.evaluate(t + 10000);
        assertEquals(PowerManager.Level.标准, pm.level(), "锁定期后流畅应升档");
    }

    @Test
    void ultraStaysMax() {
        PowerManager pm = new PowerManager();
        pm.setLevel(PowerManager.Level.超频, 0);
        long t = 0;
        // 超频档卡顿 → 降一档 (高性能), 不能超过最低
        for (int i = 0; i < 3; i++) {
            pm.reportFrame(150, t);
            t += 200;
        }
        pm.evaluate(t + 6000);
        assertEquals(PowerManager.Level.高性能, pm.level());
        // 节能档继续卡顿 → 停在节能 (不越界)
        pm.setLevel(PowerManager.Level.节能, t);
        for (int i = 0; i < 3; i++) {
            pm.reportFrame(150, t);
            t += 200;
        }
        pm.evaluate(t + 6000);
        assertEquals(PowerManager.Level.节能, pm.level(), "节能档不应再降");
    }

    @Test
    void profileAppliesToBrain() {
        // 功率配置 → 大脑参数生效 (防卡死核心: 降低模型负载)
        Brain brain = Brain.simpleBrain();
        assertEquals(20, brain.powerProfile().learnTimeSteps, "默认标准20步");
        // 切节能
        PowerManager pm = new PowerManager();
        pm.setLevel(PowerManager.Level.节能, 0);
        brain.applyPowerProfile(pm.profile());
        assertEquals(8, brain.powerProfile().learnTimeSteps, "节能应8步");
        // 学习仍工作 (步数少但功能不丢)
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        assertEquals("你好", brain.recognizeVisual(img), "节能档学习仍应工作");
        // 切超频
        pm.setLevel(PowerManager.Level.超频, 0);
        brain.applyPowerProfile(pm.profile());
        assertEquals(40, brain.powerProfile().learnTimeSteps, "超频应40步");
        assertEquals(6, brain.powerProfile().hubCyclesPerSync, "超频中枢环路6次");
    }

    @Test
    void statisticsTracked() {
        PowerManager pm = new PowerManager();
        long t = 0;
        for (int i = 0; i < 10; i++) {
            pm.reportFrame(i < 5 ? 100 : 10, t);
            t += 100;
        }
        assertEquals(10, pm.frameCount());
        assertTrue(pm.avgFrameMs() > 0);
        assertEquals(100, pm.peakFrameMs());
        assertEquals(5, pm.stutters());
        assertTrue(pm.summary().contains("功率"));
    }
}
