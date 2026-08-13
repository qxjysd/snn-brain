package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 双通路视觉 (What+Where/How) + 丘脑中继 (脑干降噪/LGN过滤) 验证。
 */
public class DualPathwayTest {

    @Test
    void dorsalPositionDetectsLeft() {
        DorsalPathway dp = new DorsalPathway();
        // 物体在左侧 → 质心X 应 < 0.5
        double[] img = new double[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 32; x++) img[y * 64 + x] = 255;
        }
        double[] pos = dp.encodePosition(img, 64, 64);
        assertTrue(pos[0] < 0.4, "左侧物体质心X应<0.4, got=" + pos[0]);
        assertTrue(pos[2] > 0.3, "覆盖度应>0.3, got=" + pos[2]);
    }

    @Test
    void dorsalPositionDetectsRight() {
        DorsalPathway dp = new DorsalPathway();
        double[] img = new double[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 32; x < 64; x++) img[y * 64 + x] = 255;
        }
        double[] pos = dp.encodePosition(img, 64, 64);
        assertTrue(pos[0] > 0.6, "右侧物体质心X应>0.6, got=" + pos[0]);
    }

    @Test
    void dorsalMotionDetected() {
        DorsalPathway dp = new DorsalPathway();
        // 前一帧物体在左, 当前帧在右 → 运动检测
        double[] prev = new double[64 * 64];
        double[] curr = new double[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 20; x++) { prev[y * 64 + x] = 255; curr[y * 64 + x + 30] = 255; }
        }
        double[] motion = dp.encodeMotion(prev, curr, 64, 64);
        assertTrue(motion[2] > 0.1, "应有运动, strength=" + motion[2]);
        // 静态帧 → 无运动
        double[] staticMotion = dp.encodeMotion(prev, prev.clone(), 64, 64);
        assertTrue(staticMotion[2] < 0.1, "静态应无运动, got=" + staticMotion[2]);
    }

    @Test
    void brainstemFiltersNoise() {
        ThalamicRelay relay = new ThalamicRelay();
        // 弱信号 (噪声) 抑制
        double[] signal = {0.01, 0.9, 0.02, 0.8, 0.0};
        double[] filtered = relay.brainstemFilter(signal);
        assertEquals(0.0, filtered[0], "噪声频带应抑制");
        assertEquals(0.9, filtered[1], 1e-9, "显著信号保留");
        assertEquals(0.0, filtered[2], "噪声频带应抑制");
    }

    @Test
    void thalamicRelayAmplifies() {
        ThalamicRelay relay = new ThalamicRelay();
        relay.setAttention(1.0);  // 高注意力
        double[] signal = {0.5, 0.3, 0.0};
        double[] out = relay.thalamicRelay(signal);
        assertTrue(out[0] > 0.5, "高注意力应放大信号, got=" + out[0]);
        assertEquals(0.0, out[2], "零信号保持零");
    }

    @Test
    void brainDualPathway() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 32; x++) img[y * 64 + x] = 255;  // 左半亮
        }
        double[][] paths = brain.dualPathwayVisual(img, null, 64, 64);
        assertEquals(io.brainx.core.VisualNeuralEncoder.OUTPUT_DIM, paths[0].length, "腹侧 What 维度");
        assertEquals(io.brainx.core.DorsalPathway.OUTPUT_DIM, paths[1].length, "背侧 Where 维度");
        assertTrue(paths[1][0] < 0.5, "背侧应感知物体在左侧, got=" + paths[1][0]);
        // 识别仍工作 (腹侧通路)
        double[] what = paths[0];
        for (int e = 0; e < 3; e++) brain.learnVisualWord(what, 0);
        String r = brain.recognizeVisual(what);
        assertTrue(r.equals("你好") || r.equals("苹果"), "腹侧识别应工作, got=" + r);
    }

    @Test
    void dualPathwaySummary() {
        Brain brain = Brain.simpleBrain();
        assertNotNull(brain.dorsalPathway());
        assertNotNull(brain.visualRelay());
        assertNotNull(brain.auditoryRelay());
        assertTrue(DorsalPathway.summary().contains("背侧"));
        assertTrue(ThalamicRelay.summary().contains("丘脑"));
    }
}
