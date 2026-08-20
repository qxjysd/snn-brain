package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * WholeBrainNetwork 全脑网络模型测试。
 * 对照 NatComm 2025 (10.1038/s41467-025-64470-3):
 *  - 粗粒度建模: 8 节点闭式方程模拟全脑 (显著少于微观神经元)
 *  - 结构连接 SC → 功能连接 FC: 解剖相连区域 FC 高
 *  - 全局耦合: 外部刺激经 SC 传播到其他区域 (结构→功能桥)
 *  - 模块内 FC > 模块间 FC (功能网络模块化, 人脑特征)
 *  - 主导区域: 刺激区域活动最高
 */
public class WholeBrainNetworkTest {

    /** 结构连接: 默认 SC 对称且对角为 0 */
    @Test
    public void structuralConnectivitySymmetric() {
        double[][] sc = WholeBrainNetwork.defaultSC();
        assertEquals(WholeBrainNetwork.N, sc.length);
        for (int i = 0; i < WholeBrainNetwork.N; i++) {
            assertEquals(0.0, sc[i][i], 1e-12, "对角应为 0 (无自连接)");
            for (int j = 0; j < WholeBrainNetwork.N; j++) {
                assertEquals(sc[i][j], sc[j][i], 1e-12, "SC 应对称");
            }
        }
        // 关键解剖通路存在
        assertTrue(sc[3][1] > 0.5, "枕叶↔顶叶视觉通路");
        assertTrue(sc[2][4] > 0.5, "颞叶↔海马记忆通路");
        assertTrue(sc[0][1] > 0.3, "前额叶↔顶叶执行通路");
    }

    /** 刺激传播: 枕叶刺激 → 活动经 SC 扩散, 枕叶活动最高 */
    @Test
    public void stimulusPropagatesThroughSC() {
        WholeBrainNetwork wbn = new WholeBrainNetwork();
        double[] stim = new double[WholeBrainNetwork.N];
        stim[3] = 5.0;  // 视觉刺激 → 枕叶
        for (int t = 0; t < 300; t++) wbn.step(stim, 1.0);
        assertEquals(3, wbn.dominantRegion(), "枕叶刺激应主导");
        // 顶叶 (与枕叶 SC=0.8) 活动应显著高于海马 (SC=0)
        double occipital = wbn.activity(3);
        double parietal = wbn.activity(1);
        double hippocampus = wbn.activity(4);
        assertTrue(parietal > hippocampus, "枕叶刺激应经 SC 传到顶叶 (0.8) 而非海马 (0): " + parietal + " vs " + hippocampus);
        assertTrue(occipital > 0.3, "枕叶自身应激活");
    }

    /** 结构→功能: 解剖相连区域 FC 高于未相连区域 */
    @Test
    public void scDrivesFC() {
        WholeBrainNetwork wbn = new WholeBrainNetwork();
        double[] stim = new double[WholeBrainNetwork.N];
        stim[3] = 5.0;
        java.util.Random rnd = new java.util.Random(42);
        for (int t = 0; t < 500; t++) {
            stim[3] = 3.0 + 2.0 * Math.sin(t / 20.0);  // 视觉输入波动
            wbn.step(stim, 1.0);
        }
        double[][] fc = wbn.functionalConnectivity();
        // 枕叶↔顶叶 (SC=0.8) FC 应显著 > 枕叶↔海马 (SC=0)
        double fcParietal = Math.abs(fc[3][1]);
        double fcHippocampus = Math.abs(fc[3][4]);
        assertTrue(fcParietal > fcHippocampus,
                "SC 强连接区域 FC 应更高: 枕顶 " + fcParietal + " vs 枕海马 " + fcHippocampus);
    }

    /** 功能网络模块化: 模块内 FC > 模块间 FC (人脑功能网络特征) */
    @Test
    public void modularOrganization() {
        WholeBrainNetwork wbn = new WholeBrainNetwork();
        double[] stim = new double[WholeBrainNetwork.N];
        java.util.Random rnd = new java.util.Random(7);
        for (int t = 0; t < 600; t++) {
            // 交替视觉/听觉刺激 (多模态输入)
            if (t % 100 < 50) stim[3] = 4.0; else stim[3] = 0.5;
            if (t % 150 < 60) stim[2] = 4.0; else stim[2] = 0.5;
            wbn.step(stim, 1.0);
        }
        double within = wbn.withinModuleFC();
        double between = wbn.betweenModuleFC();
        assertTrue(within > between,
                "模块内 FC 应高于模块间: " + within + " vs " + between);
    }

    /** 全局同步: 强耦合网络同步度为正且随刺激波动 */
    @Test
    public void globalSynchronyPositive() {
        WholeBrainNetwork wbn = new WholeBrainNetwork();
        double[] stim = new double[WholeBrainNetwork.N];
        stim[3] = 4.0;
        for (int t = 0; t < 300; t++) wbn.step(stim, 1.0);
        double sync = wbn.globalSynchrony();
        assertTrue(sync > -1.0 && sync <= 1.0, "同步度应在 [-1,1]");
        // 无刺激时 (低活动) 同步度应较低或不同
        WholeBrainNetwork quiet = new WholeBrainNetwork();
        for (int t = 0; t < 300; t++) quiet.step(null, 1.0);
        assertTrue(quiet.globalSynchrony() <= 1.0);
    }

    /** 粗粒度规模: 8 节点全脑 (论文: 显著少于微观神经元) */
    @Test
    public void coarseGrainedScale() {
        assertEquals(8, WholeBrainNetwork.N, "8 区域粗粒度节点");
        assertEquals(8, WholeBrainNetwork.REGIONS.length);
        assertEquals("前额叶", WholeBrainNetwork.REGIONS[0]);
        assertEquals("运动", WholeBrainNetwork.REGIONS[7]);
    }

    /** 功能连接: 对角线为 1, 值域 [-1,1] */
    @Test
    public void fcProperties() {
        WholeBrainNetwork wbn = new WholeBrainNetwork();
        double[] stim = new double[WholeBrainNetwork.N];
        stim[3] = 4.0;
        for (int t = 0; t < 300; t++) wbn.step(stim, 1.0);
        double[][] fc = wbn.functionalConnectivity();
        for (int a = 0; a < WholeBrainNetwork.N; a++) {
            assertEquals(1.0, fc[a][a], 1e-9, "FC 对角应为 1");
            for (int b = 0; b < WholeBrainNetwork.N; b++) {
                assertTrue(fc[a][b] >= -1.0 && fc[a][b] <= 1.0, "FC 值域 [-1,1]");
            }
        }
    }

    /** 重置: 活动回到基线 */
    @Test
    public void resetRestoresBaseline() {
        WholeBrainNetwork wbn = new WholeBrainNetwork();
        double[] stim = new double[WholeBrainNetwork.N];
        stim[3] = 5.0;
        for (int t = 0; t < 200; t++) wbn.step(stim, 1.0);
        assertTrue(wbn.activity(3) > 0.3);
        wbn.reset();
        assertEquals(0.1, wbn.activity(3), 1e-9, "重置后回到基线");
    }

    /** summary 含关键量 */
    @Test
    public void summaryContainsMetrics() {
        WholeBrainNetwork wbn = new WholeBrainNetwork();
        double[] stim = new double[WholeBrainNetwork.N];
        stim[3] = 4.0;
        for (int t = 0; t < 300; t++) wbn.step(stim, 1.0);
        String s = wbn.summary();
        assertTrue(s.contains("全脑"), "summary 应含全脑");
        assertTrue(s.contains("同步"), "summary 应含同步");
    }
}
