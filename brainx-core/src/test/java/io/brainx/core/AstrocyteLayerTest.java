package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * AstrocyteLayer 星形胶质细胞三突触可塑性测试。
 * 对照 AGMP 论文 (Frontiers 2026, 10.3389/fnins.2025.1768235):
 *  - 慢速积分: τa≫τm,τe, 持续活动下胶质状态缓慢上升 (论文 Eq.8)
 *  - 门控方向: 高活动神经元 gate→高, 静止神经元 gate→低 (论文 Eq.10 归一化)
 *  - 稳定-可塑平衡: 门控低 → 权重更新被抑制 (防灾难性遗忘)
 *  - 门控因子范围 [0,1]
 */
public class AstrocyteLayerTest {

    /** 慢速积分: 持续活动 2000 步(每步1ms)胶质状态缓升且未饱和 */
    @Test
    public void slowIntegrationOverMillis() {
        AstrocyteLayer layer = new AstrocyteLayer(4, 5000.0, 0.2, 0.3, 0.5, 4.0, 0.0);
        boolean[] spikes = {true, false, true, false};
        double[] v = {1.0, 0.0, 1.0, 0.0};
        for (int t = 0; t < 2000; t++) layer.step(v, null, spikes, 1.0);
        double[] a = layer.states();
        // 持续活跃的神经元胶质状态 > 静止神经元 (慢速积分区分活动水平)
        assertTrue(a[0] > a[1], "活跃神经元胶质状态应更高");
        assertTrue(a[2] > a[3], "活跃神经元胶质状态应更高");
        // 5 秒慢积分后状态仍远未饱和 (τa=5000ms, 时间常数内)
        assertTrue(a[0] < 0.5, "慢速积分不应立即饱和");
    }

    /** 门控方向: 高活动神经元 gate 高, 静止神经元 gate 低 */
    @Test
    public void gateFollowsActivity() {
        AstrocyteLayer layer = new AstrocyteLayer(2, 2000.0, 0.2, 0.3, 0.5, 6.0, 0.0);
        boolean[] spikes = {true, false};
        double[] v = {2.0, 0.0};
        for (int t = 0; t < 3000; t++) layer.step(v, null, spikes, 1.0);
        double gActive = layer.gate(0);
        double gQuiet = layer.gate(1);
        assertTrue(gActive > gQuiet, "活跃神经元门控应更高: " + gActive + " vs " + gQuiet);
        assertTrue(gActive >= 0 && gActive <= 1, "门控应在 [0,1]: " + gActive);
        assertTrue(gQuiet >= 0 && gQuiet <= 1, "门控应在 [0,1]: " + gQuiet);
    }

    /** 稳定-可塑平衡: 静止后门控回落到低值, 保护旧权重 */
    @Test
    public void quietNeuronGateDrops() {
        AstrocyteLayer layer = new AstrocyteLayer(2, 3000.0, 0.2, 0.3, 0.5, 5.0, 0.0);
        boolean[] spikes = {true, false};
        double[] v = {2.0, 0.0};
        for (int t = 0; t < 4000; t++) layer.step(v, null, spikes, 1.0);
        double gPeak = layer.gate(0);
        // 之后全部静止 6000 步 → 胶质状态衰减 → 门控回落
        boolean[] quiet = {false, false};
        double[] vq = {0.0, 0.0};
        for (int t = 0; t < 6000; t++) layer.step(vq, null, quiet, 1.0);
        double gAfter = layer.gate(0);
        assertTrue(gAfter < gPeak, "静止后门控应回落: " + gPeak + " -> " + gAfter);
    }

    /** 门控缩放权重更新: gatedUpdate = lr·g·M·e */
    @Test
    public void gatedUpdateScalesByGate() {
        AstrocyteLayer layer = new AstrocyteLayer(2, 2000.0, 0.2, 0.3, 0.5, 6.0, 0.0);
        boolean[] spikes = {true, false};
        double[] v = {2.0, 0.0};
        for (int t = 0; t < 3000; t++) layer.step(v, null, spikes, 1.0);
        double gActive = layer.gate(0);
        double gQuiet = layer.gate(1);
        double updActive = layer.gatedUpdate(1.0, 0.5, 0.01, 0);
        double updQuiet = layer.gatedUpdate(1.0, 0.5, 0.01, 1);
        assertEquals(0.01 * gActive * 1.0 * 0.5, updActive, 1e-9, "活跃神经元更新 = lr·g·M·e");
        assertEquals(0.01 * gQuiet * 1.0 * 0.5, updQuiet, 1e-9, "静止神经元更新 = lr·g·M·e");
        assertTrue(updActive > updQuiet, "活跃神经元更新应更大");
    }

    /** 突触输入项: ηs·Σ|w_ij|·s_j 贡献胶质状态 */
    @Test
    public void synapticInputContributes() {
        AstrocyteLayer layer = new AstrocyteLayer(2, 2000.0, 0.0, 0.3, 0.0, 6.0, 0.0);
        double[] synIn = {3.0, 0.0};
        boolean[] spikes = {false, false};
        for (int t = 0; t < 3000; t++) layer.step(null, synIn, spikes, 1.0);
        double[] a = layer.states();
        assertTrue(a[0] > a[1], "突触输入应驱动胶质状态");
    }

    /** 重置: 状态归零, 门控回中性 0.5 */
    @Test
    public void resetClearsState() {
        AstrocyteLayer layer = new AstrocyteLayer(3, 2000.0, 0.2, 0.3, 0.5, 6.0, 0.0);
        boolean[] spikes = {true, true, false};
        double[] v = {2.0, 2.0, 0.0};
        for (int t = 0; t < 3000; t++) layer.step(v, null, spikes, 1.0);
        layer.reset();
        for (int i = 0; i < 3; i++) {
            assertEquals(0.0, layer.states()[i], 1e-12, "重置后状态归零");
            assertEquals(0.5, layer.gate(i), 1e-12, "重置后门控中性");
        }
    }

    /** 高活动与低活动神经元的门控差值随区分度扩大 (归一化对比) */
    @Test
    public void gateSeparationGrowsWithContrast() {
        // 对照组: 活动差异小
        AstrocyteLayer lowContrast = new AstrocyteLayer(2, 2000.0, 0.2, 0.3, 0.5, 6.0, 0.0);
        boolean[] spikesL = {true, true};
        double[] vL = {2.0, 1.0};
        for (int t = 0; t < 3000; t++) lowContrast.step(vL, null, spikesL, 1.0);
        // 实验组: 活动差异大
        AstrocyteLayer highContrast = new AstrocyteLayer(2, 2000.0, 0.2, 0.3, 0.5, 6.0, 0.0);
        boolean[] spikesH = {true, false};
        double[] vH = {2.0, 0.0};
        for (int t = 0; t < 3000; t++) highContrast.step(vH, null, spikesH, 1.0);
        double sepL = Math.abs(lowContrast.gate(0) - lowContrast.gate(1));
        double sepH = Math.abs(highContrast.gate(0) - highContrast.gate(1));
        assertTrue(sepH > sepL, "活动对比越大门控分离越明显");
    }
}
