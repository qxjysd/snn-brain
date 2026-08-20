package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * DiffusionApproximation 扩散近似测试。
 * 对照 bioRxiv 2024.12.17.628339 (NH-PDMP → 扩散近似):
 *  - 泊松基准: 相互独立脉冲 Fano≈1 (方差=均值)
 *  - 亚泊松: 规则发放 Fano<1 (生物皮层特征)
 *  - 超泊松: 爆发性发放 Fano>1 (同步/爆发)
 *  - 宏观率: 窗口脉冲数/窗口秒数 解析一致
 *  - 稳定性: Fano 超泊松+高活动 → 不稳定倾向; 亚泊松 → 稳定
 */
public class DiffusionApproximationTest {

    /** 泊松脉冲: Fano≈1 (方差≈均值) */
    @Test
    public void poissonFanoIsOne() {
        DiffusionApproximation da = new DiffusionApproximation(1.0, 1.0, 1);
        double[] counts = {3, 5, 2, 4, 3, 6, 1, 4, 2, 5};
        da.estimate(counts);
        double mean = 3.5;
        assertEquals(mean, da.meanRate(), 0.1, "宏观率 = 窗口平均脉冲数/秒");
        assertTrue(Math.abs(da.fanoFactor() - 1.0) < 0.6, "泊松 Fano 应≈1, got " + da.fanoFactor());
    }

    /** 亚泊松 (规则发放): Fano < 1 */
    @Test
    public void regularSpikingIsSubPoisson() {
        DiffusionApproximation da = new DiffusionApproximation(1.0, 1.0, 1);
        // 规则: 每神经元几乎相同脉冲数 (方差小)
        double[] counts = {4, 4, 4, 5, 4, 4, 4, 4, 5, 4};
        da.estimate(counts);
        assertTrue(da.fanoFactor() < 1.0, "规则发放应亚泊松 Fano<1, got " + da.fanoFactor());
        assertTrue(da.isStable(), "亚泊松网络应稳定");
    }

    /** 超泊松 (爆发): Fano > 1 */
    @Test
    public void burstySpikingIsSuperPoisson() {
        DiffusionApproximation da = new DiffusionApproximation(1.0, 1.0, 1);
        // 爆发: 一半神经元高脉冲一半低 (方差大)
        double[] counts = {12, 15, 1, 0, 14, 2, 0, 16, 1, 0};
        da.estimate(counts);
        assertTrue(da.fanoFactor() > 1.0, "爆发发放应超泊松 Fano>1, got " + da.fanoFactor());
        assertFalse(da.isStable(), "爆发网络应有不稳定倾向");
    }

    /** 静默网络: 状态标签=静默 */
    @Test
    public void silenceLabel() {
        DiffusionApproximation da = new DiffusionApproximation(1.0, 1.0, 1);
        da.estimate(new double[]{0, 0, 0, 0});
        assertEquals("静默", da.stateLabel());
        assertEquals(0.0, da.meanRate(), 1e-9);
    }

    /** 宏观率解析: 窗口内平均脉冲数 / 窗口秒数 */
    @Test
    public void rateIsMeanOverWindow() {
        DiffusionApproximation da = new DiffusionApproximation(0.5, 1.0, 1);
        double[] counts = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};  // mean=11
        da.estimate(counts);
        assertEquals(11.0 / 0.5, da.meanRate(), 1e-9, "宏观率 = 11脉冲 / 0.5s = 22Hz");
    }

    /** 解析路径: 直接给理论统计量对照公式 */
    @Test
    public void analyticPathMatches() {
        DiffusionApproximation da = new DiffusionApproximation(2.0, 1.0, 1);
        da.estimateFromStats(10.0, 5.0);
        assertEquals(5.0, da.meanRate(), 1e-9, "μ=10, 窗口2s → 5Hz");
        assertEquals(0.5, da.fanoFactor(), 1e-9, "Fano = σ²/μ = 5/10");
    }

    /** 波动强度: 高波动网络 fluctuation 大 */
    @Test
    public void fluctuationSeparates() {
        DiffusionApproximation regular = new DiffusionApproximation(1.0, 1.0, 1);
        regular.estimate(new double[]{4, 4, 4, 5, 4, 4});
        DiffusionApproximation bursty = new DiffusionApproximation(1.0, 1.0, 1);
        bursty.estimate(new double[]{12, 15, 1, 0, 14, 2});
        assertTrue(bursty.fluctuation() > regular.fluctuation(),
                "爆发网络波动强度应更大: " + regular.fluctuation() + " vs " + bursty.fluctuation());
    }

    /** interpret 输出包含关键量 */
    @Test
    public void interpretContainsMetrics() {
        DiffusionApproximation da = new DiffusionApproximation(1.0, 1.0, 1);
        da.estimate(new double[]{4, 4, 4, 5, 4, 4});
        String s = da.interpret();
        assertTrue(s.contains("Hz"), "interpret 应含率");
        assertTrue(s.contains("Fano"), "interpret 应含 Fano");
        assertTrue(s.contains("固定点"), "interpret 应含固定点");
    }

    /** 重置清空统计 */
    @Test
    public void resetClears() {
        DiffusionApproximation da = new DiffusionApproximation(1.0, 1.0, 1);
        da.estimate(new double[]{12, 15, 1, 0});
        da.reset();
        assertEquals(0.0, da.meanRate(), 1e-9);
        assertTrue(da.isStable());
    }
}
