package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * brainunit 物理单位系统验证 (NC 2025 物理单位感知计算):
 *   - 运算量纲+标度推导 (乘/除/幂)
 *   - 加减量纲校验 (防 ms+mV 错误) + 标度换算相加
 *   - 单位换算 (ms→s)
 *   - 物理关系验证 (tau = Cm/gL, V = I/G 量纲自动推导)
 */
public class UnitTest {

    @Test
    void dimensionalDerivation() {
        // 电压 = 电流/电导 (复合维度 I·V⁻¹ → I/G = V)
        Unit.Value v = Unit.nA(10).div(Unit.nS(2));
        assertEquals(5.0, v.v, 1e-9);
        assertTrue(v.unit.dim.equals(Unit.VOLTAGE), "I/G → V, got " + v.unit.dim);
        // 幂次: (Hz)² = T⁻²
        Unit.Value h2 = Unit.Hz(3).pow(2);
        assertTrue(h2.unit.dim.equals(Unit.FREQUENCY.pow(2)), "Hz² 量纲");
        // 无量纲
        assertTrue(Unit.nA(2).div(Unit.nA(2)).isDimensionless());
        // 频率 = 1/时间 (量纲推导; 数值换算到 Hz 基准: 1/10ms = 100Hz)
        Unit.Value freq = Unit.unitless(1).div(Unit.ms(10));
        assertTrue(freq.unit.dim.equals(Unit.FREQUENCY), "1/时间 → 频率量纲");
        assertEquals(100.0, freq.to(Unit.HZ), 1e-9, "1/10ms = 100Hz");
    }

    @Test
    void dimensionCheckRejectsMismatch() {
        // 加减量纲不一致 → 抛异常 (防止 ms+mV 类单位错误)
        Unit.Value time = Unit.ms(10);
        Unit.Value volt = Unit.mV(5);
        assertThrows(IllegalArgumentException.class, () -> time.add(volt), "ms+mV 应拒绝");
        assertThrows(IllegalArgumentException.class, () -> volt.sub(time), "mV-ms 应拒绝");
        // 一致量纲正常运算
        assertEquals(15.0, Unit.ms(10).add(Unit.ms(5)).v, 1e-9);
    }

    @Test
    void unitConversion() {
        // 同量纲换算: 10ms = 0.01s (标度换算)
        double secs = Unit.ms(10).to(Unit.S);
        assertEquals(0.01, secs, 1e-9, "ms→s 换算");
        // 反方向: 0.01s = 10ms
        assertEquals(10.0, Unit.s(0.01).to(Unit.MS), 1e-9);
        // 换算量纲不一致 → 拒绝
        assertThrows(IllegalArgumentException.class,
                () -> Unit.ms(10).to(Unit.MV), "ms→mV 换算应拒绝");
        // 跨标度相加: 10ms + 1s = 1.01s
        assertEquals(1.01, Unit.ms(10).add(Unit.s(1)).to(Unit.S), 1e-9);
    }

    @Test
    void physicalRelations() {
        // tau = Cm/gL: Cm = 电荷/电压 (1nA·ms/1mV, 量纲 I·T·V⁻¹), gL = 10nS (I·V⁻¹)
        // → Cm/gL 量纲自动推出时间, 数值 0.1s = 100ms
        Unit.Value cm = Unit.nA(1).mul(Unit.ms(1)).div(Unit.mV(1));
        Unit.Value gl = Unit.nS(10);
        Unit.Value tau = Unit.tauFromCmG(cm, gl);
        assertTrue(tau.unit.dim.equals(Unit.TIME), "Cm/gL 量纲应为时间, got " + tau.unit.dim);
        assertEquals(100.0, tau.to(Unit.MS), 1e-6, "τ = 100ms (Cm=1µF, gL=10nS)");
        // V = I/G
        Unit.Value v = Unit.voltageFromCurrentConductance(Unit.nA(5), Unit.nS(1));
        assertEquals(5.0, v.v, 1e-9);
        assertTrue(v.unit.dim.equals(Unit.VOLTAGE), "V=I/G 量纲");
    }

    @Test
    void membraneConstants() {
        // 神经元参数带量纲: 时间常数 10ms, 静息 -65mV, 阈值 -50mV
        Unit.Value tau = Unit.ms(10);
        Unit.Value rest = Unit.mV(-65);
        Unit.Value thresh = Unit.mV(-50);
        assertEquals(15.0, thresh.sub(rest).v, 1e-9, "阈值-静息 = 15mV");
        // 发放率 8Hz 量纲正确
        assertTrue(Unit.Hz(8).unit.dim.equals(Unit.FREQUENCY));
        // 电导复合维度: I·V⁻¹
        assertTrue(Unit.NS.dim.equals(Unit.CURRENT.div(Unit.VOLTAGE)));
    }
}
