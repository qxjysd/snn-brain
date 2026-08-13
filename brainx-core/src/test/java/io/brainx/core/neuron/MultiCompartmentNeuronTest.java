package io.brainx.core.neuron;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 多腔室神经元 (braincell) 验证:
 *   - NMDA 电压依赖非线性 (Jahr & Stevens 1990)
 *   - 树突超线性整合 (同输入下 NMDA 发放 > 线性树突)
 *   - 输入位置效应 (耦合 gc 小 = 远端输入衰减, 点神经元做不到)
 *   - 胞体 LIF 发放行为
 */
public class MultiCompartmentNeuronTest {

    /** 恒流输入下发放计数 */
    private int fireCount(MultiCompartmentNeuron n, double current, int ms) {
        int count = 0;
        for (int t = 0; t < ms; t++) {
            n.step(current, 1.0);
            if (n.fired()) count++;
        }
        return count;
    }

    @Test
    void nmdaVoltageDependence() {
        // 去极化 → 镁离子解除阻断 → 电导增大 (Jahr & Stevens 曲线单调上升)
        double gNeg = MultiCompartmentNeuron.nmdaConductance(-70);
        double gMid = MultiCompartmentNeuron.nmdaConductance(-30);
        double gZero = MultiCompartmentNeuron.nmdaConductance(0);
        double gPos = MultiCompartmentNeuron.nmdaConductance(30);
        assertTrue(gNeg < gMid && gMid < gZero && gZero < gPos,
                "NMDA 电导应随去极化单调上升: " + gNeg + " < " + gMid + " < " + gZero + " < " + gPos);
        assertTrue(gZero > 0.5 && gPos > 0.9, "去极化时电导应接近 1");
    }

    @Test
    void nmdaSuperlinearIntegration() {
        // 相同输入: NMDA (β=2) 树突超线性放大 → 发放显著多于线性树突 (β=0)
        MultiCompartmentNeuron nmda = new MultiCompartmentNeuron(10, -65, -50, -65, 1, 20, 4, 0.8, 2.0);
        MultiCompartmentNeuron linear = new MultiCompartmentNeuron(10, -65, -50, -65, 1, 20, 4, 0.8, 0.0);
        int nmdaFires = fireCount(nmda, 8.0, 300);
        int linearFires = fireCount(linear, 8.0, 300);
        assertTrue(nmdaFires > linearFires,
                "NMDA 树突应超线性放大: nmda=" + nmdaFires + " linear=" + linearFires);
        // 弱输入下两者都不应误发放
        MultiCompartmentNeuron quiet = new MultiCompartmentNeuron(10, -65, -50, -65, 1, 20, 4, 0.8, 2.0);
        assertEquals(0, fireCount(quiet, 1.0, 100), "弱输入不应发放");
    }

    @Test
    void inputLocationEffect() {
        // 输入位置效应: 耦合 gc 小 (远端树突输入, 经胞体耦合衰减) → 发放少;
        // gc 大 (近端输入) → 贡献大。点神经元无法表达此差异。
        MultiCompartmentNeuron distal = new MultiCompartmentNeuron(10, -65, -50, -65, 1, 20, 4, 0.2, 0.0);
        MultiCompartmentNeuron proximal = new MultiCompartmentNeuron(10, -65, -50, -65, 1, 20, 4, 1.2, 0.0);
        int distalFires = fireCount(distal, 12.0, 300);
        int proximalFires = fireCount(proximal, 12.0, 300);
        assertTrue(proximalFires > distalFires,
                "近端输入 (大 gc) 应比远端输入 (小 gc) 更有效: proximal="
                        + proximalFires + " distal=" + distalFires);
    }

    @Test
    void somaBehavesLikeLIF() {
        // 胞体 LIF 行为: 发放后重置, 状态可读
        MultiCompartmentNeuron n = MultiCompartmentNeuron.defaultParams();
        assertEquals(2, n.stateDim(), "双腔室状态 (胞体+树突)");
        // 强输入驱动发放
        boolean fired = false;
        for (int t = 0; t < 200 && !fired; t++) {
            n.step(20.0, 1.0);
            fired = n.fired();
        }
        assertTrue(fired, "强输入应驱动胞体发放");
        // 发放后膜电位回到重置水平 (<= 阈值)
        assertTrue(n.membranePotential() < -50, "发放后应重置: v=" + n.membranePotential());
        // 树突电位是独立状态 (树突整合可见)
        assertTrue(Double.isFinite(n.dendritePotential()));
    }
}
