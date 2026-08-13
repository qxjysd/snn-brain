package io.brainx.core.neuron;

import io.brainx.core.Neuron;

/**
 * 多腔室神经元 (Multi-Compartment, 对应 brain 生态 braincell 的详细细胞路线)。
 *
 * 真实神经元区别于点模型的三大特性:
 *   1. 树突整合: 输入先在树突局部整合, 非线性放大后再传胞体
 *   2. NMDA 受体电压依赖非线性 (Jahr & Stevens 1990):
 *      g(V) = 1 / (1 + 0.28·exp(-0.062·V)) — 镁离子阻断, 去极化时电导增大 → 超线性放大
 *   3. 输入位置效应: 远端树突输入经耦合电导衰减 → 距胞体越远贡献越小
 *
 * 简化双腔室模型 (EE 模型):
 *   树突: τd·dVd/dt = -(Vd-Vrest) + Rd·I_dend·(1+β·g_nmda(Vd)) + gc·(Vs-Vd)
 *   胞体: τs·dVs/dt  = -(Vs-Vrest) + Rs·gc·(Vd-Vs)         (LIF 发放)
 *   发放: Vs ≥ Vth → 发放 + 重置
 *
 * 算力友好: 每神经元仅 +1 个状态变量 (树突电位), 计算量 O(1) 与点神经元同级。
 */
public class MultiCompartmentNeuron implements Neuron {
    // 胞体 (LIF) 参数
    private final double tauSomaMs, vRest, vThreshold, vReset, resistance;
    // 树突参数
    private final double tauDendMs, dendResistance;
    // 树突↔胞体耦合电导 (gc 越大耦合越强, 输入位置效应)
    private final double gc;
    // NMDA 增强系数 (β=0 退化为线性树突)
    private final double nmdaBeta;

    private double vSoma;
    private double vDend;
    private boolean firedFlag;

    public MultiCompartmentNeuron(double tauSomaMs, double vRest, double vThreshold, double vReset,
                                  double resistance, double tauDendMs, double dendResistance,
                                  double gc, double nmdaBeta) {
        this.tauSomaMs = tauSomaMs;
        this.vRest = vRest;
        this.vThreshold = vThreshold;
        this.vReset = vReset;
        this.resistance = resistance;
        this.tauDendMs = tauDendMs;
        this.dendResistance = dendResistance;
        this.gc = gc;
        this.nmdaBeta = nmdaBeta;
        this.vSoma = vRest;
        this.vDend = vRest;
    }

    public static MultiCompartmentNeuron defaultParams() {
        return new MultiCompartmentNeuron(10.0, -65.0, -50.0, -65.0, 1.0,
                20.0, 4.0, 0.8, 2.0);
    }

    /** NMDA 电压依赖电导 (Jahr & Stevens): 去极化 → 电导增大 (镁离子解除阻断) */
    public static double nmdaConductance(double vMv) {
        return 1.0 / (1.0 + 0.28 * Math.exp(-0.062 * vMv));
    }

    /** 树突电位 (mV) — 树突整合状态 */
    public double dendritePotential() { return vDend; }

    @Override public void reset() { vSoma = vRest; vDend = vRest; firedFlag = false; }

    @Override
    public void step(double inputCurrent, double dtMs) {
        // ===== 树突腔室: 输入 + NMDA 非线性放大 + 漏 + 胞体耦合 =====
        double gNmda = nmdaConductance(vDend);
        double dendCurrent = inputCurrent * (1.0 + nmdaBeta * gNmda);  // 超线性放大
        double dvDend = (-(vDend - vRest) + dendResistance * dendCurrent
                + gc * (vSoma - vDend)) / tauDendMs;
        vDend += dvDend * dtMs;

        // ===== 胞体腔室: 树突电流 + 漏 + LIF 发放 =====
        double somaCurrent = gc * (vDend - vSoma);
        double dvSoma = (-(vSoma - vRest) + resistance * somaCurrent) / tauSomaMs;
        vSoma += dvSoma * dtMs;

        firedFlag = (vSoma >= vThreshold);
        if (firedFlag) vSoma = vReset;
    }

    @Override public boolean fired() { return firedFlag; }
    @Override public double membranePotential() { return vSoma; }
    @Override public int size() { return 1; }
    @Override public int stateDim() { return 2; }   // vSoma, vDend
    @Override public double state(int i) { return i == 0 ? vSoma : vDend; }
    @Override public double inputDerivative(int i) {
        // dVdend/dI_dend ≈ dendResistance·(1+β·g_nmda)/tauDend (近似)
        return i == 0 ? 0.0 : dendResistance * (1.0 + nmdaBeta * nmdaConductance(vDend)) / tauDendMs;
    }

    public double nmdaBeta() { return nmdaBeta; }
}
