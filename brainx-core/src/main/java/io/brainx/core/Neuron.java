package io.brainx.core;

/**
 * 神经元接口（对应 brain.state 的 Neuron 基类）。
 * 所有神经元模型实现此接口，统一的状态演化与脉冲读取。
 */
public interface Neuron {
    /** 重置到初始状态 */
    void reset();
    /** 前向一步：给定总输入电流 (nA)，更新内部状态 */
    void step(double inputCurrent, double dtMs);
    /** 本步是否发放脉冲 */
    boolean fired();
    /** 膜电位 (mV) 用于监测/可视化 */
    double membranePotential();
    /** 神经元数量（向量化神经元返回>1；单神经元返回1） */
    int size();
    /** 内部状态数（用于在线学习的资格迹维度） */
    int stateDim();
    /** 读取第 i 个内部状态值（资格迹计算用） */
    double state(int i);
    /** 输入电流对状态的导数 diag(D_f) (pp-prop 用) */
    double inputDerivative(int i);
}
