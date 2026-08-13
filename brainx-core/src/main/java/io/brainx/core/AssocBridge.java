package io.brainx.core;

/**
 * 联想皮层桥接器 —— 联想层接入中枢脉冲网络。
 *
 * 书中依据 (全局爆发 global ignition, 德阿纳):
 *   - 感觉输入 → 联想层(前额叶) → 跨阈值 → 全局爆发 → 广播全脑
 *   - 联想层是"全局工作空间"的核心节点 (高阶层翻译低层表征)
 *
 * 实现:
 *   - emitPulses: 联想层激活值 (tanh 0-1) → 中枢 (高阶表征广播)
 *   - receiveBroadcast: 中枢广播 → 联想增益调制 (全局爆发增强联想)
 */
public class AssocBridge implements PulseModule {
    /** 联想层激活向量 (由 Brain 更新: tanh 输出 0-1) */
    private double[] activation;
    /** 联想层大小 */
    private final int size;
    /** 下行增益 */
    private double topDownGain = 1.0;

    public AssocBridge(int size) {
        this.size = size;
        this.activation = new double[size];
    }

    /** 更新激活向量 (Brain 在联想计算后调用) */
    public void updateActivation(double[] act) {
        for (int i = 0; i < size && i < act.length; i++) {
            activation[i] = Math.max(0, Math.min(1, act[i]));
        }
    }

    /** 联想层当前激活 (供 Brain 读取) */
    public double[] activation() { return activation; }

    /** 联想增益 (中枢广播调制) */
    public double topDownGain() { return topDownGain; }

    @Override public String moduleName() { return "联想皮层"; }
    @Override public int pulseDim() { return size; }

    /** 发射脉冲: 联想激活 (tanh) → 中枢 */
    @Override
    public double[] emitPulses() { return activation.clone(); }

    /** 接收中枢广播: 全局爆发 → 联想增益增强 (高阶表征强化) */
    @Override
    public boolean receiveBroadcast(double[] broadcastRates) {
        if (broadcastRates.length == 0) return false;
        double avg = 0;
        for (double r : broadcastRates) avg += r;
        avg /= broadcastRates.length;
        if (avg > 0.3) {
            topDownGain = Math.min(1.5, topDownGain + 0.05);
        } else {
            topDownGain = Math.max(1.0, topDownGain - 0.02);
        }
        return avg > 0.3;
    }
}
