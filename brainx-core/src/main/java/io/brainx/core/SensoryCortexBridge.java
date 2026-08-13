package io.brainx.core;

import io.brainx.core.neuron.LIF;

/**
 * 感觉皮层桥接器 —— 视觉/听觉皮层接入中枢脉冲网络。
 *
 * 书中依据 (丘脑-皮层环路, 全局爆发):
 *   - 感觉输入通过皮层→丘脑→皮层环路广播到全脑 (全局爆发含感觉)
 *   - 中枢下行调制感觉皮层 (双向连接: 上行脉冲 + 下行增益)
 *   - 注意力/意识状态通过下行连接增强或抑制感觉处理
 *
 * 实现:
 *   - emitPulses: 皮层神经元发放率 → 中枢 (感觉输入广播)
 *   - receiveBroadcast: 中枢活跃 → 下行增益调制 (注意增强)
 */
public class SensoryCortexBridge implements PulseModule {
    /** 皮层神经元 */
    private final LIF[] cortex;
    /** 皮层发放率 (滑动平均, 供中枢读取) */
    private final double[] firingRates;
    /** 下行调制增益 (中枢广播 → 感觉增益) */
    private double topDownGain = 1.0;
    /** 模块名 */
    private final String name;

    public SensoryCortexBridge(LIF[] cortex, String name) {
        this.cortex = cortex;
        this.name = name;
        this.firingRates = new double[cortex.length];
    }

    /** 皮层 step 后更新发放率 (由 Brain 每步调用) */
    public void updateRates() {
        for (int i = 0; i < cortex.length; i++) {
            if (cortex[i].fired()) {
                firingRates[i] = Math.min(1.0, firingRates[i] + 0.3);
            } else {
                firingRates[i] *= 0.9;
            }
        }
    }

    /** 下行增益 (中枢广播调制, 供皮层输入使用) */
    public double topDownGain() { return topDownGain; }

    @Override public String moduleName() { return name; }
    @Override public int pulseDim() { return cortex.length; }

    /** 发射脉冲: 皮层发放率 → 中枢 (感觉输入广播) */
    @Override
    public double[] emitPulses() {
        return firingRates.clone();
    }

    /** 接收中枢广播: 中枢活跃 → 下行增益 (注意增强感觉) */
    @Override
    public boolean receiveBroadcast(double[] broadcastRates) {
        if (broadcastRates.length == 0) return false;
        double avg = 0;
        for (double r : broadcastRates) avg += r;
        avg /= broadcastRates.length;
        // 中枢活跃 → 感觉增益 (注意聚焦), 缓慢回落
        if (avg > 0.3) {
            topDownGain = Math.min(1.5, topDownGain + 0.05);
        } else {
            topDownGain = Math.max(1.0, topDownGain - 0.02);
        }
        return avg > 0.3;
    }
}
