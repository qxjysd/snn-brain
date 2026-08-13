package io.brainx.core;

/**
 * 脉冲事件（对应 brain 的 brainevent 事件驱动计算）。
 * 稀疏表示：只有发放的神经元才产生事件，时间步+神经元索引+时间戳。
 * 事件驱动 = 不遍历全部神经元，只处理有事件的连接（对应脑中的稀疏性）。
 */
public final class SpikeEvent {
    public final int neuronIndex;
    public final int timeStep;
    public final double spikeTimeMs;
    /** 可选的脉冲强度（突触权重乘法用） */
    public final double amplitude;

    public SpikeEvent(int neuronIndex, int timeStep, double spikeTimeMs) {
        this(neuronIndex, timeStep, spikeTimeMs, 1.0);
    }

    public SpikeEvent(int neuronIndex, int timeStep, double spikeTimeMs, double amplitude) {
        this.neuronIndex = neuronIndex;
        this.timeStep = timeStep;
        this.spikeTimeMs = spikeTimeMs;
        this.amplitude = amplitude;
    }

    @Override
    public String toString() {
        return "SpikeEvent[n=" + neuronIndex + ", t=" + timeStep + ", tms=" + spikeTimeMs + "]";
    }
}
