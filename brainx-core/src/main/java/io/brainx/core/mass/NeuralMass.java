package io.brainx.core.mass;

/**
 * 神经群模型接口（对应 brainmass 的 Neural Mass Model）。
 * 每个神经群表示一群同质神经元的平均发放率动态。
 * 对应 BrainMass 生态 (Jansen-Rit / Wong-Wang / Wilson-Cowan / Kuramoto)。
 */
public interface NeuralMass {
    /** 重置 */
    void reset();
    /** 前进一步, 返回当前发放率/活动值 */
    double step(double externalInput, double dtMs);
    /** 当前活动值 */
    double activity();
    /** 状态变量数 */
    int stateDim();
    /** 第 i 个状态 */
    double state(int i);
}
