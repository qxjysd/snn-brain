package io.brainx.core;

/**
 * 中枢脉冲接口 —— 各模块通过中枢脉冲网络统一联动的契约。
 *
 * 每个认知/记忆模块实现此接口:
 *   - emitPulses(): 将模块当前状态编码为脉冲率 (状态→脉冲发射到中枢)
 *   - receiveBroadcast(): 接收中枢广播的整合脉冲 (广播→调制模块状态)
 *
 * 这就是"各模块通过大脑中枢脉冲网络统一联动"：
 *   模块状态 → 脉冲 → 中枢整合 → 广播 → 调制各模块 → 循环
 */
public interface PulseModule {
    /**
     * 发射脉冲: 当前状态 → 脉冲率数组 (0-1, 长度=模块脉冲维度)。
     * 由中枢在每步收集。
     */
    double[] emitPulses();

    /**
     * 接收中枢广播: 整合脉冲 → 调制模块状态。
     * @param broadcastRates 中枢发放率向量 (广播)
     * @return 是否产生显著调制 (供诊断)
     */
    boolean receiveBroadcast(double[] broadcastRates);

    /** 模块名 (总线/诊断标识) */
    String moduleName();

    /** 脉冲维度 (发射向量长度) */
    int pulseDim();
}
