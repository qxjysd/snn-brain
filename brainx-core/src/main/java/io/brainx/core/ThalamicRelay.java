package io.brainx.core;

/**
 * 脑干/LGN 中继 —— 感觉信号进入皮层前的降噪分拣与过滤。
 *
 * 资料依据 (听觉传导通路 + 视觉信息通路):
 *   - 听觉: 听神经 → 脑干(耳蜗核/上橄榄核/下丘: 分拣/降噪/声源定位)
 *     → 丘脑(过滤整合) → 听觉皮层
 *   - 视觉: 视网膜 → 视神经 → 外侧膝状体 LGN (丘脑, 中继过滤) → V1
 *   - 中继站功能: 抑制弱噪声, 增强显著信号, 选择性传递 (注意力门控)
 *
 * 实现:
 *   - 脑干降噪: 低于门限的频带抑制 (静音/背景噪声过滤)
 *   - 丘脑过滤: 显著信号增益放大, 注意力调制
 *   - 输出: 过滤后的神经信号 (更干净的皮层输入)
 */
public class ThalamicRelay {
    /** 噪声门限 (低于此的频带视为噪声抑制) */
    private double noiseFloor = 0.05;
    /** 增益 (丘脑放大) */
    private double gain = 1.0;
    /** 注意力门控 (0-1: 注意力聚焦时增益高) */
    private double attentionGate = 0.5;

    /**
     * 脑干降噪分拣: 抑制弱频带, 保留显著信号。
     * @param neuralSignal 原始神经信号 (0-1)
     * @return 降噪后的信号
     */
    public double[] brainstemFilter(double[] neuralSignal) {
        if (neuralSignal == null) return new double[0];
        double[] out = neuralSignal.clone();
        for (int i = 0; i < out.length; i++) {
            if (out[i] < noiseFloor) {
                out[i] = 0;  // 噪声抑制 (静音频带)
            }
        }
        return out;
    }

    /**
     * 丘脑 (LGN) 中继过滤: 增益放大 + 注意力调制。
     * 显著信号增强, 注意力高时整体增益提升。
     * @param neuralSignal 脑干过滤后的信号
     * @return 丘脑输出 (进入皮层)
     */
    public double[] thalamicRelay(double[] neuralSignal) {
        if (neuralSignal == null) return new double[0];
        double[] out = neuralSignal.clone();
        double effectiveGain = gain * (0.5 + attentionGate);
        for (int i = 0; i < out.length; i++) {
            if (out[i] > 0) {
                out[i] = Math.min(1.0, out[i] * effectiveGain);
            }
        }
        return out;
    }

    /** 设置注意力 (0-1): 高注意力 → 信号增强 (顶叶注意网络) */
    public void setAttention(double a) { this.attentionGate = Math.max(0, Math.min(1, a)); }

    /** 设置噪声门限 */
    public void setNoiseFloor(double f) { this.noiseFloor = Math.max(0, Math.min(0.5, f)); }

    public double attentionGate() { return attentionGate; }
    public double noiseFloor() { return noiseFloor; }

    /** 摘要 */
    public static String summary() {
        return "🔀 丘脑中继: 脑干降噪(噪声门限) + LGN过滤(注意力增益) → 皮层";
    }
}
