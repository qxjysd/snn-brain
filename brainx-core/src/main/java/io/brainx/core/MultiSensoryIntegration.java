package io.brainx.core;

/**
 * 多感觉整合模块 —— 贝叶斯因果推断模型。
 *
 * 理论依据 (Körding et al. 2007, PLoS ONE "Causal Inference in Multisensory
 * Perception"; Scikit-neuromsi 框架的理论基础):
 *   - 大脑将视觉与听觉线索按可靠性(精度)加权整合:
 *     s_hat = Σ w_i · s_i, w_i = 精度_i / Σ 精度
 *   - 因果推断: 判断两线索是否来自同一来源 (P(common cause))
 *     - 同源概率高 → 强制融合 (Φ 高, 感知更精确)
 *     - 同源概率低 → 分离估计 (各自保留)
 *   - 一致性越高, 整合度越高 (对应意识模块的 Φ)
 *
 * 实现:
 *   - 视觉/听觉各自给出估计 + 可靠性 (置信度)
 *   - 贝叶斯加权整合 → 融合估计
 *   - 因果推断: 基于线索差异与先验, 计算 P(common)
 *   - 整合度 Φ = P(common) × 精度加权和 (输出给意识模块)
 */
public class MultiSensoryIntegration {
    /** 同源先验 P(common) */
    private final double priorCommon;
    /** 线索差异阈值 (超过则倾向分离) */
    private final double sigmaDiff;

    public MultiSensoryIntegration(double priorCommon, double sigmaDiff) {
        this.priorCommon = priorCommon;
        this.sigmaDiff = sigmaDiff;
    }

    public static MultiSensoryIntegration defaultParams() {
        return new MultiSensoryIntegration(0.5, 0.35);
    }

    /**
     * 整合视觉+听觉估计。
     * @param visualEstimate   视觉估计值 (如识别得分 0-1)
     * @param visualReliability 视觉可靠性 (置信度 0-1)
     * @param auditoryEstimate  听觉估计值
     * @param auditoryReliability 听觉可靠性 (置信度 0-1)
     * @return [融合估计, P(common), Φ整合度]
     */
    public double[] integrate(double visualEstimate, double visualReliability,
                              double auditoryEstimate, double auditoryReliability) {
        double vAcc = Math.max(0.01, visualReliability);
        double aAcc = Math.max(0.01, auditoryReliability);

        // 1. 因果推断: P(common) = f(线索差异, 先验)
        double diff = Math.abs(visualEstimate - auditoryEstimate);
        // 似然: 差异越小越可能同源 (高斯似然)
        double likelihoodCommon = Math.exp(-diff * diff / (2 * sigmaDiff * sigmaDiff));
        double pCommon = priorCommon * likelihoodCommon
                / (priorCommon * likelihoodCommon + (1 - priorCommon) * 0.2);

        // 2. 贝叶斯加权整合 (精度加权)
        double wV = vAcc / (vAcc + aAcc);
        double wA = aAcc / (vAcc + aAcc);
        double fused = wV * visualEstimate + wA * auditoryEstimate;

        // 3. 整合度 Φ: P(common) × 总精度
        double phi = pCommon * (vAcc + aAcc) / 2.0;
        return new double[]{fused, pCommon, Math.min(1.0, phi)};
    }

    /**
     * 整合标签级信息 (识别词): 视觉和听觉识别是否一致。
     * @param visualLabel  视觉识别标签
     * @param auditoryLabel 听觉识别标签
     * @return 一致则高 Φ
     */
    public double integrateLabels(String visualLabel, String auditoryLabel) {
        if (visualLabel.isEmpty() || auditoryLabel.isEmpty()) return 0.3;
        if (visualLabel.equals(auditoryLabel)) return 0.95;   // 跨模态一致
        return 0.15;                                          // 冲突
    }

    /**
     * γ 振荡绑定: 跨模态信号在 γ 频段 (30-100Hz) 同步 = 绑定同一对象。
     * (fWBM/GWT: γ 绑定是跨模态整合的神经机制)
     * 视觉+听觉一致 → 同步 γ → 高绑定强度; 冲突 → 不同步
     */
    public double gammaBinding(String visualLabel, String auditoryLabel, double gammaHz) {
        if (visualLabel.isEmpty() || auditoryLabel.isEmpty()) return 0;
        if (visualLabel.equals(auditoryLabel)) {
            // 一致: γ 同步 (绑定强度 = 频率精度 × 一致性)
            return 0.8 + 0.2 * Math.sin(gammaHz / 40.0 * Math.PI);  // γ 相位同步
        }
        return 0.05;  // 冲突: 无绑定
    }

    /** 跨模态绑定频率 (γ 带内, 一致时输出绑定频率) */
    public double bindingFrequency(boolean consistent) {
        return consistent ? 40.0 + (Math.random() * 20) : 0;  // γ 30-60Hz
    }
}
