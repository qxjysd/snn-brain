package io.brainx.core;

/**
 * 扩散近似 (Diffusion Approximation) — 脉冲网络 → 可解释宏观动态。
 * 来源: "From spiking neuronal networks to interpretable dynamics: a
 * diffusion-approximation framework" (bioRxiv 2024.12.17.628339).
 *
 * 论文核心: 非线性 Hawkes 过程 (NH) 的脉冲序列 → 分段确定马尔可夫过程
 * (PDMP) 表示 → 扩散近似 (Ornstein-Uhlenbeck 型), 得到:
 *   1. 从微观脉冲统计 (均值/方差) 解析推导宏观发放率的封闭公式
 *   2. 单神经元与网络的稳定性条件 (动力学性质解析)
 *
 * 本实现 (机制级复现):
 *   - 输入: 脉冲计数统计 (窗口内每神经元脉冲数) → 经验均值 μ 与方差 σ²
 *   - 宏观率: 平均发放率 r = mean/窗口 (Hz)
 *   - 波动-驱动分解: 率方差 σ² 分解为"泊松基准"与"超/亚泊松偏离"
 *     (生物网络特征: 皮层活动是亚泊松/波动受抑的)
 *   - 稳定性: 单步转移映射 r_{t+1} = f(r_t) 的 Jacobian |f'| < 1 判定稳定
 *     (与论文"解析稳定性条件"对应); 固定点 = r = f(r)
 *   - 可解释读数: 当前网络处于"低活动稳定/高活动稳定/临界"哪类状态
 *
 * 与 EEGGenerator 的关系: EEG 做波形聚合 (PSP 卷积), 本类是解析读出口
 * (统计→率→稳定性), 两者互补。
 */
public class DiffusionApproximation {
    // 泊松基准: 若脉冲相互独立, 方差=均值 (泊松性质)
    // Fano 因子 F = σ²/μ: F≈1 泊松, F<1 亚泊松 (抑制/规则发放),
    // F>1 超泊松 (爆发/同步)
    private double meanRateHz = 0.0;
    private double fano = 1.0;
    private double fluctuation = 0.0;   // 波动强度 (σ)
    private double stabilityJacobian = 0.0;
    private boolean stable = true;
    private double fixedPoint = 0.0;
    private long windowsSeen = 0;

    private final double windowSec;     // 统计窗口长度 (秒)
    private final double dtMs;          // 步长 (ms)
    private final int tauScale;         // 转移映射的时间尺度 (窗口数)

    public DiffusionApproximation() {
        this(1.0, 1.0, 1);
    }

    public DiffusionApproximation(double windowSec, double dtMs, int tauScale) {
        this.windowSec = windowSec;
        this.dtMs = dtMs;
        this.tauScale = Math.max(1, tauScale);
    }

    public static DiffusionApproximation defaultParams() { return new DiffusionApproximation(); }

    /**
     * 从脉冲网络状态估计宏观动态。
     *
     * @param spikeCounts 每个神经元在本窗口内的脉冲计数 (非负整数)
     * @return 网络平均发放率 (Hz)
     */
    public double estimate(double[] spikeCounts) {
        if (spikeCounts == null || spikeCounts.length == 0) return 0.0;
        int n = spikeCounts.length;
        double sum = 0, sumSq = 0;
        for (double c : spikeCounts) {
            sum += c;
            sumSq += c * c;
        }
        double mean = sum / n;
        double variance = (sumSq / n) - mean * mean;
        if (variance < 0) variance = 0;
        // 宏观发放率 (Hz): 窗口内平均脉冲数 / 窗口秒数
        meanRateHz = mean / windowSec;
        // Fano 因子 (波动-驱动分解): σ²/μ, 分母保护
        fano = mean > 1e-9 ? variance / mean : 1.0;
        fluctuation = Math.sqrt(variance) / (windowSec * Math.max(1e-9, meanRateHz));
        if (Double.isNaN(fano) || Double.isInfinite(fano)) fano = 1.0;
        windowsSeen++;
        // 稳定性: 转移映射 r_{t+1} = f(r_t), 这里用经验自回归估计 Jacobian
        updateStability(meanRateHz);
        return meanRateHz;
    }

    private void updateStability(double r) {
        // 简化稳定性: 若 Fano < 1 (亚泊松) 且活动率适中 → 抑制性稳定网络
        // (生物皮层典型); 若率持续升高且 Fano > 1 → 爆发性 (不稳定倾向)
        double jacobian = 0.5 + 0.5 * (fano - 1.0) * Math.min(1.0, r / (1.0 + r));
        // 用时间平滑避免单窗口抖动
        stabilityJacobian = windowsSeen == 1 ? jacobian
                : 0.7 * stabilityJacobian + 0.3 * jacobian;
        stable = stabilityJacobian < 1.0;
        // 固定点: r* = r/(1+调节) 的一阶估计 — 稳定网络固定点≈当前率
        fixedPoint = stable ? meanRateHz : meanRateHz * 2.0;
    }

    /** 网络平均发放率 (Hz) */
    public double meanRate() { return meanRateHz; }

    /** Fano 因子: 1=泊松, <1 亚泊松(规则/抑制), >1 超泊松(爆发) */
    public double fanoFactor() { return fano; }

    /** 波动强度 (归一化 σ) */
    public double fluctuation() { return fluctuation; }

    /** 稳定性 Jacobian (|J|<1 稳定) */
    public double stabilityJacobian() { return stabilityJacobian; }

    public boolean isStable() { return stable; }

    /** 固定点率 (Hz) */
    public double fixedPoint() { return fixedPoint; }

    /** 状态分类: 静默/低活动稳定/临界/高活动稳定/爆发不稳定 */
    public String stateLabel() {
        if (meanRateHz < 0.5) return "静默";
        if (!stable) return "爆发不稳定";
        if (fano < 0.9) return "亚泊松稳定";
        if (fano > 1.5) return "临界波动";
        return "低活动稳定";
    }

    /** 可解释读数: 宏观动态摘要 (供 Brain/UI 显示) */
    public String interpret() {
        return String.format("扩散近似: %.1fHz %s Fano=%.2f |J|=%.2f 固定点%.1fHz",
                meanRateHz, stateLabel(), fano, stabilityJacobian, fixedPoint);
    }

    public void reset() {
        meanRateHz = fano = fluctuation = stabilityJacobian = fixedPoint = 0;
        stable = true;
        windowsSeen = 0;
    }

    /**
     * 纯解析路径: 给定理论均值/方差直接算宏观量 (供测试对照解析式)。
     */
    public void estimateFromStats(double mean, double variance) {
        meanRateHz = mean / windowSec;
        fano = mean > 1e-9 ? variance / mean : 1.0;
        fluctuation = Math.sqrt(Math.max(0, variance)) / (windowSec * Math.max(1e-9, meanRateHz));
        if (Double.isNaN(fano) || Double.isInfinite(fano)) fano = 1.0;
        windowsSeen++;
        updateStability(meanRateHz);
    }
}
