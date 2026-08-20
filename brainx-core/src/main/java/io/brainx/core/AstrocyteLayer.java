package io.brainx.core;

/**
 * 星形胶质细胞三突触可塑性层 (Astrocyte-Gated Multi-Timescale Plasticity, AGMP)。
 * 来源: Dong & He, "Astrocyte-gated multi-timescale plasticity for online
 * continual learning in deep spiking neural networks", Frontiers in
 * Neuroscience 2026 (10.3389/fnins.2025.1768235).
 *
 * 生物学动机: 人脑中胶质细胞约占总细胞一半, 星形胶质细胞包裹突触(三突触
 * 结构), 以秒-分钟级慢时间常数整合局部活动, 释放胶质递质门控突触可塑性。
 * 这解决了"稳定-可塑困境": 新任务快速学习的同时不灾难性遗忘旧知识。
 *
 * 算法 (论文 Eq.8-10):
 *   a_i[t+1] = λa·a_i[t] + (1-λa)·φ_i[t]          (慢速胶质状态, τa ≫ τe)
 *   φ_i[t]   = ηu·|v_i[t]| + ηs·Σ_j|w_ij|·s_j[t] + ηo·s_i[t]  (活动负载)
 *   â_i      = (a_i - μa)/(σa + ε)                  (层内移动统计归一化)
 *   g_i[t]   = σ(kg·â_i[t] + βg)                    (sigmoid 门控因子 ∈ [0,1])
 *   Δw_ij    = η·g_i[t]·M_i[t]·e_ij[t] - ηdecay·w_ij (四因子更新: 门控×误差×迹)
 *
 * 复杂度 O(N) 每步, 与突触数无关; 门控因子按突触后神经元共享 (对应星形
 * 胶质细胞域包裹局部突触簇)。
 */
public class AstrocyteLayer {
    // 慢速整合参数 (论文: τa 秒级, τm/τe 毫秒级)
    private final double tauA;       // 星形胶质细胞时间常数 (ms), 默认 5000 (5s)
    private final double etaU, etaS, etaO;  // 活动负载系数: 膜电位/突触输入/输出脉冲
    private final double kg, betaG;  // 门控 sigmoid 参数
    private final double eps = 1e-6; // 归一化稳定性常数

    private final int n;             // 神经元数
    private final double[] a;        // 星形胶质细胞状态 (慢速积分)
    private final double[] gate;     // 门控因子 g_i
    private double muA, sigmaA;      // 层内移动统计
    private long steps = 0;

    public AstrocyteLayer(int n) {
        this(n, 5000.0, 0.2, 0.3, 0.5, 4.0, 0.0);
    }

    public AstrocyteLayer(int n, double tauA, double etaU, double etaS, double etaO,
                          double kg, double betaG) {
        this.n = n;
        this.tauA = tauA;
        this.etaU = etaU;
        this.etaS = etaS;
        this.etaO = etaO;
        this.kg = kg;
        this.betaG = betaG;
        this.a = new double[n];
        this.gate = new double[n];
        java.util.Arrays.fill(gate, 0.5);  // 初始中性门控
    }

    public static AstrocyteLayer defaultParams(int n) { return new AstrocyteLayer(n); }

    /**
     * 单步更新星形胶质细胞状态并计算门控因子。
     *
     * @param membranePotentials 膜电位 v_i (可传 null 表示无膜电位项)
     * @param synapticInputs     突触输入 Σ|w_ij|·s_j (论文 ηs 项, 可传 null)
     * @param outputSpikes       输出脉冲 s_i ∈ {0,1}
     * @param dtMs               时间步长 (ms)
     */
    public void step(double[] membranePotentials, double[] synapticInputs,
                     boolean[] outputSpikes, double dtMs) {
        double lambdaA = Math.exp(-dtMs / tauA);
        double inv = 1.0 - lambdaA;
        for (int i = 0; i < n; i++) {
            double phi = 0.0;
            if (membranePotentials != null) phi += etaU * Math.abs(membranePotentials[i]);
            if (synapticInputs != null)     phi += etaS * Math.abs(synapticInputs[i]);
            if (outputSpikes != null && outputSpikes[i]) phi += etaO;
            a[i] = lambdaA * a[i] + inv * phi;
        }
        // 层内移动统计归一化 (论文 Eq. 层归一化)
        double sum = 0;
        for (int i = 0; i < n; i++) sum += a[i];
        muA = sum / n;
        double var = 0;
        for (int i = 0; i < n; i++) {
            double d = a[i] - muA;
            var += d * d;
        }
        sigmaA = Math.sqrt(var / n + eps);
        for (int i = 0; i < n; i++) {
            double norm = (a[i] - muA) / sigmaA;
            gate[i] = sigmoid(kg * norm + betaG);
        }
        steps++;
    }

    private double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }

    /** 门控因子数组 g_i ∈ [0,1]: 高活动神经元门控→1(允许学习), 低活动→低(保护旧知识) */
    public double[] gates() { return gate.clone(); }

    public double gate(int i) { return gate[i]; }

    /** 星形胶质细胞慢速状态 a_i */
    public double[] states() { return a.clone(); }

    public double meanState() { return muA; }
    public double stdState() { return sigmaA; }
    public long steps() { return steps; }

    /**
     * AGMP 权重更新: Δw_ij = η·g_i·M_i·e_ij (论文 Eq.10, 无衰减项版本,
     * 衰减由调用方学习规则负责, 避免双重重衰减)。
     *
     * @param learningSignal M_i (广播误差/教学信号)
     * @param eligibility    e_ij (资格迹)
     * @param lr             学习率 η
     * @param rowIndex       突触后神经元索引 i
     */
    public double gatedUpdate(double learningSignal, double eligibility,
                              double lr, int rowIndex) {
        return lr * gate[rowIndex] * learningSignal * eligibility;
    }

    /** 重置为初始状态 (a=0, gate=0.5) */
    public void reset() {
        java.util.Arrays.fill(a, 0.0);
        java.util.Arrays.fill(gate, 0.5);
        muA = sigmaA = 0.0;
        steps = 0;
    }

    /** 摘要: 平均门控/高门控神经元数/平均胶质状态 */
    public String summary() {
        int high = 0;
        double gSum = 0;
        for (int i = 0; i < n; i++) {
            gSum += gate[i];
            if (gate[i] > 0.5) high++;
        }
        return String.format("胶质: %d神经元 门控均值%.2f 高门控%d 状态均值%.3f",
                n, gSum / n, high, muA);
    }
}
