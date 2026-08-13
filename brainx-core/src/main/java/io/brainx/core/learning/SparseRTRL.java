package io.brainx.core.learning;

import java.util.Random;

/**
 * Sparse RTRL — 稀疏路径实时递归学习 (arXiv 2603.15195, brain 生态)。
 *
 * 核心发现 (论文): RTRL 的递归雅可比大规模冗余——
 *   随机传播 k=4 条路径 (n=64 时仅 6%) 恢复 78-84% 全 RTRL 适应能力;
 *   **绝对 k 定律**: k=4 在 n=64/128/256 均有效, 网络越大相对越便宜;
 *   从 k=0 到 k>0 是阶梯式跃迁。机制: 雅可比满秩但近各向同性 (条件数 2.6-6.5)。
 *
 * 实现 (论文机制):
 *   - 路径 = 隐藏元的雅可比行 ∂h_i/∂w (W 维); 随机选 k 个隐藏元传播, 其余行冻结
 *   - 递归耦合保留: J[p][j] = f'·(δ + Σ_{p'} wRec[hid_p][hid_p']·J[p'][j])
 *     (只对选中的 k 个隐藏元求和 — 稀疏递归)
 *   - 内存 O(k·W) vs 全 RTRL O(H·W); 传播 O(k²·W) vs O(H²·W)
 *   - k >= H → 全 RTRL; k=0 → 无雅可比传播 (仅输出层 LMS, 论文基线)
 *
 * 网络: input → hidden (tanh, 递归自连接) → output (线性)。
 */
public class SparseRTRL {
    private final int numInput, numHidden;
    private final double lr;
    private final double[] wIn;             // 输入权重: hidden × input
    private final double[] wRec;            // 递归权重: hidden × hidden
    private final double[] wOut;            // 输出权重: hidden
    private final double[] hidden;
    private final double[] hiddenPrev;

    private final int weightsPerRow;        // W = H·I + H·H (每隐藏元的雅可比行长度)
    private final int k;                    // 传播路径数 = 选中的隐藏元数
    private final int[] pathRows;           // k 个选中隐藏元
    private final double[][] jTrace;        // [k][W] 稀疏雅可比行
    private final Random rnd;
    private long steps = 0;
    private static final int RESELECT_EVERY = 200;

    public SparseRTRL(int numInput, int numHidden, int k, double lr, long seed) {
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.lr = lr;
        this.weightsPerRow = numHidden * numInput + numHidden * numHidden;
        this.k = Math.max(0, Math.min(k, numHidden));   // k>=H → 全 RTRL
        this.wIn = new double[numHidden * numInput];
        this.wRec = new double[numHidden * numHidden];
        this.wOut = new double[numHidden];
        this.hidden = new double[numHidden];
        this.hiddenPrev = new double[numHidden];
        this.rnd = new Random(seed);
        for (int i = 0; i < wIn.length; i++) wIn[i] = (rnd.nextDouble() - 0.5) * 0.4;
        for (int i = 0; i < wRec.length; i++) wRec[i] = (rnd.nextDouble() - 0.5) * 0.4;
        for (int i = 0; i < wOut.length; i++) wOut[i] = (rnd.nextDouble() - 0.5) * 0.4;
        // 注意: 构造体内裸 k 是参数, 必须用 this.k (字段已 clamp 到 numHidden)
        this.pathRows = new int[this.k];
        this.jTrace = new double[this.k][weightsPerRow];
        for (int p = 0; p < this.k; p++) pathRows[p] = p % numHidden;
    }

    /** 选中路径数 (k) */
    public int pathCount() { return k; }

    /** 雅可比迹内存条目数: k×W vs 全 RTRL H×W */
    public int traceMemory() { return k * weightsPerRow; }

    /** 全 RTRL 迹内存 (H×W) */
    public int fullTraceMemory() { return numHidden * weightsPerRow; }

    /** 重选随机隐藏元路径 (论文: 任意随机子集都是方向代表) */
    private void reselectPaths() {
        for (int p = 0; p < k; p++) {
            pathRows[p] = rnd.nextInt(numHidden);
            java.util.Arrays.fill(jTrace[p], 0);
        }
    }

    /**
     * 在线一步: 前向 + 稀疏雅可比传播 + 权重更新 (含输出层 LMS)。
     * @return 预测误差 (target - output)
     */
    public double learnStep(double[] input, double target) {
        // ===== 前向 =====
        for (int i = 0; i < numHidden; i++) {
            double sum = 0;
            int baseIn = i * numInput;
            for (int j = 0; j < numInput; j++) sum += wIn[baseIn + j] * input[j];
            int baseRec = i * numHidden;
            for (int j = 0; j < numHidden; j++) sum += wRec[baseRec + j] * hiddenPrev[j];
            hidden[i] = Math.tanh(sum);
        }
        double out = 0;
        for (int i = 0; i < numHidden; i++) out += wOut[i] * hidden[i];
        double err = target - out;

        // ===== 稀疏雅可比传播: J[p][j] = f'·(δ_pj + Σ_{p'} wRec[p][p']·J[p'][j]) =====
        if (k > 0) {
            double[][] newTrace = new double[k][weightsPerRow];
            for (int p = 0; p < k; p++) {
                int hid = pathRows[p];
                double fp = 1 - hidden[hid] * hidden[hid];
                // 递归耦合: 只对选中的 k 个隐藏元求和 (稀疏递归)
                double[] recurTerm = new double[weightsPerRow];
                for (int p2 = 0; p2 < k; p2++) {
                    double w = wRec[hid * numHidden + pathRows[p2]];
                    if (w != 0) {
                        double[] jp2 = jTrace[p2];
                        for (int j = 0; j < weightsPerRow; j++) recurTerm[j] += w * jp2[j];
                    }
                }
                // 输入侧 δ: 只有"连接到 hid_p 的权重"才有直接贡献 (其余权重 δ=0)
                int baseIn = hid * numInput;
                int baseRec = hid * numHidden;
                for (int j = 0; j < weightsPerRow; j++) {
                    int wHid;   // 权重 j 连接的目标隐藏元
                    if (j < numHidden * numInput) wHid = j / numInput;
                    else wHid = (j - numHidden * numInput) / numHidden;
                    double delta = 0;
                    if (wHid == hid) {
                        if (j < numHidden * numInput) delta = input[j % numInput];
                        else delta = hiddenPrev[(j - numHidden * numInput) % numHidden];
                    }
                    newTrace[p][j] = fp * (delta + recurTerm[j]);
                }
            }
            for (int p = 0; p < k; p++) System.arraycopy(newTrace[p], 0, jTrace[p], 0, weightsPerRow);
        }

        // ===== 权重更新: Δw_j = lr · Σ_p hidErr_p · J[p][j] =====
        if (k > 0) {
            double[] dw = new double[weightsPerRow];
            for (int p = 0; p < k; p++) {
                int hid = pathRows[p];
                double hidErr = err * (1 - hidden[hid] * hidden[hid]) * wOut[hid];
                if (hidErr != 0) {
                    double[] jp = jTrace[p];
                    for (int j = 0; j < weightsPerRow; j++) dw[j] += hidErr * jp[j];
                }
            }
            double lrs = lr;
            for (int i = 0; i < numHidden; i++) {
                int baseIn = i * numInput;
                for (int j = 0; j < numInput; j++) wIn[baseIn + j] += lrs * dw[i * numInput + j];
                int baseRec = i * numHidden;
                for (int j = 0; j < numHidden; j++) wRec[baseRec + j] += lrs * dw[numHidden * numInput + i * numHidden + j];
            }
        }
        // 输出层 LMS (全量, O(H))
        for (int i = 0; i < numHidden; i++) wOut[i] += lr * err * hidden[i];

        System.arraycopy(hidden, 0, hiddenPrev, 0, numHidden);
        if (++steps % RESELECT_EVERY == 0) reselectPaths();
        return err;
    }

    /** 纯预测 (状态推进) */
    public double predict(double[] input) {
        for (int i = 0; i < numHidden; i++) {
            double sum = 0;
            int baseIn = i * numInput;
            for (int j = 0; j < numInput; j++) sum += wIn[baseIn + j] * input[j];
            int baseRec = i * numHidden;
            for (int j = 0; j < numHidden; j++) sum += wRec[baseRec + j] * hiddenPrev[j];
            hidden[i] = Math.tanh(sum);
        }
        double out = 0;
        for (int i = 0; i < numHidden; i++) out += wOut[i] * hidden[i];
        System.arraycopy(hidden, 0, hiddenPrev, 0, numHidden);
        return out;
    }
}
