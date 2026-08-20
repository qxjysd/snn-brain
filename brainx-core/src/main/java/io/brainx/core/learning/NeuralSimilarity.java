package io.brainx.core.learning;

/**
 * 神经表征相似度 (Neural Similarity) — Procrustes 距离。
 * 来源: "Can Biologically Plausible Temporal Credit Assignment Rules Match
 * BPTT for Neural Similarity? E-prop as an Example" (arXiv 2506.06904).
 *
 * 论文核心方法: 用 Procrustes 距离比较两个模型的神经表征几何 (状态
 * 表示矩阵 X 与 Y), 通过正交变换对齐后测残差:
 *   min_Q ||X·Q - Y||_F,  Q 正交
 *   Procrustes 距离 = ||X·Q* - Y||_F (对齐后的 Frobenius 残差)
 *
 * 闭式解: X^T·Y 的 SVD = U·S·V^T → Q* = U·V^T (Schönemann 1966).
 * 论文为什么选 Procrustes: 对高维偏置稳健, 是真正的度量, 不像 CCA
 * 依赖低方差噪声分量 / RSA 受维度影响。
 *
 * 用途: 对比 e-prop / pp-prop / 全 RTRL 等规则学到的内部表征与参考
 * 表征 (BPTT 或真实神经数据) 的对齐度 — 论文结论: e-prop 匹配 BPTT。
 */
public final class NeuralSimilarity {

    private NeuralSimilarity() {}

    /**
     * Procrustes 距离: 两个表征矩阵的正交对齐残差。
     * 矩阵行=样本 (时间点), 列=神经元/特征。
     *
     * @return 归一化距离 [0,∞), 越小越相似; 完全一致=0
     */
    public static double procrustesDistance(double[][] X, double[][] Y) {
        int rows = Math.min(X.length, Y.length);
        int cols = Math.min(X[0].length, Y[0].length);
        if (rows == 0 || cols == 0) return Double.POSITIVE_INFINITY;
        // 中心化 (论文标准预处理: 表征先去均值)
        double[][] Xc = center(X, rows, cols);
        double[][] Yc = center(Y, rows, cols);
        // M = X^T·Y
        double[][] M = new double[cols][cols];
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                double s = 0;
                for (int r = 0; r < rows; r++) s += Xc[r][i] * Yc[r][j];
                M[i][j] = s;
            }
        }
        // SVD: M = U·S·V^T → Q* = U·V^T
        double[][] U = new double[cols][cols];
        double[] S = new double[cols];
        double[][] Vt = new double[cols][cols];
        svd(M, U, S, Vt);
        double[][] Q = new double[cols][cols];
        for (int i = 0; i < cols; i++)
            for (int j = 0; j < cols; j++) {
                double s = 0;
                for (int k = 0; k < cols; k++) s += U[i][k] * Vt[k][j];  // Q = U·Vt (Vt 已是 V^T)
                Q[i][j] = s;
            }
        // 残差 ||X·Q - Y||_F
        double res = 0, yNorm = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double xq = 0;
                for (int k = 0; k < cols; k++) xq += Xc[r][k] * Q[k][c];
                double d = xq - Yc[r][c];
                res += d * d;
                yNorm += Yc[r][c] * Yc[r][c];
            }
        }
        if (yNorm < 1e-12) return Double.POSITIVE_INFINITY;
        return Math.sqrt(res) / Math.sqrt(yNorm);
    }

    /** 相似度 [0,1]: 1=完全对齐, 0=正交/无关 */
    public static double similarity(double[][] X, double[][] Y) {
        double d = procrustesDistance(X, Y);
        if (Double.isInfinite(d)) return 0.0;
        return 1.0 / (1.0 + d);
    }

    private static double[][] center(double[][] M, int rows, int cols) {
        double[][] out = new double[rows][cols];
        for (int c = 0; c < cols; c++) {
            double mean = 0;
            for (int r = 0; r < rows; r++) mean += M[r][c];
            mean /= rows;
            for (int r = 0; r < rows; r++) out[r][c] = M[r][c] - mean;
        }
        return out;
    }

    /** 幂迭代 SVD (取主分量, 对小矩阵足够精确) — M = U·S·V^T */
    static void svd(double[][] M, double[][] U, double[] S, double[][] Vt) {
        int n = M.length;
        // 对称矩阵 A = M^T·M → 特征分解得 V, S; 再 U = M·V·S^-1
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += M[k][i] * M[k][j];
                A[i][j] = s;
            }
        // Jacobi 特征分解 (对称矩阵, n 小)
        double[][] V = new double[n][n];
        for (int i = 0; i < n; i++) V[i][i] = 1.0;
        double[] ev = new double[n];
        System.arraycopy(extractDiag(A), 0, ev, 0, n);
        jacobiEigen(A, ev, V, 200);
        // 排序 (降序), 同步 V
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (ev[j] > ev[i]) {
                    double t = ev[i]; ev[i] = ev[j]; ev[j] = t;
                    for (int r = 0; r < n; r++) {
                        double tv = V[r][i]; V[r][i] = V[r][j]; V[r][j] = tv;
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            S[i] = Math.sqrt(Math.max(0, ev[i]));
            for (int j = 0; j < n; j++) Vt[i][j] = V[j][i];  // Vt = V^T
        }
        // U = M·V·S^-1; 退化奇异方向 (σ≈0) 用正交补填充 (防除0)
        double[][] Uraw = new double[n][n];
        for (int j = 0; j < n; j++) {
            if (S[j] > 1e-10) {
                for (int i = 0; i < n; i++) {
                    double s = 0;
                    for (int k = 0; k < n; k++) s += M[i][k] * V[k][j];
                    Uraw[i][j] = s / S[j];
                }
            }
        }
        // Gram-Schmidt 正交化 U 的列 (含退化方向补基) — U 是方法参数(输出)
        for (int j = 0; j < n; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = Uraw[i][j];  // 取第 j 列
            // 若退化方向无内容, 用单位向量作起点
            double nrm = 0;
            for (double v : col) nrm += v * v;
            if (nrm < 1e-12) {
                col[j] = 1.0;
            }
            // 减去已正交列投影
            for (int k = 0; k < j; k++) {
                double dot = 0;
                for (int i = 0; i < n; i++) dot += col[i] * U[i][k];
                for (int i = 0; i < n; i++) col[i] -= dot * U[i][k];
            }
            nrm = 0;
            for (double v : col) nrm += v * v;
            nrm = Math.sqrt(nrm);
            if (nrm > 1e-12) {
                for (int i = 0; i < n; i++) U[i][j] = col[i] / nrm;
            } else {
                // 全零列 (理论不可能, 防御)
                for (int i = 0; i < n; i++) U[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    private static double[] extractDiag(double[][] A) {
        double[] d = new double[A.length];
        for (int i = 0; i < A.length; i++) d[i] = A[i][i];
        return d;
    }

    /** 对称矩阵 Jacobi 特征分解 (Numerical Recipes 标准版, tau 简化旋转) */
    static void jacobiEigen(double[][] A, double[] d, double[][] V, int maxSweeps) {
        int n = A.length;
        double[] b = new double[n], z = new double[n];
        System.arraycopy(d, 0, b, 0, n);
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double off = 0;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
            if (off < 1e-18) break;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double apq = A[p][q];
                    if (Math.abs(apq) < 1e-14) continue;
                    double theta = 0.5 * (d[q] - d[p]) / apq;
                    double t = 1.0 / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
                    if (theta < 0.0) t = -t;
                    double c = 1.0 / Math.sqrt(t * t + 1.0);
                    double s = t * c;
                    double tau = s / (1.0 + c);
                    double h = t * apq;
                    z[p] -= h; z[q] += h;
                    d[p] -= h; d[q] += h;
                    A[p][q] = A[q][p] = 0.0;
                    // 三段式旋转: 只更新非对角元素, 对角线由 d/z 维护
                    for (int j = 0; j < p; j++) {
                        double g = A[j][p], hh = A[j][q];
                        A[j][p] = g - s * (hh + g * tau);
                        A[j][q] = hh + s * (g - hh * tau);
                    }
                    for (int j = p + 1; j < q; j++) {
                        double g = A[p][j], hh = A[j][q];
                        A[p][j] = g - s * (hh + g * tau);
                        A[j][q] = hh + s * (g - hh * tau);
                    }
                    for (int j = q + 1; j < n; j++) {
                        double g = A[p][j], hh = A[q][j];
                        A[p][j] = g - s * (hh + g * tau);
                        A[q][j] = hh + s * (g - hh * tau);
                    }
                    // 特征向量累积
                    for (int j = 0; j < n; j++) {
                        double g = V[j][p], hh = V[j][q];
                        V[j][p] = g - s * (hh + g * tau);
                        V[j][q] = hh + s * (g - hh * tau);
                    }
                }
            }
            // 每轮 sweep 结束: d += z, z 清零 (NR 标准)
            for (int i = 0; i < n; i++) { b[i] += z[i]; d[i] = b[i]; z[i] = 0.0; }
        }
    }
}
