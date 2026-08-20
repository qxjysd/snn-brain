package io.brainx.core;

/**
 * 全脑网络模型 (Whole-Brain Network Model)。
 * 来源: "Modeling macroscopic brain dynamics with brain-inspired computing
 * architecture" (Nature Communications 2025, 10.1038/s41467-025-64470-3).
 *
 * 论文核心 (粗粒度建模): 每个节点 = 一个脑区 (神经群平均发放率动态,
 * 闭式方程), 节点间通过结构连接 (SC, 来自 dMRI) 耦合, 整体产生宏观
 * 功能动态; 通过模型反演对接 fMRI/dMRI/EEG 多模态数据。
 *
 * 本实现 (机制级复现):
 *   - 8 个脑区节点 (前额叶/顶叶/颞叶/枕叶/海马/丘脑/小脑/运动), 每节点
 *     用 Wilson-Cowan 兴奋-抑制神经群 (闭式平均场方程, 论文代表模型之一)
 *   - 结构连接矩阵 SC (8×8, 默认基于真实脑区解剖连接的简化模式:
 *     枕叶↔顶叶视觉通路、颞叶↔海马记忆通路、丘脑全局中继等)
 *   - 全局耦合: 每节点输入 = 外部刺激 + G·Σ SC_ij·r_j (其他区域活动加权)
 *   - 功能连接 FC: 区域间活动的时间相关性 (Pearson), 由 SC 驱动的宏观
 *     动力学自然涌现 — 论文"结构→功能"桥的核心验证
 *   - 粗粒度优势: 8 节点 vs 微观 5488 神经元, 验证"显著更少节点建模全脑"
 *
 * 与 Brain 集成: 全脑图为宏观镜像 (大脑"全局状态"), 各模块活动上报
 * 对应脑区, 输出 FC 矩阵供 EEG/意识/显示层解读。
 */
public class WholeBrainNetwork {
    /** 脑区名称 (中文, 与项目语言风格一致) */
    public static final String[] REGIONS = {
            "前额叶", "顶叶", "颞叶", "枕叶", "海马", "丘脑", "小脑", "运动"
    };
    public static final int N = REGIONS.length;

    // 每区域 Wilson-Cowan 状态 (兴奋性活动水平 E)
    private final double[] e = new double[N];
    private final double[] i = new double[N];
    // 结构连接矩阵 SC (对称, 0-1 归一化强度)
    private final double[][] sc;
    // 全局耦合强度 G (论文: 可调参数)
    private final double gGlobal;
    // 功能连接时间序列 (滚动窗口, 用于 Pearson 相关)
    private final double[][] history;   // [region][time]
    private final int window;
    private int tick = 0;

    public WholeBrainNetwork() {
        this(defaultSC(), 0.6, 200);
    }

    public WholeBrainNetwork(double[][] sc, double gGlobal, int fcWindow) {
        this.sc = new double[N][N];
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) this.sc[r][c] = sc[r][c];
        this.gGlobal = gGlobal;
        this.window = fcWindow;
        this.history = new double[N][window];
        reset();
    }

    public static WholeBrainNetwork defaultParams() { return new WholeBrainNetwork(); }

    /** 默认结构连接: 基于解剖学的简化模式 (真实脑区连接拓扑) */
    public static double[][] defaultSC() {
        double[][] sc = new double[N][N];
        // 视觉通路: 枕叶↔顶叶 (背侧), 枕叶↔颞叶 (腹侧)
        sc[3][1] = sc[1][3] = 0.8;   // 枕叶↔顶叶
        sc[3][2] = sc[2][3] = 0.7;   // 枕叶↔颞叶
        // 记忆通路: 颞叶↔海马
        sc[2][4] = sc[4][2] = 0.9;
        // 前额叶: 全局执行控制 (与顶/颞/海马)
        sc[0][1] = sc[1][0] = 0.6;
        sc[0][2] = sc[2][0] = 0.5;
        sc[0][4] = sc[4][0] = 0.4;
        // 丘脑: 皮层中继 (与所有皮层区域弱连接)
        for (int r = 0; r < 6; r++) {
            if (r != 5) { sc[r][5] = sc[5][r] = 0.3; }
        }
        // 小脑: 运动协调 (与运动/前额)
        sc[6][7] = sc[7][6] = 0.6;
        sc[6][0] = sc[0][6] = 0.3;
        sc[6][3] = sc[3][6] = 0.2;
        // 运动: 接收顶叶 (感觉运动整合)
        sc[7][1] = sc[1][7] = 0.5;
        // 对角置 0
        for (int r = 0; r < N; r++) sc[r][r] = 0;
        return sc;
    }

    /**
     * 单步全脑动力学 (Wilson-Cowan 耦合).
     *
     * @param externalInput 每区域外部刺激 (视觉→枕叶, 听觉→颞叶等, 可传 null)
     * @param dtMs          时间步长 (ms)
     */
    public void step(double[] externalInput, double dtMs) {
        double[] input = new double[N];
        for (int r = 0; r < N; r++) {
            double ext = (externalInput != null) ? externalInput[r] * 0.1 : 0.0;
            // 全局耦合: 其他区域活动经 SC 加权 (论文: G·Σ SC_ij·r_j)
            double coupling = 0;
            for (int j = 0; j < N; j++) {
                if (j != r) coupling += sc[r][j] * e[j];
            }
            input[r] = ext + gGlobal * coupling;
            // Wilson-Cowan: τE·dE/dt = -E + S(wEE·E - wIE·I + input)
            // 温和参数 (wEE<5) 防自激饱和, 活动稳定在 0.2-0.8 动态范围
            double de = (-e[r] + sigmoid(4.0 * e[r] - 2.5 * i[r] + input[r])) / 8.0;
            double di = (-i[r] + sigmoid(2.5 * e[r] - 2.0 * i[r])) / 8.0;
            e[r] += de * dtMs;
            i[r] += di * dtMs;
            if (e[r] < 0) e[r] = 0;
            if (i[r] < 0) i[r] = 0;
        }
        // 记录功能连接历史
        for (int r = 0; r < N; r++) history[r][tick % window] = e[r];
        tick++;
    }

    private double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }

    /** 区域活动水平 (E, 0-1) */
    public double activity(int region) { return e[region]; }

    public double[] activities() { return e.clone(); }

    /** 结构连接矩阵 */
    public double[][] structuralConnectivity() { return sc.clone(); }

    /**
     * 功能连接 FC 矩阵: 区域活动时间序列 Pearson 相关 (结构→功能桥)。
     * 对角 = 1; 范围 [-1,1]。
     */
    public double[][] functionalConnectivity() {
        double[][] fc = new double[N][N];
        int n = Math.min(window, tick);
        if (n < 5) {
            for (int i = 0; i < N; i++) {
                java.util.Arrays.fill(fc[i], 0);
                fc[i][i] = 1.0;
            }
            return fc;
        }
        for (int a = 0; a < N; a++) {
            for (int b = 0; b < N; b++) {
                if (a == b) { fc[a][b] = 1.0; continue; }
                fc[a][b] = pearson(history[a], history[b], n);
            }
        }
        return fc;
    }

    /** 模块内 vs 模块间 FC: 解剖模块 (视觉/记忆/执行) 内相关应高于跨模块 */
    public double withinModuleFC() {
        double[][] fc = functionalConnectivity();
        double within = 0;
        int cnt = 0;
        int[][] pairs = {{1, 3}, {2, 3}, {2, 4}, {0, 1}, {6, 7}};  // 解剖连接对
        for (int[] p : pairs) { within += Math.abs(fc[p[0]][p[1]]); cnt++; }
        return cnt > 0 ? within / cnt : 0;
    }

    public double betweenModuleFC() {
        double[][] fc = functionalConnectivity();
        double sum = 0;
        int cnt = 0;
        for (int a = 0; a < N; a++)
            for (int b = a + 1; b < N; b++) {
                boolean inModule = false;
                for (int[] p : new int[][]{{1, 3}, {2, 3}, {2, 4}, {0, 1}, {6, 7}}) {
                    if ((p[0] == a && p[1] == b) || (p[0] == b && p[1] == a)) { inModule = true; break; }
                }
                if (!inModule) { sum += Math.abs(fc[a][b]); cnt++; }
            }
        return cnt > 0 ? sum / cnt : 0;
    }

    private double pearson(double[] x, double[] y, int n) {
        double mx = 0, my = 0;
        for (int t = 0; t < n; t++) { mx += x[t]; my += y[t]; }
        mx /= n; my /= n;
        double cov = 0, vx = 0, vy = 0;
        for (int t = 0; t < n; t++) {
            double dx = x[t] - mx, dy = y[t] - my;
            cov += dx * dy; vx += dx * dx; vy += dy * dy;
        }
        if (vx < 1e-12 || vy < 1e-12) return 0;
        return cov / Math.sqrt(vx * vy);
    }

    /** 主导激活区域 (全局工作空间中的"焦点") */
    public int dominantRegion() {
        int best = 0;
        for (int r = 1; r < N; r++) if (e[r] > e[best]) best = r;
        return best;
    }

    /** 全脑同步度: 区域活动的平均成对相关 (全局整合度) */
    public double globalSynchrony() {
        double[][] fc = functionalConnectivity();
        double sum = 0;
        int cnt = 0;
        for (int a = 0; a < N; a++)
            for (int b = a + 1; b < N; b++) { sum += fc[a][b]; cnt++; }
        return cnt > 0 ? sum / cnt : 0;
    }

    /** 宏观状态摘要 */
    public String summary() {
        int dom = dominantRegion();
        return String.format("全脑: %s主导 同步%.2f 模块内FC%.2f/模块间%.2f",
                REGIONS[dom], globalSynchrony(), withinModuleFC(), betweenModuleFC());
    }

    public void reset() {
        java.util.Arrays.fill(e, 0.1);
        java.util.Arrays.fill(i, 0.1);
        for (double[] h : history) java.util.Arrays.fill(h, 0.1);
        tick = 0;
    }
}
