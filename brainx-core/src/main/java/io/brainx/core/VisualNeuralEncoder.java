package io.brainx.core;

/**
 * 视觉神经编码器 —— 图像直接转为神经信号 (视网膜→视觉皮层编码)。
 *
 * 神经科学依据 (《醉醺醺的脑科学》第12篇 "眼睛知道什么对我们有好处"):
 *   - 视网膜神经节细胞: 中心-周边拮抗感受野 (ON/OFF)
 *     (Difference of Gaussians, DoG) — 增强边缘对比, 压缩冗余
 *   - 初级视觉皮层 V1: 方向选择性神经元 (边缘/朝向检测)
 *   - 输出 = 神经节细胞/V1 神经元的发放率 (0-1) — 直接神经信号
 *
 * 编码链:
 *   灰度图 → 中心-周边拮抗(DoG) → 感受野网格发放率
 *          → 方向边缘(4方向 Sobel × 4尺度) → V1 方向神经元发放率
 *   输出: 5120 维神经信号 (64×64 感受野拮抗 + 4方向×4尺度×64 空间方向特征)
 */
public class VisualNeuralEncoder {
    /** 输入灰度图边长 (128×128 采样, 超采样空间分辨率) */
    public static final int GRID = 128;
    /** 感受野输出网格 (64×64 = 4096 神经信号) */
    public static final int RF_GRID = 64;
    /** 方向尺度数 (细/中/粗/极粗 4 尺度金字塔, V1 复杂细胞) */
    public static final int DIR_SCALES = 4;
    /** 总输出维度: 4096 感受野 + 4方向×4尺度×64 = 5120 (1280×4 精确4倍) */
    public static final int OUTPUT_DIM = RF_GRID * RF_GRID + 4 * DIR_SCALES * RF_GRID;

    /**
     * 图像灰度像素 (任意尺寸) → 神经信号向量。
     * @param grayPixels 灰度像素数组 (0-255)
     * @param width 图像宽
     * @param height 图像高
     * @return 5120 维神经信号 (0-1 发放率)
     */
    public double[] encode(double[] grayPixels, int width, int height) {
        // 1. 缩放到 128×128 灰度网格
        double[][] grid = resizeToGrid(grayPixels, width, height);

        // 2. 中心-周边拮抗感受野 (DoG): ON/OFF 通道
        double[][] onRF = new double[RF_GRID][RF_GRID];
        double[][] offRF = new double[RF_GRID][RF_GRID];
        // 感受野: 每 2×2 网格块 = 一个神经节细胞感受野
        for (int gy = 0; gy < RF_GRID; gy++) {
            for (int gx = 0; gx < RF_GRID; gx++) {
                // 中心区 (2×2) 和周边区 (4×4 环)
                double center = 0, surround = 0;
                int cn = 0, sn = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = gy * 2 + dy, xx = gx * 2 + dx;
                        if (yy >= 0 && yy < GRID && xx >= 0 && xx < GRID) {
                            double v = grid[yy][xx];
                            // 中心 1×1 内圈 vs 周边环
                            if (Math.abs(dy) <= 0 && Math.abs(dx) <= 0) { center += v; cn++; }
                            else { surround += v; sn++; }
                        }
                    }
                }
                if (cn == 0) cn = 1;
                if (sn == 0) sn = 1;
                center /= cn;
                surround /= sn;
                // DoG 拮抗: ON 通道 = 中心-周边 (正), OFF = 周边-中心 (负)
                double dog = center - surround;
                onRF[gy][gx] = clamp01(dog);       // ON 神经节细胞发放率
                offRF[gy][gx] = clamp01(-dog);     // OFF 神经节细胞发放率
            }
        }

        // 3. 方向边缘 (V1 方向选择性): 4 方向 Sobel 能量 × 4 尺度 (细/中/粗/极粗 金字塔)
        double[] direction = new double[4 * DIR_SCALES * RF_GRID];
        // 每方向×每尺度 → GRID×GRID 边缘能量图 (s=0 细: 原分辨率, s=1..3: 2/4/8 倍降采样)
        double[][][] dirEdges = new double[4 * DIR_SCALES][][];   // [d*DIR_SCALES+s] → 边缘能量图
        for (int d = 0; d < 4; d++) {
            dirEdges[d * DIR_SCALES] = sobel(grid, d);
            for (int s = 1; s < DIR_SCALES; s++) {
                dirEdges[d * DIR_SCALES + s] = sobelCoarse(grid, d, 1 << s);   // 降采样 2/4/8 倍
            }
        }
        // 降采样到 RF_GRID 每方向每尺度 (多尺度空间位置特征)
        int scaleStep = GRID / RF_GRID;
        for (int d = 0; d < 4; d++) {
            for (int s = 0; s < DIR_SCALES; s++) {
                double[][] src = dirEdges[d * DIR_SCALES + s];
                for (int pos = 0; pos < RF_GRID; pos++) {
                    double sum = 0;
                    for (int y = pos * scaleStep; y < pos * scaleStep + scaleStep && y < GRID; y++) {
                        for (int x = 0; x < GRID; x++) sum += src[y][x];
                    }
                    direction[(d * DIR_SCALES + s) * RF_GRID + pos] =
                            clamp01(sum / (scaleStep * GRID));
                }
            }
        }

        // 4. 拼接神经信号: 4096 感受野拮抗 + 1024 方向特征 = 5120 维
        //    增益放大: 生物神经信号需要动态范围 (DoG 原始响应小)
        double[] out = new double[OUTPUT_DIM];
        for (int i = 0; i < RF_GRID * RF_GRID; i++) {
            int y = i / RF_GRID, x = i % RF_GRID;
            out[i] = onRF[y][x] - offRF[y][x];  // 拮抗净信号
            if (out[i] < 0) out[i] = 0;
            out[i] = clamp01(out[i] * 2.5);  // 感受野增益
        }
        for (int i = 0; i < direction.length; i++) {
            out[RF_GRID * RF_GRID + i] = clamp01(direction[i] * 4.0);  // 方向增益
        }
        return out;
    }

    /** 多尺度 Sobel: block 均值降采样 factor 倍 → sobel → 放大回 GRID×GRID (V1 复杂细胞大感受野金字塔) */
    private double[][] sobelCoarse(double[][] grid, int dir, int factor) {
        int n = GRID / factor;
        double[][] reduced = new double[n][n];
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                double sum = 0;
                for (int dy = 0; dy < factor; dy++) {
                    for (int dx = 0; dx < factor; dx++) sum += grid[y * factor + dy][x * factor + dx];
                }
                reduced[y][x] = sum / (factor * factor);
            }
        }
        double[][] edge = sobel(reduced, dir);
        // 放大回 GRID×GRID (后续带区求和按 GRID 尺寸统一处理)
        double[][] out = new double[GRID][GRID];
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                for (int dy = 0; dy < factor; dy++) {
                    for (int dx = 0; dx < factor; dx++) out[y * factor + dy][x * factor + dx] = edge[y][x];
                }
            }
        }
        return out;
    }

    /** 灰度数组 → 16×16 网格 (块均值采样: 每格取原图块平均, 采样充分) */
    private double[][] resizeToGrid(double[] pixels, int w, int h) {
        double[][] grid = new double[GRID][GRID];
        if (pixels == null || pixels.length == 0 || w <= 0 || h <= 0) return grid;
        for (int gy = 0; gy < GRID; gy++) {
            for (int gx = 0; gx < GRID; gx++) {
                // 每格对应原图块 [y0..y1) × [x0..x1) — 取块内所有像素均值
                int y0 = gy * h / GRID, y1 = Math.max(y0 + 1, (gy + 1) * h / GRID);
                int x0 = gx * w / GRID, x1 = Math.max(x0 + 1, (gx + 1) * w / GRID);
                double sum = 0;
                int cnt = 0;
                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        int idx = y * w + x;
                        if (idx < pixels.length) {
                            sum += pixels[idx];
                            cnt++;
                        }
                    }
                }
                grid[gy][gx] = cnt > 0 ? Math.max(0, Math.min(1, sum / cnt / 255.0)) : 0;
            }
        }
        return grid;
    }

    /** Sobel 边缘 (方向: 0=水平边缘检测[垂直梯度], 1=垂直边缘[水平梯度], 2=对角\, 3=对角/) */
    private double[][] sobel(double[][] grid, int dir) {
        int g = grid.length;  // 动态尺寸 (支持降采样图)
        double[][] out = new double[g][g];
        // 标准 Sobel: Gx=水平梯度(检测垂直边缘), Gy=垂直梯度(检测水平边缘)
        int[] gx = new int[]{-1,0,1,-2,0,2,-1,0,1};  // 水平梯度 → 垂直边缘
        int[] gy = new int[]{-1,-2,-1,0,0,0,1,2,1};  // 垂直梯度 → 水平边缘
        int[] kx, ky;
        if (dir == 0) { kx = gy; ky = gx; }          // 水平边缘: |Gy|
        else if (dir == 1) { kx = gx; ky = gy; }     // 垂直边缘: |Gx|
        else if (dir == 2) { kx = new int[]{0,-1,-2,1,0,-1,2,1,0}; ky = new int[]{-2,-1,0,-1,0,1,0,1,2}; }
        else { kx = new int[]{-2,-1,0,-1,0,1,0,1,2}; ky = new int[]{0,-1,-2,1,0,-1,2,1,0}; }
        for (int y = 1; y < g - 1; y++) {
            for (int x = 1; x < g - 1; x++) {
                double sum = 0;
                int k = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        sum += grid[y + dy][x + dx] * kx[k] / 8.0;
                        k++;
                    }
                }
                out[y][x] = Math.abs(sum);
            }
        }
        return out;
    }

    private static double clamp01(double v) {
        return Math.max(0, Math.min(1, v));
    }

    /** 摘要 */
    public static String summary() {
        return String.format("👁️ 视觉神经编码: %d×%d采样 → 感受野拮抗+方向选择性 → %d维神经信号",
                GRID, GRID, OUTPUT_DIM);
    }
}
