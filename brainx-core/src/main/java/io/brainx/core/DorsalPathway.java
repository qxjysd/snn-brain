package io.brainx.core;

/**
 * 背侧通路 (Where/How) —— 空间位置与运动感知。
 *
 * 资料依据 (大脑双通路系统):
 *   背侧通路: 枕叶→顶叶, 负责空间位置/运动轨迹/深度 → "物体在哪里/如何行动"
 *   与腹侧通路 (What: 形状/颜色/识别) 协同 → 完整立体视觉
 *
 * 实现:
 *   - 空间位置编码: 16×16 灰度图 → 亮度质心 (物体在视野哪侧/上下)
 *   - 运动检测: 两帧差异 → 运动方向/强度 (顶叶 MT 区)
 *   - 输出: 空间特征 (6维: 质心x/y/覆盖/运动x/y/运动强度)
 */
public class DorsalPathway {
    /** 输出维度: [质心X, 质心Y, 覆盖度, 运动X, 运动Y, 运动强度] */
    public static final int OUTPUT_DIM = 6;

    /**
     * 空间编码: 当前帧 → 位置/覆盖特征。
     * @param grayPixels 灰度 (0-255)
     * @param width 宽
     * @param height 高
     * @return [质心X(0-1左-右), 质心Y(0-1上-下), 覆盖度(0-1)]
     */
    public double[] encodePosition(double[] grayPixels, int width, int height) {
        double[] out = new double[3];
        if (grayPixels == null || width <= 0 || height <= 0) return out;
        double sum = 0, sumX = 0, sumY = 0;
        int n = Math.min(grayPixels.length, width * height);
        for (int y = 0; y < height && y * width < n; y++) {
            for (int x = 0; x < width && y * width + x < n; x++) {
                double v = grayPixels[y * width + x] / 255.0;
                if (v > 0.3) {  // 显著物体像素
                    sum += v;
                    sumX += v * x;
                    sumY += v * y;
                }
            }
        }
        if (sum > 0) {
            out[0] = sumX / sum / width;   // 质心X 0-1
            out[1] = sumY / sum / height;  // 质心Y 0-1
            out[2] = Math.min(1.0, sum / (width * height * 0.5));  // 覆盖度
        }
        return out;
    }

    /**
     * 运动检测: 两帧差异 → 运动方向/强度 (顶叶 MT 区)。
     * @param prev 前一帧灰度
     * @param curr 当前帧灰度
     * @param width 宽
     * @param height 高
     * @return [运动X(-1..1), 运动Y(-1..1), 运动强度(0-1)]
     */
    public double[] encodeMotion(double[] prev, double[] curr, int width, int height) {
        double[] out = new double[3];
        if (prev == null || curr == null || width <= 0 || height <= 0) return out;
        int n = Math.min(Math.min(prev.length, curr.length), width * height);
        double moveX = 0, moveY = 0, strength = 0;
        int count = 0;
        for (int y = 1; y < height - 1 && y * width < n; y++) {
            for (int x = 1; x < width - 1 && y * width + x < n; x++) {
                int idx = y * width + x;
                double diff = Math.abs(curr[idx] / 255.0 - prev[idx] / 255.0);
                if (diff > 0.15) {  // 显著变化像素
                    // 估计局部运动方向: 与左/右/上/下邻居差异
                    double dl = Math.abs(curr[idx] - curr[idx - 1]);
                    double dr = Math.abs(curr[idx] - curr[idx + 1]);
                    double du = Math.abs(curr[idx] - curr[idx - width]);
                    double dd = Math.abs(curr[idx] + (idx + width < n ? curr[idx + width] : curr[idx]));
                    // 简化: 用当前帧梯度方向估计运动倾向
                    moveX += (dl > dr ? -1 : 1) * diff;
                    moveY += (du > dd ? -1 : 1) * diff;
                    strength += diff;
                    count++;
                }
            }
        }
        if (count > 0) {
            out[0] = Math.max(-1, Math.min(1, moveX / count));
            out[1] = Math.max(-1, Math.min(1, moveY / count));
            out[2] = Math.min(1.0, strength / count * 2);
        }
        return out;
    }

    /** 摘要 */
    public static String summary() {
        return "🧭 背侧通路: 空间位置(质心/覆盖) + 运动检测(MT区) → Where/How";
    }
}
