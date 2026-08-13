package io.brainx.core.encoding;

import java.util.Random;

/**
 * 输入编码器（对应 brain 的 braintools + SpikingGamma 的 sigma-delta）。
 * 1. PoissonEncoder: 速率→脉冲概率 (视觉/通用)
 * 2. SigmaDeltaEncoder: 连续信号→脉冲流 (听觉/时序)
 * 3. ITD: 双耳时间差→空间位置 (SpikingGamma 论文的猫头鹰听觉定位)
 */
public final class Encoders {
    private Encoders() {}

    /** Poisson 编码: 输入值 (0-1) 作为发放概率，逐时间步伯努利采样 */
    public static final class Poisson {
        private final Random rnd;
        public Poisson(long seed) { this.rnd = new Random(seed); }
        public Poisson() { this.rnd = new Random(); }

        /** 返回 1 表示发放 (概率 = value 0..1) */
        public int encode(double value) {
            return rnd.nextDouble() < value ? 1 : 0;
        }

        /** 向量版 */
        public int[] encode(double[] values) {
            int[] out = new int[values.length];
            for (int i = 0; i < values.length; i++) out[i] = encode(values[i]);
            return out;
        }
    }

    /**
     * Sigma-Delta 编码器: 连续信号 → 脉冲流。
     * y(t) 积分误差超阈值 → 发放 (带符号)，误差复位。
     * 对应 SpikingGamma 论文的 sigma-delta spike coding。
     */
    public static final class SigmaDelta {
        private final double theta;
        private double integral = 0;

        public SigmaDelta(double theta) { this.theta = theta; }
        public static SigmaDelta defaultParams() { return new SigmaDelta(0.1); }

        /** 输入连续值, 返回 -1/0/+1 脉冲 */
        public int encode(double value) {
            integral += value;
            int spike = 0;
            if (integral >= theta) { spike = 1; integral -= theta; }
            else if (integral <= -theta) { spike = -1; integral += theta; }
            return spike;
        }

        public void reset() { integral = 0; }
        public double integral() { return integral; }
    }

    /**
     * ITD (Interaural Time Difference) 双耳时间差定位。
     * 来源: SpikingGamma 论文 "Learning delayed coincidence detection" ——
     * 猫头鹰通过双耳脉冲到达时间差 (ITD) 定位声源方位。
     *
     * 实现: 左右耳脉冲流的互相关峰值延迟 → 方位角。
     * ITD = (t_right - t_left)，延迟窗口内做互相关。
     */
    public static final class ITD {
        private final int maxDelaySteps;   // 最大延迟 (时间步)
        private final double speedOfSound; // 声速 (mm/ms)
        private final double headRadius;   // 头半径 (mm)

        public ITD(int maxDelaySteps, double speedOfSound, double headRadius) {
            this.maxDelaySteps = maxDelaySteps;
            this.speedOfSound = speedOfSound;
            this.headRadius = headRadius;
        }

        public static ITD defaultParams() { return new ITD(20, 340.0, 87.0); }

        /** 理论 ITD: 方位角 theta (弧度) → 时间差 (ms) */
        public double azimuthToITD(double azimuthRad) {
            return headRadius / speedOfSound * (Math.sin(azimuthRad) + azimuthRad);
        }

        /** ITD → 方位角 (近似反演) */
        public double itdToAzimuth(double itdMs) {
            // 近似: theta ≈ itd * c / r (小角度)
            return Math.asin(Math.max(-1, Math.min(1, itdMs * speedOfSound / headRadius)));
        }

        /**
         * 双耳脉冲流互相关定位。
         * @param leftSpikes  左耳脉冲 (0/1 序列)
         * @param rightSpikes 右耳脉冲 (0/1 序列)
         * @return 估计 ITD (ms), 正=声源偏右
         */
        public double locate(int[] leftSpikes, int[] rightSpikes, double dtMs) {
            double bestItd = 0;
            double bestCorr = -1;
            int n = Math.min(leftSpikes.length, rightSpikes.length);
            for (int delay = -maxDelaySteps; delay <= maxDelaySteps; delay++) {
                double corr = 0;
                for (int t = 0; t < n; t++) {
                    int lt = t, rt = t + delay;
                    if (rt >= 0 && rt < n) {
                        corr += leftSpikes[lt] * rightSpikes[rt];
                    }
                }
                if (corr > bestCorr) {
                    bestCorr = corr;
                    bestItd = -delay * dtMs;  // 右耳晚到(右延迟正)→声源偏左
                }
            }
            return bestItd;
        }

        /**
         * 生成测试信号: 给定方位角, 生成带 ITD 的双耳脉冲。
         * 物理: 声源偏右(+) → 右耳更近 → 右耳先到 (t_right < t_left)。
         * 返回: [left[], right[]] 双通道脉冲。
         */
        public int[][] generateTestSignal(double azimuthRad, int durationSteps, double dtMs, long seed) {
            Random rnd = new Random(seed);
            int[] left = new int[durationSteps];
            int[] right = new int[durationSteps];
            double itd = azimuthToITD(azimuthRad);   // >0: 右耳早到
            int itdSteps = (int) Math.round(itd / dtMs);
            for (int t = 0; t < durationSteps; t += 5) {
                if (rnd.nextDouble() < 0.3) {
                    right[t] = 1;
                    int tLeft = t + itdSteps;   // 左耳晚到
                    if (tLeft >= 0 && tLeft < durationSteps) left[tLeft] = 1;
                }
            }
            return new int[][]{left, right};
        }
    }
}
