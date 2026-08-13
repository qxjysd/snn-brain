package io.brainx.core;

import java.util.Random;

/**
 * 随机脉冲信号源 —— 模拟大脑自发神经活动 (spontaneous neural activity)。
 *
 * 神经科学依据 (fWBM 论文 critical brain hypothesis):
 *   - 大脑即使在无刺激时也有随机放电 (背景噪声/自发活动)
 *   - 大脑在临界点运行, 自发活动具有高效信息传输优势
 *   - 随机共激活驱动突触形成 (fire together, wire together)
 *
 * 三种噪声模式:
 *   POISSON  泊松发放 (速率λ, 模拟背景突触输入)
 *   GAUSSIAN 高斯电流噪声 (膜电位抖动, Ornstein-Uhlenbeck 简化)
 *   BURST    爆发性噪声 (模拟皮层 up-state 自发爆发)
 */
public class NoiseSource {
    public enum Mode { POISSON, GAUSSIAN, BURST }

    private final Mode mode;
    private final double rate;        // 泊松速率 (spikes/s)
    private final double amplitude;   // 噪声幅度
    private final double dtMs;
    private final Random rnd;
    private int burstCounter = 0;

    public NoiseSource(Mode mode, double rate, double amplitude, double dtMs, long seed) {
        this.mode = mode;
        this.rate = rate;
        this.amplitude = amplitude;
        this.dtMs = dtMs;
        this.rnd = new Random(seed);
    }

    /** 背景泊松噪声: 8Hz, 幅度 0.3 */
    public static NoiseSource poissonBackground(double dtMs) {
        return new NoiseSource(Mode.POISSON, 8.0, 0.3, dtMs, 42);
    }

    /** 高斯电流噪声 */
    public static NoiseSource gaussian(double sigma, double dtMs) {
        return new NoiseSource(Mode.GAUSSIAN, 0, sigma, dtMs, 7);
    }

    /** 爆发性噪声 (模拟皮层 up-state) */
    public static NoiseSource burst(double rate, double amplitude, double dtMs) {
        return new NoiseSource(Mode.BURST, rate, amplitude, dtMs, 99);
    }

    /** 每步采样: 返回噪声电流 (nA) 或 0/1 脉冲 */
    public double sample() {
        switch (mode) {
            case POISSON:
                // 泊松: 每步发放概率 = rate * dt/1000
                return rnd.nextDouble() < rate * dtMs / 1000.0 ? amplitude : 0;
            case GAUSSIAN:
                return rnd.nextGaussian() * amplitude;
            case BURST:
                // up-state 爆发: 随机开始, 持续 20-50ms
                if (burstCounter > 0) {
                    burstCounter--;
                    return amplitude;
                }
                if (rnd.nextDouble() < rate * dtMs / 1000.0) {
                    burstCounter = 20 + rnd.nextInt(30);
                    return amplitude;
                }
                return 0;
            default:
                return 0;
        }
    }

    /** 泊松脉冲 (0/1) —— 用于突触输入 */
    public int spike() {
        return rnd.nextDouble() < rate * dtMs / 1000.0 ? 1 : 0;
    }

    public Mode mode() { return mode; }
    public double rate() { return rate; }
}
