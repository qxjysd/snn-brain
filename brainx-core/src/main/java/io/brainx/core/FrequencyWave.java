package io.brainx.core;

/**
 * 频率波发生器 —— 脉冲信号=频率波 (类脑核心)。
 *
 * 神经科学依据:
 *   - 神经元之间的信号本质是发放频率 (firing rate), 以波形式传播
 *   - 群体神经元同步振荡形成脑电波 (EEG): θ(4-8Hz)记忆/α(8-13Hz)静息/
 *     β(13-30Hz)警觉/γ(30-100Hz)注意绑定
 *   - 记忆检索 = 频率共振: 注入查询频率模式 → 与记忆存储的频率共振
 *   - 突触传递的是频率调制波, 不是离散计数
 *
 * 实现:
 *   - 信号强度 (0-1) → 发放频率 (Hz) 线性/非线性映射
 *   - 生成正弦振荡波 (幅度=强度, 频率=编码频率)
 *   - 神经元驱动: 频率波 → 输入电流 (随波幅变化)
 *   - 脑电节律生成: θ/α/β/γ 多频段叠加
 */
public class FrequencyWave {
    /** 信号→频率映射范围 (Hz) */
    private final double minHz, maxHz;
    /** 当前相位 (波形连续性) */
    private double phase = 0;

    public FrequencyWave(double minHz, double maxHz) {
        this.minHz = minHz;
        this.maxHz = maxHz;
    }

    /** 神经元发放频率范围: 5-40 Hz (对应 LIF 神经元敏感区) */
    public static FrequencyWave neuralRange() { return new FrequencyWave(5.0, 40.0); }
    /** 记忆共振频率范围: θ-α 带 (4-13 Hz, 记忆检索) */
    public static FrequencyWave memoryRange() { return new FrequencyWave(4.0, 13.0); }

    /** 信号强度 (0-1) → 发放频率 (Hz) */
    public double intensityToHz(double intensity) {
        double clamped = Math.max(0, Math.min(1, intensity));
        // 非线性: 弱信号慢发放, 强信号快发放
        return minHz + (maxHz - minHz) * clamped * clamped;
    }

    /** 频率 (Hz) → 归一化强度 (0-1) */
    public double hzToIntensity(double hz) {
        double v = (hz - minHz) / (maxHz - minHz);
        return Math.max(0, Math.min(1, Math.sqrt(Math.max(0, v))));
    }

    /**
     * 生成正弦频率波 (模拟神经元群体同步振荡)。
     * @param intensity 信号强度 (0-1) → 决定频率和幅度
     * @param dtMs      时间步长 (ms)
     * @return 当前波值 (-1..1), 作为输入电流调制
     */
    public double wave(double intensity, double dtMs) {
        double hz = intensityToHz(intensity);
        double amplitude = 0.3 + 0.7 * intensity;  // 强信号波幅大
        phase += 2 * Math.PI * hz * dtMs / 1000.0;
        return amplitude * Math.sin(phase);
    }

    /**
     * 频率波驱动神经元: 波值 → 输入电流。
     * 正半周兴奋, 负半周抑制 (突触频率调制)。
     * 加基础偏置保证强信号能过阈值 (模拟背景兴奋输入)。
     */
    public double driveNeuron(double intensity, double dtMs) {
        double w = wave(intensity, dtMs);
        // 正波→兴奋电流 (nA), 负波→抑制; 基础偏置随强度 (保证发放)
        return w * 20.0 + intensity * 12.0;
    }

    /**
     * 记忆共振: 查询频率与存储频率的匹配度 (0-1)。
     * 频率越接近 → 共振越强 (谐振曲线, 高斯型)。
     */
    public static double resonance(double queryHz, double storedHz, double bandwidthHz) {
        double diff = Math.abs(queryHz - storedHz);
        return Math.exp(-diff * diff / (2 * bandwidthHz * bandwidthHz));
    }

    /** 多频段脑电节律生成 (θ/α/β/γ 叠加, 模拟 EEG), 每次调用推进相位 */
    public double eegRhythm(double thetaAmp, double alphaAmp, double gammaAmp, double dtMs) {
        double t = phase;
        phase += dtMs;
        // θ 4-8Hz, α 8-13Hz, γ 30-100Hz
        double theta = thetaAmp * Math.sin(2 * Math.PI * 6 * t / 1000.0);
        double alpha = alphaAmp * Math.sin(2 * Math.PI * 10 * t / 1000.0);
        double gamma = gammaAmp * Math.sin(2 * Math.PI * 40 * t / 1000.0);
        return theta + alpha + gamma;
    }

    /** 特征频率分配: 索引 → 频率 (记忆/词条均匀分布在频带) */
    public double featureHz(int index, int total) {
        return minHz + (maxHz - minHz) * index / Math.max(1, total - 1);
    }

    public void reset() { phase = 0; }
    public double phase() { return phase; }
}
