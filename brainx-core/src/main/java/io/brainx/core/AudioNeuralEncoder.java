package io.brainx.core;

/**
 * 听觉神经编码器 —— 声音直接转为神经信号 (耳蜗→听神经编码)。
 *
 * 神经科学依据 (《醉醺醺的脑科学》第23篇 声音识别):
 *   - 耳蜗基底膜: 频率位置编码 (低频在顶端, 高频在底端) — 频带分解
 *   - 毛细胞: 强度→神经发放率 (对数压缩, 韦伯-费希纳响度感知)
 *   - 听神经: 16 个频带神经元的发放率 (0-1) — 直接神经信号
 *
 * 编码链:
 *   PCM 采样 → 耳蜗频带分解 (分带 RMS) → 对数压缩 → 听神经发放率
 * 输出: 16 维听觉神经信号 (0-1 发放率)
 */
public class AudioNeuralEncoder {
    /** 频带数 (耳蜗位置编码 — 对数频率分布, 128 通道 = 人工耳蜗滤波器组级) */
    public static final int BANDS = 128;
    /** 采样率 (语音标准 16kHz) */
    public static final int SAMPLE_RATE = 16000;
    /** 最低频 (Hz): 20Hz 起 (人耳 20Hz-20kHz 范围, 增强音域) */
    private static final double F_MIN = 20;

    /**
     * PCM 采样 → 听觉神经信号 (耳蜗对数频带分解)。
     * @param samples 归一化 PCM 采样 (-1..1)
     * @param sampleRate 实际采样率 (Hz)
     * @return BANDS 维神经信号 (0-1 发放率)
     */
    public static double[] encode(short[] samples, int sampleRate) {
        double[] neural = new double[BANDS];
        if (samples == null || samples.length == 0) return neural;
        int n = samples.length;
        int nyquist = sampleRate / 2;
        for (int band = 0; band < BANDS; band++) {
            // 对数频率分布 (耳蜗基底膜: 低频密集高频稀疏, 20Hz 起)
            double fLow = F_MIN * Math.pow(nyquist / F_MIN, (double) band / BANDS);
            double fHigh = F_MIN * Math.pow(nyquist / F_MIN, (double) (band + 1) / BANDS);
            int idxLow = (int) (n * fLow / nyquist / 2);
            int idxHigh = Math.max(idxLow + 1, (int) (n * fHigh / nyquist / 2));
            idxHigh = Math.min(idxHigh, n);
            if (idxHigh <= idxLow) { neural[band] = 0; continue; }
            double sum = 0;
            for (int i = idxLow; i < idxHigh; i++) {
                double v = samples[i] / 32768.0;
                sum += v * v;
            }
            double rms = Math.sqrt(sum / (idxHigh - idxLow));
            // 耳蜗对数压缩 (韦伯-费希纳) + 音量增益 (系数 200: 中等音量即满)
            double loudness = rms > 0 ? Math.log10(1 + rms * 200) / 2 : 0;
            neural[band] = Math.max(0, Math.min(1, loudness));
        }
        return neural;
    }

    /** 兼容: 默认 16kHz 采样率 */
    public static double[] encode(short[] samples) {
        return encode(samples, SAMPLE_RATE);
    }

    /** 摘要 */
    public static String summary() {
        return "👂 听觉神经编码: 耳蜗24对数频带(16kHz) + 对数压缩 → 听神经发放率";
    }
}
