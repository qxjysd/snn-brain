package io.brainx.core;

import java.util.ArrayList;
import java.util.List;

/**
 * 声带学习进化 —— 从听到的声音学习发声 (听觉-发声回路)。
 *
 * 神经科学依据 (婴儿语言习得 / 运动学习):
 *   - 婴儿先听大人说话 → 听觉皮层提取声学参数
 *   - 声带运动模仿: 听-说回路 (布罗卡区-运动皮层-声带)
 *   - 自我监听: 自己发声 → 听到 → 与目标对比 → 调整 (误差驱动)
 *   - 咿呀学语 → 音素库积累 → 组合成词 (自主发声)
 *
 * 实现 (非硬编码, 从真实音频学习):
 *   - learnFromAudio(pcm): 自相关 F0 + 频谱峰共振峰 + 时长 → 声带模板
 *   - 模板库: 相似合并增强, 不同新增 (进化积累)
 *   - mimic(pcm): 听到 → 提取 → 立即用学到参数模仿 (鹦鹉学舌)
 *   - speakLearned(text): 用学到的模板库组合发声 (自主)
 *   - 进化指标: 模板数 / 模仿误差
 */
public class VoiceLearner {
    /** 声带模板: 从实际声音学到的发声参数 */
    public static class VoiceTemplate {
        public double f0;        // 基频 (Hz) — 声带振动
        public double f1, f2;    // 共振峰 (Hz) — 声道形状
        public double durationMs; // 时长
        public int heardCount;   // 听到次数 (增强)

        public double similarity(VoiceTemplate o) {
            double df = Math.abs(f0 - o.f0) / Math.max(40, (f0 + o.f0) / 2);
            double d1 = Math.abs(f1 - o.f1) / Math.max(100, (f1 + o.f1) / 2);
            double d2 = Math.abs(f2 - o.f2) / Math.max(200, (f2 + o.f2) / 2);
            return Math.exp(-(df * df + d1 * d1 + d2 * d2) * 8);
        }

        public String describe() {
            return String.format("F0=%.0fHz F1=%.0f F2=%.0f %.0fms 听到%d次",
                    f0, f1, f2, durationMs, heardCount);
        }
    }

    /** 声带模板库 (学到的最小语音单元) */
    private final List<VoiceTemplate> library = new ArrayList<>();
    /** 合并阈值: 相似度 > 0.7 视为同一模板 */
    private static final double MERGE_THRESHOLD = 0.7;
    /** 学习率: 新声音影响程度 */
    private double learningRate = 0.5;
    /** 采样率 */
    private int sampleRate = 16000;

    public VoiceLearner() {}
    public VoiceLearner(int sampleRate) { this.sampleRate = sampleRate; }

    /** 模板库 */
    public List<VoiceTemplate> library() { return library; }
    public int templateCount() { return library.size(); }
    public double learningRate() { return learningRate; }

    /**
     * 学习: 从听到的音频提取声学参数 → 声带模板 (进化积累)。
     * @param pcm 听到的声音
     * @return 学到的模板 (合并或新增)
     */
    public VoiceTemplate learnFromAudio(short[] pcm) {
        if (pcm == null || pcm.length < 100) return null;
        VoiceTemplate t = new VoiceTemplate();
        t.f0 = estimateF0(pcm, sampleRate);
        double[] fs = estimateFormants(pcm, sampleRate);
        t.f1 = fs[0];
        t.f2 = fs[1];
        t.durationMs = pcm.length * 1000.0 / sampleRate;
        t.heardCount = 1;

        // 静音/无特征 → 不学 (F0 范围收紧: 排除 80Hz 以下环境低频噪声)
        if (t.f0 < 80 || t.f0 > 400) return null;

        // 相似合并增强 (进化: 反复听到 → 模板更稳)
        VoiceTemplate best = null;
        double bestSim = 0;
        for (VoiceTemplate v : library) {
            double sim = v.similarity(t);
            if (sim > bestSim) { bestSim = sim; best = v; }
        }
        if (best != null && bestSim > MERGE_THRESHOLD) {
            // 合并: 学习率加权平均 (听到越多越稳定)
            best.f0 = best.f0 * (1 - learningRate) + t.f0 * learningRate;
            best.f1 = best.f1 * (1 - learningRate) + t.f1 * learningRate;
            best.f2 = best.f2 * (1 - learningRate) + t.f2 * learningRate;
            best.durationMs = best.durationMs * (1 - learningRate) + t.durationMs * learningRate;
            best.heardCount++;
            return best;
        }
        // 新增模板
        library.add(t);
        return t;
    }

    /**
     * 鹦鹉学舌: 听到 → 学习 → 立即用学到的参数模仿。
     */
    public short[] mimic(short[] input) {
        VoiceTemplate t = learnFromAudio(input);
        if (t == null) return new short[0];
        return synthesizeFromTemplate(t);
    }

    /**
     * 用学到模板合成 (声带按学到的参数振动)。
     */
    public short[] synthesizeFromTemplate(VoiceTemplate t) {
        int n = (int) (t.durationMs * sampleRate / 1000.0);
        if (n <= 0) n = sampleRate / 5;
        double[] out = new double[n];
        double period = sampleRate / Math.max(50, t.f0);
        // 三共振峰滤波 (声道 = 学到的形状)。加强阻尼 (BW 250+): 共振峰振荡快速衰减,
        // 不掩盖声门基频 —— 否则听感锁到共振峰频率而非 F0 (坑14)。
        FormantFilter f1f = new FormantFilter(t.f1, 260, sampleRate);
        FormantFilter f2f = new FormantFilter(t.f2, 300, sampleRate);
        FormantFilter f3f = new FormantFilter(t.f2 * 1.6, 340, sampleRate);
        for (int i = 0; i < n; i++) {
            double pulsePhase = (i % period) / period;
            double glottal = pulsePhase < 0.45 ? Math.sin(pulsePhase / 0.45 * Math.PI) : 0;
            // 声门主导 (权重 2.2) + 共振峰着色 (弱权重) → 听感基频 = t.f0
            double sig = glottal * 2.2
                    + f1f.filter(glottal) * 0.35
                    + f2f.filter(glottal) * 0.2
                    + f3f.filter(glottal) * 0.1;
            double env = Math.min(1.0, i / (n * 0.08)) * Math.min(1.0, (n - i) / (n * 0.15));
            out[i] = sig * env;
        }
        short[] pcm = new short[n];
        for (int i = 0; i < n; i++) {
            double v = Math.max(-1, Math.min(1, out[i] * 0.6));
            pcm[i] = (short) (v * 32767);
        }
        return pcm;
    }

    /**
     * 自主发声: 用学到的模板库组合 (学得多 → 组合越丰富)。
     * 每字符轮转使用库中模板; 库空 → 返回空 (还不会说话)。
     */
    public short[] speakLearned(String text) {
        if (library.isEmpty()) return new short[0];
        if (text == null || text.isEmpty()) text = "啊";
        int chars = text.length();
        short[] all = new short[0];
        for (int i = 0; i < chars; i++) {
            // 轮转选模板 (学到的声音)
            VoiceTemplate t = library.get(i % library.size());
            short[] seg = synthesizeFromTemplate(t);
            short[] merged = new short[all.length + seg.length];
            System.arraycopy(all, 0, merged, 0, all.length);
            System.arraycopy(seg, 0, merged, all.length, seg.length);
            all = merged;
        }
        return all;
    }

    /** 进化摘要 */
    public String summary() {
        return String.format("🗣️ 声带学习: %d个音模板 | 学习率%.0f%% | %s",
                library.size(), learningRate * 100,
                library.isEmpty() ? "咿呀期(还没学会)" : "可模仿发声");
    }

    // ============ 声学参数提取 (从真实音频学习) ============

    /**
     * 自相关 F0 估计 (语音标准方法, 多窗择优)。
     *
     * 整段录音 (常含静音/清音/环境音) 只取中段会锁到非语音段 → 发声频率偏离输入。
     * 修复: 滑窗扫描 (每窗 250ms 步进 125ms), 每窗低通+自相关取首个显著峰,
     *       取"自相关峰值质量最高"的窗 (最周期段 = 语音元音段), 静音/噪声窗被质量门槛滤掉。
     */
    public static double estimateF0(short[] pcm, int sr) {
        if (pcm == null || pcm.length < 200) return 0;
        int win = sr / 4;                      // 每窗 250ms
        int step = sr / 8;                     // 步进 125ms
        double bestF0 = 0, bestQ = 0.35;      // 质量门槛: 自相关峰 < 0.35 视为无周期 (静音/噪声); 真实语音峰 0.6+, 含共振峰合成音 0.35+
        for (int start = 0; start + win <= pcm.length && start < sr * 2; start += step) {
            double[] r = windowF0(pcm, start, win, sr);
            if (r[0] > bestQ && r[1] >= 80 && r[1] <= 400) {   // F0 范围: 排除环境低频/过高
                bestQ = r[0];
                bestF0 = r[1];
            }
        }
        return bestF0;
    }

    /** 单窗 F0 估计: [自相关峰值质量, F0 Hz] (低通 → 自相关 → 首个显著局部峰) */
    private static double[] windowF0(short[] pcm, int start, int win, int sr) {
        int e = Math.min(start + win, pcm.length);
        // 一阶低通 (fc≈900Hz): 去掉高频共振峰干扰, 保留基频周期
        double rc = 1.0 / (2 * Math.PI * 900);
        double dt = 1.0 / sr;
        double alpha = dt / (rc + dt);
        double[] lp = new double[e - start];
        double prev = 0;
        for (int i = 0; i < e - start; i++) {
            prev = prev + alpha * (pcm[start + i] - prev);
            lp[i] = prev;
        }
        int minLag = sr / 400;   // 400Hz 上限
        int maxLag = sr / 50;    // 50Hz 下限
        int s = 0;
        int en = e - start;
        if (en - s < minLag * 2) return new double[]{0, 0};
        int maxL = Math.min(maxLag, en - s);
        double[] norms = new double[maxL + 1];
        for (int lag = minLag; lag <= maxL; lag++) {
            double sum = 0, energy = 0;
            for (int i = s; i < en - lag; i++) {
                sum += lp[i] * lp[i + lag];
                energy += lp[i] * lp[i];
            }
            norms[lag] = energy > 0 ? sum / energy : 0;
        }
        // 找第一个显著局部峰 (最短真实周期 = 基频; 包络周期更长被跳过)
        for (int lag = minLag + 1; lag < maxL; lag++) {
            if (norms[lag] > norms[lag-1] && norms[lag] >= norms[lag+1]
                    && norms[lag] > 0.3) {
                return new double[]{norms[lag], (double) sr / lag};
            }
        }
        return new double[]{0, 0};
    }

    /** FFT 共振峰提取: 频谱两个最大峰 → [F1, F2] */
    public static double[] estimateFormants(short[] pcm, int sr) {
        double[] out = new double[]{500, 1500};
        if (pcm == null || pcm.length < 256) return out;
        // 取中段 256 采样 + 汉宁窗 → FFT
        int N = 256;
        int mid = pcm.length / 2;
        int start = Math.max(0, mid - N / 2);
        double[] re = new double[N], im = new double[N];
        for (int i = 0; i < N && start + i < pcm.length; i++) {
            double w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));
            re[i] = pcm[start + i] * w;
        }
        fft(re, im);
        // 找频谱峰 (100Hz - 4000Hz, 排除 DC)
        double[] mag = new double[N / 2];
        for (int k = 1; k < N / 2; k++) {
            mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
        }
        // 局部峰
        double f1 = 0, f2 = 0, m1 = 0, m2 = 0;
        for (int k = 2; k < N / 2 - 1; k++) {
            double freq = (double) k * sr / N;
            if (freq < 100 || freq > 4000) continue;
            if (mag[k] > mag[k-1] && mag[k] > mag[k+1] && mag[k] > 0) {
                // 峰值 (抛物线插值)
                double denom = mag[k-1] - 2*mag[k] + mag[k+1];
                double offset = denom != 0 ? 0.5 * (mag[k-1] - mag[k+1]) / denom : 0;
                double peakFreq = (k + offset) * sr / N;
                if (mag[k] > m1) {
                    m2 = m1; f2 = f1;
                    m1 = mag[k]; f1 = peakFreq;
                } else if (mag[k] > m2) {
                    m2 = mag[k]; f2 = peakFreq;
                }
            }
        }
        if (f1 > 0) out[0] = Math.max(150, Math.min(1500, f1));
        if (f2 > 0) out[1] = Math.max(400, Math.min(3800, f2));
        return out;
    }

    /** 基-2 FFT (迭代实现) */
    private static void fft(double[] re, double[] im) {
        int n = re.length;
        // 位反转
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double tr = re[i]; re[i] = re[j]; re[j] = tr;
                double ti = im[i]; im[i] = im[j]; im[j] = ti;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2 * Math.PI / len;
            double wRe = Math.cos(ang), wIm = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double curRe = 1, curIm = 0;
                for (int k = 0; k < len / 2; k++) {
                    int a = i + k, b = i + k + len / 2;
                    double tRe = curRe * re[b] - curIm * im[b];
                    double tIm = curRe * im[b] + curIm * re[b];
                    re[b] = re[a] - tRe;
                    im[b] = im[a] - tIm;
                    re[a] += tRe;
                    im[a] += tIm;
                    double nRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = nRe;
                }
            }
        }
    }

    /** 数字谐振滤波器 (二阶 IIR) */
    private static class FormantFilter {
        private final double r, theta;
        private double y1 = 0, y2 = 0;

        FormantFilter(double freq, double bandwidth, int sr) {
            this.r = Math.exp(-Math.PI * bandwidth / sr);
            this.theta = 2 * Math.PI * freq / sr;
        }

        double filter(double x) {
            double y = x + 2 * r * Math.cos(theta) * y1 - r * r * y2;
            y2 = y1;
            y1 = y;
            return y;
        }
    }
}
