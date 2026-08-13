package io.brainx.core;

/**
 * 模拟声带 —— 人声合成 (源-滤波器模型 + 音节结构 + 声调)。
 *
 * 资料依据 (语音产生机制):
 *   - 声带振动 (声源): 基频 F0 决定音高, 4 声调 = F0 轮廓
 *   - 声道共鸣 (滤波器): 韵母 = 共振峰 (F1/F2/F3), 复韵母 = 共振峰轨迹
 *   - 辅音 (声母): 气流噪声 (清音) 或 浊音 (m/n/l/r)
 *   - 音节结构: 声母 + 韵母 (汉语拼音)
 *
 * 实现:
 *   - 汉字 → 拼音 (PinyinMap) → 音节 [声母段 + 韵母段]
 *   - 韵母: 20+ 韵母共振峰表 (含复韵母轨迹)
 *   - 声调: 4 声 F0 轮廓 (阴平55/阳平35/上声214/去声51)
 *   - 输出: PCM (16kHz)
 */
public class VocalCordSimulator {
    /** 采样率 */
    public static final int SAMPLE_RATE = 16000;
    /** 每字音节时长 (ms) */
    private static final double SYLLABLE_MS = 320;
    /** 声母段占比 (前段为辅音) */
    private static final double INITIAL_RATIO = 0.25;

    /** 声带基频 (Hz): 可调音高 */
    private double baseF0 = 220;

    public void setBaseF0(double f0) { this.baseF0 = Math.max(60, Math.min(400, f0)); }
    public double baseF0() { return baseF0; }

    /**
     * 文本 → 人声 PCM (汉字→拼音→音节合成)。
     * @param text 中文文本
     * @return PCM 采样 (16-bit, 16kHz)
     */
    public short[] synthesize(String text) {
        if (text == null || text.isEmpty()) text = "嗯";
        java.util.ArrayList<Character> chars = new java.util.ArrayList<>();
        for (char c : text.toCharArray()) {
            if (PinyinMap.lookup(c) != null) chars.add(c);
        }
        if (chars.isEmpty()) {
            // 无字典字 → 用元音近似
            for (char c : text.toCharArray()) if (isVowelChar(c)) chars.add(c);
        }
        if (chars.isEmpty()) chars.add('嗯');

        int total = (int) (chars.size() * SYLLABLE_MS * SAMPLE_RATE / 1000.0);
        short[] pcm = new short[total];
        int offset = 0;
        for (int ci = 0; ci < chars.size(); ci++) {
            char c = chars.get(ci);
            int n = (int) (SYLLABLE_MS * SAMPLE_RATE / 1000.0);
            short[] seg = synthesizeSyllable(c, ci);
            int copy = Math.min(n, seg.length);
            System.arraycopy(seg, 0, pcm, offset, copy);
            offset += copy;
        }
        return pcm;
    }

    /** 音节合成: 声母(辅音/浊音) + 韵母(共振峰轨迹 + 声调) */
    private short[] synthesizeSyllable(char c, int index) {
        int n = (int) (SYLLABLE_MS * SAMPLE_RATE / 1000.0);
        double[] out = new double[n];

        String[] py = PinyinMap.lookup(c);
        if (py == null) {
            // 未知字 → 元音近似
            int vi = vowelCharIndex(c);
            if (vi < 0) vi = 0;
            synthVowelSegment(out, 0, n, VOWELS[vi], 220, 0);
            return toPcm(out);
        }
        String initial = py[0], final_ = py[1];
        int tone = Integer.parseInt(py[2]);

        double[] f1f2f3 = finalFormants(final_);
        int initialLen = (int) (n * INITIAL_RATIO);
        int finalLen = n - initialLen;

        // 声母段: 清音=噪声, 浊音(m/n/l/r/y/w)=低频脉冲
        boolean voiced = "mnlryw".contains(initial);
        if (!initial.isEmpty()) {
            java.util.Random rnd = new java.util.Random(c * 131 + index);
            FormantFilter if1 = new FormantFilter(voiced ? 300 : 1800, 200);
            FormantFilter if2 = new FormantFilter(voiced ? 1200 : 3000, 250);
            for (int i = 0; i < initialLen; i++) {
                double src = voiced
                        ? (i % (SAMPLE_RATE / (baseF0 * 0.9)) < (SAMPLE_RATE / (baseF0 * 0.9)) * 0.4 ? 0.8 : 0)
                        : (rnd.nextDouble() * 2 - 1) * 0.7;
                double sig = src * 0.6 + if1.filter(src) * 0.5 + if2.filter(src) * 0.3;
                double env = Math.min(1.0, i / (initialLen * 0.3)) * Math.min(1.0, (initialLen - i) / (initialLen * 0.4));
                out[i] = sig * env;
            }
        }

        // 韵母段: 共振峰轨迹 + 声调 F0 轮廓
        synthVowelSegment(out, initialLen, n, f1f2f3, baseF0, tone);
        return toPcm(out);
    }

    /** 韵母合成: F0 轮廓(声调) + 共振峰滤波 */
    private void synthVowelSegment(double[] out, int start, int end, double[] f1f2f3, double f0Base, int tone) {
        int n = end - start;
        if (n <= 0) return;
        FormantFilter f1 = new FormantFilter(f1f2f3[0], 80);
        FormantFilter f2 = new FormantFilter(f1f2f3[1], 100);
        FormantFilter f3 = new FormantFilter(f1f2f3[2], 120);
        for (int i = 0; i < n; i++) {
            double t = (double) i / n;
            // 声调 F0 轮廓 (5度制)
            double f0 = f0Base * toneContour(tone, t);
            double period = SAMPLE_RATE / Math.max(50, f0);
            double pulsePhase = (i % period) / period;
            double glottal = pulsePhase < 0.45 ? Math.sin(pulsePhase / 0.45 * Math.PI) : 0;
            double sig = glottal * 1.6
                    + f1.filter(glottal) * 0.5
                    + f2.filter(glottal) * 0.3
                    + f3.filter(glottal) * 0.15;
            double env = Math.min(1.0, i / (n * 0.08)) * Math.min(1.0, (n - i) / (n * 0.15));
            out[start + i] = sig * env;
        }
    }

    /** 声调轮廓 (归一化 F0 倍率, 5度制→Hz) */
    private double toneContour(int tone, double t) {
        switch (tone) {
            case 1: return 1.0;                       // 阴平 55: 平
            case 2: return 1.0 + 0.3 * t;             // 阳平 35: 升
            case 3: return 1.0 - 0.25 * Math.sin(Math.PI * t) + (t < 0.5 ? -0.1 : 0.1); // 上声 214: 降升
            case 4: return 1.2 - 0.5 * t;             // 去声 51: 快降
            default: return 1.0 - 0.1 * t;            // 轻声
        }
    }

    /** 韵母 → 共振峰 [F1,F2,F3] */
    private double[] finalFormants(String final_) {
        switch (final_) {
            case "a": return VOWELS[0];
            case "o": return VOWELS[1];
            case "e": return VOWELS[2];
            case "i": return VOWELS[3];
            case "u": return VOWELS[4];
            case "v": case "ü": return VOWELS[5];
            case "ai": return new double[]{750, 1300, 2800};
            case "ei": return new double[]{550, 1700, 2600};
            case "ao": return new double[]{750, 950, 2700};
            case "ou": return new double[]{500, 880, 2500};
            case "an": return new double[]{800, 1250, 2500};
            case "en": return new double[]{550, 1500, 2400};
            case "ang": return new double[]{800, 1050, 2400};
            case "eng": return new double[]{550, 1300, 2300};
            case "ong": return new double[]{500, 850, 2300};
            case "ia": return new double[]{450, 1900, 2900};
            case "ie": return new double[]{450, 2000, 2800};
            case "iao": return new double[]{500, 1500, 2700};
            case "iu": return new double[]{450, 1500, 2600};
            case "ian": return new double[]{500, 1800, 2700};
            case "in": return new double[]{400, 2200, 2900};
            case "iang": return new double[]{500, 1600, 2500};
            case "ing": return new double[]{400, 2100, 2800};
            case "iong": return new double[]{450, 1400, 2400};
            case "ua": return new double[]{450, 1000, 2800};
            case "uo": return new double[]{450, 900, 2600};
            case "uai": return new double[]{550, 1300, 2700};
            case "ui": return new double[]{450, 1500, 2600};
            case "uan": return new double[]{500, 1400, 2500};
            case "un": return new double[]{450, 1600, 2500};
            case "uang": return new double[]{500, 1200, 2400};
            case "ue": case "üe": return new double[]{450, 1900, 2700};
            case "uan2": case "üan": return new double[]{500, 1700, 2600};
            case "un2": case "ün": return new double[]{450, 2000, 2700};
            case "er": return new double[]{600, 1500, 2500};
            case "ve": return new double[]{450, 1900, 2700};
            default: return new double[]{600, 1300, 2600};
        }
    }

    /** 单韵母共振峰表 */
    private static final double[][] VOWELS = {
        {800, 1150, 2900},  // a
        {500, 880, 2500},   // o
        {550, 1450, 2450},  // e
        {300, 2300, 3000},  // i
        {350, 900, 2400},   // u
        {300, 1900, 2600}   // ü
    };
    private static final String VOWEL_CHARS = "aoeiuüAOEIUÜ";

    private static boolean isVowelChar(char c) { return VOWEL_CHARS.indexOf(c) >= 0; }
    private static int vowelCharIndex(char c) {
        int idx = VOWEL_CHARS.indexOf(c);
        return idx < 0 ? -1 : idx % 6;
    }

    /** 数字谐振滤波器 (二阶 IIR, 模拟声道共振峰) */
    private static class FormantFilter {
        private final double r, theta;
        private double y1 = 0, y2 = 0;

        FormantFilter(double freq, double bandwidth) {
            this.r = Math.exp(-Math.PI * bandwidth / SAMPLE_RATE);
            this.theta = 2 * Math.PI * freq / SAMPLE_RATE;
        }

        double filter(double x) {
            double y = x + 2 * r * Math.cos(theta) * y1 - r * r * y2;
            y2 = y1;
            y1 = y;
            return y;
        }
    }

    private short[] toPcm(double[] out) {
        short[] pcm = new short[out.length];
        for (int i = 0; i < out.length; i++) {
            double v = Math.max(-1, Math.min(1, out[i] * 0.6));
            pcm[i] = (short) (v * 32767);
        }
        return pcm;
    }

    /** 摘要 */
    public static String summary() {
        return "🗣️ 人声声带: 汉字→拼音(声母+韵母) + 4声调F0轮廓 + 共振峰轨迹 → PCM";
    }
}
