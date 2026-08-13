package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 声带学习进化验证: 从听到的声音学习 (非硬编码)。
 */
public class VoiceLearnerTest {

    /** 生成测试音: 特定基频+共振峰的合成音 (模拟听到的人声) */
    private short[] synthTone(double f0, double f1, double f2, int sr, int ms) {
        int n = sr * ms / 1000;
        short[] pcm = new short[n];
        double period = sr / f0;
        double r1 = Math.exp(-Math.PI * 100 / sr), t1 = 2 * Math.PI * f1 / sr;
        double r2 = Math.exp(-Math.PI * 120 / sr), t2 = 2 * Math.PI * f2 / sr;
        double y1a = 0, y2a = 0, y1b = 0, y2b = 0;
        for (int i = 0; i < n; i++) {
            double ph = (i % period) / period;
            double g = ph < 0.45 ? Math.sin(ph / 0.45 * Math.PI) : 0;
            double y1 = g + 2 * r1 * Math.cos(t1) * y1a - r1 * r1 * y2a;
            y2a = y1a; y1a = y1;
            double y2 = g + 2 * r2 * Math.cos(t2) * y1b - r2 * r2 * y2b;
            y2b = y1b; y1b = y2;
            pcm[i] = (short) (Math.max(-1, Math.min(1, (g * 1.5 + y1 * 0.5 + y2 * 0.3) * 0.5)) * 32767);
        }
        return pcm;
    }

    @Test
    void learnsFromAudio() {
        VoiceLearner vl = new VoiceLearner(16000);
        short[] heard = synthTone(200, 800, 1200, 16000, 300);
        VoiceLearner.VoiceTemplate t = vl.learnFromAudio(heard);
        assertNotNull(t, "应学到模板");
        assertEquals(1, vl.templateCount(), "应新增1个模板");
        // 学到的 F0 应接近 200Hz (自相关提取)
        assertTrue(Math.abs(t.f0 - 200) < 60, "F0应≈200, got=" + t.f0);
        // 共振峰应接近
        assertTrue(Math.abs(t.f1 - 800) < 400, "F1应≈800, got=" + t.f1);
    }

    @Test
    void similarSoundsMerge() {
        VoiceLearner vl = new VoiceLearner(16000);
        // 两个相近声音 → 合并增强 (heardCount=2)
        short[] a = synthTone(200, 800, 1200, 16000, 300);
        short[] b = synthTone(205, 820, 1180, 16000, 300);
        vl.learnFromAudio(a);
        vl.learnFromAudio(b);
        assertEquals(1, vl.templateCount(), "相近声音应合并");
        VoiceLearner.VoiceTemplate t = vl.library().get(0);
        assertEquals(2, t.heardCount, "听到2次应增强");
    }

    @Test
    void differentSoundsAdd() {
        VoiceLearner vl = new VoiceLearner(16000);
        short[] low = synthTone(150, 700, 1100, 16000, 300);
        short[] high = synthTone(350, 300, 2300, 16000, 300);
        vl.learnFromAudio(low);
        vl.learnFromAudio(high);
        assertEquals(2, vl.templateCount(), "不同声音应新增");
    }

    @Test
    void mimicProducesSound() {
        VoiceLearner vl = new VoiceLearner(16000);
        short[] heard = synthTone(220, 900, 1300, 16000, 300);
        short[] mimic = vl.mimic(heard);  // 听到→立即模仿
        assertTrue(mimic.length > 1000, "模仿应产生声音, len=" + mimic.length);
        // 模仿音的基频应接近听到的
        double f0 = VoiceLearner.estimateF0(mimic, 16000);
        assertTrue(Math.abs(f0 - 220) < 80, "模仿F0应≈220, got=" + f0);
    }

    @Test
    void speakLearnedNeedsLibrary() {
        VoiceLearner vl = new VoiceLearner(16000);
        // 没学过 → 说不出
        short[] empty = vl.speakLearned("你好");
        assertEquals(0, empty.length, "没学应说不出");
        // 学一个 → 能说
        vl.learnFromAudio(synthTone(200, 800, 1200, 16000, 300));
        short[] speech = vl.speakLearned("你");
        assertTrue(speech.length > 1000, "学后应能发声");
    }

    @Test
    void silenceNotLearned() {
        VoiceLearner vl = new VoiceLearner(16000);
        short[] silence = new short[4000];  // 全 0
        VoiceLearner.VoiceTemplate t = vl.learnFromAudio(silence);
        assertNull(t, "静音不应学习");
        assertEquals(0, vl.templateCount());
    }

    @Test
    void maturationSummary() {
        VoiceLearner vl = new VoiceLearner(16000);
        assertTrue(vl.summary().contains("咿呀"), "初始应咿呀期");
        vl.learnFromAudio(synthTone(200, 800, 1200, 16000, 300));
        assertTrue(vl.summary().contains("模板"), "学后应显示模板数");
    }

    /** 对应输入频率发声: 输入150Hz → 输出≈150Hz; 输入300Hz → 输出≈300Hz */
    @Test
    void mimicMatchesInputFrequency() {
        VoiceLearner vl = new VoiceLearner(16000);
        // 低音 150Hz
        short[] low = synthTone(150, 700, 1100, 16000, 300);
        short[] mimicLow = vl.mimic(low);
        double f0Low = VoiceLearner.estimateF0(mimicLow, 16000);
        assertTrue(Math.abs(f0Low - 150) < 70, "低音模仿应≈150Hz, got=" + f0Low);
        // 高音 300Hz
        short[] high = synthTone(300, 900, 2300, 16000, 300);
        short[] mimicHigh = vl.mimic(high);
        double f0High = VoiceLearner.estimateF0(mimicHigh, 16000);
        assertTrue(Math.abs(f0High - 300) < 110, "高音模仿应≈300Hz, got=" + f0High);
        // 高低音应有差异 (频率对应)
        assertTrue(f0High > f0Low + 40, "高频模仿应高于低频: low=" + f0Low + " high=" + f0High);
    }

    /** 模仿时长匹配输入 (听到300ms → 模仿~300ms) */
    @Test
    void mimicMatchesDuration() {
        VoiceLearner vl = new VoiceLearner(16000);
        short[] heard = synthTone(220, 800, 1200, 16000, 300);
        short[] mimic = vl.mimic(heard);
        double inMs = heard.length * 1000.0 / 16000;
        double outMs = mimic.length * 1000.0 / 16000;
        assertTrue(Math.abs(outMs - inMs) < 100, "时长应匹配: in=" + inMs + " out=" + outMs);
    }

    /** 模仿→自主演进: 模板积累后可用学到的声音自主说 */
    @Test
    void mimicToAutonomousEvolution() {
        VoiceLearner vl = new VoiceLearner(16000);
        // 模仿期: 只能回响学过的
        assertEquals(0, vl.templateCount(), "初始无模板");
        // 学 3 个不同音
        vl.learnFromAudio(synthTone(150, 700, 1100, 16000, 250));
        vl.learnFromAudio(synthTone(220, 800, 1200, 16000, 250));
        vl.learnFromAudio(synthTone(300, 900, 2300, 16000, 250));
        assertEquals(3, vl.templateCount(), "学到3个模板");
        // 自主期: 用学到的声音组合说话
        short[] speech = vl.speakLearned("你好");
        assertTrue(speech.length > 2000, "自主发声应产生声音, len=" + speech.length);
        // 说话的音高来自学到的模板 (150-300Hz 范围)
        double f0 = VoiceLearner.estimateF0(speech, 16000);
        assertTrue(f0 > 100 && f0 < 400, "自主发声音高应来自所学模板, f0=" + f0);
    }
}
