package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 发声频率保真度验证 (v5.4, 用户报"发声频率相对输入音频过低")。
 *
 * 覆盖真实场景:
 *   1. 含静音的整段录音 → F0 滑窗应锁到语音段 (旧代码只取中段 → 锁到静音/低频)
 *   2. 合成发声输出 → F0 应与学到的一致 (声门主导听感, 共振峰不掩盖基频)
 *   3. 环境低频噪声 (60-70Hz) → 拒绝学习 (F0 范围收紧 80-400)
 */
public class VoiceFidelityTest {

    /** 合成带共振峰人声 (模仿真实语音) */
    private short[] synthVoice(double f0, double f1, double f2, int sr, int ms) {
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

    /** 拼接: 静音 + 语音 + 静音 (模拟"录音里人只说了短句") */
    private short[] withSilence(short[] voice, int sr, int preMs, int postMs) {
        int pre = sr * preMs / 1000, post = sr * postMs / 1000;
        short[] all = new short[pre + voice.length + post];
        for (int i = 0; i < voice.length; i++) all[pre + i] = voice[i];
        return all;
    }

    @Test
    void slidingWindowLocksToVoicedSegment() {
        int sr = 16000;
        // 1.5s 录音: 前 700ms 静音 + 300ms 语音(250Hz) + 后 500ms 静音
        short[] voice = synthVoice(250, 800, 1400, sr, 300);
        short[] rec = withSilence(voice, sr, 700, 500);
        // 旧代码只取中段 (375-625ms) = 纯静音 → 检测失败/低频
        // 新代码滑窗扫描 → 锁到 700-950ms 语音段
        double f0 = VoiceLearner.estimateF0(rec, sr);
        assertTrue(f0 >= 200 && f0 <= 300,
                "含静音录音应锁到语音段 F0≈250, got=" + f0);
    }

    @Test
    void synthesizedOutputKeepsLearnedF0() {
        int sr = 16000;
        VoiceLearner vl = new VoiceLearner(sr);
        // 学到真实语音 (F0=250)
        short[] heard = synthVoice(250, 800, 1400, sr, 400);
        VoiceLearner.VoiceTemplate t = vl.learnFromAudio(heard);
        assertNotNull(t);
        // 合成发声 → 输出 F0 应与学到一致 (声门主导, 共振峰不掩盖)
        short[] out = vl.synthesizeFromTemplate(t);
        double outF0 = VoiceLearner.estimateF0(out, sr);
        assertTrue(Math.abs(outF0 - t.f0) < 40,
                "发声输出 F0 应≈学到 F0 (" + t.f0 + "), got=" + outF0);
    }

    @Test
    void lowFreqEnvNoiseRejected() {
        VoiceLearner vl = new VoiceLearner(16000);
        // 60Hz 环境低频音 (风扇/空调) — 纯低频无语音共振峰结构
        short[] noise = new short[16000 / 4];
        double period = 16000 / 60.0;
        for (int i = 0; i < noise.length; i++) {
            double ph = (i % period) / period;
            noise[i] = (short) (Math.sin(2 * Math.PI * ph) * 8000);
        }
        VoiceLearner.VoiceTemplate t = vl.learnFromAudio(noise);
        assertNull(t, "60Hz 环境噪声不应入模 (F0 范围 80-400)");
        assertEquals(0, vl.templateCount(), "库应为空");
    }

    @Test
    void normalMaleVoiceLearned() {
        VoiceLearner vl = new VoiceLearner(16000);
        // 男声下限 85Hz 应正常学习
        short[] male = synthVoice(110, 500, 900, 16000, 300);
        VoiceLearner.VoiceTemplate t = vl.learnFromAudio(male);
        assertNotNull(t, "男声 110Hz 应正常学习");
        assertTrue(Math.abs(t.f0 - 110) < 40, "男声 F0≈110, got=" + t.f0);
    }
}
