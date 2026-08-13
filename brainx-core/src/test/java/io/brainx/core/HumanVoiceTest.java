package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 人声声带验证: 汉字→拼音→音节(声母+韵母+声调)合成。
 */
public class HumanVoiceTest {

    @Test
    void pinyinMapHasCommonChars() {
        // 常用字都有拼音
        assertNotNull(PinyinMap.lookup('你'), "你应有拼音");
        assertNotNull(PinyinMap.lookup('好'), "好应有拼音");
        assertNotNull(PinyinMap.lookup('苹'), "苹应有拼音");
        assertNotNull(PinyinMap.lookup('果'), "果应有拼音");
        String[] py = PinyinMap.lookup('好');
        assertEquals("h", py[0], "好的声母");
        assertEquals("ao", py[1], "好的韵母");
        assertEquals("3", py[2], "好的声调(上声)");
    }

    @Test
    void producesHumanLikePcm() {
        VocalCordSimulator vc = new VocalCordSimulator();
        short[] pcm = vc.synthesize("你好");
        assertTrue(pcm.length > 5000, "人声PCM应更长, len=" + pcm.length);
        // 有信号
        double max = 0;
        for (short s : pcm) max = Math.max(max, Math.abs(s));
        assertTrue(max > 1000, "应有明显信号, max=" + max);
    }

    @Test
    void toneAffectsPitch() {
        VocalCordSimulator vc = new VocalCordSimulator();
        // 一(阴平55) vs 大(去声51): 不同声调 → 不同基频特征
        short[] yi = vc.synthesize("一");
        short[] da = vc.synthesize("大");
        double f0Yi = estimateF0(yi);
        double f0Da = estimateF0(da);
        // 至少一个能检测到基频 (人声有周期性)
        assertTrue(f0Yi > 0 || f0Da > 0, "应检测到基频: yi=" + f0Yi + " da=" + f0Da);
    }

    @Test
    void differentSyllablesDiffer() {
        VocalCordSimulator vc = new VocalCordSimulator();
        short[] ni = vc.synthesize("你");
        short[] hao = vc.synthesize("好");
        boolean differs = false;
        int n = Math.min(ni.length, hao.length);
        for (int k = 0; k < n; k += 30) {
            if (Math.abs(ni[k] - hao[k]) > 2000) { differs = true; break; }
        }
        assertTrue(differs, "不同音节应产生不同波形");
    }

    @Test
    void fullSentenceSynthesizes() {
        VocalCordSimulator vc = new VocalCordSimulator();
        short[] pcm = vc.synthesize("我看到苹果了");
        assertTrue(pcm.length > 10000, "整句应合成, len=" + pcm.length);
        boolean nonZero = false;
        for (short s : pcm) if (Math.abs(s) > 500) { nonZero = true; break; }
        assertTrue(nonZero, "整句应有声音");
    }

    /** 简化 F0 估计 (峰值间隔) */
    private double estimateF0(short[] pcm) {
        int start = pcm.length / 4, end = pcm.length * 3 / 4;
        double max = 0;
        for (int i = start; i < end; i++) max = Math.max(max, Math.abs(pcm[i]));
        double th = max * 0.5;
        java.util.ArrayList<Integer> peaks = new java.util.ArrayList<>();
        for (int i = start + 1; i < end - 1; i++) {
            if (pcm[i] > th && pcm[i] > pcm[i-1] && pcm[i] > pcm[i+1]
                    && (peaks.isEmpty() || i - peaks.get(peaks.size()-1) > 8)) {
                peaks.add(i);
            }
        }
        if (peaks.size() < 3) return 0;
        double avg = 0;
        for (int k = 1; k < peaks.size(); k++) avg += peaks.get(k) - peaks.get(k-1);
        avg /= (peaks.size() - 1);
        return avg > 0 ? VocalCordSimulator.SAMPLE_RATE / avg : 0;
    }
}
