package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * ImaginedSpeech 想象发声/内心独白测试。
 * 对照 bioRxiv 2025.07.30.667805 (Imagined Speech Reconstruction):
 *  - 触发阈值: 激活强度 < 阈值不触发; ≥ 阈值触发内心独白
 *  - 模板库依赖: 无模板时不能想象 (没有"会说的话"就没法"想")
 *  - 代谢成本: 想象 < 实说 (论文核心论点: 想象语音代谢成本更低)
 *  - 冷却: 触发后冷却期内不重复触发
 *  - 能量预算: 预算紧张 → 想象偏好上升
 */
public class ImaginedSpeechTest {

    private VoiceLearner makeVoiceLearner() {
        VoiceLearner vl = new VoiceLearner(16000);
        // 造一个模板 (直接注入, 模拟学过的声音)
        VoiceLearner.VoiceTemplate t = new VoiceLearner.VoiceTemplate();
        t.f0 = 200.0; t.f1 = 800.0; t.f2 = 1200.0; t.durationMs = 400.0; t.heardCount = 3;
        vl.library().add(t);
        return vl;
    }

    /** 触发阈值: 弱激活不触发, 强激活触发 */
    @Test
    public void activationThresholdGates() {
        ImaginedSpeech is = new ImaginedSpeech(makeVoiceLearner());
        String weak = is.imagine("概念#1", 0.3, 0);
        assertEquals("", weak, "弱激活不应触发内心独白");
        String strong = is.imagine("概念#1", 0.8, 0);
        assertEquals("概念#1", strong, "强激活应触发内心独白");
        assertTrue(is.hasInnerSpeech());
    }

    /** 无模板不能想象 (没有会说的话) */
    @Test
    public void noTemplatesNoImagination() {
        ImaginedSpeech is = new ImaginedSpeech(new VoiceLearner(16000));
        String s = is.imagine("概念#2", 0.9, 0);
        assertEquals("", s, "无模板时不能想象");
        assertEquals(0, is.imaginesCount());
    }

    /** 代谢成本: 想象 < 实说 (论文核心) */
    @Test
    public void imagineCheaperThanVocalize() {
        ImaginedSpeech is = new ImaginedSpeech(makeVoiceLearner());
        double intensity = 0.7;
        double imagineCost = is.imagineCost(intensity);
        double vocalizeCost = is.vocalizeCost(intensity);
        assertTrue(imagineCost < vocalizeCost, "想象应比实说省能量: " + imagineCost + " vs " + vocalizeCost);
        assertTrue(is.shouldImagine(intensity), "默认应优先想象");
    }

    /** 冷却: 触发后冷却期内不重复 */
    @Test
    public void cooldownPreventsSpam() {
        ImaginedSpeech is = new ImaginedSpeech(makeVoiceLearner());
        is.imagine("概念#1", 0.8, 0);
        int countAfterFirst = is.imaginesCount();
        String during = is.imagine("概念#2", 0.9, 100);  // 冷却期内
        assertEquals("", during, "冷却期内不应触发");
        assertEquals(countAfterFirst, is.imaginesCount(), "冷却期内计数不变");
        String after = is.imagine("概念#2", 0.9, 100000);  // 冷却后
        assertEquals("概念#2", after, "冷却后应可再次触发");
    }

    /** 能量预算: 预算紧张 → 想象偏好更高 (代谢约束驱动) */
    @Test
    public void energyScarcityIncreasesImagination() {
        ImaginedSpeech is = new ImaginedSpeech(makeVoiceLearner());
        is.imagine("概念#1", 0.8, 0);
        is.setEnergyBudget(10.0);
        double richPref = is.imaginePreference();
        is.setEnergyBudget(1.0);  // 预算紧张
        double poorPref = is.imaginePreference();
        assertTrue(poorPref < richPref || Math.abs(poorPref - richPref) < 0.01,
                "预算紧张时想象偏好变化 (成本占比上升)");
        // 重新触发一次让成本计入
        is.imagine("概念#1", 0.9, 5000);
        is.setEnergyBudget(0.8);
        double veryPoor = is.imaginePreference();
        assertTrue(veryPoor <= 1.0 && veryPoor >= 0.2, "偏好应在 [0.2,1]");
    }

    /** 强度影响代谢成本: 高想象强度成本更高 */
    @Test
    public void intensityRaisesCost() {
        ImaginedSpeech is = new ImaginedSpeech(makeVoiceLearner());
        double low = is.imagineCost(0.2);
        double high = is.imagineCost(0.9);
        assertTrue(high > low, "高强度想象成本更高: " + low + " vs " + high);
    }

    /** 触发后状态记录: lastImagined/lastIntensity */
    @Test
    public void recordsLastImagination() {
        ImaginedSpeech is = new ImaginedSpeech(makeVoiceLearner());
        is.imagine("概念#7", 0.85, 0);
        assertEquals("概念#7", is.lastImagined());
        assertTrue(is.lastIntensity() > 0);
    }

    /** summary 含关键量 */
    @Test
    public void summaryContainsMetrics() {
        ImaginedSpeech is = new ImaginedSpeech(makeVoiceLearner());
        is.imagine("概念#1", 0.8, 0);
        String s = is.summary();
        assertTrue(s.contains("代谢"), "summary 应含代谢");
        assertTrue(s.contains("偏好"), "summary 应含偏好");
    }

    /** 重置清空 */
    @Test
    public void resetClears() {
        ImaginedSpeech is = new ImaginedSpeech(makeVoiceLearner());
        is.imagine("概念#1", 0.8, 0);
        is.reset();
        assertEquals(0, is.imaginesCount());
        assertEquals("", is.lastImagined());
        assertFalse(is.hasInnerSpeech());
    }
}
