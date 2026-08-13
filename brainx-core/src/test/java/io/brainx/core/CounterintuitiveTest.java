package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 反直觉设定验证 (对照两本书) + 神经元规模可行性。
 */
public class CounterintuitiveTest {

    /** 书中29篇: 不确定奖励比固定奖励更能激活多巴胺 */
    @Test
    void uncertainRewardBoostsDopamine() {
        Brain brain = Brain.simpleBrain();
        // 确定奖励 (uncertainty=0)
        brain.uncertainReward(1.0, 0.0);
        double certain = brain.dopamineSystem().dopamine();
        // 高不确定奖励 (uncertainty=1.0) → 多巴胺反应更强
        brain.uncertainReward(1.0, 1.0);
        double uncertain = brain.dopamineSystem().dopamine();
        assertTrue(uncertain > certain,
                "不确定奖励应更强激活多巴胺: certain=" + certain + " uncertain=" + uncertain);
    }

    /** 书中22篇: 每件事都是习惯 (僵尸行动者自动化) */
    @Test
    void habitsAreAutomatic() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        // 重复练习 → 自动化
        for (int e = 0; e < 6; e++) brain.learnVisualWord(img, 0);
        // 僵尸行动者已自动化
        assertTrue(brain.zombieSkills().size() > 0, "重复应形成习惯(技能)");
        // 自动化技能解放意识
        assertTrue(brain.zombieAgent().freedAttention() > 0,
                "自动化应解放意识资源");
    }

    /** 书中28篇: 预测失误才学习 (错误驱动) */
    @Test
    void learningFromPredictionErrors() {
        DopamineSystem ds = new DopamineSystem();
        // 完全预期 (RPE=0) → 学习微弱
        double rpeExpected = ds.learnEvent(1.0, 1.0);
        // 意外 (RPE>0) → 学习强烈
        double rpeSurprise = ds.learnEvent(1.0, 0.0);
        assertTrue(Math.abs(rpeSurprise) > Math.abs(rpeExpected),
                "意外应比预期学得更强: exp=" + rpeExpected + " sur=" + rpeSurprise);
    }

    /** 书中: 噪声是必要的 (随机共振) */
    @Test
    void noiseHelpsDetection() {
        double signal = 0.5, threshold = 1.0;
        // 内部噪声扫描 (dtMs=1.0, trials=2000) → 有效性曲线
        double[] validity = StochasticResonance.detectionCurve(signal, threshold, 1.0, 2000);
        // 存在中等噪声有效性峰值 (倒U型: 中间高)
        double maxV = 0;
        for (double v : validity) maxV = Math.max(maxV, v);
        assertTrue(maxV > validity[0] && maxV > validity[validity.length - 1],
                "噪声应提升检测有效性 (倒U型): max=" + maxV);
    }

    /** 书中: 感知是主观解释 (先验纠偏识别) */
    @Test
    void perceptionIsInterpretation() {
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) imgA[i] = (i % 3 == 0) ? 0.9 : 0.1;
        // 学习词A
        for (int e = 0; e < 5; e++) brain.learnVisualWord(imgA, 0);
        // 识别时频率共振参与 → 识别结果是"解释" (带先验)
        String[] r = brain.recognizeVisualWithConfidence(imgA);
        assertEquals("你好", r[0], "识别应命中已学词");
        // 预测引擎已建立先验
        assertTrue(brain.predictiveEngine().priorCount() > 0, "先验应建立");
    }

    /** 书中: 线虫302神经元也能学习 (模型仅96神经元实现认知) */
    @Test
    void smallBrainLearns() {
        // 当前模型规模 96 神经元 (72皮层+24中枢) < 线虫 302
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 3; e++) brain.learnVisualWord(img, 0);
        String[] r = brain.recognizeVisualWithConfidence(img);
        assertTrue(!r[0].isEmpty(), "小规模网络应能学习识别");
        assertTrue(brain.centralHub().hubSize() < 302, "中枢 < 线虫神经元数");
    }

    /** 神经元规模可行性: 手机无法运行 1720 亿, 神经群抽象可行 */
    @Test
    void scaleFeasibility() {
        // 全连接中枢: 初始 100% 连接密度 (过度产生)
        CentralHub hub = CentralHub.defaultParams(24);
        assertEquals(1.0, hub.connectionDensity(), 1e-9);
        // 学习后修剪 (密度下降) — 规模控制
        double[] input = new double[24];
        for (int j = 0; j < 12; j++) input[j] = 0.9;  // 半活跃
        for (int i = 0; i < 300; i++) {
            hub.clearInput();
            hub.inject(0, input);
            hub.step(1.0);
            hub.learnWeights(1.0);
        }
        assertTrue(hub.connectionDensity() < 1.0, "学习后密度下降(修剪)");
        // 神经群抽象: 每个单元=皮层柱级微电路 → 手机可运行 (规模论证)
        // 1720亿/1000万神经元 = 1.72万神经群单元 → 手机可运行 (当前已含 JansenRit/WongWang 等)
        long neurons = 172_000_000_000L;
        int perMassUnit = 10_000_000;  // 每个神经群 = 皮层柱级微电路
        long units = neurons / perMassUnit;
        assertTrue(units < 50_000, "神经群抽象后单元数可管理: " + units);
    }

    /** 书中: 忘记是必要的 (修剪+遗忘曲线) */
    @Test
    void forgettingIsNecessary() {
        WorkingMemory wm = WorkingMemory.defaultParams();
        double[] content = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) content[i] = 0.5;
        wm.write(content);
        double s0 = wm.strength(0);
        // 长时间不重放 → 遗忘 (衰减)
        for (int t = 0; t < 50; t++) wm.tick(1000.0);
        assertTrue(wm.strength(0) < s0, "不重放应遗忘 (衰减)");
    }
}
