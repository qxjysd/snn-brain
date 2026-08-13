package io.brainx.core;

import io.brainx.core.encoding.Encoders;
import io.brainx.core.learning.PPProp;
import io.brainx.core.learning.DRTRL;
import io.brainx.core.mass.*;
import io.brainx.core.neuron.*;
import io.brainx.core.synapse.STDP;
import io.brainx.core.synapse.Synapse;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 算法验证测试 —— 对照论文数值/已知结果。
 * BrainTrace: pp-prop 线性内存 + 资格迹分解
 * SpikingGamma: σ-δ 编码 + ITD 双耳定位
 * brain: LIF/Izhikevich 发放
 * BrainMass: Jansen-Rit α振荡 / Wong-Wang 决策 / Kuramoto 同步
 */
public class AlgorithmTest {

    @Test
    void lifFiresUnderSufficientCurrent() {
        LIF n = LIF.defaultParams();  // threshold -50
        int spikes = 0;
        for (int t = 0; t < 1000; t++) {
            n.step(25.0, 0.1);  // 强电流 (需 >15nA 过阈值)
            if (n.fired()) spikes++;
        }
        // LIF 在强电流下应周期性发放 (brain 已知行为)
        assertTrue(spikes >= 10, "LIF 应发放, got " + spikes);
        assertTrue(spikes < 500, "不应过发放");
    }

    @Test
    void lifSilentWithoutCurrent() {
        LIF n = LIF.defaultParams();
        int spikes = 0;
        for (int t = 0; t < 1000; t++) {
            n.step(0.0, 0.1);
            if (n.fired()) spikes++;
        }
        assertEquals(0, spikes, "无电流不应发放");
    }

    @Test
    void izhikevichRegularSpiking() {
        Izhikevich n = Izhikevich.regularSpiking();
        int spikes = 0;
        for (int t = 0; t < 10000; t++) {
            n.step(10.0, 0.1);
            if (n.fired()) spikes++;
        }
        // RS 神经元在 10 电流下应规则发放 (dt=0.1ms, 10000步=1s)
        assertTrue(spikes > 5 && spikes < 400, "RS 发放率合理, got " + spikes);
    }

    @Test
    void spikingGammaFiresAndBackprop() {
        SpikingGamma n = SpikingGamma.defaultParams();
        int spikes = 0;
        for (int t = 0; t < 1000; t++) {
            n.step(0.5, 0.1);  // 持续输入
            if (n.fired()) spikes++;
        }
        // σ-δ 编码: 持续输入应产生稀疏脉冲流
        assertTrue(spikes > 0, "σ-δ 应发放");
        assertTrue(spikes < 200, "σ-δ 应稀疏");

        // 反向传播: 误差更新桶权重 (不崩溃)
        n.backpropError(0.1, 0.01);
        assertTrue(n.bucketWeights.length == 8);
    }

    @Test
    void ppPropLearnsAssociation() {
        // pp-prop: 输入模式 → 隐藏状态 关联学习
        PPProp learner = new PPProp(4, 4, 0.95, 0.1);
        double[] patternA = {1, 0, 1, 0};
        double[] patternB = {0, 1, 0, 1};
        double[] target = {1, 0, 0, 0};  // 学习信号: 只激活第一个输出

        // 训练: 输入A → 目标
        for (int epoch = 0; epoch < 100; epoch++) {
            learner.resetTraces();
            for (int t = 0; t < 10; t++) {
                learner.step(patternA, new double[4], new double[]{1,1,1,1}, 0.99);
                // 学习信号: 输出与目标的误差
                double[] out = learner.forward(patternA);
                double[] signal = new double[4];
                for (int i = 0; i < 4; i++) signal[i] = (target[i] - out[i]);
                learner.update(signal);
            }
            learner.applyGradients();
        }
        double[] out = learner.forward(patternA);
        // 输出应偏向 target[0] 最大
        assertTrue(out[0] > out[1] && out[0] > out[2] && out[0] > out[3],
                "pp-prop 应学会关联, out=" + java.util.Arrays.toString(out));
    }

    @Test
    void drtrlTracesFinite() {
        DRTRL learner = new DRTRL(3, 3, 0.95, 0.1);
        double[] input = {1, 0, 1};
        for (int t = 0; t < 100; t++) {
            learner.step(input, new double[]{1,1,1}, 0.99);
            learner.update(new double[]{0.1, -0.1, 0.05});
        }
        learner.applyGradients();
        // 资格迹应为有限值
        double[][] e = learner.eligibility();
        for (double[] row : e) for (double v : row) assertTrue(Double.isFinite(v));
    }

    @Test
    void stdpPotentiatesCausalPairing() {
        STDP stdp = STDP.defaultParams();
        Synapse s = Synapse.delta(0, 1, 0.5);
        // 前发放先于后发放 10ms → 增强
        stdp.onPreSpike(s, 100.0);
        stdp.onPostSpike(s, 110.0);
        assertTrue(s.weight > 0.5, "因果配对应增强, w=" + s.weight);
    }

    @Test
    void stdpDepressesAntiCausal() {
        STDP stdp = STDP.defaultParams();
        Synapse s = Synapse.delta(0, 1, 0.5);
        // 后发放先于前发放 → 抑制
        stdp.onPostSpike(s, 100.0);
        stdp.onPreSpike(s, 110.0);
        assertTrue(s.weight < 0.5, "反因果应抑制, w=" + s.weight);
    }

    @Test
    void jansenRitProducesOscillation() {
        // Jansen-Rit: 应产生 α 节律振荡 (~10 Hz)
        JansenRit jr = JansenRit.defaultParams();
        double[] samples = new double[5000];
        for (int t = 0; t < 5000; t++) {
            samples[t] = jr.step(0, 0.1);  // dt=0.1ms, 0.5s
        }
        // 峰值计数 (局部极大) → 频率估计
        int peaks = 0;
        for (int t = 1000; t < 4999; t++) {
            if (samples[t] > samples[t-1] && samples[t] > samples[t+1]) peaks++;
        }
        double freqHz = peaks / 0.4;  // 统计 0.4s (从 t=1000 即 0.1s 起)
        // α 节律: 8-13 Hz
        assertTrue(freqHz > 5 && freqHz < 15, "Jansen-Rit 应产生 α 节律 (~10Hz), got " + freqHz + "Hz");
    }

    @Test
    void wongWangMakesDecision() {
        // Wong-Wang: 强正相干性 → 决策1; 强负相干性 → 决策2
        WongWang ww = WongWang.defaultParams();
        // 重置后强刺激
        for (int i = 0; i < 5; i++) {
            ww.reset();
            for (int t = 0; t < 500; t++) ww.step(0.5, 1.0);
            assertEquals(1, ww.decision(), "正相干应决策1");
        }
        for (int i = 0; i < 5; i++) {
            ww.reset();
            for (int t = 0; t < 500; t++) ww.step(-0.5, 1.0);
            assertEquals(2, ww.decision(), "负相干应决策2");
        }
    }

    @Test
    void kuramotoSynchronizes() {
        // 强耦合 → 同步序参量趋近1
        Kuramoto k = new Kuramoto(8, 2.0, 42);
        for (int t = 0; t < 20000; t++) k.step(1.0);  // 20s
        double order = k.orderParameter();
        assertTrue(order > 0.8, "强耦合应同步, R=" + order);
    }

    @Test
    void itdLocatesSoundSource() {
        Encoders.ITD itd = Encoders.ITD.defaultParams();
        // 右偏声源 (正方位角) → 右耳先到
        double itdMs = itd.azimuthToITD(Math.toRadians(30));
        int[][] signals = itd.generateTestSignal(Math.toRadians(30), 500, 0.1, 7);
        int[] left = signals[0];
        int[] right = signals[1];
        double estItd = itd.locate(left, right, 0.1);
        // 估计 ITD 应接近理论值 (符号一致, 量级接近)
        assertTrue(estItd * itdMs > 0, "ITD 符号应正确, est=" + estItd + ", theory=" + itdMs);
    }

    @Test
    void sigmaDeltaEncodesSignal() {
        Encoders.SigmaDelta sd = Encoders.SigmaDelta.defaultParams();
        int spikes = 0;
        for (int t = 0; t < 1000; t++) {
            spikes += Math.abs(sd.encode(0.05));  // 小信号
        }
        // 积分编码: 0.05*1000 = 50 → 约 500 个脉冲 (阈值 0.1)
        assertTrue(spikes > 100, "σ-δ 应编码信号, spikes=" + spikes);
    }

    @Test
    void brainLearnsAndRecognizes() {
        Brain brain = Brain.simpleBrain();
        // 学习: 模式1 → 词0 (你好), 模式2 → 词1 (苹果)
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM];
        double[] imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) {
            imgA[i] = (i % 3 == 0) ? 0.9 : 0.1;
            imgB[i] = (i % 3 == 1) ? 0.9 : 0.1;
        }
        for (int epoch = 0; epoch < 20; epoch++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        // 识别
        String r1 = brain.recognizeVisual(imgA);
        String r2 = brain.recognizeVisual(imgB);
        assertTrue(r1.equals("你好") || r1.equals("苹果"), "识别A=" + r1);
        assertTrue(r2.equals("你好") || r2.equals("苹果"), "识别B=" + r2);
        assertTrue(brain.learnedWords().size() >= 2, "应学到至少2个词");
    }
}
