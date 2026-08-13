package io.brainx.core.network;

import io.brainx.core.Neuron;
import io.brainx.core.SpikeEvent;
import io.brainx.core.synapse.Synapse;
import io.brainx.core.synapse.STDP;

import java.util.*;

/**
 * 脉冲神经网络 (对应 brain.state 的 Network + Projection)。
 * 支持: 多层神经元 + 稀疏突触连接 + 事件驱动更新 + STDP。
 * 每个神经元的输入电流 = Σ 到达的突触电流（脉冲经延迟/时间常数处理）。
 */
public class SNN {
    /** 神经元列表（按层组织，全局索引连续） */
    private final List<Neuron> neurons = new ArrayList<>();
    private final Map<Integer, List<Synapse>> incoming = new HashMap<>();  // post → synapses
    private final Map<Integer, List<Synapse>> outgoing = new HashMap<>();  // pre → synapses
    private final List<SpikeEvent> spikeEvents = new ArrayList<>();
    private final List<double[]> layerStates = new ArrayList<>();  // 每层发放率历史

    private STDP stdp;
    private int timeStep = 0;
    private double timeMs = 0;
    private double dtMs = 0.1;
    private boolean stdpEnabled = false;

    /** 突触电流缓冲: postNeuron → 当前输入电流 */
    private final Map<Integer, Double> inputCurrent = new HashMap<>();
    /** 指数突触状态: synapse → 当前电流值 */
    private final Map<Synapse, Double> synCurrent = new HashMap<>();

    public SNN(double dtMs) {
        this.dtMs = dtMs;
    }

    /** 添加一层神经元，返回起始索引 */
    public int addLayer(Neuron[] layer) {
        int start = neurons.size();
        Collections.addAll(neurons, layer);
        for (int i = 0; i < layer.length; i++) {
            incoming.put(start + i, new ArrayList<>());
            outgoing.put(start + i, new ArrayList<>());
        }
        return start;
    }

    /** 全连接两个层（含抑制比例） */
    public void connectAll(int preStart, int preCount, int postStart, int postCount,
                           double weight, double prob, long seed, double tauSynMs) {
        Random rnd = new Random(seed);
        for (int i = 0; i < preCount; i++) {
            for (int j = 0; j < postCount; j++) {
                if (rnd.nextDouble() < prob) {
                    connect(preStart + i, postStart + j, weight, tauSynMs);
                }
            }
        }
    }

    public void connect(int pre, int post, double weight, double tauSynMs) {
        Synapse s = weight >= 0
            ? Synapse.exp(pre, post, weight, tauSynMs)
            : Synapse.coba(pre, post, weight, tauSynMs, true);
        connect(s);
    }

    public void connect(Synapse s) {
        incoming.get(s.postNeuron).add(s);
        outgoing.get(s.preNeuron).add(s);
    }

    public void enableSTDP(STDP stdp) { this.stdp = stdp; this.stdpEnabled = true; }

    /** 前向一步: 所有神经元 + 突触传递 */
    public void step() {
        timeMs += dtMs;
        timeStep++;
        spikeEvents.clear();

        // 1. 清空输入
        for (Integer k : inputCurrent.keySet()) inputCurrent.put(k, 0.0);

        // 2. 突触电流衰减 (指数突触)
        for (Map.Entry<Synapse, Double> e : synCurrent.entrySet()) {
            Synapse s = e.getKey();
            if (s.type == Synapse.Type.EXP || s.type == Synapse.Type.COBA_AMPA || s.type == Synapse.Type.COBA_GABA) {
                double decay = Math.exp(-dtMs / Math.max(s.tauSynMs, 1e-6));
                e.setValue(e.getValue() * decay);
                inputCurrent.merge(s.postNeuron, e.getValue(), Double::sum);
            }
        }

        // 3. 处理本步新脉冲
        //    前一步的脉冲通过延迟传导 —— 简化: 同步传递
        //    使用 lastStepSpikes 记录上一时刻脉冲
        if (!lastStepSpikes.isEmpty()) {
            for (int pre : lastStepSpikes) {
                List<Synapse> syns = outgoing.get(pre);
                if (syns == null) continue;
                for (Synapse s : syns) {
                    if (s.type == Synapse.Type.DELTA) {
                        inputCurrent.merge(s.postNeuron, s.weight, Double::sum);
                    } else {
                        double val = synCurrent.getOrDefault(s, 0.0) + s.weight;
                        synCurrent.put(s, val);
                        inputCurrent.merge(s.postNeuron, val, Double::sum);
                    }
                    if (stdpEnabled && stdp != null) stdp.onPreSpike(s, timeMs);
                }
            }
        }
        lastStepSpikes.clear();

        // 4. 更新所有神经元
        for (int i = 0; i < neurons.size(); i++) {
            double I = inputCurrent.getOrDefault(i, 0.0);
            Neuron n = neurons.get(i);
            n.step(I, dtMs);
            if (n.fired()) {
                spikeEvents.add(new SpikeEvent(i, timeStep, timeMs));
                lastStepSpikes.add(i);
                // STDP 后脉冲
                if (stdpEnabled && stdp != null) {
                    List<Synapse> ins = incoming.get(i);
                    if (ins != null) for (Synapse s : ins) stdp.onPostSpike(s, timeMs);
                }
            }
        }
    }

    private final List<Integer> lastStepSpikes = new ArrayList<>();

    /** 模拟 T 步 */
    public void run(int steps) {
        for (int i = 0; i < steps; i++) step();
    }

    /** 当前步脉冲事件 */
    public List<SpikeEvent> spikeEvents() { return spikeEvents; }

    /** 累计发放数 */
    public long totalSpikes() {
        long count = 0;
        for (Neuron n : neurons) count += n.size();
        return count;
    }

    public int neuronCount() { return neurons.size(); }
    public int timeStep() { return timeStep; }
    public double timeMs() { return timeMs; }
    public List<Neuron> neurons() { return neurons; }
}
