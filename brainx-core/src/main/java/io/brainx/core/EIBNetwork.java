package io.brainx.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * E/I 平衡网络 + 证据累积决策任务。
 * 来源: BrainTrace 论文 (NC 2026) Fig.5 ——
 *   "800 GIF neurons, 4:1 excitatory-to-inhibitory ratio, 10% random connection,
 *    conductance-based synapses, evidence-accumulation task:
 *    animals navigate a virtual linear track and encounter seven visual cues,
 *    then select direction at a T-junction based on majority cue distribution."
 *
 * 实现 (简化):
 *   - E/I 平衡: 兴奋神经元占 80%, 抑制占 20%, 稀疏随机连接 (10%)
 *   - 证据累积: 左/右线索逐步累积到两个证据池 (模拟后顶叶皮层)
 *   - T 字路口决策: 证据累积超过阈值 → 决策方向
 *   - 后训练分析: 选择性种群发放模式 (时间到首次脉冲编码 TTFS)
 */
public class EIBNetwork {
    /** 神经元类型 */
    public enum Type { EXCITATORY, INHIBITORY }

    private final int totalNeurons;
    private final double[] membrane;       // 膜电位
    private final Type[] types;
    private final double[][] weights;      // 突触权重
    private final boolean[] firing;
    private final double[] inputCurrent;

    // E/I 参数 (BrainTrace 论文: 4:1 E/I, 10% 连接)
    private final double tauE = 10.0, tauI = 8.0;        // ms
    private final double vRest = -65.0, vThresh = -50.0, vReset = -65.0;
    private final double wEE = 1.0, wEI = -1.8, wIE = 1.5, wII = -1.2;  // 权重 (E 兴奋 I 抑制)

    // 证据累积 (decision variables)
    private double evidenceLeft = 0, evidenceRight = 0;
    private final double decisionThreshold = 0.3;

    private final Random rnd = new Random(42);
    private final List<double[]> activityTraces = new ArrayList<>();  // 每步发放率

    public EIBNetwork(int totalNeurons, long seed) {
        this.totalNeurons = totalNeurons;
        this.membrane = new double[totalNeurons];
        this.types = new Type[totalNeurons];
        this.weights = new double[totalNeurons][totalNeurons];
        this.firing = new boolean[totalNeurons];
        this.inputCurrent = new double[totalNeurons];
        Random r = new Random(seed);
        // 4:1 E/I 比例 (BrainTrace 论文)
        for (int i = 0; i < totalNeurons; i++) {
            types[i] = (i < totalNeurons * 4 / 5) ? Type.EXCITATORY : Type.INHIBITORY;
            membrane[i] = vRest;
        }
        // 10% 稀疏随机连接 (BrainTrace 论文)
        for (int i = 0; i < totalNeurons; i++) {
            for (int j = 0; j < totalNeurons; j++) {
                if (i != j && r.nextDouble() < 0.1) {
                    boolean preE = types[i] == Type.EXCITATORY;
                    boolean postE = types[j] == Type.EXCITATORY;
                    if (preE && postE) weights[i][j] = wEE;
                    else if (preE && !postE) weights[i][j] = wEI;
                    else if (!preE && postE) weights[i][j] = wIE;
                    else weights[i][j] = wII;
                }
            }
        }
    }

    public static EIBNetwork defaultParams() { return new EIBNetwork(800, 42); }

    /**
     * 前进一步: 施加左右证据线索 → 网络更新 → 证据累积。
     * @param cueLeft  左侧线索 (0-1)
     * @param cueRight 右侧线索 (0-1)
     */
    public void step(double cueLeft, double cueRight, double dtMs) {
        // 1. 输入: 线索分配到对应神经元群
        for (int i = 0; i < totalNeurons; i++) {
            inputCurrent[i] = 0;
            if (types[i] == Type.EXCITATORY) {
                // 前 2/5 兴奋神经元响应左线索, 后 2/5 响应右线索 (证据池)
                double frac = (double) i / totalNeurons;
                if (frac < 0.4) inputCurrent[i] += cueLeft * 30.0;
                else if (frac < 0.8) inputCurrent[i] += cueRight * 30.0;
            }
        }
        // 2. 网络脉冲传播
        for (int pre = 0; pre < totalNeurons; pre++) {
            if (firing[pre]) {
                for (int post = 0; post < totalNeurons; post++) {
                    if (weights[pre][post] != 0) {
                        inputCurrent[post] += weights[pre][post];
                    }
                }
            }
        }
        // 3. 神经元更新
        double rate = 0;
        for (int i = 0; i < totalNeurons; i++) {
            double tau = types[i] == Type.EXCITATORY ? tauE : tauI;
            membrane[i] += (-(membrane[i] - vRest) + inputCurrent[i]) / tau * dtMs;
            firing[i] = membrane[i] >= vThresh;
            if (firing[i]) {
                membrane[i] = vReset;
                rate++;
            }
        }
        // 4. 证据累积: 左/右兴奋群的发放率差
        double leftRate = 0, rightRate = 0;
        for (int i = 0; i < totalNeurons; i++) {
            double frac = (double) i / totalNeurons;
            if (firing[i] && types[i] == Type.EXCITATORY) {
                if (frac < 0.4) leftRate++;
                else if (frac < 0.8) rightRate++;
            }
        }
        // 证据 = 累计发放差 (每秒发放数, dtMs=1ms)
        evidenceLeft += leftRate * dtMs / 1000.0 * 0.5;
        evidenceRight += rightRate * dtMs / 1000.0 * 0.5;

        // 记录活动 (脑图用)
        activityTraces.add(new double[]{leftRate, rightRate, evidenceLeft, evidenceRight});
        if (activityTraces.size() > 500) activityTraces.remove(0);
    }

    /** 决策状态: 0=未决, 1=左, 2=右 (T字路口) */
    public int decision() {
        if (evidenceLeft >= decisionThreshold) return 1;
        if (evidenceRight >= decisionThreshold) return 2;
        return 0;
    }

    /** 完整决策: 运行整个证据累积序列 (全部线索呈现后到 T 字路口才决策) */
    public int runTrial(double[] cueSequence) {
        reset();
        // 虚拟跑道: 依次呈现所有视觉线索 (BrainTrace Fig5: 7 cues)
        for (int t = 0; t < cueSequence.length; t += 2) {
            double left = cueSequence[t];
            double right = cueSequence[t + 1];
            for (int sub = 0; sub < 10; sub++) step(left, right, 1.0);
        }
        // T 字路口: 基于累积证据决策
        return evidenceLeft >= evidenceRight ? 1 : 2;
    }

    public void reset() {
        for (int i = 0; i < totalNeurons; i++) {
            membrane[i] = vRest;
            firing[i] = false;
        }
        evidenceLeft = evidenceRight = 0;
        activityTraces.clear();
    }

    public double evidenceLeft() { return evidenceLeft; }
    public double evidenceRight() { return evidenceRight; }
    public int totalNeurons() { return totalNeurons; }
    public boolean[] firingState() { return firing; }
    public List<double[]> activityTraces() { return activityTraces; }
    public Type typeOf(int i) { return types[i]; }

    /** 统计: 兴奋/抑制发放率 (Hz, 假设 1ms 步) */
    public double[] firingRates() {
        int eFire = 0, iFire = 0, eCount = 0, iCount = 0;
        for (int i = 0; i < totalNeurons; i++) {
            if (types[i] == Type.EXCITATORY) { eCount++; if (firing[i]) eFire++; }
            else { iCount++; if (firing[i]) iFire++; }
        }
        return new double[]{(double) eFire / eCount * 1000, (double) iFire / iCount * 1000};
    }
}
