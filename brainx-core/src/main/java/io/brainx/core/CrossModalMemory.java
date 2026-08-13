package io.brainx.core;

/**
 * 跨模态记忆 — 视觉原型 ↔ 听觉原型绑定 (多模态学习核心)。
 *
 * 神经科学依据 (多感觉整合/跨模态可塑性):
 *   - 概念 = 多模态特征的绑定 (看到猫 + 听到猫叫 → 同一概念的视觉/听觉两面)
 *   - 跨模态回忆: 视觉线索 → 激活听觉表征 (联想皮层的模态间投射)
 *   - 原型滑动平均: 反复绑定 → 原型更稳 (Hebbian 累积)
 *
 * 每词维护 [视觉原型, 听觉原型]:
 *   - bind: 学习时用当前模态特征滑动平均更新原型 (学习率 α)
 *   - recallVisual/recallAudio: 跨模态回忆 (返回另一模态的原型特征)
 *   - 相似度: 跨模态验证 (视觉输入 vs 回忆的视觉原型)
 */
public class CrossModalMemory {
    /** 词数上限 */
    private final int vocabSize;
    /** 视觉原型: [词][视觉特征维] */
    private final double[][] visualProtos;
    /** 听觉原型: [词][听觉特征维] */
    private final double[][] audioProtos;
    /** 绑定次数 (学习率衰减: 反复绑定更稳) */
    private final int[] bindCount;
    /** 学习率 */
    private static final double ALPHA = 0.3;

    public CrossModalMemory(int vocabSize, int visualDim, int audioDim) {
        this.vocabSize = vocabSize;
        this.visualProtos = new double[vocabSize][visualDim];
        this.audioProtos = new double[vocabSize][audioDim];
        this.bindCount = new int[vocabSize];
    }

    /** 绑定: 视觉+听觉特征同时关联到词 (滑动平均更新原型) */
    public void bind(int wordIndex, double[] visualFeatures, double[] audioFeatures) {
        if (wordIndex < 0 || wordIndex >= vocabSize) return;
        if (visualFeatures != null) {
            double lr = ALPHA / (1.0 + bindCount[wordIndex] * 0.1);   // 反复绑定学习率衰减
            double[] proto = visualProtos[wordIndex];
            for (int i = 0; i < Math.min(proto.length, visualFeatures.length); i++) {
                proto[i] = proto[i] * (1 - lr) + visualFeatures[i] * lr;
            }
        }
        if (audioFeatures != null) {
            double lr = ALPHA / (1.0 + bindCount[wordIndex] * 0.1);
            double[] proto = audioProtos[wordIndex];
            for (int i = 0; i < Math.min(proto.length, audioFeatures.length); i++) {
                proto[i] = proto[i] * (1 - lr) + audioFeatures[i] * lr;
            }
        }
        bindCount[wordIndex]++;
    }

    /** 跨模态回忆: 视觉原型 (听到声音 → 唤起视觉形象) */
    public double[] recallVisual(int wordIndex) {
        return wordIndex >= 0 && wordIndex < vocabSize ? visualProtos[wordIndex].clone() : null;
    }

    /** 跨模态回忆: 听觉原型 (看到物体 → 唤起声音) */
    public double[] recallAudio(int wordIndex) {
        return wordIndex >= 0 && wordIndex < vocabSize ? audioProtos[wordIndex].clone() : null;
    }

    /** 绑定次数 */
    public int bindCount(int wordIndex) { return bindCount[wordIndex]; }

    /** 是否已绑定 (有原型) */
    public boolean isBound(int wordIndex) {
        return wordIndex >= 0 && wordIndex < vocabSize && bindCount[wordIndex] > 0;
    }

    /** 相似度 (0-1, 余弦): 输入特征 vs 词的原型 */
    public static double similarity(double[] a, double[] b) {
        if (a == null || b == null || a.length == 0 || b.length == 0) return 0;
        double dot = 0, na = 0, nb = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return Math.max(0, Math.min(1, dot / (Math.sqrt(na) * Math.sqrt(nb))));
    }

    /** 摘要 */
    public String summary() {
        int bound = 0;
        for (int c : bindCount) if (c > 0) bound++;
        return String.format("🔗 跨模态记忆: %d/%d 词已绑定 (视觉↔听觉原型)", bound, vocabSize);
    }
}
