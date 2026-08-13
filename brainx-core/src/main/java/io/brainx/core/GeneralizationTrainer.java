package io.brainx.core;

import java.util.Random;

/**
 * 泛化训练器 —— 变体训练 → 原型提取 → 未见变体验证。
 *
 * 理论依据:
 *   - 人类学习泛化: 见过苹果的各种样子(不同光照/角度/大小)后,
 *     能认出没见过的苹果 —— 原型提取 (prototype theory, Rosch)
 *   - 突触可塑性: 多变异训练 → 突触权重形成稳定原型模式
 *   - 训练/测试分离: 训练集上的表现 ≠ 泛化能力, 需未见样本验证
 *
 * 实现:
 *   - 变体生成: 对原型特征加扰动 (噪声/位移/缩放) → 模拟真实世界变异
 *   - 原型提取: 变体训练后, 联想记忆收敛到原型 (均值)
 *   - 泛化评估: 用训练中未见的变体测试识别率
 */
public class GeneralizationTrainer {
    private final Random rnd;

    public GeneralizationTrainer(long seed) { this.rnd = new Random(seed); }
    public static GeneralizationTrainer defaultParams() { return new GeneralizationTrainer(42); }

    /**
     * 生成变体: 对基础特征加扰动, 模拟真实世界变异。
     * @param base      基础特征向量 (0-1)
     * @param noiseLevel 扰动幅度 (0-0.5)
     * @param variants  生成数量
     * @return 变体数组 [variants][len]
     */
    public double[][] generateVariants(double[] base, double noiseLevel, int variants) {
        double[][] out = new double[variants][base.length];
        for (int v = 0; v < variants; v++) {
            for (int i = 0; i < base.length; i++) {
                double noise = (rnd.nextDouble() * 2 - 1) * noiseLevel;
                double val = base[i] + noise;
                // 位移效果: 相邻像素干扰 (模拟视角变化)
                if (rnd.nextDouble() < 0.2 && i + 1 < base.length) {
                    val = base[i + 1] + (rnd.nextDouble() - 0.5) * noiseLevel;
                }
                out[v][i] = Math.max(0, Math.min(1, val));
            }
        }
        return out;
    }

    /**
     * 用变体训练大脑: 每类多个变体 → 泛化学习。
     * @param brain     大脑
     * @param bases     每类的原型特征
     * @param labels    每类的词索引
     * @param noiseLevel 扰动幅度
     * @param variantsPerClass 每类变体数
     */
    public void trainWithVariants(Brain brain, double[][] bases, int[] labels,
                                  double noiseLevel, int variantsPerClass) {
        for (int c = 0; c < bases.length; c++) {
            double[][] variants = generateVariants(bases[c], noiseLevel, variantsPerClass);
            for (double[] variant : variants) {
                brain.learnVisualWord(variant, labels[c]);
            }
        }
    }

    /**
     * 泛化评估: 用训练中未见的新变体测试识别率。
     * @param brain   大脑
     * @param bases   每类原型
     * @param labels  每类词索引 (用于比对)
     * @param vocab   词表
     * @param noiseLevel 测试扰动 (可高于训练噪声 = 更强泛化挑战)
     * @param testPerClass 每类测试数
     * @return [识别率 0-1, 命中数, 总数]
     */
    public double[] evaluateGeneralization(Brain brain, double[][] bases, int[] labels,
                                           String[] vocab, double noiseLevel, int testPerClass) {
        int hits = 0, total = 0;
        for (int c = 0; c < bases.length; c++) {
            double[][] testVariants = generateVariants(bases[c], noiseLevel, testPerClass);
            for (double[] variant : testVariants) {
                String guess = brain.recognizeVisual(variant);
                if (guess.equals(vocab[labels[c]])) hits++;
                total++;
            }
        }
        return new double[]{(double) hits / total, hits, total};
    }

    /**
     * 对比实验: 单样本训练 vs 多样本(变体)训练 的泛化率。
     * 证明: 变体训练 → 泛化更好 (原型提取)
     */
    public double[] compareGeneralization(double[][] bases, int[] labels,
                                          String[] vocab, int variantsPerClass) {
        // 基线: 每类只训练 1 个干净样本
        Brain baseline = Brain.simpleBrain();
        for (int c = 0; c < bases.length; c++) {
            baseline.learnVisualWord(bases[c], labels[c]);
        }
        double[] baseGen = evaluateGeneralization(baseline, bases, labels, vocab, 0.2, 10);

        // 变体训练
        Brain variantBrain = Brain.simpleBrain();
        trainWithVariants(variantBrain, bases, labels, 0.15, variantsPerClass);
        double[] variantGen = evaluateGeneralization(variantBrain, bases, labels, vocab, 0.2, 10);

        return new double[]{baseGen[0], variantGen[0]};
    }
}
