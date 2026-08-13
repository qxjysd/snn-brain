package io.brainx.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 自发突触形成器 —— 随机共激活驱动新连接建立 (synaptogenesis)。
 *
 * 神经科学依据:
 *   - 婴儿早期大脑有大量自发神经活动 (自发波/retinal waves)
 *   - Hebbian 学习: "fire together, wire together" ——
 *     两个神经元频繁同时放电 → 它们之间的连接被建立/增强
 *   - 随机信号为尚未建立连接的神经元提供共激活机会 →
 *     促进网络连接形成 (模拟人类神经链接的发育方式)
 *   - 用进废退: 从不共激活的候选连接被修剪 (synaptic pruning)
 *
 * 模型: 维护候选连接池 (随机初始化, 稀疏), 每次共激活增强,
 *   长期无共激活则修剪; 新候选随机产生 (持续探索)。
 */
public class SynapseFormation {
    /** 候选连接: [pre, post, weight] */
    private final List<double[]> candidates = new ArrayList<>();
    /** 候选连接最后共激活时间 */
    private final List<Integer> lastCoactivate = new ArrayList<>();
    /** 神经元数量 */
    private final int numNeurons;
    /** 候选连接上限 (8N, 防无界增长) */
    private final int maxCandidates;
    /** 空间网格边长 (>0 启用局部连接: 前 grid² 神经元按皮层网格距离衰减) */
    private final int spatialGrid;
    /** 神经元 2D 位置 (局部连接用) */
    private final int[] posX, posY;
    /** 候选索引: (pre,post) → candidates 下标 (JIT 优化: coactivate O(1) 替代线性扫描) */
    private final java.util.Map<Long, Integer> connIndex = new java.util.HashMap<>();
    private final double growRate;    // 共激活增强率
    private final double pruneRate;   // 修剪阈值
    private final int pruneAfterMs;   // 无共激活 N ms 后修剪
    private final Random rnd;
    private int timeMs = 0;

    /** 连接对 → 哈希键 (无向: 取小值在前) */
    private static long key(int a, int b) {
        int lo = Math.min(a, b), hi = Math.max(a, b);
        return ((long) lo << 32) | (hi & 0xffffffffL);
    }

    public SynapseFormation(int numNeurons, int initialCandidates, long seed,
                            double growRate, int pruneAfterMs) {
        this(numNeurons, initialCandidates, seed, growRate, pruneAfterMs, 0);
    }

    /**
     * @param spatialGrid >0 时启用皮层局部连接: 前 grid² 神经元按 2D 网格,
     *                   连接生成距离衰减 (局部微柱 ~80%, 少量长程投射) —— 符合皮层解剖
     */
    public SynapseFormation(int numNeurons, int initialCandidates, long seed,
                            double growRate, int pruneAfterMs, int spatialGrid) {
        this.numNeurons = numNeurons;
        this.maxCandidates = numNeurons * 8;
        this.spatialGrid = spatialGrid;
        this.posX = new int[numNeurons];
        this.posY = new int[numNeurons];
        this.growRate = growRate;
        this.pruneAfterMs = pruneAfterMs;
        this.pruneRate = 0.02;
        this.rnd = new Random(seed);
        int grid = spatialGrid > 0 ? spatialGrid : 64;   // 非空间模式也分配随机位置 (供距离统计)
        for (int i = 0; i < numNeurons; i++) {
            if (spatialGrid > 0 && i < grid * grid) { posX[i] = i % grid; posY[i] = i / grid; }
            else { posX[i] = rnd.nextInt(grid); posY[i] = rnd.nextInt(grid); }
        }
        for (int i = 0; i < initialCandidates; i++) {
            int pre = rnd.nextInt(numNeurons);
            int post;
            // 局部连接占主导 (~80%): 距离衰减选邻居 (皮层微柱/列结构)
            if (spatialGrid > 0 && rnd.nextDouble() < 0.8) post = localNeighbor(pre);
            else post = rnd.nextInt(numNeurons);
            if (pre != post) {
                candidates.add(new double[]{pre, post, 0.01 + rnd.nextDouble() * 0.05});
                lastCoactivate.add(0);
                connIndex.put(key(pre, post), candidates.size() - 1);
            }
        }
    }

    /** 距离衰减邻居: 8 次随机尝试取距离最近 (指数衰减核, λ≈0.3×grid) */
    private int localNeighbor(int pre) {
        int best = rnd.nextInt(numNeurons);
        double bestP = -1;
        double lambda = Math.max(1.0, spatialGrid * 0.3);
        for (int t = 0; t < 8; t++) {
            int cand = rnd.nextInt(numNeurons);
            double dx = posX[pre] - posX[cand], dy = posY[pre] - posY[cand];
            double p = Math.exp(-Math.sqrt(dx * dx + dy * dy) / lambda);
            if (p > bestP) { bestP = p; best = cand; }
        }
        return best;
    }

    /** 连接平均距离 (局部结构验证: spatial 模式应显著小于全局随机) */
    public double averageDistance() {
        double sum = 0; int cnt = 0;
        for (double[] c : candidates) {
            int a = (int) c[0], b = (int) c[1];
            double dx = posX[a] - posX[b], dy = posY[a] - posY[b];
            sum += Math.sqrt(dx * dx + dy * dy);
            cnt++;
        }
        return cnt > 0 ? sum / cnt : 0;
    }

    public static SynapseFormation defaultParams(int numNeurons) {
        return new SynapseFormation(numNeurons, numNeurons * 4, 42, 0.05, 5000);
    }

    /** 默认空间版: 视觉皮层按网格局部连接 (皮层微柱结构) */
    public static SynapseFormation defaultSpatialParams(int numNeurons, int spatialGrid) {
        return new SynapseFormation(numNeurons, numNeurons * 4, 42, 0.05, 5000, spatialGrid);
    }

    /** 每时间步: 随机自发活动 → 检查共激活 → 增强/修剪 */
    public void step(boolean[] firing, int dtMs) {
        timeMs += dtMs;
        // 自发共激活: 随机选两对共激活神经元
        for (int trial = 0; trial < 2; trial++) {
            int a = rnd.nextInt(numNeurons);
            int b = rnd.nextInt(numNeurons);
            // 模拟自发共激活 (即使无外部刺激)
            if (rnd.nextDouble() < 0.3) {
                coactivate(a, b);
            }
        }
        // 真实发放共激活
        for (int i = 0; i < numNeurons; i++) {
            if (firing[i]) {
                int partner = rnd.nextInt(numNeurons);
                if (partner != i && firing[partner]) coactivate(i, partner);
            }
        }
        // 修剪: 长期无共激活的连接 (swap-remove 保持索引一致)
        for (int i = candidates.size() - 1; i >= 0; i--) {
            if (timeMs - lastCoactivate.get(i) > pruneAfterMs) {
                removeAt(i);
            }
        }
        // 探索: 随机产生新候选
        if (rnd.nextDouble() < 0.01) {
            int pre = rnd.nextInt(numNeurons);
            int post = rnd.nextInt(numNeurons);
            if (pre != post && !connIndex.containsKey(key(pre, post))) {
                candidates.add(new double[]{pre, post, 0.01});
                lastCoactivate.add(timeMs);
                connIndex.put(key(pre, post), candidates.size() - 1);
            }
        }
    }

    /** 删除候选 i: 交换末尾元素补位 (只影响一个下标, 同步索引) */
    private void removeAt(int i) {
        double[] c = candidates.get(i);
        connIndex.remove(key((int) c[0], (int) c[1]));
        int last = candidates.size() - 1;
        if (i != last) {
            candidates.set(i, candidates.get(last));
            lastCoactivate.set(i, lastCoactivate.get(last));
            double[] moved = candidates.get(i);
            connIndex.put(key((int) moved[0], (int) moved[1]), i);
        }
        candidates.remove(last);
        lastCoactivate.remove(last);
    }

    /**
     * 两神经元共激活 → 若存在候选连接则增强 (Hebbian);
     * 若未命中候选 → 直接建立新连接 (synaptogenesis: fire together wire together)。
     * 大网络 (数千神经元) 下随机候选池命中率极低 (~0.07%), 若只增强不新建,
     * 连接永不成熟 → 突触形成机制失效 (v5.4 连接审计发现)。
     */
    public void coactivate(int a, int b) {
        if (a == b) return;
        // 哈希索引 O(1) 查找 (无向连接)
        Integer idx = connIndex.get(key(a, b));
        if (idx != null) {
            double[] c = candidates.get(idx);
            c[2] += growRate;
            if (c[2] > 1.0) c[2] = 1.0;
            lastCoactivate.set(idx, timeMs);
            return;
        }
        // 共激活未命中候选 → 建立新连接 (受 8N 上限约束, 修剪控制网络规模)
        if (candidates.size() < maxCandidates) {
            candidates.add(new double[]{a, b, 0.05 + growRate});
            lastCoactivate.add(timeMs);
            connIndex.put(key(a, b), candidates.size() - 1);
        }
    }

    /** 读取成熟连接 (权重超过阈值) 的列表 */
    public List<double[]> matureConnections(double minWeight) {
        List<double[]> out = new ArrayList<>();
        for (double[] c : candidates) {
            if (c[2] >= minWeight) out.add(c.clone());
        }
        return out;
    }

    /** 导出全部候选连接: [pre, post, weight, lastCoactivate] 列表 (模型快照) */
    public List<double[]> exportConnections() {
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) {
            double[] c = candidates.get(i);
            out.add(new double[]{c[0], c[1], c[2], lastCoactivate.get(i)});
        }
        return out;
    }

    /** 导入候选连接 (模型恢复): 清空后重建 */
    public void importConnections(List<double[]> conns) {
        candidates.clear();
        lastCoactivate.clear();
        connIndex.clear();
        for (double[] c : conns) {
            if (c.length >= 3) {
                candidates.add(new double[]{c[0], c[1], c[2]});
                lastCoactivate.add((int) c[3]);
                connIndex.put(key((int) c[0], (int) c[1]), candidates.size() - 1);
            }
        }
    }

    /** 当前连接总数 (导出统计) */
    public int connectionCount() { return candidates.size(); }

    public int candidateCount() { return candidates.size(); }
    public int matureCount(double minWeight) { return matureConnections(minWeight).size(); }
    public int timeMs() { return timeMs; }
}
