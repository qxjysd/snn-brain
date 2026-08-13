package io.brainx.core;

import java.util.HashMap;
import java.util.Map;

/**
 * 僵尸行动者系统 —— 技能自动化 (zombie agents)。
 *
 * 理论依据 (Koch & Crick, 《意识与脑》第5-6章):
 *   - 僵尸行动者: 无意识的感觉-运动程序, "在意识雷达屏幕下"自动执行
 *     惯例任务 (眨眼、开车、打字、回击网球)
 *   - 技能获取: "只有经过几百小时的专门训练, 这一系列动作执行才成为
 *     自动的, 即转变为俗称的肌肉记忆" —— 重复 → 自动化
 *   - 自动化解放意识: 意识资源有限, 惯例任务交给僵尸行动者,
 *     意识专注于新情况/意外 (黑天鹅事件)
 *
 * 实现:
 *   - 每个技能/词条有熟练度 (练习次数)
 *   - 熟练度 ≥ 阈值 → 自动化: 识别不占意识资源, 置信度稳定高
 *   - 自动化技能处理快 (低延迟), 意识资源解放 → 可处理意外
 */
public class ZombieAgent {
    /** 自动化阈值 (练习次数) */
    private final int automationThreshold;
    /** 技能熟练度: 标签 → 练习次数 */
    private final Map<String, Integer> skill = new HashMap<>();
    /** 自动化技能集 */
    private final Map<String, Double> automated = new HashMap<>();
    /** 已解放的意识资源 (0-1) */
    private double freedAttention = 0;

    public ZombieAgent(int automationThreshold) {
        this.automationThreshold = Math.max(2, automationThreshold);
    }

    public static ZombieAgent defaultParams() {
        return new ZombieAgent(5);  // 5 次练习 → 自动化 (模拟"几百小时"的压缩)
    }

    /** 练习一次技能 (学习/识别该词) */
    public void practice(String label) {
        if (label == null || label.isEmpty()) return;
        int count = skill.getOrDefault(label, 0) + 1;
        skill.put(label, count);
        if (count >= automationThreshold && !automated.containsKey(label)) {
            // 技能自动化: 解放意识资源
            automated.put(label, 0.95);  // 自动化识别置信度 (稳定高)
            freedAttention = Math.min(1.0, freedAttention + 0.15);
        }
    }

    /** 该技能是否已自动化 (僵尸行动者接管) */
    public boolean isAutomated(String label) {
        return automated.containsKey(label);
    }

    /** 自动化技能识别: 快速、稳定、不占意识 */
    public double automatedConfidence(String label) {
        return automated.getOrDefault(label, 0.0);
    }

    /**
     * 识别处理: 自动化技能走僵尸通道 (快, 无需意识),
     * 否则需意识参与 (慢, 占资源)。
     * @return [置信度, 是否自动化, 意识占用]
     */
    public double[] process(String label, double consciousConfidence) {
        if (isAutomated(label)) {
            // 僵尸行动者: 无意识快速处理, 解放意识
            return new double[]{automatedConfidence(label), 1.0, 0.0};
        }
        // 有意识处理: 占意识资源
        return new double[]{consciousConfidence, 0.0, 1.0};
    }

    /** 可用的意识资源 (总1.0 - 被占用) */
    public double availableAttention() {
        return Math.max(0.1, 1.0 - freedAttention * 0.3);  // 自动化释放部分意识
    }

    /** 意外处理: Φ高时, 未自动化技能也能可靠处理 (黑天鹅) */
    public double handleUnexpected(double phi, String label) {
        if (isAutomated(label)) return 0.95;  // 自动化技能无需Φ
        // 非自动化 + 新情况: Φ 越高处理越好 (PHI 理论)
        return Math.min(0.95, 0.3 + phi * 0.6);
    }

    public int practiceCount(String label) { return skill.getOrDefault(label, 0); }
    public int automatedCount() { return automated.size(); }
    public double freedAttention() { return freedAttention; }
    public Map<String, Double> automated() { return automated; }

    /** 导出全部技能熟练度: "词:次数" 映射 (模型快照) */
    public Map<String, Integer> exportSkills() {
        return new HashMap<>(skill);
    }

    /** 导入技能熟练度 (模型恢复) */
    public void importSkills(Map<String, Integer> skills) {
        skill.clear();
        automated.clear();
        freedAttention = 0;
        for (Map.Entry<String, Integer> e : skills.entrySet()) {
            skill.put(e.getKey(), e.getValue());
            if (e.getValue() >= automationThreshold) {
                automated.put(e.getKey(), 0.95);
                freedAttention = Math.min(1.0, freedAttention + 0.15);
            }
        }
    }

    /** 摘要 (APK 显示) */
    public String summary() {
        return String.format("🤖 僵尸行动者: %d个技能已自动化 (解放意识%.0f%%)",
                automated.size(), freedAttention * 100);
    }
}
