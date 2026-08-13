package io.brainx.core;

import java.util.*;

/**
 * 大脑状态快照 —— 完整模型导出/导入 (跨平台分享训练成果)。
 *
 * 导出内容 = 整个神经网络的连接参数 + 记忆参数:
 *   【神经网络连接参数】
 *   - pp-prop 学习器权重 (视觉→联想, 听觉→联想) —— 真正的网络连接
 *   - 自发突触形成器连接池 (pre/post/weight)
 *   - 联想记忆权重 (Hebbian 概念→词)
 *   【记忆参数】
 *   - 分层记忆: 中期(情景) + 长期(语义)
 *   - 工作记忆内容 + 强度
 *   - 僵尸行动者技能熟练度 (肌肉记忆)
 *   - 预测引擎先验 (世界模型)
 *   - 多巴胺预期 (学习历史)
 *   【养成状态】
 *   - 等级/点数/XP/成就/物品/认知模式/自我意识
 *
 * 格式: 紧凑纯文本 (跨平台: 任何语言/平台可解析), 以 BASE64 存矩阵。
 */
public class BrainSnapshot {
    private static final String MAGIC = "BRAINX-SNAP-2";

    /** 从 Brain 导出完整模型状态 */
    public static String export(Brain brain, int trainerLevel, int trainerPoints,
                                int trainerXp, String achievements, String inventory) {
        StringBuilder sb = new StringBuilder();
        sb.append(MAGIC).append("\n");
        sb.append("level:").append(trainerLevel).append("\n");
        sb.append("points:").append(trainerPoints).append("\n");
        sb.append("xp:").append(trainerXp).append("\n");
        sb.append("achv:").append(achievements).append("\n");
        sb.append("inv:").append(inventory).append("\n");
        sb.append("words:").append(String.join(",", brain.learnedWords())).append("\n");

        // 【神经网络连接参数】
        // 1. 联想记忆权重 (assoc x vocab)
        sb.append("mem:").append(matrixToStr(brain.assocWeights())).append("\n");
        // 2. pp-prop 视觉→联想权重
        sb.append("nn-vis:").append(matrixToStr(brain.visualToAssocWeights())).append("\n");
        // 3. pp-prop 听觉→联想权重
        sb.append("nn-aud:").append(matrixToStr(brain.auditoryToAssocWeights())).append("\n");
        // 4. 突触形成连接池: "pre,post,w,t;..."
        sb.append("syn:").append(connectionsToStr(brain.synapseFormation().exportConnections())).append("\n");
        // 5. 中枢脉冲网络权重 (初始全连接→学习修剪后的连接参数)
        sb.append("hubw:").append(matrixToStr(brain.centralHub().cloneWeights())).append("\n");

        // 【记忆参数】
        // 5. 分层记忆: 长期标签 + 中期条目
        sb.append("ltmem:").append(brain.longTermLabels()).append("\n");
        sb.append("epmem:").append(brain.episodicSummary()).append("\n");
        // 6. 工作记忆: 内容向量 (BASE64 每槽)
        sb.append("wm:").append(workingMemoryToStr(brain)).append("\n");
        // 7. 僵尸行动者熟练度: "词:次数,..."
        sb.append("zombie:").append(skillsToStr(brain)).append("\n");
        // 8. 预测先验: "词:概率,..."
        sb.append("priors:").append(priorsToStr(brain)).append("\n");

        // 【其他状态】
        sb.append("mode:").append(String.format(Locale.US, "%.3f,%.3f",
                brain.cognitiveMode().upperBrain(), brain.cognitiveMode().lowerBrain())).append("\n");
        sb.append("dopa:").append(String.format(Locale.US, "%.3f", brain.dopamineExpected())).append("\n");
        sb.append("self:").append(brain.selfConfidenceSnapshot()).append("\n");
        return sb.toString();
    }

    /** 解析快照 → 恢复大脑完整状态 (神经网络+记忆+养成) */
    public static RestoreInfo importSnapshot(Brain brain, String snapshot) {
        if (snapshot == null || (!snapshot.startsWith(MAGIC) && !snapshot.startsWith("BRAINX-SNAP-1"))) {
            return null;
        }
        RestoreInfo info = new RestoreInfo();
        try {
            String[] lines = snapshot.split("\n");
            for (String line : lines) {
                if (line.isEmpty() || line.startsWith("BRAINX-SNAP")) continue;
                int colon = line.indexOf(':');
                if (colon < 0) continue;
                String key = line.substring(0, colon);
                String val = line.substring(colon + 1);
                switch (key) {
                    case "level": info.level = parseInt(val); break;
                    case "points": info.points = parseInt(val); break;
                    case "xp": info.xp = parseInt(val); break;
                    case "achv": info.achievements = val; break;
                    case "inv": info.inventory = val; break;
                    case "words":
                        if (!val.isEmpty()) {
                            for (String w : val.split(",")) {
                                if (!w.isEmpty()) brain.learnWordOnly(w);
                            }
                            info.wordCount = brain.learnedWords().size();
                        }
                        break;
                    case "mem": brain.setAssocWeights(strToMatrix(val, brain.assocSize(), brain.vocabularySize()));
                        info.memoryRestored = true; break;
                    case "nn-vis": brain.setVisualToAssocWeights(strToMatrix(val,
                            brain.visualToAssocRows(), brain.visualToAssocCols()));
                        info.nnRestored = true; break;
                    case "nn-aud": brain.setAuditoryToAssocWeights(strToMatrix(val,
                            brain.auditoryToAssocRows(), brain.auditoryToAssocCols()));
                        break;
                    case "syn": brain.synapseFormation().importConnections(strToConnections(val));
                        info.synapseRestored = true; break;
                    case "hubw": brain.centralHub().setWeights(strToMatrix(val,
                            brain.centralHub().hubSize(), brain.centralHub().inputDim()));
                        break;
                    case "ltmem":
                        if (!val.isEmpty()) {
                            for (String label : val.split(",")) {
                                if (!label.isEmpty()) brain.addLongTermLabel(label);
                            }
                            info.longTermCount = brain.longTermCount();
                        }
                        break;
                    case "epmem": brain.restoreEpisodic(val); break;
                    case "wm": restoreWorkingMemory(brain, val); break;
                    case "zombie": brain.importZombieSkills(val); break;
                    case "priors": brain.importPriors(val); break;
                    case "mode":
                        String[] mv = val.split(",");
                        if (mv.length == 2) {
                            brain.setCognitiveMode(Double.parseDouble(mv[0]), Double.parseDouble(mv[1]));
                        }
                        break;
                    case "dopa": break;
                    case "self":
                        String[] sv = val.split(",");
                        if (sv.length >= 1 && !sv[0].isEmpty()) {
                            brain.setSelfConfidence(Double.parseDouble(sv[0]));
                        }
                        break;
                    default: break;
                }
            }
            return info;
        } catch (Exception e) {
            return null;
        }
    }

    // ============ 序列化工具 ============

    /** 矩阵 → 行用 ; 分隔, 列用 , 分隔 (BASE64 压缩可选, 先直接存) */
    static String matrixToStr(double[][] m) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < m.length; i++) {
            if (i > 0) sb.append(';');
            for (int j = 0; j < m[i].length; j++) {
                if (j > 0) sb.append(',');
                sb.append(String.format(Locale.US, "%.6f", m[i][j]));
            }
        }
        return sb.toString();
    }

    static double[][] strToMatrix(String s, int rows, int cols) {
        double[][] m = new double[rows][cols];
        String[] rowStrs = s.split(";");
        for (int i = 0; i < rows && i < rowStrs.length; i++) {
            String[] colStrs = rowStrs[i].split(",");
            for (int j = 0; j < cols && j < colStrs.length; j++) {
                if (!colStrs[j].isEmpty()) {
                    m[i][j] = Double.parseDouble(colStrs[j]);
                }
            }
        }
        return m;
    }

    static String connectionsToStr(List<double[]> conns) {
        StringBuilder sb = new StringBuilder();
        for (double[] c : conns) {
            if (sb.length() > 0) sb.append(';');
            sb.append((int) c[0]).append(',').append((int) c[1]).append(',')
              .append(String.format(Locale.US, "%.4f", c[2])).append(',')
              .append((int) c[3]);
        }
        return sb.toString();
    }

    static List<double[]> strToConnections(String s) {
        List<double[]> out = new ArrayList<>();
        if (s.isEmpty()) return out;
        for (String part : s.split(";")) {
            String[] p = part.split(",");
            if (p.length >= 4) {
                out.add(new double[]{Integer.parseInt(p[0]), Integer.parseInt(p[1]),
                        Double.parseDouble(p[2]), Integer.parseInt(p[3])});
            }
        }
        return out;
    }

    static String workingMemoryToStr(Brain brain) {
        double[][] contents = brain.workingMemory().exportContents();
        double[] strengths = brain.workingMemory().strengths();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < contents.length; i++) {
            if (i > 0) sb.append(';');
            sb.append(String.format(Locale.US, "%.3f", strengths[i])).append(':');
            for (int j = 0; j < contents[i].length; j++) {
                if (j > 0) sb.append(',');
                sb.append(String.format(Locale.US, "%.4f", contents[i][j]));
            }
        }
        return sb.toString();
    }

    static void restoreWorkingMemory(Brain brain, String val) {
        if (val.isEmpty()) return;
        String[] slots = val.split(";");
        for (int i = 0; i < slots.length && i < brain.workingMemory().capacity(); i++) {
            int colon = slots[i].indexOf(':');
            if (colon < 0) continue;
            double strength = Double.parseDouble(slots[i].substring(0, colon));
            String[] vals = slots[i].substring(colon + 1).split(",");
            double[] content = new double[vals.length];
            for (int j = 0; j < vals.length; j++) content[j] = Double.parseDouble(vals[j]);
            brain.workingMemory().importContents(
                    new double[][]{content}, new double[]{strength});
        }
    }

    static String skillsToStr(Brain brain) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Integer> e : brain.zombieSkills().entrySet()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(e.getKey()).append(':').append(e.getValue());
        }
        return sb.toString();
    }

    static String priorsToStr(Brain brain) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Double> e : brain.predictivePriors().entrySet()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(e.getKey()).append(':').append(String.format(Locale.US, "%.4f", e.getValue()));
        }
        return sb.toString();
    }

    private static int parseInt(String s) {
        try { return Integer.parseInt(s.trim()); } catch (Exception e) { return 0; }
    }

    /** 恢复信息 (供 UI 展示) */
    public static class RestoreInfo {
        public int level = 0, points = 0, xp = 0;
        public int wordCount = 0, longTermCount = 0;
        public boolean memoryRestored = false, nnRestored = false, synapseRestored = false;
        public String achievements = "", inventory = "";

        public String describe() {
            return String.format("🧠 模型恢复: %d词 | 联想记忆%s | 神经网络%s | 突触连接%s | 等级%d | %d点",
                    wordCount, memoryRestored ? "✓" : "✗",
                    nnRestored ? "✓" : "✗",
                    synapseRestored ? "✓(" + longTermCount + "条)" : "✗",
                    level, points);
        }
    }
}
