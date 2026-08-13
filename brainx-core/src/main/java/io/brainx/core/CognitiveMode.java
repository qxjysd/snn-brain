package io.brainx.core;

/**
 * 认知模式系统 —— 上脑/下脑双系统理论 (Kosslyn, 《上脑与下脑》)。
 *
 * 理论依据:
 *   - 上脑系统 (额叶上端): 计划、预测、执行、根据反馈调整计划
 *   - 下脑系统 (枕叶→颞/顶叶): 感知分类、与记忆比较、识别
 *   - 两系统使用度差异 → 四种认知模式 (第7章):
 *     行动者(mover):   上脑+下脑都高 → 计划+观察后果+调整 (领导者)
 *     感知者(perceiver): 下脑高/上脑低 → 观察理解、沉着 (军师)
 *     刺激者(stimulator): 上脑高/下脑低 → 创新计划, 难纠错 (猛张飞)
 *     适应者(adaptor):  两者都低 → 随波逐流、易适应、执行 (骨干)
 *
 * 实现:
 *   - 上脑/下脑使用度 (0-1) 随活动动态变化
 *   - 活动→系统调用: 学习/识别→下脑, 决策/探索→上脑
 *   - 四模式 = 二维空间的象限
 *   - 可培养: 教育者选择培养方向 (训练活动偏向)
 */
public class CognitiveMode {
    /** 四种认知模式 (Kosslyn 第7章) */
    public enum Mode {
        行动者("🎯 行动者模式", "上脑+下脑都高: 计划、行动、观察后果并调整 (适合领导者)", 0.8, 0.8),
        感知者("👁️ 感知者模式", "下脑高上脑低: 观察全盘、沉着冷静、见解独特 (军师)", 0.3, 0.8),
        刺激者("💡 刺激者模式", "上脑高下脑低: 跳出框架、创意无限, 但难纠错 (猛张飞)", 0.8, 0.3),
        适应者("🌊 适应者模式", "两者都低: 随波逐流、易适应、执行骨干 (关羽)", 0.3, 0.3);
        final String name; final String desc; final double upRef, lowRef;
        Mode(String n, String d, double up, double low) { this.name = n; this.desc = d; this.upRef = up; this.lowRef = low; }
    }

    /** 上脑使用度 (计划/预测/决策) 0-1 */
    private double upperBrain = 0.5;
    /** 下脑使用度 (感知/识别/记忆) 0-1 */
    private double lowerBrain = 0.5;
    /** 培养偏向: -1=纯下脑培养, 0=均衡, +1=纯上脑培养 */
    private double trainingBias = 0;

    public CognitiveMode() {}

    /** 活动: 学习新东西 (下脑: 感知分类) */
    public void learnActivity(double amount) {
        lowerBrain = clamp(lowerBrain + amount * (1 - trainingBias));
        upperBrain = clamp(upperBrain + amount * 0.2);
    }

    /** 活动: 识别/回忆 (下脑: 与记忆比较) */
    public void recognizeActivity(double amount) {
        lowerBrain = clamp(lowerBrain + amount);
    }

    /** 活动: 决策/计划 (上脑: 预测后果) */
    public void decideActivity(double amount) {
        upperBrain = clamp(upperBrain + amount * (1 + trainingBias));
        lowerBrain = clamp(lowerBrain + amount * 0.2);
    }

    /** 活动: 探索未知 (上脑计划 + 下脑感知) */
    public void exploreActivity(double amount) {
        upperBrain = clamp(upperBrain + amount);
        lowerBrain = clamp(lowerBrain + amount);
    }

    /** 睡眠 (巩固, 双系统微调) */
    public void sleepActivity() {
        upperBrain = clamp(upperBrain - 0.02);
        lowerBrain = clamp(lowerBrain - 0.02);
    }

    /** 当前认知模式 (二维象限匹配) */
    public Mode currentMode() {
        Mode best = Mode.适应者;
        double bestDist = Double.MAX_VALUE;
        for (Mode m : Mode.values()) {
            double d = Math.sqrt((upperBrain - m.upRef) * (upperBrain - m.upRef)
                    + (lowerBrain - m.lowRef) * (lowerBrain - m.lowRef));
            if (d < bestDist) { bestDist = d; best = m; }
        }
        return best;
    }

    /** 培养: 设置培养偏向 (教育者引导认知风格) */
    public void setTrainingBias(double bias) {
        this.trainingBias = Math.max(-1, Math.min(1, bias));
    }
    public double trainingBias() { return trainingBias; }
    public double upperBrain() { return upperBrain; }
    public double lowerBrain() { return lowerBrain; }

    /** 模式摘要 (APK 显示) */
    public String describe() {
        Mode m = currentMode();
        return String.format("%s (上脑%.0f%%/下脑%.0f%%)\n%s",
                m.name, upperBrain * 100, lowerBrain * 100, m.desc);
    }

    private static double clamp(double v) {
        return Math.max(0.05, Math.min(1.0, v));
    }
}
