package io.brainx.core;

/**
 * 意识模型 —— 全局工作空间理论 (Global Workspace Theory, GWT)。
 *
 * 理论依据:
 *   - Baars (1988) / Dehaene & Naccache (2001) GWT:
 *     意识 = 信息被"广播"到全局工作空间 (global workspace),
 *     供所有模块访问 —— 未广播的信息处于无意识加工
 *   - 竞争机制: 各感觉/认知模块争夺工作空间访问权 (斗争→胜者广播)
 *   - 注意力门控: 丘脑-皮层回路决定什么进入意识 (对应 fWBM 的 thalamic gating)
 *   - 整合信息理论 (IIT, Tononi): 意识程度 = 系统整合信息量
 *
 * 实现:
 *   - 意识状态: 无意识/潜意识/意识/专注 (四级)
 *   - 工作空间广播: 输入信号强度竞争, 最强胜出进入意识
 *   - 整合度 Φ (简化): 多模态一致性 (视觉+听觉同物体 → 高整合)
 *   - 睡眠时: 意识关闭 (外部输入无法广播), 记忆重放主导
 */
public class Consciousness implements PulseModule {
    /** 意识状态 */
    public enum State {
        无意识("😴", 0.0),   // 深度睡眠: 无外部感知
        潜意识("🌙", 0.3),   // 浅睡/默认: 弱处理
        意识("👁️", 0.7),    // 清醒: 感知广播
        专注("🎯", 1.0);     // 注意力集中: 单点深度处理
        final String emoji; final double level;
        State(String e, double l) { this.emoji = e; this.level = l; }
    }

    private State state = State.意识;
    /** 当前广播内容 (意识内容) */
    private String broadcast = "";
    /** 广播强度 */
    private double broadcastStrength = 0;
    /** 广播频率 (意识内容=主导频率: γ=绑定内容) */
    private double broadcastHz = 40.0;  // 默认 γ (意识内容绑定)
    /** 整合信息度 Φ (0-1): 多模态一致度 */
    private double phi = 0;
    /** 注意力焦点 (丘脑门控: 0=发散, 1=高度聚焦) */
    private double attention = 0.5;

    public Consciousness() {}

    /**
     * 每步感知处理: 各模态输入竞争进入意识。
     * @param visualStrength  视觉输入强度
     * @param auditoryStrength 听觉输入强度
     * @param visualLabel     视觉识别标签
     * @param auditoryLabel   听觉识别标签
     */
    public void perceive(double visualStrength, double auditoryStrength,
                         String visualLabel, String auditoryLabel) {
        if (state == State.无意识) return;  // 睡眠中无外部广播

        // 竞争: 注意力调制的加权强度
        double vWeighted = visualStrength * (1.0 - attention * 0.5);
        double aWeighted = auditoryStrength * (0.5 + attention * 0.5);

        if (vWeighted >= aWeighted && vWeighted > 0.2) {
            broadcast = visualLabel;
            broadcastStrength = vWeighted;
        } else if (aWeighted > 0.2) {
            broadcast = auditoryLabel;
            broadcastStrength = aWeighted;
        } else {
            broadcastStrength *= 0.9;  // 弱输入衰减
        }

        // 整合信息度: 视觉+听觉识别一致 → 高 Φ (多感觉整合)
        if (!visualLabel.isEmpty() && visualLabel.equals(auditoryLabel)) {
            phi = Math.min(1.0, phi + 0.1);
        } else {
            phi = Math.max(0.0, phi - 0.02);
        }

        // 意识状态动态
        if (attention > 0.8) state = State.专注;
        else if (broadcastStrength > 0.5) state = State.意识;
        else state = State.潜意识;
    }

    /** 睡眠: 意识关闭, 外部输入无法广播 */
    public void sleep() { state = State.无意识; }
    /** 唤醒 */
    public void wake() { state = State.意识; }

    public State state() { return state; }
    public String stateEmoji() { return state.emoji; }
    public String broadcast() { return broadcast; }
    public double broadcastStrength() { return broadcastStrength; }
    public double phi() { return phi; }
    public double attention() { return attention; }
    public void setAttention(double a) { this.attention = Math.max(0, Math.min(1, a)); }

    /** 广播频率 (意识输出到频率总线: 内容绑定频率) */
    public double broadcastHz() { return broadcastHz; }
    /** 接收外部频率注入 (频率总线→意识: 如记忆检索θ/注意γ) */
    public void receiveFrequency(double hz, double strength) {
        if (state == State.无意识) return;  // 睡眠不接收
        broadcastHz = hz;
        if (strength > broadcastStrength) broadcastStrength = strength;
        // 频率状态: θ=记忆检索, γ=注意绑定 (仅增强, 不降级已唤醒状态)
        if (state != State.意识 && state != State.专注) {
            if (hz < 13) state = State.潜意识;
        }
        if (attention > 0.7) state = State.专注;
        else if (hz >= 30) state = State.意识;
    }

    /**
     * 全局爆发广播 (书中: global ignition 传播至所有远隔位置)。
     * 活动跨阈值 → 意识内容扩散到全局 → 广播强度/Φ 增强。
     */
    public void broadcastIgnition(double strength) {
        if (state == State.无意识) return;
        broadcastStrength = Math.min(1.0, broadcastStrength + strength * 0.3);
        phi = Math.min(1.0, phi + strength * 0.2);
        broadcastHz = 40.0 + strength * 20;  // γ 带 (意识绑定强化)
        state = State.意识;
    }

    /** 意识内容摘要 (APK 显示) */
    public String describe() {
        if (state == State.无意识) return "💤 无意识 (睡眠中, 记忆重放中)";
        String content = broadcast.isEmpty() ? "空白" : broadcast;
        return String.format("%s %s | 意识内容: %s | 整合度Φ: %.2f",
                state.emoji, state.name(), content, phi);
    }

    // ============ PulseModule: 中枢脉冲联动 ============

    @Override public String moduleName() { return "意识"; }
    @Override public int pulseDim() { return 2; }  // [广播强度, Φ整合度]

    /** 发射脉冲: [广播强度, 整合度Φ] (意识内容→中枢) */
    @Override
    public double[] emitPulses() {
        return new double[]{broadcastStrength, phi};
    }

    /** 接收中枢广播: 中枢活跃 → 意识广播增强 (全局整合提升意识) */
    @Override
    public boolean receiveBroadcast(double[] broadcastRates) {
        if (broadcastRates.length == 0) return false;
        double avg = 0;
        for (double r : broadcastRates) avg += r;
        avg /= broadcastRates.length;
        if (avg > 0.3 && state != State.无意识) {
            broadcastStrength = Math.min(1.0, broadcastStrength + avg * 0.05);
            phi = Math.min(1.0, phi + avg * 0.02);  // 全局整合提升Φ
            return true;
        }
        return false;
    }
}
