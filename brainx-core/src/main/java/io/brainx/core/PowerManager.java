package io.brainx.core;

/**
 * 运行功率管理器 —— 根据手机性能自动调整模型功率, 避免卡死。
 *
 * 设计: 检测 → 决策 → 调参 → 反馈 自适应闭环
 *   - 检测: 帧耗时 (脑循环单步耗时) + 可用内存 + CPU 负载
 *   - 决策: 4 档功率 (节能/标准/高性能/超频), 卡顿降档, 流畅试探升档
 *   - 调参: 学习时间步数 / 中枢环路步数 / 刷新间隔 / 突触学习开关
 *   - 反馈: 新帧耗时进入下一轮决策 (防抖锁定避免振荡)
 *
 * 防卡死策略:
 *   - 帧耗时 > 阈值 → 立即降档 (保护主线程)
 *   - 可用内存 < 阈值 → 降档 (防 OOM)
 *   - 降档后锁定一段时间 (防抖动)
 *   - 长时间稳定流畅 → 试探性升档 (充分利用性能)
 */
public class PowerManager {
    /** 功率档位 */
    public enum Level {
        节能("ECO", 0),
        标准("BALANCED", 1),
        高性能("PERFORMANCE", 2),
        超频("ULTRA", 3);

        public final String tag;
        public final int rank;
        Level(String tag, int rank) { this.tag = tag; this.rank = rank; }
    }

    /** 运行配置 (每个档位的模型参数) */
    public static class Profile {
        public final Level level;
        public final int learnTimeSteps;     // 学习时间步数 (皮层模拟时长)
        public final int hubCyclesPerSync;   // 每次同步的中枢脉冲环路步数
        public final long loopDelayMs;       // 脑循环刷新间隔 (UI 帧率)
        public final boolean synapseLearning;// 突触学习开关 (重负载暂缓)

        public Profile(Level level, int steps, int cycles, long delayMs, boolean synLearn) {
            this.level = level;
            this.learnTimeSteps = steps;
            this.hubCyclesPerSync = cycles;
            this.loopDelayMs = delayMs;
            this.synapseLearning = synLearn;
        }
    }

    // 档位参数表 (低→高)
    private static final Profile[] PROFILES = {
        new Profile(Level.节能,  8,  1, 500, false),   // 节能: 少步/慢刷/关突触学习
        new Profile(Level.标准, 20,  2, 200, true),    // 标准: 默认
        new Profile(Level.高性能, 30, 4, 100, true),   // 高性能: 多步/快刷
        new Profile(Level.超频, 40,  6,  66, true)     // 超频: 全速
    };

    /** 标准档配置 (Brain 默认) */
    public static final Profile PROFILES_STANDARD = PROFILES[1];

    // 检测阈值
    private static final long STUTTER_MS = 80;      // 单帧 >80ms = 卡顿
    private static final long SLOW_MS = 50;         // 单帧 >50ms = 偏慢
    private static final long SMOOTH_MS = 25;       // 单帧 <25ms = 流畅
    private static final int  STUTTER_COUNT = 3;    // 连续卡顿次数 → 降档
    private static final int  SMOOTH_COUNT = 30;    // 连续流畅次数 → 升档试探
    private static final double MEMORY_LOW_RATIO = 0.15;  // 可用内存 <15% → 降档
    private static final long LOCK_MS = 5000;       // 降档后锁定 5s 防抖

    private Level level = Level.标准;
    private int stutterStreak = 0;
    private int smoothStreak = 0;
    private long lastChangeMs = 0;
    private long frameCount = 0;
    private long totalFrameMs = 0;
    private double avgFrameMs = 0;
    private long peakFrameMs = 0;
    private long stutters = 0;
    private int adjustCount = 0;
    private long nowMs = 0;

    /** 上报一次帧耗时 (脑循环每次迭代调用) */
    public void reportFrame(long elapsedMs, long nowMs) {
        this.nowMs = nowMs;
        frameCount++;
        totalFrameMs += elapsedMs;
        avgFrameMs = (double) totalFrameMs / frameCount;
        peakFrameMs = Math.max(peakFrameMs, elapsedMs);
        if (elapsedMs > STUTTER_MS) {
            stutters++;
            stutterStreak++;
            smoothStreak = 0;
        } else if (elapsedMs < SMOOTH_MS) {
            smoothStreak++;
            stutterStreak = 0;
        } else {
            stutterStreak = 0;
            smoothStreak = 0;
        }
    }

    /** 上报内存状态 (可用/总量 MB) */
    public void reportMemory(long freeMb, long totalMb, long nowMs) {
        this.nowMs = nowMs;
        double ratio = totalMb > 0 ? (double) freeMb / totalMb : 1.0;
        if (ratio < MEMORY_LOW_RATIO && level.rank > 0) {
            changeLevel(Level.values()[level.rank - 1], nowMs, "内存不足(" + freeMb + "MB)");
        }
    }

    /** 评估并调整档位 (周期性调用, 如每 2s) */
    public void evaluate(long nowMs) {
        this.nowMs = nowMs;
        // 防抖: 刚降档锁定期间不动作
        if (nowMs - lastChangeMs < LOCK_MS) return;

        if (stutterStreak >= STUTTER_COUNT) {
            // 连续卡顿 → 降档 (防卡死核心)
            if (level.rank > 0) {
                changeLevel(Level.values()[level.rank - 1], nowMs, "连续卡顿(" + stutterStreak + "帧)");
            }
            stutterStreak = 0;
        } else if (smoothStreak >= SMOOTH_COUNT) {
            // 长时间流畅 → 试探升档
            if (level.rank < Level.超频.rank) {
                changeLevel(Level.values()[level.rank + 1], nowMs, "持续流畅(" + smoothStreak + "帧)");
            }
            smoothStreak = 0;
        }
    }

    private void changeLevel(Level newLevel, long nowMs, String reason) {
        level = newLevel;
        lastChangeMs = nowMs;
        adjustCount++;
        // 换档后重置统计
        stutterStreak = 0;
        smoothStreak = 0;
    }

    /** 当前档位 */
    public Level level() { return level; }

    /** 当前档位配置 */
    public Profile profile() { return PROFILES[level.rank]; }

    /** 强制设置档位 (手动/初始) */
    public void setLevel(Level l, long nowMs) {
        this.nowMs = nowMs;
        level = l;
        lastChangeMs = nowMs;
        stutterStreak = 0;
        smoothStreak = 0;
    }

    // 统计
    public long frameCount() { return frameCount; }
    public double avgFrameMs() { return avgFrameMs; }
    public long peakFrameMs() { return peakFrameMs; }
    public long stutters() { return stutters; }
    public int adjustCount() { return adjustCount; }

    /** 摘要 (APK 显示) */
    public String summary() {
        return String.format("📊 功率: %s | 平均帧%.0fms | 峰值%dms | 卡顿%d次 | 调档%d次",
                level.tag, avgFrameMs, peakFrameMs, stutters, adjustCount);
    }
}
