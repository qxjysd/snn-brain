package io.brainx.app;

/**
 * 语言工具 — 中英文切换 (全局静态)。
 * 界面文字精简: clip() 截断长摘要, t() 双语返回。
 */
public final class Lang {
    private static boolean en = false;

    private Lang() {}

    /** 切换语言 (zh ↔ en) */
    public static void toggle() { en = !en; }

    /** 当前是否英文 */
    public static boolean isEn() { return en; }

    /** 双语返回: t("中文", "English") */
    public static String t(String zh, String enText) { return en ? enText : zh; }

    /** 截断长文字 (精简显示) */
    public static String clip(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max) + "…";
    }

    /** 语言标签 */
    public static String label() { return en ? "🌐 EN" : "🌐 中"; }
}
